"""
record_outcomes.py
------------------
Fetches actual game outcomes from MLB Stats API for every game where we
logged a prediction, then writes the truth to game_outcomes / player_outcomes.

Designed to run unattended via GitHub Actions on a daily schedule. Standalone
script — does NOT import from app.py or any Streamlit code, because Streamlit
will not be available in the GitHub Actions runner.

Idempotent: re-running on the same day skips already-resolved predictions.
Safe to run multiple times (e.g., a morning pass picking up overnight games,
plus a retry for any postponed games that finished later).

Usage
-----
    # Resolve everything from yesterday (default)
    python record_outcomes.py

    # Resolve a specific date
    python record_outcomes.py --date 2026-04-15

    # Resolve a date range (inclusive)
    python record_outcomes.py --start 2026-04-01 --end 2026-04-15

    # Retry pending/postponed only (don't re-fetch finalized games)
    python record_outcomes.py --retry-pending

Environment
-----------
    DATABASE_URL  Required. SQLAlchemy connection string for Supabase.
                  Same value as the Streamlit secret, but loaded from env.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_TIMEOUT = 30          # seconds; box scores are small JSON, but be patient
HTTP_RETRIES = 3           # MLB API occasionally blips, retry a few times
HTTP_RETRY_BACKOFF = 2.0   # seconds; doubles each attempt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("record_outcomes")


# ---------------------------------------------------------------------------
#  HTTP helpers
# ---------------------------------------------------------------------------
def _get_json(url: str, *, params: dict | None = None) -> dict | None:
    """
    GET with retry. Returns None on persistent failure rather than raising —
    the caller decides how to handle a single missing game (skip, mark pending).
    """
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            log.warning("GET %s -> HTTP %d (attempt %d)", url, resp.status_code, attempt)
        except requests.RequestException as e:
            log.warning("GET %s failed: %s (attempt %d)", url, e, attempt)
        if attempt < HTTP_RETRIES:
            time.sleep(HTTP_RETRY_BACKOFF ** attempt)
    return None


# ---------------------------------------------------------------------------
#  MLB Stats API parsing
# ---------------------------------------------------------------------------
def fetch_boxscore(game_id: str) -> dict | None:
    """
    Pull the live feed for a game. The /live endpoint is more verbose than
    /boxscore but gives us first-inning play-by-play for NRFI resolution
    in one request, plus the full boxscore.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    return _get_json(url)


def _game_status(feed: dict) -> str:
    """
    MLB statusCode values we care about:
        F   = Final
        FT  = Final, tiebreaker (still resolved)
        FR  = Final, rain-shortened (still resolved)
        DI  = Delayed, rain
        PO/PR = Postponed
        S   = Scheduled
        I/IH/IR = In progress / mid-inning
        CR  = Cancelled
        SU  = Suspended

    Map to our coarse status enum used in DB.
    """
    code = (feed.get("gameData", {})
                 .get("status", {})
                 .get("statusCode", "")
                 .upper())
    if code in {"F", "FT", "FR", "FO"}:
        return "final"
    if code.startswith("PO") or code == "PR":
        return "postponed"
    if code == "SU":
        return "suspended"
    if code == "CR":
        return "cancelled"
    return "pending"  # in-progress, scheduled, delayed — try again later


def _team_runs_through_inning(plays: list, half: str, max_inning: int) -> int:
    """
    Sum runs scored by `half` ('top' for away, 'bottom' for home) through
    the end of `max_inning`. Used for F5 score and NRFI resolution.

    We sum from result.awayScore / result.homeScore at the end of each
    qualifying half-inning's last play to avoid double-counting.
    """
    total = 0
    last_seen = 0
    for play in plays:
        about = play.get("about") or {}
        inning = about.get("inning")
        half_inning = (about.get("halfInning") or "").lower()
        if inning is None or inning > max_inning:
            continue
        if half_inning != half:
            continue
        # Each play has the running score AFTER the play resolves.
        result = play.get("result") or {}
        score_key = "awayScore" if half == "top" else "homeScore"
        running = result.get(score_key)
        if running is not None:
            last_seen = running
    total = last_seen
    return total


def _runs_in_first_inning(plays: list) -> int:
    """NRFI = runs in 1st inning (both halves combined)."""
    runs = 0
    for play in plays:
        about = play.get("about") or {}
        if about.get("inning") != 1:
            continue
        result = play.get("result") or {}
        # rbi is the cleanest indicator of runs scored ON this play; the
        # running score deltas would be more robust but rbi is sufficient
        # for the standard NRFI definition (any earned/unearned run counts).
        runs += int(result.get("rbi") or 0)
    return runs


def parse_game_outcome(feed: dict, prediction_row: dict) -> dict:
    """
    Build a game_outcomes row from a live feed payload. prediction_row carries
    the IDs we need for joining (game_prediction_id, starter IDs).

    Returns a dict ready to INSERT.
    """
    status = _game_status(feed)
    base = {
        "game_prediction_id": prediction_row["id"],
        "game_id":            prediction_row["game_id"],
        "game_date":          prediction_row["game_date"],
        "status":             status,
        "away_score":         None,
        "home_score":         None,
        "away_won":           None,
        "f5_away_score":      None,
        "f5_home_score":      None,
        "f5_away_won":        None,
        "f5_tied":            None,
        "nrfi_hit":           None,
        "total_runs":         None,
        "away_starter_actual_k":  None,
        "away_starter_pitched":   None,
        "home_starter_actual_k":  None,
        "home_starter_pitched":   None,
    }
    if status != "final":
        return base

    live   = feed.get("liveData", {}) or {}
    plays  = (live.get("plays") or {}).get("allPlays") or []
    line   = live.get("linescore") or {}
    teams  = (live.get("boxscore") or {}).get("teams") or {}

    # Final score lives on the linescore.teams object ----------------------
    ls_away = (line.get("teams") or {}).get("away", {}) or {}
    ls_home = (line.get("teams") or {}).get("home", {}) or {}
    away_score = ls_away.get("runs")
    home_score = ls_home.get("runs")

    # F5 — sum runs through inning 5 from play-by-play --------------------
    f5_away = _team_runs_through_inning(plays, "top", 5)
    f5_home = _team_runs_through_inning(plays, "bottom", 5)

    base.update({
        "away_score":    away_score,
        "home_score":    home_score,
        "away_won":      (away_score > home_score) if (away_score is not None and home_score is not None) else None,
        "f5_away_score": f5_away,
        "f5_home_score": f5_home,
        "f5_away_won":   f5_away > f5_home,
        "f5_tied":       f5_away == f5_home,
        "nrfi_hit":      _runs_in_first_inning(plays) == 0,
        "total_runs":    (away_score + home_score) if (away_score is not None and home_score is not None) else None,
    })

    # Pitcher Ks — find the starters by ID in each team's pitcher list ----
    base["away_starter_actual_k"], base["away_starter_pitched"] = \
        _starter_strikeouts(teams.get("away") or {}, prediction_row.get("away_starter_id"))
    base["home_starter_actual_k"], base["home_starter_pitched"] = \
        _starter_strikeouts(teams.get("home") or {}, prediction_row.get("home_starter_id"))

    return base


def _starter_strikeouts(team_box: dict, starter_id: int | None) -> tuple[int | None, bool | None]:
    """
    Return (strikeouts, pitched) for the listed starter. If the starter never
    appeared (scratched, traded, etc.), pitched=False and props should be
    treated as voided — NOT zero. Distinguishing "0 Ks" from "did not pitch"
    matters for calibration.
    """
    if starter_id is None:
        return (None, None)

    players = team_box.get("players") or {}
    key = f"ID{starter_id}"
    p = players.get(key)
    if not p:
        return (None, False)

    pitching = (p.get("stats") or {}).get("pitching") or {}
    if not pitching:
        # Listed on roster but didn't pitch
        return (None, False)

    # 'strikeOuts' is MLB's key; defensive default to 0 if pitched but key absent.
    return (int(pitching.get("strikeOuts") or 0), True)


def parse_player_outcomes(feed: dict, predictions: list[dict]) -> list[dict]:
    """
    Build player_outcomes rows for every batter we logged a prediction for.

    Joins on player_id when present (most reliable), falls back to name match
    within the correct team's roster (unreliable for shared names — log a
    warning). If a logged player never appeared in the box, status='did_not_play'.
    """
    status = _game_status(feed)
    if status != "final":
        return [
            {**_blank_player_row(p), "status": status if status != "final" else "pending"}
            for p in predictions
        ]

    teams = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    away_box = teams.get("away") or {}
    home_box = teams.get("home") or {}

    rows = []
    for p in predictions:
        team_box = away_box if p["side"] == "away" else home_box
        stats = _find_batter_stats(team_box, p)
        row = _blank_player_row(p)

        if stats is None:
            row["status"] = "did_not_play"
            rows.append(row)
            continue

        bat = stats.get("batting") or {}
        run = stats.get("baseRunning") or {}

        pa  = int(bat.get("plateAppearances") or 0)
        ab  = int(bat.get("atBats") or 0)
        h   = int(bat.get("hits") or 0)
        d   = int(bat.get("doubles") or 0)
        t   = int(bat.get("triples") or 0)
        hr  = int(bat.get("homeRuns") or 0)
        rbi = int(bat.get("rbi") or 0)
        sb  = int(run.get("stolenBases") or bat.get("stolenBases") or 0)
        singles = h - d - t - hr
        tb = singles + 2 * d + 3 * t + 4 * hr

        row.update({
            "status":            "final",
            "plate_appearances": pa,
            "at_bats":           ab,
            "hits":              h,
            "doubles":           d,
            "triples":           t,
            "home_runs":         hr,
            "rbi":               rbi,
            "stolen_bases":      sb,
            "total_bases":       tb,
            "hit_1h":            h   >= 1,
            "hit_2tb":           tb  >= 2,
            "hit_1hr":           hr  >= 1,
            "hit_1rbi":          rbi >= 1,
            "hit_1sb":           sb  >= 1,
        })
        rows.append(row)

    return rows


def _blank_player_row(prediction: dict) -> dict:
    """Skeleton with all NULLs except identity fields."""
    return {
        "player_prediction_id": prediction["id"],
        "game_id":              prediction["game_id"],
        "game_date":            prediction["game_date"],
        "player_id":            prediction.get("player_id"),
        "player_name":          prediction["player_name"],
        "status":               "pending",
        "plate_appearances":    None, "at_bats": None, "hits": None,
        "doubles": None, "triples": None, "home_runs": None,
        "rbi": None, "stolen_bases": None, "total_bases": None,
        "hit_1h": None, "hit_2tb": None, "hit_1hr": None,
        "hit_1rbi": None, "hit_1sb": None,
    }


def _find_batter_stats(team_box: dict, prediction: dict) -> dict | None:
    """
    Locate a batter's stats inside team_box['players']. Prefer player_id;
    fall back to a normalized-name match. Returns the raw 'stats' dict or
    None if not found in the box.
    """
    players = team_box.get("players") or {}

    pid = prediction.get("player_id")
    if pid is not None:
        p = players.get(f"ID{pid}")
        if p:
            return p.get("stats")

    # Name fallback — last resort, unreliable for shared names.
    target = (prediction.get("player_name") or "").strip().lower()
    for p in players.values():
        person = p.get("person") or {}
        name = (person.get("fullName") or "").strip().lower()
        if name and name == target:
            log.warning("Player matched by name fallback: %s (no ID match)", target)
            return p.get("stats")
    return None


# ---------------------------------------------------------------------------
#  Database I/O
# ---------------------------------------------------------------------------
def _engine() -> Engine:
    url = os.environ.get("DATABASE_URL")
    if not url:
        log.error("DATABASE_URL environment variable is not set.")
        sys.exit(2)
    return create_engine(url, pool_pre_ping=True)


def fetch_unresolved_predictions(
    engine: Engine,
    *,
    target_dates: list[date] | None,
    retry_pending: bool,
) -> list[dict]:
    """
    Returns game_predictions rows that don't yet have a 'final' game_outcomes
    row. If retry_pending=True, also includes ones whose existing outcome is
    pending/postponed/suspended (in case the game finished after our last pass).
    """
    where = []
    params: dict[str, Any] = {}

    if target_dates:
        where.append("gp.game_date = ANY(:dates)")
        params["dates"] = target_dates

    if retry_pending:
        outcome_filter = """
            (go.id IS NULL
             OR go.status IN ('pending', 'postponed', 'suspended'))
        """
    else:
        outcome_filter = "go.id IS NULL"

    sql = f"""
        SELECT gp.id, gp.game_id, gp.game_date,
               gp.away_starter_id, gp.home_starter_id
        FROM game_predictions gp
        LEFT JOIN game_outcomes go ON go.game_prediction_id = gp.id
        WHERE {outcome_filter}
          {('AND ' + ' AND '.join(where)) if where else ''}
        ORDER BY gp.game_date, gp.game_id
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return [dict(r._mapping) for r in result]


def fetch_player_predictions_for_game(engine: Engine, game_prediction_id: int) -> list[dict]:
    sql = """
        SELECT pp.id, pp.game_id, pp.game_date,
               pp.player_id, pp.player_name, pp.side
        FROM player_predictions pp
        LEFT JOIN player_outcomes po ON po.player_prediction_id = pp.id
        WHERE pp.game_prediction_id = :gpid
          AND po.id IS NULL
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"gpid": game_prediction_id})
        return [dict(r._mapping) for r in result]


def upsert_game_outcome(engine: Engine, row: dict) -> None:
    """
    Insert or update by game_prediction_id (UNIQUE constraint enforces 1:1).
    Update lets us refresh a previously-pending row once the game finalizes.
    """
    sql = text("""
        INSERT INTO game_outcomes (
            game_prediction_id, game_id, game_date, status,
            away_score, home_score, away_won,
            f5_away_score, f5_home_score, f5_away_won, f5_tied,
            nrfi_hit, total_runs,
            away_starter_actual_k, away_starter_pitched,
            home_starter_actual_k, home_starter_pitched
        ) VALUES (
            :game_prediction_id, :game_id, :game_date, :status,
            :away_score, :home_score, :away_won,
            :f5_away_score, :f5_home_score, :f5_away_won, :f5_tied,
            :nrfi_hit, :total_runs,
            :away_starter_actual_k, :away_starter_pitched,
            :home_starter_actual_k, :home_starter_pitched
        )
        ON CONFLICT (game_prediction_id) DO UPDATE SET
            recorded_at           = NOW(),
            status                = EXCLUDED.status,
            away_score            = EXCLUDED.away_score,
            home_score            = EXCLUDED.home_score,
            away_won              = EXCLUDED.away_won,
            f5_away_score         = EXCLUDED.f5_away_score,
            f5_home_score         = EXCLUDED.f5_home_score,
            f5_away_won           = EXCLUDED.f5_away_won,
            f5_tied               = EXCLUDED.f5_tied,
            nrfi_hit              = EXCLUDED.nrfi_hit,
            total_runs            = EXCLUDED.total_runs,
            away_starter_actual_k = EXCLUDED.away_starter_actual_k,
            away_starter_pitched  = EXCLUDED.away_starter_pitched,
            home_starter_actual_k = EXCLUDED.home_starter_actual_k,
            home_starter_pitched  = EXCLUDED.home_starter_pitched
    """)
    with engine.begin() as conn:
        conn.execute(sql, row)


def upsert_player_outcomes(engine: Engine, rows: list[dict]) -> None:
    if not rows:
        return
    sql = text("""
        INSERT INTO player_outcomes (
            player_prediction_id, game_id, game_date,
            player_id, player_name, status,
            plate_appearances, at_bats, hits,
            doubles, triples, home_runs, rbi, stolen_bases, total_bases,
            hit_1h, hit_2tb, hit_1hr, hit_1rbi, hit_1sb
        ) VALUES (
            :player_prediction_id, :game_id, :game_date,
            :player_id, :player_name, :status,
            :plate_appearances, :at_bats, :hits,
            :doubles, :triples, :home_runs, :rbi, :stolen_bases, :total_bases,
            :hit_1h, :hit_2tb, :hit_1hr, :hit_1rbi, :hit_1sb
        )
        ON CONFLICT (player_prediction_id) DO UPDATE SET
            recorded_at       = NOW(),
            status            = EXCLUDED.status,
            plate_appearances = EXCLUDED.plate_appearances,
            at_bats           = EXCLUDED.at_bats,
            hits              = EXCLUDED.hits,
            doubles           = EXCLUDED.doubles,
            triples           = EXCLUDED.triples,
            home_runs         = EXCLUDED.home_runs,
            rbi               = EXCLUDED.rbi,
            stolen_bases      = EXCLUDED.stolen_bases,
            total_bases       = EXCLUDED.total_bases,
            hit_1h            = EXCLUDED.hit_1h,
            hit_2tb           = EXCLUDED.hit_2tb,
            hit_1hr           = EXCLUDED.hit_1hr,
            hit_1rbi          = EXCLUDED.hit_1rbi,
            hit_1sb           = EXCLUDED.hit_1sb
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------
def resolve_predictions(target_dates: list[date] | None, retry_pending: bool) -> dict:
    engine = _engine()
    log.info("Looking for unresolved predictions (dates=%s, retry_pending=%s)",
             target_dates, retry_pending)

    games = fetch_unresolved_predictions(
        engine,
        target_dates=target_dates,
        retry_pending=retry_pending,
    )
    log.info("Found %d unresolved game prediction(s).", len(games))

    counters = {"resolved": 0, "pending": 0, "failed": 0, "players": 0}

    for g in games:
        game_id = g["game_id"]
        feed = fetch_boxscore(game_id)
        if feed is None:
            log.error("Could not fetch feed for game_id=%s, skipping", game_id)
            counters["failed"] += 1
            continue

        game_row = parse_game_outcome(feed, g)
        upsert_game_outcome(engine, game_row)

        player_preds = fetch_player_predictions_for_game(engine, g["id"])
        player_rows = parse_player_outcomes(feed, player_preds)
        upsert_player_outcomes(engine, player_rows)

        if game_row["status"] == "final":
            counters["resolved"] += 1
            counters["players"]  += sum(1 for r in player_rows if r["status"] == "final")
            log.info("RESOLVED game_id=%s (%d→%d, %d players)",
                     game_id, game_row["away_score"] or 0,
                     game_row["home_score"] or 0,
                     sum(1 for r in player_rows if r["status"] == "final"))
        else:
            counters["pending"] += 1
            log.info("PENDING  game_id=%s status=%s", game_id, game_row["status"])

    log.info("Done. resolved=%d pending=%d failed=%d players=%d",
             counters["resolved"], counters["pending"],
             counters["failed"],   counters["players"])
    return counters


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve MLB prediction outcomes.")
    p.add_argument("--date",  type=date.fromisoformat,
                   help="Single date YYYY-MM-DD (default: yesterday).")
    p.add_argument("--start", type=date.fromisoformat, help="Range start (inclusive).")
    p.add_argument("--end",   type=date.fromisoformat, help="Range end (inclusive).")
    p.add_argument("--retry-pending", action="store_true",
                   help="Also re-fetch outcomes currently in pending/postponed status.")
    p.add_argument("--all-unresolved", action="store_true",
                   help="Resolve every unresolved prediction regardless of date.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.all_unresolved:
        target_dates = None
    elif args.start and args.end:
        days = (args.end - args.start).days
        if days < 0:
            log.error("--end must be on/after --start")
            return 2
        target_dates = [args.start + timedelta(days=i) for i in range(days + 1)]
    elif args.date:
        target_dates = [args.date]
    else:
        target_dates = [date.today() - timedelta(days=1)]

    counters = resolve_predictions(target_dates, args.retry_pending)
    # Exit non-zero if we had API failures so GitHub Actions surfaces a red X.
    return 1 if counters["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())