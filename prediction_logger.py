"""
prediction_logger.py
--------------------
Persists every Monte Carlo prediction the MLB engine generates to Supabase
Postgres so we can later compare predictions to actual outcomes, measure
calibration, and apply correction models (isotonic regression in Phase 4).

One log_prediction() call writes:
    - 1 row in game_predictions   (game markets + pitcher Ks + weather audit)
    - N rows in player_predictions (one per batter in each lineup)

Idempotency
-----------
log_prediction() is a "last-write-wins" upsert keyed on
(game_id, game_date, model_version). Re-running the slate — whether from a
Streamlit reclick or a cron retry — replaces any prior prediction for the
same game/day/version rather than piling up duplicates. The most recent
prediction (closest to first pitch, with confirmed lineups and the tightest
weather forecast) is the one that survives. The delete-and-reinsert runs in
a single SQL transaction, so a crash mid-write leaves the DB in its prior
state, not partially erased.

Designed to never crash the UI — the caller wraps this in try/except so a
transient DB hiccup degrades to a soft toast warning instead of a stack
trace.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from typing import Any

import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Bump this when the simulator or resolver changes in a way that meaningfully
# shifts predictions. Lets calibration analysis segment results by version so
# we don't mix pre- and post-change predictions in the same reliability curve.
MODEL_VERSION = "v1"

# K thresholds we report on. Must match what's rendered in app.py so the
# logged probability == the displayed probability (no drift between UI & DB).
K_THRESHOLDS = (3, 4, 5, 6, 7)


# ---------------------------------------------------------------------------
#  Connection (works in both Streamlit and headless / GitHub Actions contexts)
# ---------------------------------------------------------------------------
def _running_under_streamlit() -> bool:
    """
    Detect whether we're inside a real Streamlit script context.

    `import streamlit as st` works anywhere — but `st.connection` and
    related plumbing only work when there's an active ScriptRunContext.
    Detecting that lets prediction_logger gracefully switch between:
      • Streamlit runtime → uses st.connection (reads secrets.toml)
      • Cron / scripts    → uses raw SQLAlchemy from DATABASE_URL env var

    Both code paths return an object whose `.session` is a context-manager
    yielding a SQLAlchemy session, so the rest of log_prediction() doesn't
    care which one we got.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


class _HeadlessSQLConn:
    """
    Minimal stand-in for Streamlit's SQLConnection in non-Streamlit
    contexts (GitHub Actions, local scripts, Jupyter notebooks).

    Exposes only the surface area prediction_logger uses: a `.session`
    property that returns a SQLAlchemy session usable as a context
    manager. The session supports `s.execute(...)`, `s.commit()`, and
    auto-rollback on exception — same as Streamlit's wrapper.
    """
    def __init__(self, url: str):
        self._engine = create_engine(url, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)

    @property
    def session(self):
        # SQLAlchemy ORM sessions support __enter__/__exit__ since 1.4:
        # exit auto-rollbacks if an exception occurred, otherwise leaves
        # the session intact (we still call s.commit() ourselves).
        return self._Session()


@st.cache_resource(show_spinner=False)
def _get_conn():
    """
    Returns a SQL connection for prediction logging.

    In Streamlit: returns st.connection("predictions_db", type="sql"),
    which reads [connections.predictions_db] from secrets.toml — same as
    before. The @st.cache_resource decorator memoizes per Streamlit
    runtime so we don't rebuild on every rerun.

    In headless mode: returns _HeadlessSQLConn built from the
    DATABASE_URL environment variable. The cache_resource decorator is
    a no-op outside Streamlit (it just returns the wrapped function's
    result), so each cron invocation gets a fresh connection — fine for
    a short-lived process.
    """
    if _running_under_streamlit():
        return st.connection("predictions_db", type="sql")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Required when running prediction_logger outside Streamlit "
            "(e.g. from GitHub Actions cron jobs)."
        )
    return _HeadlessSQLConn(db_url)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _k_threshold_probs(k_dist: list, median_k: float) -> dict[int, float]:
    """
    Mirror of the rendering logic in app.py: empirical CDF when we have a
    full K distribution, Poisson fallback on the median otherwise. We log
    exactly what the user sees on screen.
    """
    if k_dist:
        arr = np.asarray(k_dist)
        return {t: float(np.mean(arr >= t)) for t in K_THRESHOLDS}

    lam = max(float(median_k or 0.0), 0.1)
    out = {}
    for t in K_THRESHOLDS:
        cdf = sum((math.exp(-lam) * lam**k) / math.factorial(k) for k in range(t))
        out[t] = max(1.0 - cdf, 0.0)
    return out


def _player_id(player: dict) -> int | None:
    """
    Be defensive — different roster scrapers stash the player ID under
    different keys. Try the common ones, return None if absent.
    """
    for k in ("id", "player_id", "mlb_id", "personId"):
        v = player.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _parse_game_date(game_datetime: str | None, fallback: str) -> date:
    """
    MLB Stats API returns ISO 8601 like '2025-04-15T23:05:00Z'. We just want
    the date. Falls back to as_of_date, then today's date, before giving up.
    """
    if game_datetime:
        try:
            cleaned = game_datetime.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).date()
        except (ValueError, AttributeError):
            pass
    try:
        return date.fromisoformat(fallback)
    except (ValueError, AttributeError):
        return date.today()


def _parse_game_datetime(game_datetime: str | None) -> datetime | None:
    """
    Parse the MLB Stats API ISO 8601 first-pitch timestamp into a
    timezone-aware UTC datetime. SQLAlchemy/psycopg2 binds aware datetimes
    cleanly to TIMESTAMPTZ. Returns None on missing or unparseable input so
    the column lands as NULL rather than aborting the whole insert — the
    read-only viewer's _to_central() helper renders missing times as a
    blank header anyway.
    """
    if not game_datetime:
        return None
    try:
        cleaned = game_datetime.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, AttributeError):
        return None


def _native(v: Any) -> Any:
    """Convert numpy scalars to native Python so SQL parameter binding is clean."""
    if v is None:
        return None
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _native_list(seq) -> list:
    """Coerce every element of a sequence to native Python types. Used
    before json.dumps on K-distribution arrays so numpy ints don't crash
    serialization."""
    return [_native(v) for v in seq]


def _native_dict(d) -> dict | None:
    """Coerce every value of a dict to native Python. None-safe — passes
    None straight through, which json.dumps serializes as `null`. Keys are
    assumed to be strings already (which they are for first-inning stats)."""
    if d is None:
        return None
    return {k: _native(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def log_prediction(
    *,
    game: dict,
    away: dict,
    home: dict,
    results: dict,
    iterations: int,
    as_of_date: str,
    weather: dict | None = None,
) -> int:
    """
    Persist one full game prediction.

    All parameters mirror variables already in scope at the call site in
    app.py — no extra plumbing required. Keyword-only to prevent argument-
    order bugs when the call site evolves.

    Returns
    -------
    int
        The id of the inserted game_predictions row. Useful for testing and
        for chaining outcome ingestion later.
    """
    away_k_probs = _k_threshold_probs(
        results.get("away_k_dist", []),
        results.get("away_pitcher_median_k", 0),
    )
    home_k_probs = _k_threshold_probs(
        results.get("home_k_dist", []),
        results.get("home_pitcher_median_k", 0),
    )

    away_win = float(results.get("away_win_prob", 0.0))
    f5_away  = float(results.get("f5_away_win_prob", 0.0))
    f5_tie   = float(results.get("f5_tie_prob", 0.0))

    game_row = {
        "game_id":           str(game.get("game_id", "")),
        "game_date":         _parse_game_date(game.get("game_datetime"), as_of_date),
        "away_team":         away.get("team_name"),
        "home_team":         home.get("team_name"),
        "away_starter_id":   _native(away.get("starter_id")),
        "away_starter_name": away.get("starter_name"),
        "home_starter_id":   _native(home.get("starter_id")),
        "home_starter_name": home.get("starter_name"),

        "iterations":        int(iterations),
        "as_of_date":        as_of_date,
        "model_version":     MODEL_VERSION,

        # weather snapshot (None for missing or dome where appropriate) ------
        "weather_park":           _native((weather or {}).get("park")),
        "weather_is_dome":        _native((weather or {}).get("is_dome")),
        "weather_temp_f":         _native((weather or {}).get("temp_f")),
        "weather_wind_mph":       _native((weather or {}).get("wind_speed_mph")),
        "weather_carry_delta_ft": _native((weather or {}).get("carry_delta_ft")),
        "weather_hr_score":       _native((weather or {}).get("score")),

        # weather snapshot — Priority 3 additions ----------------------------
        # These columns let the read-only viewer render the full HR-conditions
        # readout (humidity, pressure, wind direction + CF axis component +
        # tier label) without re-running the simulator. Same defensive
        # pattern as the original weather block: `(weather or {}).get(...)`
        # so a None weather dict (e.g. weather API timeout) doesn't crash
        # the write — the columns just land as NULL and the viewer's
        # hotfixed render path shows '—' chips.
        "weather_humidity_pct":      _native((weather or {}).get("humidity_pct")),
        "weather_pressure_inhg":     _native((weather or {}).get("pressure_inhg")),
        "weather_wind_from_compass": _native((weather or {}).get("wind_from_compass")),
        "weather_wind_label":        _native((weather or {}).get("wind_label")),
        "weather_wind_out_mph":      _native((weather or {}).get("wind_out_mph")),
        "weather_label":             _native((weather or {}).get("label")),

        # First-pitch UTC timestamp. Stored as TIMESTAMPTZ (aware datetime);
        # converted to America/Chicago at render time by app.py's
        # _to_central() helper. Source is the same MLB Stats API
        # game_datetime string the engine already uses for game_date.
        "game_datetime_utc":         _parse_game_datetime(game.get("game_datetime")),

        # game markets ------------------------------------------------------
        "away_win_prob":  away_win,
        "home_win_prob":  1.0 - away_win,
        "f5_away_prob":   f5_away,
        "f5_home_prob":   max(1.0 - f5_away - f5_tie, 0.0),
        "f5_tie_prob":    f5_tie,
        "nrfi_prob":      float(results.get("nrfi_prob", 0.0)),
        "median_total":   float(results.get("median_total", 0.0)),

        # totals distribution & threshold ladder -----------------------------
        # Falls back to median_total for total_mean if the simulator predates
        # the dist-aware return shape — keeps old slates loggable. Same idea
        # for total_std (0.0 default) and total_thresholds ({} default).
        "total_mean":       float(results.get("total_mean",
                                              results.get("median_total", 0.0))),
        "total_std":        float(results.get("total_std", 0.0)),
        # JSONB column. Serialize here so SQLAlchemy binds a string and the
        # ::jsonb cast in the INSERT statement parses it server-side. Keys
        # are stringified (Postgres JSONB doesn't support numeric keys).
        "total_thresholds": json.dumps({
            str(k): float(v)
            for k, v in (results.get("total_thresholds") or {}).items()
        }),
        "run_line_home_minus_1_5_prob": float(
            results.get("run_line_home_minus_1_5", 0.0)
        ),
        "run_line_away_minus_1_5_prob": float(
            results.get("run_line_away_minus_1_5", 0.0)
        ),

        # pitcher Ks --------------------------------------------------------
        "away_pitcher_median_k": float(results.get("away_pitcher_median_k", 0.0)),
        "away_p_3k": away_k_probs[3],
        "away_p_4k": away_k_probs[4],
        "away_p_5k": away_k_probs[5],
        "away_p_6k": away_k_probs[6],
        "away_p_7k": away_k_probs[7],

        "home_pitcher_median_k": float(results.get("home_pitcher_median_k", 0.0)),
        "home_p_3k": home_k_probs[3],
        "home_p_4k": home_k_probs[4],
        "home_p_5k": home_k_probs[5],
        "home_p_6k": home_k_probs[6],
        "home_p_7k": home_k_probs[7],

        # Full K distributions and 1st-inning stat dicts ---------------------
        # Persisted so the read-only Streamlit Cloud viewer can render the
        # K histogram and the 1st-inning section in the pitcher card without
        # re-running the simulator. Coerced to lists/dicts of native types
        # so json.dumps doesn't choke on numpy values. None-safe: pitchers
        # without enough starts get null, and the renderer's existing
        # fallback path handles that.
        "away_k_dist": json.dumps(_native_list(results.get("away_k_dist") or [])),
        "home_k_dist": json.dumps(_native_list(results.get("home_k_dist") or [])),
        "away_starter_first_inn": json.dumps(
            _native_dict(results.get("away_starter_first_inn"))
        ),
        "home_starter_first_inn": json.dumps(
            _native_dict(results.get("home_starter_first_inn"))
        ),
    }

    conn = _get_conn()
    with conn.session as s:
        # ---- Idempotency: clear any prior prediction for this slot --------
        # Keyed on (game_id, game_date, model_version) so re-runs replace
        # rather than duplicate. player_predictions is wiped first via its
        # FK back to game_predictions.id — this is precise (only THIS
        # version's player rows) so a future v2 run alongside v1 wouldn't
        # clobber v1's batter rows. The whole block runs in the same
        # transaction as the inserts below; commit happens once at the end.
        idempotency_keys = {
            "gid": game_row["game_id"],
            "gd":  game_row["game_date"],
            "mv":  game_row["model_version"],
        }
        s.execute(
            text("""
                DELETE FROM player_predictions
                WHERE game_prediction_id IN (
                    SELECT id FROM game_predictions
                    WHERE game_id = :gid
                      AND game_date = :gd
                      AND model_version = :mv
                )
            """),
            idempotency_keys,
        )
        s.execute(
            text("""
                DELETE FROM game_predictions
                WHERE game_id = :gid
                  AND game_date = :gd
                  AND model_version = :mv
            """),
            idempotency_keys,
        )

        # ---- Insert game-level row, capture the new id --------------------
        # NOTE: predicted_at is INTENTIONALLY omitted from this INSERT.
        # The column has DEFAULT NOW() in the schema, so Postgres assigns
        # the timestamp at the moment of insertion — which is what we want
        # for calibration ("when did this prediction actually land in the
        # DB?"). Letting Python pass a datetime would risk drift if the
        # caller built game_row early and inserted late.
        result = s.execute(
            text("""
                INSERT INTO game_predictions (
                    game_id, game_date, away_team, home_team,
                    away_starter_id, away_starter_name,
                    home_starter_id, home_starter_name,
                    iterations, as_of_date, model_version,
                    weather_park, weather_is_dome, weather_temp_f,
                    weather_wind_mph, weather_carry_delta_ft, weather_hr_score,
                    weather_humidity_pct, weather_pressure_inhg,
                    weather_wind_from_compass, weather_wind_label,
                    weather_wind_out_mph, weather_label,
                    game_datetime_utc,
                    away_win_prob, home_win_prob,
                    f5_away_prob, f5_home_prob, f5_tie_prob,
                    nrfi_prob, median_total,
                    total_mean, total_std, total_thresholds,
                    run_line_home_minus_1_5_prob, run_line_away_minus_1_5_prob,
                    away_pitcher_median_k,
                    away_p_3k, away_p_4k, away_p_5k, away_p_6k, away_p_7k,
                    home_pitcher_median_k,
                    home_p_3k, home_p_4k, home_p_5k, home_p_6k, home_p_7k,
                    away_k_dist, home_k_dist,
                    away_starter_first_inn, home_starter_first_inn
                ) VALUES (
                    :game_id, :game_date, :away_team, :home_team,
                    :away_starter_id, :away_starter_name,
                    :home_starter_id, :home_starter_name,
                    :iterations, :as_of_date, :model_version,
                    :weather_park, :weather_is_dome, :weather_temp_f,
                    :weather_wind_mph, :weather_carry_delta_ft, :weather_hr_score,
                    :weather_humidity_pct, :weather_pressure_inhg,
                    :weather_wind_from_compass, :weather_wind_label,
                    :weather_wind_out_mph, :weather_label,
                    :game_datetime_utc,
                    :away_win_prob, :home_win_prob,
                    :f5_away_prob, :f5_home_prob, :f5_tie_prob,
                    :nrfi_prob, :median_total,
                    :total_mean, :total_std, CAST(:total_thresholds AS JSONB),
                    :run_line_home_minus_1_5_prob, :run_line_away_minus_1_5_prob,
                    :away_pitcher_median_k,
                    :away_p_3k, :away_p_4k, :away_p_5k, :away_p_6k, :away_p_7k,
                    :home_pitcher_median_k,
                    :home_p_3k, :home_p_4k, :home_p_5k, :home_p_6k, :home_p_7k,
                    CAST(:away_k_dist AS JSONB), CAST(:home_k_dist AS JSONB),
                    CAST(:away_starter_first_inn AS JSONB),
                    CAST(:home_starter_first_inn AS JSONB)
                )
                RETURNING id
            """),
            game_row,
        )
        game_pred_id = result.scalar_one()

        # ---- Build per-player rows from both lineups ----------------------
        player_props = results.get("player_props", {}) or {}
        player_rows = []
        for side, roster in (("away", away), ("home", home)):
            for order_idx, player in enumerate(roster.get("lineup_details", []), start=1):
                name = player.get("name")
                if not name:
                    continue
                props = player_props.get(name, {}) or {}
                player_rows.append({
                    "game_prediction_id": game_pred_id,
                    "game_id":            game_row["game_id"],
                    "game_date":          game_row["game_date"],
                    "player_id":          _player_id(player),
                    "player_name":        name,
                    "side":               side,
                    "batting_order":      order_idx,
                    "p_1h":   float(props.get("1+ Hits", 0.0)),
                    "p_2tb":  float(props.get("2+ TB",   0.0)),
                    "p_1hr":  float(props.get("1+ HR",   0.0)),
                    "p_1rbi": float(props.get("1+ RBI",  0.0)),
                    "p_1sb":  float(props.get("1+ SB",   0.0)),
                })

        if player_rows:
            s.execute(
                text("""
                    INSERT INTO player_predictions (
                        game_prediction_id, game_id, game_date,
                        player_id, player_name, side, batting_order,
                        p_1h, p_2tb, p_1hr, p_1rbi, p_1sb
                    ) VALUES (
                        :game_prediction_id, :game_id, :game_date,
                        :player_id, :player_name, :side, :batting_order,
                        :p_1h, :p_2tb, :p_1hr, :p_1rbi, :p_1sb
                    )
                """),
                player_rows,
            )

        s.commit()

    return game_pred_id

# ===========================================================================
#  READ-SIDE PUBLIC API
# ===========================================================================
# Functions used by the read-only Streamlit Cloud viewer (app.py). The
# viewer never simulates — it queries Supabase for the cron-logged
# predictions and renders them. These functions are the inverse of
# log_prediction(): they reconstruct the dicts the renderer expects from
# the rows on disk.
#
# Two design points worth noting:
#   1. We DON'T fetch the K distribution arrays in fetch_predictions_for_date
#      because at slate-overview level we don't need them — they're only
#      needed for the per-game pitcher histogram. fetch_game_detail() pulls
#      them on demand. Keeps the slate query light (~8KB/game vs ~50KB/game).
#   2. JSONB columns come back as already-parsed Python objects from
#      psycopg2 — no manual json.loads needed. This is true for both the
#      Streamlit SQLConnection path and the raw SQLAlchemy path.
# ===========================================================================

def fetch_predictions_for_date(game_date_iso: str,
                               model_version: str = MODEL_VERSION) -> list[dict]:
    """
    Fetch every game prediction logged for a given date. Returns a list
    of dicts, one per game, in game_id order. Each dict mimics the shape
    the renderer expects — same key names as the in-memory `results` dict
    from the simulator, plus the game/team metadata.

    K distribution arrays and 1st-inning stat dicts are NOT included here
    (use fetch_game_detail for those). This keeps the slate-overview
    query small.
    """
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(
            text("""
                SELECT
                    id,
                    game_id, game_date, predicted_at,
                    away_team, home_team,
                    away_starter_id, away_starter_name,
                    home_starter_id, home_starter_name,
                    iterations,
                    weather_park, weather_is_dome, weather_temp_f,
                    weather_wind_mph, weather_carry_delta_ft, weather_hr_score,
                    weather_humidity_pct, weather_pressure_inhg,
                    weather_wind_from_compass, weather_wind_label,
                    weather_wind_out_mph, weather_label,
                    game_datetime_utc,
                    away_win_prob, home_win_prob,
                    f5_away_prob, f5_home_prob, f5_tie_prob,
                    nrfi_prob, median_total,
                    total_mean, total_std, total_thresholds,
                    run_line_home_minus_1_5_prob, run_line_away_minus_1_5_prob,
                    away_pitcher_median_k,
                    away_p_3k, away_p_4k, away_p_5k, away_p_6k, away_p_7k,
                    home_pitcher_median_k,
                    home_p_3k, home_p_4k, home_p_5k, home_p_6k, home_p_7k
                FROM game_predictions
                WHERE game_date = :d
                  AND model_version = :mv
                ORDER BY game_id
            """),
            {"d": game_date_iso, "mv": model_version},
        ).mappings().all()
    return [dict(r) for r in rows]


def fetch_game_detail(game_prediction_id: int) -> dict | None:
    """
    Fetch the full prediction for a single game, including the heavy
    JSONB columns (K dists, 1st-inning stat dicts) needed for the per-game
    pitcher card. Used when rendering an individual game card.

    Returns None if the row doesn't exist (e.g. stale link, deleted row).
    """
    conn = _get_conn()
    with conn.session as s:
        row = s.execute(
            text("""
                SELECT
                    away_k_dist, home_k_dist,
                    away_starter_first_inn, home_starter_first_inn
                FROM game_predictions
                WHERE id = :id
            """),
            {"id": game_prediction_id},
        ).mappings().first()
    return dict(row) if row else None


def fetch_player_props_for_game(game_prediction_id: int) -> dict[str, dict]:
    """
    Reconstruct the per-batter player_props dict for one game. Output
    matches the simulator's `results['player_props']` shape exactly — keys
    are batter names, values are dicts with '1+ Hits', '2+ TB', '1+ HR',
    '1+ RBI', '1+ SB' probabilities. Empty dict if no rows.
    """
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(
            text("""
                SELECT player_name, p_1h, p_2tb, p_1hr, p_1rbi, p_1sb
                FROM player_predictions
                WHERE game_prediction_id = :id
            """),
            {"id": game_prediction_id},
        ).mappings().all()
    return {
        r["player_name"]: {
            "1+ Hits": float(r["p_1h"]),
            "2+ TB":   float(r["p_2tb"]),
            "1+ HR":   float(r["p_1hr"]),
            "1+ RBI":  float(r["p_1rbi"]),
            "1+ SB":   float(r["p_1sb"]),
        }
        for r in rows
    }


def fetch_lineups_for_game(game_prediction_id: int) -> dict[str, list[dict]]:
    """
    Reconstruct the home/away lineup-detail lists from player_predictions,
    sorted by batting order. Returns:
        {"away": [{...}, ...], "home": [{...}, ...]}
    Each entry has 'name', 'id' (player_id), 'order' — the minimum the
    lineup-table renderer needs. Empty side lists if no rows.
    """
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(
            text("""
                SELECT player_name, player_id, side, batting_order
                FROM player_predictions
                WHERE game_prediction_id = :id
                ORDER BY side, batting_order
            """),
            {"id": game_prediction_id},
        ).mappings().all()

    lineups = {"away": [], "home": []}
    for r in rows:
        lineups[r["side"]].append({
            "name":  r["player_name"],
            "id":    r["player_id"],
            "order": r["batting_order"],
        })
    return lineups


def fetch_top_batter_props(game_date_iso: str,
                           prop_key: str = "p_1h",
                           limit: int = 5,
                           model_version: str = MODEL_VERSION) -> list[dict]:
    """
    Top-N batters by a single prop probability across the entire slate.
    Used by the Beat the Streak sidebar (default: top 5 by 1+ Hits).

    Each row includes the batter, both teams (so the user sees who's
    playing whom), the probability, and predicted_at so they can see
    how fresh the prediction is. Joined to game_predictions for team
    context and to filter by date + version in one shot.

    `prop_key` must be one of: 'p_1h', 'p_2tb', 'p_1hr', 'p_1rbi', 'p_1sb'.
    Validated against an allow-list to keep this safe from SQL injection
    (prop_key gets interpolated into the SELECT/ORDER BY, not bound).
    """
    ALLOWED_PROPS = {"p_1h", "p_2tb", "p_1hr", "p_1rbi", "p_1sb"}
    if prop_key not in ALLOWED_PROPS:
        raise ValueError(
            f"prop_key must be one of {ALLOWED_PROPS}, got {prop_key!r}"
        )

    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(
            text(f"""
                SELECT
                    pp.player_name,
                    pp.{prop_key} AS prob,
                    gp.away_team,
                    gp.home_team,
                    pp.side,
                    gp.predicted_at
                FROM player_predictions pp
                JOIN game_predictions gp ON gp.id = pp.game_prediction_id
                WHERE gp.game_date = :d
                  AND gp.model_version = :mv
                ORDER BY pp.{prop_key} DESC
                LIMIT :n
            """),
            {"d": game_date_iso, "mv": model_version, "n": limit},
        ).mappings().all()
    return [dict(r) for r in rows]