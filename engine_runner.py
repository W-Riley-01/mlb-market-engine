"""
engine_runner.py
----------------
Shared orchestration: fetch slate → fetch rosters → run simulations →
apply weather → log to DB. Used by:

  - app.py            (interactive Streamlit UI; passes a callback to render
                       game cards as each game finishes)
  - auto_log_predictions.py
                      (headless GitHub Actions cron; no callback, just logs)

This is the single source of truth for "running a slate." If the pipeline
needs to change (new market, different weather logic, etc.), you change it
here and both call sites pick up the change.

Critical design points
----------------------
- Every game on the slate is processed regardless of status — Scheduled,
  Pre-Game, In Progress, Final, Delayed, Postponed. The UI renders a card
  for every one of them, with the live-scoreboard helper indicating
  current state.
- Logging is gated to PRE_FIRST_PITCH_STATUSES only. Once a game starts,
  the prediction window has closed: re-logging the same game later would
  overwrite the morning's prediction (via prediction_logger's last-
  write-wins upsert) with one made AFTER first pitch, polluting the
  calibration data. Sim still runs for already-started games so the UI
  shows fresh markets next to the live scoreboard, but no DB write.
- Idempotent: re-running on the same date does NOT create duplicate
  predictions in Supabase. prediction_logger.log_prediction performs a
  delete-then-insert keyed on (game_id, game_date, model_version), so
  multiple cron retries or UI clicks converge to one row per game.
- Rendering-agnostic: the on_game_complete callback is optional. Pass one
  to display per-game UI, omit it for headless runs.
- Logging is best-effort: a Supabase failure on one game does NOT abort
  the rest of the slate (the prediction is still computed, just not
  persisted; the failure is captured on GameRun.log_error).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from sqlalchemy import create_engine, text

# Engine modules — same imports as app.py
from resolver import MatchupResolver
from advanced_simulator import run_advanced_monte_carlo
from daily_scraper import fetch_todays_schedule, fetch_game_rosters
from weather import get_game_weather, apply_weather_to_props
from prediction_logger import log_prediction, MODEL_VERSION

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Status taxonomy
# ---------------------------------------------------------------------------
# Statuses where the game has NOT yet started — predictions for games in
# this set are valid pre-first-pitch predictions and get logged to Supabase.
# Anything outside this set (In Progress, Final, Postponed, etc.) gets
# rendered in the UI but is NOT logged: doing so would overwrite the
# morning's prediction with one made after the prediction window closed,
# polluting calibration data. Update this set if MLB Stats API surfaces a
# new pre-game status code we want to include.
PRE_FIRST_PITCH_STATUSES = frozenset({
    "Scheduled",
    "Pre-Game",
    "Warmup",
    "Delayed Start",
})


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GameRun:
    """One game's worth of computed results, passed to the UI callback."""
    game: dict
    rosters: Optional[dict]            # None if rosters/lineups not posted
    weather: Optional[dict]            # None if no weather data
    away: Optional[dict] = None        # None if no rosters
    home: Optional[dict] = None
    results: Optional[dict] = None     # Sim output; None if no rosters
    logged: bool = False
    log_error: Optional[str] = None
    skipped_reason: Optional[str] = None   # 'no_lineups', 'already_logged', 'in_progress'


@dataclass
class SlateRunSummary:
    """Roll-up returned at the end of a slate run."""
    total_games: int = 0
    simulated:   int = 0
    logged:      int = 0
    skipped_no_lineups:    int = 0
    skipped_already_logged: int = 0
    skipped_in_progress:   int = 0
    log_failures:          int = 0
    failed_game_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Idempotency check
# ---------------------------------------------------------------------------
def _already_logged_game_ids(db_url: str, game_date: str,
                              model_version: str = MODEL_VERSION) -> set[str]:
    """
    Returns the set of game_ids that already have a prediction logged for
    this date + model version. Used to skip games we've already processed.
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT game_id
                FROM game_predictions
                WHERE game_date = :d
                  AND model_version = :mv
            """),
            {"d": game_date, "mv": model_version},
        ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def run_slate(
    *,
    resolver: MatchupResolver,
    iterations: int = 5000,
    db_url: Optional[str] = None,
    skip_already_logged: bool = True,
    on_game_complete: Optional[Callable[[GameRun, int, int], None]] = None,
    log_predictions: bool = True,
) -> SlateRunSummary:
    """
    Run today's slate end-to-end.

    Parameters
    ----------
    resolver : MatchupResolver
        Pre-loaded resolver (its parquet matrices are big — load once, reuse).
    iterations : int
        Monte Carlo iterations per game.
    db_url : str | None
        SQLAlchemy URL for Supabase. Required when log_predictions=True
        AND skip_already_logged=True; otherwise optional. The actual
        log_prediction() call uses Streamlit's st.connection in app.py
        and falls through to the same DB; this parameter is only used
        for the idempotency lookup.
    skip_already_logged : bool
        If True, skip games whose game_id already has a logged prediction
        for today's date. Set False to force re-logging (the UI uses this
        so reclicks always re-render fresh cards; idempotency is handled
        downstream by prediction_logger's delete-then-insert upsert).
    on_game_complete : callable | None
        Optional callback invoked after each game with
        (GameRun, current_index, total_games). Use to render UI live.
    log_predictions : bool
        If False, skip the DB write entirely (useful for dry runs). When
        True (the default), only games whose status is in
        PRE_FIRST_PITCH_STATUSES are logged — see the module docstring
        for the rationale.

    Returns
    -------
    SlateRunSummary
        Counts of what happened. Inspect for monitoring / alerting.
    """
    summary = SlateRunSummary()
    today_str = datetime.today().strftime("%Y-%m-%d")

    # ---- Step 1: Fetch slate ----------------------------------------------
    slate = fetch_todays_schedule()
    if not slate:
        log.info("No games on today's slate.")
        return summary

    # No status filter — every game on the slate gets a card in the UI,
    # whether it's Scheduled, In Progress, or Final. The render-vs-log
    # split happens further down: simulation runs for every game with
    # confirmed lineups, but DB logging is gated by status.
    games_to_run = list(slate)
    summary.total_games = len(games_to_run)

    log.info("Slate: %d total games, %d to process", len(slate), summary.total_games)

    # ---- Step 2: Idempotency check (which games are already in DB?) -----
    already_logged: set[str] = set()
    if skip_already_logged and log_predictions and db_url:
        try:
            already_logged = _already_logged_game_ids(db_url, today_str)
            if already_logged:
                log.info("%d game(s) already logged for %s, skipping",
                         len(already_logged), today_str)
        except Exception as e:
            log.warning("Idempotency check failed (will not skip duplicates): %s", e)

    # ---- Step 3: Process each game ----------------------------------------
    for i, game in enumerate(games_to_run):
        game_id = str(game.get("game_id", ""))
        run = GameRun(game=game, rosters=None, weather=None)

        # 3a. Skip if already logged today
        if game_id in already_logged:
            run.skipped_reason = "already_logged"
            summary.skipped_already_logged += 1
            _safe_callback(on_game_complete, run, i, summary.total_games)
            continue

        # 3b. Weather (rendered even if lineups not yet up — useful for UI)
        run.weather = _safe_get_weather(game)

        # 3c. Rosters / lineups
        # The simulator's AdvancedSimulatedGame indexes lineup[0..8], so
        # both sides need at least 9 batters in `lineup_details`. Earlier
        # versions only checked the away `lineup` field — but the sim
        # consumes `lineup_details`, and only validated one side. A real-
        # world half-posted slate (away lineup up, home lineup partial)
        # crashed game 822987 with IndexError on 2026-05-03. This stricter
        # gate guarantees we never feed a malformed roster to the sim.
        rosters = fetch_game_rosters(game_id)
        away_lu = (rosters or {}).get("Away", {}).get("lineup_details") or []
        home_lu = (rosters or {}).get("Home", {}).get("lineup_details") or []
        if not rosters or len(away_lu) < 9 or len(home_lu) < 9:
            run.skipped_reason = "no_lineups"
            summary.skipped_no_lineups += 1
            log.info(
                "Skipping game_id=%s: insufficient lineups (away=%d, home=%d)",
                game_id, len(away_lu), len(home_lu),
            )
            _safe_callback(on_game_complete, run, i, summary.total_games)
            continue

        run.rosters = rosters
        run.away = rosters["Away"]
        run.home = rosters["Home"]

        # 3d. Simulate
        try:
            results = run_advanced_monte_carlo(
                resolver=resolver,
                away_lineup=run.away["lineup_details"],
                home_lineup=run.home["lineup_details"],
                away_starter=run.away["starter_id"],
                home_starter=run.home["starter_id"],
                away_bullpen=run.away["bullpen_ids"],
                home_bullpen=run.home["bullpen_ids"],
                # density_ratio is INTENTIONALLY hardcoded to 1.0 (no
                # in-sim environmental boost). Two reasons:
                #   1) The resolver's environmental physics step
                #      (`merged['hr'] *= density_ratio`) is sign-
                #      inverted relative to enviroment_merger.py.
                #      The merger gives Coors a ratio of ~0.945 (low
                #      = thin), but the resolver expects values > 1.0
                #      for thin air. Wiring live weather in via this
                #      parameter would propagate the bug to every game.
                #   2) Live weather effects DO reach the prediction —
                #      via the post-sim adjustments below
                #      (apply_weather_to_props for batter HR/2+TB,
                #      _adjust_totals_for_weather for game totals and
                #      threshold ladders). That path is correctly
                #      signed (positive carry_delta → more runs).
                # When the resolver/merger physics get reconciled in a
                # dedicated session, this parameter can carry the
                # full carry_delta. Until then: leave at 1.0.
                density_ratio=1.0,
                iterations=iterations,
                as_of_date=today_str,
            )
        except Exception as e:
            log.exception("Simulation failed for game_id=%s: %s", game_id, e)
            summary.failed_game_ids.append(game_id)
            run.skipped_reason = "sim_error"
            _safe_callback(on_game_complete, run, i, summary.total_games)
            continue

        summary.simulated += 1

        # 3e. Weather adjustment — applied to OUTDOOR games only.
        # Dome games skip weather overlay because the carry delta is zero
        # and there's no useful adjustment to make. The PREDICTION itself
        # is still logged either way, unlike the previous bug in app.py.
        # The adjustment touches player props (HR, 2+TB) AND game totals
        # (median, mean, std, threshold ladder) so weather effects flow
        # through to over/under markets too — earlier versions only
        # boosted player props, leaving median_total inconsistent with
        # the per-batter HR boost.
        if run.weather and not run.weather.get("is_dome"):
            carry_delta = run.weather.get("carry_delta_ft", 0.0)
            results["player_props"] = apply_weather_to_props(
                results["player_props"], carry_delta
            )
            results = _adjust_totals_for_weather(results, carry_delta)

        run.results = results

        # 3f. Log to Supabase — only for games that haven't started yet.
        # Logging an in-progress or final game would overwrite the morning's
        # pre-first-pitch prediction (last-write-wins upsert in
        # prediction_logger) with one made AFTER the prediction window
        # closed. For calibration we want the DB row to be the LAST
        # prediction made BEFORE first pitch, not whatever the user (or
        # cron) happened to compute most recently. So sim everything, log
        # only the pre-game ones. Best-effort: a Supabase failure on one
        # game does not abort the rest of the slate.
        is_pre_first_pitch = game.get("status") in PRE_FIRST_PITCH_STATUSES
        if log_predictions and is_pre_first_pitch:
            try:
                log_prediction(
                    game=game,
                    away=run.away,
                    home=run.home,
                    results=results,
                    iterations=iterations,
                    as_of_date=today_str,
                    weather=run.weather,
                )
                run.logged = True
                summary.logged += 1
            except Exception as e:
                run.log_error = str(e)
                summary.log_failures += 1
                log.warning("log_prediction failed for game_id=%s: %s", game_id, e)

        # 3g. Callback for UI rendering (no-op in headless mode)
        _safe_callback(on_game_complete, run, i, summary.total_games)

    log.info(
        "Slate run complete: simulated=%d logged=%d "
        "skip_no_lineups=%d skip_already_logged=%d log_failures=%d",
        summary.simulated, summary.logged,
        summary.skipped_no_lineups, summary.skipped_already_logged,
        summary.log_failures,
    )
    return summary


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _safe_get_weather(game: dict) -> Optional[dict]:
    """Wrap weather call so a flaky weather API can't kill the run."""
    try:
        return get_game_weather(game.get("home_team"), game.get("game_datetime"))
    except Exception as e:
        log.warning("Weather fetch failed for game_id=%s: %s",
                    game.get("game_id"), e)
        return None


def _safe_callback(cb, run, i, total):
    """Ignore exceptions in the UI callback so headless callers aren't affected."""
    if cb is None:
        return
    try:
        cb(run, i, total)
    except Exception as e:
        log.exception("on_game_complete callback raised: %s", e)


# ---------------------------------------------------------------------------
#  Weather → totals coupling
# ---------------------------------------------------------------------------
# Each foot of carry shifts game totals by ~0.05 runs. Derivation:
#   - At neutral conditions the slate-wide HR rate is ~1.1 HR/team/game,
#     i.e. ~2.2 HR/game total.
#   - Empirical / physics rule-of-thumb: ~2.5% more HRs per ft of carry.
#   - 1 ft × 2.5% × 2.2 HR ≈ 0.055 extra HRs per game.
#   - Each HR scores ~1.5 runs on average (mix of solo & runners-on).
#   - 0.055 × 1.5 ≈ 0.08 extra runs per HR component.
#   - Plus a smaller contribution from 2B / fly-ball-to-double conversion
#     (~30% of the HR effect, per the resolver's existing magnitude).
#   - Round to ~0.05 runs/ft as a conservative all-in estimate.
#
# This is a CONSCIOUSLY APPROXIMATE coupling. Its magnitude is consistent
# with the player-prop adjustment (apply_weather_to_props) so the median
# total moves in step with the per-batter HR boost. After a few weeks of
# paired data we can refit this constant against actuals — it's the kind
# of thing the calibration pipeline will tune empirically.
RUNS_PER_FT_CARRY = 0.05


def _adjust_totals_for_weather(results: dict, carry_delta_ft: float) -> dict:
    """
    Shift the totals distribution by carry_delta_ft × RUNS_PER_FT_CARRY,
    then re-derive median, mean, std, and the over/under threshold ladder
    from the shifted distribution.

    Run-line probabilities are intentionally NOT adjusted: weather lifts
    BOTH teams' run output roughly symmetrically, so the margin
    distribution shifts only marginally. Skipping this avoids spurious
    weather-driven precision in run-line probs.

    No-op when carry_delta_ft is zero, when the dist is missing, or when
    the input is a dome game (caller already filters).
    """
    if not carry_delta_ft:
        return results
    raw = results.get("game_totals_dist")
    if not raw:
        return results

    shift = carry_delta_ft * RUNS_PER_FT_CARRY
    shifted = np.asarray(raw, dtype=float) + shift

    results["median_total"] = float(np.median(shifted))
    results["total_mean"]   = float(np.mean(shifted))
    results["total_std"]    = float(np.std(shifted))

    # Re-derive the threshold ladder. Reuse the existing keys so we don't
    # accidentally diverge from the simulator's threshold list.
    existing = results.get("total_thresholds", {})
    if existing:
        results["total_thresholds"] = {
            x: float(np.mean(shifted >= float(x))) for x in existing.keys()
        }
    return results