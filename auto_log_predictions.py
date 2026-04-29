"""
auto_log_predictions.py
-----------------------
Headless wrapper around engine_runner.run_slate() that runs in GitHub
Actions on a schedule. No Streamlit, no UI — just fetches today's slate,
runs sims for every pre-first-pitch game with posted lineups, and writes
predictions to Supabase.

Cron strategy: every run re-simulates and re-logs every pre-first-pitch
game on the slate (skip_already_logged=False). The upsert pattern in
prediction_logger.log_prediction (delete-then-insert keyed on game_id +
game_date + model_version) makes this safe — duplicate runs converge to
exactly one row per game, with last-write-wins semantics. The status
gate inside run_slate prevents post-first-pitch overwrites: once a game
starts, its prediction is frozen.

Why re-sim instead of skip-already-logged: as the day progresses, weather
forecasts tighten and lineups get confirmed (or scratched). The latest
pre-first-pitch prediction is the most informed one, and that's what we
want in the DB for calibration analysis. The compute cost is modest —
a 15-game slate at 5000 iterations runs in a few minutes on Actions.

Environment
-----------
    DATABASE_URL  Required. Same Supabase URL as record_outcomes.py.
    GITHUB_TOKEN  Required for private repo (provided automatically in
                  GitHub Actions; set manually for local testing).

Exit codes
----------
    0   Slate processed successfully (or empty slate — nothing to do).
    1   At least one logging failure with no successful logs (treated
        as a hard error so the Action surfaces it).
    2   Missing DATABASE_URL — config error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure data files are present BEFORE any engine module imports parquet
from bootstrap_data import ensure_data_files
ensure_data_files()

from resolver import MatchupResolver
from engine_runner import run_slate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_log_predictions")


def _build_resolver() -> MatchupResolver:
    """
    Construct the resolver with the same paths app.py uses. The resolver
    loads ~120MB of parquet on init — slow but only happens once per
    Action run.
    """
    return MatchupResolver(
        pqm_path="./data/pitch_matrix.parquet",
        cqm_path="./data/contact_matrix_env.parquet",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-log MLB predictions to Supabase.")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Monte Carlo iterations per game (default: 5000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip games already logged today (default: re-sim and "
                             "upsert every pre-first-pitch game; last-write-wins).")
    args = parser.parse_args(argv or sys.argv[1:])

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        return 2

    log.info("Loading resolver (this may take ~10s)...")
    resolver = _build_resolver()
    log.info("Resolver loaded. Running slate (iterations=%d)...", args.iterations)

    summary = run_slate(
        resolver=resolver,
        iterations=args.iterations,
        db_url=db_url,
        skip_already_logged=args.skip_existing,
        on_game_complete=None,
        log_predictions=True,
    )

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Total games on slate:       %d", summary.total_games)
    log.info("  Simulated this run:         %d", summary.simulated)
    log.info("  Logged to Supabase:         %d", summary.logged)
    log.info("  Skipped (no lineups yet):   %d", summary.skipped_no_lineups)
    log.info("  Skipped (already logged):   %d", summary.skipped_already_logged)
    log.info("  Sim/log failures:           %d", summary.log_failures + len(summary.failed_game_ids))
    log.info("=" * 60)

    # Exit code logic
    # ----------------
    # 0 = healthy. Includes empty slates and runs where everything already
    #     started (so log_predictions sim'd but didn't write — that's
    #     correct behavior, not a failure).
    # 1 = real problem. We did work and got nothing out of it: either every
    #     log attempt failed, or every game errored in simulation.
    if summary.total_games == 0:
        log.info("Empty slate — nothing to do.")
        return 0
    if summary.simulated == 0 and summary.skipped_no_lineups == summary.total_games:
        log.info("All games still awaiting lineups — will retry on next cron tick.")
        return 0
    if summary.logged == 0 and (summary.log_failures > 0 or summary.failed_game_ids):
        log.error("Logged 0 games but had failures. Treating as error.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())