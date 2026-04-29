"""
auto_log_predictions.py
-----------------------
Headless wrapper around engine_runner.run_slate() that runs in GitHub
Actions on a schedule. No Streamlit, no UI — just fetches today's slate,
runs sims for any game with posted lineups we haven't already logged,
and writes predictions to Supabase.

Designed to run multiple times per day. Idempotency in run_slate() means
each run only processes games that haven't been logged yet, so you get
incremental coverage as more lineups post throughout the afternoon.

Environment
-----------
    DATABASE_URL  Required. Same Supabase URL as record_outcomes.py.
    GITHUB_TOKEN  Required for private repo (provided automatically in
                  GitHub Actions; set manually for local testing).

Exit codes
----------
    0   At least one game logged, OR slate was empty/all-already-logged.
    1   Logged zero games AND had simulation/log failures (something's wrong).
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
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Monte Carlo iterations per game (default: 2000)")
    parser.add_argument("--include-in-progress", action="store_true",
                        help="Also process In Progress games (default: skip)")
    parser.add_argument("--force", action="store_true",
                        help="Re-log games even if already in DB (rarely useful)")
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
        skip_compute_if_already_logged=not args.force,  # ← ADD THIS LINE
        include_in_progress=args.include_in_progress,
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

    # Exit code logic: zero is healthy unless we did real work and got nothing
    if summary.simulated == 0 and summary.skipped_already_logged == summary.total_games:
        log.info("Nothing new to do — all games already logged.")
        return 0
    if summary.total_games == 0:
        log.info("Empty slate — nothing to do.")
        return 0
    if summary.logged == 0 and (summary.log_failures > 0 or summary.failed_game_ids):
        log.error("Logged 0 games but had failures. Treating as error.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())