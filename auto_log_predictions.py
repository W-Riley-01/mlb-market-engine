"""
auto_log_predictions.py
-----------------------
Headless wrapper around engine_runner.run_slate() that runs in GitHub
Actions on a schedule. No Streamlit, no UI — just fetches today's slate,
runs sims for every pre-first-pitch game with posted lineups, and writes
predictions to RDS PostgreSQL.

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
    RDS_SECRET_ARN  Optional. Overrides the default Secrets Manager ARN
                    holding the RDS master credentials. Not sensitive
                    itself — it's an identifier, not a credential.
    DATABASE_URL    Optional override. If set, used directly instead of
                    fetching from Secrets Manager — useful for local
                    testing against a different database.
    GITHUB_TOKEN    Required for private repo (provided automatically in
                    GitHub Actions; set manually for local testing).

Exit codes
----------
    0   Slate processed successfully (or empty slate — nothing to do).
    1   At least one logging failure with no successful logs (treated
        as a hard error so the Action surfaces it).
    2   Could not obtain database credentials — config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import boto3

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

# ---------------------------------------------------------------------------
#  RDS / Secrets Manager configuration
# ---------------------------------------------------------------------------
# Same pattern as prediction_logger.py and record_outcomes.py — the RDS
# master password is generated and stored by AWS itself, never set or
# seen by us in plaintext.
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
RDS_SECRET_ARN = os.environ.get(
    "RDS_SECRET_ARN",
    "arn:aws:secretsmanager:us-east-1:687050094462:secret:rds!db-16e1cf61-de84-4850-9d01-7315eaa97bcf-65Fnln",
)
# The RDS-managed secret reliably contains username/password. host/port/
# dbname are NOT guaranteed to be present in the secret's JSON payload —
# observed in practice to be absent — so we specify them explicitly here,
# matching the values Terraform actually created. .get() with these as
# fallbacks still prefers the secret's own values if a future rotation
# adds them.
RDS_ENDPOINT = os.environ.get("RDS_ENDPOINT", "mlb-engine-db.cyzm64iqm3q4.us-east-1.rds.amazonaws.com")
RDS_PORT = os.environ.get("RDS_PORT", "5432")
RDS_DB_NAME = os.environ.get("RDS_DB_NAME", "mlb_engine")


def _fetch_db_url_from_secrets_manager() -> str:
    """
    Build a full SQLAlchemy connection string from the RDS-managed
    master credentials in Secrets Manager. Identical logic to
    prediction_logger.py / record_outcomes.py — kept as a local copy
    since this script, like record_outcomes.py, is a standalone headless
    entrypoint with no dependency on Streamlit being installed.
    """
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = client.get_secret_value(SecretId=RDS_SECRET_ARN)
    creds = json.loads(secret["SecretString"])

    return (
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{creds.get('host', RDS_ENDPOINT)}:{creds.get('port', RDS_PORT)}/{creds.get('dbname', RDS_DB_NAME)}"
    )


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
    parser = argparse.ArgumentParser(description="Auto-log MLB predictions to RDS PostgreSQL.")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Monte Carlo iterations per game (default: 5000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip games already logged today (default: re-sim and "
                             "upsert every pre-first-pitch game; last-write-wins).")
    args = parser.parse_args(argv or sys.argv[1:])

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        try:
            db_url = _fetch_db_url_from_secrets_manager()
        except Exception as e:
            log.error("Could not obtain database credentials from Secrets Manager: %s", e)
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
    log.info("  Logged to RDS:         %d", summary.logged)
    log.info("  Skipped (no lineups yet):   %d", summary.skipped_no_lineups)
    log.info("  Skipped (already logged):   %d", summary.skipped_already_logged)
    log.info("  Sim/log failures:           %d", summary.log_failures + len(summary.failed_game_ids))
    log.info("=" * 60)

    # Exit code logic
    # ----------------
    # 0 = healthy. Includes:
    #     - empty slates (no games today)
    #     - all games still awaiting lineups (normal morning cron)
    #     - mostly-still-awaiting-lineups runs where ≥1 game errored
    #       (real-world: 14/15 games no-lineups + 1 sim error from a
    #       half-posted lineup hitting the cron mid-update)
    # 1 = real problem. The slate had material work to do and got
    #     nothing out of it: most games had lineups but everything we
    #     attempted failed.
    if summary.total_games == 0:
        log.info("Empty slate — nothing to do.")
        return 0
    if summary.simulated == 0 and summary.skipped_no_lineups == summary.total_games:
        log.info("All games still awaiting lineups — will retry on next cron tick.")
        return 0

    # Distinguish "couldn't sim because no lineups posted yet" from
    # "tried to sim, everything broke." The morning-gap case is normal
    # and shouldn't page us. The mass-failure case is a real bug.
    #
    # Threshold: only treat as exit 1 if ≥ 3 games had lineups AND every
    # single one failed. With a single isolated failure (e.g. a half-
    # posted lineup hitting the cron mid-update), the next cron tick
    # will retry; alerting on it would be noise. Three failures in a
    # row implies a systemic problem (resolver broken, DB down, etc.).
    games_with_lineups = summary.total_games - summary.skipped_no_lineups
    games_failed       = len(summary.failed_game_ids) + summary.log_failures
    MASS_FAILURE_THRESHOLD = 3
    if (games_with_lineups >= MASS_FAILURE_THRESHOLD
            and summary.logged == 0
            and games_failed >= games_with_lineups):
        log.error(
            "All %d game(s) with lineups failed to log. Treating as error.",
            games_with_lineups,
        )
        return 1

    # If we got here, at least some games either logged successfully or
    # are still legitimately waiting for lineups. Individual sim errors
    # on a mostly-pending slate are noise, not a cron-level failure —
    # they're already captured in summary.failed_game_ids and visible in
    # the run log. The next cron tick will retry them.
    if games_failed > 0:
        log.warning(
            "Completed with %d game-level failure(s) but slate made progress "
            "(logged=%d, awaiting_lineups=%d). Exiting 0.",
            games_failed, summary.logged, summary.skipped_no_lineups,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
