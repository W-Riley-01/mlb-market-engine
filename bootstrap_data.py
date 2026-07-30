"""
bootstrap_data.py
-----------------
Ensures large data files are present locally, downloading them from
S3 if missing. Run this at startup before any code that touches the
engine matrices.

Why this exists
---------------
Several of these parquet files are too large to comfortably commit to
the Git repo (pitch_matrix.parquet alone is ~120MB, well past GitHub's
100MB hard limit). Hosting them in S3 gives us a single, versioned
source of truth for all engine data files, with lifecycle rules
(Standard-IA at 30 days, Glacier at 90) keeping storage costs down.

Usage
-----
    # As a script (idempotent — only downloads what's missing)
    python bootstrap_data.py

    # As a module
    from bootstrap_data import ensure_data_files
    ensure_data_files()

Authentication
--------------
Uses standard boto3 credential resolution — AWS CLI config, environment
variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), or an IAM role
if running on EC2/Actions with one attached. No credentials are
hardcoded here.

Configuration
-------------
The bucket and prefix live in the constants below. DATA_FILES lists
every parquet file the engine may need locally. Add new entries here
as new files are uploaded to S3.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import NamedTuple

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "mlb-engine-data-4462"
S3_PREFIX = "parquet/"   # all data files live under this prefix in the bucket

# Where to place files locally (relative to repo root)
DATA_DIR = Path("data")

# Default sanity-check floor for files where we haven't set a specific
# threshold below — just enough to catch a zero-byte / truncated download.
DEFAULT_MIN_BYTES = 1_024


class DataFile(NamedTuple):
    """One data file to fetch from S3."""
    s3_key: str        # key under S3_PREFIX, e.g. "pitch_matrix.parquet"
    local_path: Path   # where to write it locally
    min_bytes: int      # sanity check — abort if download is suspiciously small


DATA_FILES = [
    DataFile("batter_arsenal_profiles.parquet",     DATA_DIR / "batter_arsenal_profiles.parquet",     DEFAULT_MIN_BYTES),
    DataFile("contact_matrix.parquet",               DATA_DIR / "contact_matrix.parquet",               DEFAULT_MIN_BYTES),
    DataFile("contact_matrix_env.parquet",           DATA_DIR / "contact_matrix_env.parquet",           DEFAULT_MIN_BYTES),
    DataFile("early_inning_tracker.parquet",         DATA_DIR / "early_inning_tracker.parquet",         DEFAULT_MIN_BYTES),
    DataFile("game_metadata.parquet",                DATA_DIR / "game_metadata.parquet",                DEFAULT_MIN_BYTES),
    DataFile("game_weather_log.parquet",             DATA_DIR / "game_weather_log.parquet",             DEFAULT_MIN_BYTES),
    DataFile("master_physics_vault.parquet",         DATA_DIR / "master_physics_vault.parquet",         DEFAULT_MIN_BYTES),
    DataFile("pitch_matrix.parquet",                 DATA_DIR / "pitch_matrix.parquet",                 50_000_000),  # ~120MB full file
    DataFile("pitch_type_baselines.parquet",         DATA_DIR / "pitch_type_baselines.parquet",         DEFAULT_MIN_BYTES),
    DataFile("pitcher_early_inning_record.parquet",  DATA_DIR / "pitcher_early_inning_record.parquet",  DEFAULT_MIN_BYTES),
    DataFile("recent_form_hitters.parquet",          DATA_DIR / "recent_form_hitters.parquet",          DEFAULT_MIN_BYTES),
    DataFile("recent_form_pitchers.parquet",         DATA_DIR / "recent_form_pitchers.parquet",         DEFAULT_MIN_BYTES),
    DataFile("xwoba_lookup.parquet",                 DATA_DIR / "xwoba_lookup.parquet",                 DEFAULT_MIN_BYTES),
]


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bootstrap] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  S3 download
# ---------------------------------------------------------------------------
def _s3_client():
    return boto3.client("s3")


def _download_from_s3(s3, s3_key: str, dest: Path) -> None:
    """
    Download one object from S3 to a local path, streaming to disk.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    full_key = f"{S3_PREFIX}{s3_key}"

    log.info("Downloading s3://%s/%s → %s", S3_BUCKET, full_key, dest)
    try:
        s3.download_file(S3_BUCKET, full_key, str(dest))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            raise RuntimeError(
                f"Object {full_key!r} not found in bucket {S3_BUCKET!r}. "
                f"Did you upload it under the '{S3_PREFIX}' prefix?"
            ) from e
        if code in ("403", "AccessDenied"):
            raise RuntimeError(
                f"Access denied fetching {full_key!r} from {S3_BUCKET!r}. "
                f"Check your AWS credentials / IAM permissions."
            ) from e
        raise

    log.info("Downloaded %s OK (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def ensure_data_files(force: bool = False) -> None:
    """
    Make sure every file in DATA_FILES is present locally. Downloads
    missing ones from S3.

    Parameters
    ----------
    force : bool
        If True, re-download even if local files exist. Useful when
        rolling a new matrix version.
    """
    missing = []
    for df in DATA_FILES:
        if not df.local_path.exists():
            missing.append(df)
            continue
        # Sanity check existing file — corrupt/truncated downloads from a
        # prior interrupted run would silently break the engine.
        size = df.local_path.stat().st_size
        if size < df.min_bytes:
            log.warning("Existing %s is only %.1f MB (expected ≥%.1f MB) — re-downloading",
                        df.local_path, size / 1e6, df.min_bytes / 1e6)
            missing.append(df)
        elif force:
            missing.append(df)
        else:
            log.info("✓ %s already present (%.1f MB)",
                     df.local_path.name, size / 1e6)

    if not missing:
        log.info("All required data files present.")
        return

    log.info("Need to fetch %d file(s)", len(missing))
    s3 = _s3_client()

    for df in missing:
        _download_from_s3(s3, df.s3_key, df.local_path)

        size = df.local_path.stat().st_size
        if size < df.min_bytes:
            df.local_path.unlink()  # remove the bad file
            raise RuntimeError(
                f"Downloaded {df.s3_key} is only {size/1e6:.1f}MB "
                f"(expected ≥{df.min_bytes/1e6:.1f}MB). Aborting."
            )

    log.info("Bootstrap complete.")


if __name__ == "__main__":
    try:
        ensure_data_files(force="--force" in sys.argv)
    except Exception as e:
        log.error("Bootstrap failed: %s", e)
        sys.exit(1)
