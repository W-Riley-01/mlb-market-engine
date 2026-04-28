"""
bootstrap_data.py
-----------------
Ensures large data files are present locally, downloading them from
GitHub Releases if missing. Run this at startup before any code that
touches the engine matrices.

Why this exists
---------------
pitch_matrix.parquet is ~120MB — too large to commit to the Git repo
(GitHub's hard limit is 100MB per file). Hosting it as a GitHub Release
asset gives us a free CDN with no bandwidth limits.

Usage
-----
    # As a script (idempotent — only downloads if missing)
    python bootstrap_data.py

    # As a module
    from bootstrap_data import ensure_data_files
    ensure_data_files()

Authentication
--------------
For PRIVATE repos, you must set GITHUB_TOKEN as an env var (already
available inside GitHub Actions runners; for Streamlit Cloud you'll
add it as a secret). For PUBLIC repos no token is needed.

Configuration
-------------
The release tag and asset list live in DATA_FILES below. To roll a new
matrix version, you'll bump the release tag (e.g., data-v2) and update
this file. See README_data_releases.md for the full upload workflow.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import NamedTuple

import requests

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# Update these to match YOUR repo:
GITHUB_OWNER = "W-Riley-01"          # e.g., "wrileyjr"
GITHUB_REPO  = "mlb-market-engine"      # the repo name
RELEASE_TAG  = "data-v1"                # bump when you upload new matrices

# Where to place files locally (relative to repo root)
DATA_DIR = Path("data")


class DataFile(NamedTuple):
    """One large data file to fetch."""
    asset_name: str   # filename of the asset attached to the GitHub Release
    local_path: Path  # where to write it locally
    min_bytes: int    # sanity check — abort if download is suspiciously tiny


DATA_FILES = [
    DataFile(
        asset_name="pitch_matrix.parquet",
        local_path=DATA_DIR / "pitch_matrix.parquet",
        min_bytes=50_000_000,   # full file is ~120MB; abort if <50MB
    ),
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
#  GitHub Releases API
# ---------------------------------------------------------------------------
def _auth_headers() -> dict:
    """
    Build request headers. GITHUB_TOKEN is required for private repos.
    On Streamlit Cloud, it should be added as a secret. On Actions,
    GITHUB_TOKEN is provided automatically by the runner.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_release_metadata() -> dict:
    """
    Pull the release info from GitHub's API. Returns the parsed JSON,
    which includes an `assets` list with download URLs.
    """
    url = (f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
           f"/releases/tags/{RELEASE_TAG}")
    log.info("Fetching release metadata from %s", url)
    resp = requests.get(url, headers=_auth_headers(), timeout=30)
    if resp.status_code == 404:
        raise RuntimeError(
            f"Release {RELEASE_TAG!r} not found in {GITHUB_OWNER}/{GITHUB_REPO}. "
            f"Did you create it on GitHub and attach the parquet?"
        )
    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub authentication failed. For private repos, set "
            "GITHUB_TOKEN env var with a token that has 'repo' scope."
        )
    resp.raise_for_status()
    return resp.json()


def _find_asset(release: dict, asset_name: str) -> dict | None:
    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            return asset
    return None


def _download_asset(asset: dict, dest: Path) -> None:
    """
    Stream the asset to disk. Uses the API endpoint with the Accept header
    'application/octet-stream' to get the actual binary instead of JSON.
    Streaming chunks avoids loading 100+ MB into memory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = _auth_headers()
    headers["Accept"] = "application/octet-stream"

    url = asset["url"]   # API URL, not browser_download_url — works for private repos
    log.info("Downloading %s (%.1f MB) → %s",
             asset["name"], asset["size"] / 1e6, dest)

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or asset["size"])
        downloaded = 0
        last_pct = -10
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                pct = int(downloaded * 100 / total) if total else 0
                if pct >= last_pct + 10:
                    log.info("  ... %d%% (%.1f / %.1f MB)",
                             pct, downloaded / 1e6, total / 1e6)
                    last_pct = pct

    log.info("Downloaded %s OK (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def ensure_data_files(force: bool = False) -> None:
    """
    Make sure every file in DATA_FILES is present locally. Downloads
    missing ones from the GitHub Release.

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
    release = _fetch_release_metadata()

    for df in missing:
        asset = _find_asset(release, df.asset_name)
        if asset is None:
            raise RuntimeError(
                f"Asset {df.asset_name!r} not found in release {RELEASE_TAG!r}. "
                f"Did you attach it to the release on GitHub?"
            )
        _download_asset(asset, df.local_path)

        size = df.local_path.stat().st_size
        if size < df.min_bytes:
            df.local_path.unlink()  # remove the bad file
            raise RuntimeError(
                f"Downloaded {df.asset_name} is only {size/1e6:.1f}MB "
                f"(expected ≥{df.min_bytes/1e6:.1f}MB). Aborting."
            )

    log.info("Bootstrap complete.")


if __name__ == "__main__":
    try:
        ensure_data_files(force="--force" in sys.argv)
    except Exception as e:
        log.error("Bootstrap failed: %s", e)
        sys.exit(1)