"""
Master Statcast ingestion v2.

Replaces the original data_ingestion.py and build_database.py with a single
unified script that:
  - Pulls a much richer column set (inning data, game_pk, running scores, etc.)
  - Downloads in monthly chunks with per-chunk saves (crash-safe)
  - Supports resume: re-running skips chunks already downloaded
  - Produces a clean master vault that replaces both the old vault files

IMPORTANT: This rewrites ./data/master_physics_vault.parquet when complete.
The existing vault should be deleted or renamed before running, or the
resume logic will assume existing data is current.

Usage:
    python data_ingestion.py                     # 2019-2026 full range
    python data_ingestion.py --year 2024         # single year
    python data_ingestion.py --start-year 2023 --end-year 2025
    python data_ingestion.py --consolidate-only  # stitch existing chunks only
"""

import argparse
import os
import time
import warnings

import pandas as pd
import pybaseball as pyb

warnings.filterwarnings('ignore')
pyb.cache.enable()


# ==========================================
# OUTPUT LOCATIONS
# ==========================================
DATA_DIR      = './data'
CHUNKS_DIR    = './data/statcast_chunks'
VAULT_PATH    = './data/master_physics_vault.parquet'


# ==========================================
# THE EXPANDED COLUMN SCHEMA
# Adds to the old schema:
#   - Inning context (inning, inning_topbot) — unlocks NRFI, F5 real boundaries
#   - Game ID (game_pk) — groups pitches by specific game
#   - Running scores (post_*_score) — tells us what happened on each play
#   - at_bat_number and pitch_number — for sequence analysis
#   - outs_when_up — pre-play situation
#   - on_1b, on_2b, on_3b — base state for situational analysis
# ==========================================
CORE_COLUMNS = [
    'game_pk', 'game_date', 'game_year', 'game_type',
    'home_team', 'away_team',

    'inning', 'inning_topbot',
    'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
    'at_bat_number', 'pitch_number',

    'pitcher', 'batter', 'stand', 'p_throws',

    'events', 'description', 'type',

    'post_away_score', 'post_home_score',
    'away_score', 'home_score',

    'pitch_type', 'release_speed',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'balls', 'strikes', 'zone',

    'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
]


# ==========================================
# CHUNK PLANNING
# Monthly chunks align with how Statcast data is distributed. Each chunk
# saves to its own parquet immediately, so a crash during August 2023
# doesn't lose the 4 months already downloaded.
# ==========================================
def month_chunks(year: int) -> list:
    return [
        (f"{year}-03-15", f"{year}-03-31", f"{year}-03"),
        (f"{year}-04-01", f"{year}-04-30", f"{year}-04"),
        (f"{year}-05-01", f"{year}-05-31", f"{year}-05"),
        (f"{year}-06-01", f"{year}-06-30", f"{year}-06"),
        (f"{year}-07-01", f"{year}-07-31", f"{year}-07"),
        (f"{year}-08-01", f"{year}-08-31", f"{year}-08"),
        (f"{year}-09-01", f"{year}-09-30", f"{year}-09"),
        (f"{year}-10-01", f"{year}-11-05", f"{year}-10"),
    ]


def chunk_path(label: str) -> str:
    return os.path.join(CHUNKS_DIR, f"statcast_{label}.parquet")


# ==========================================
# PER-CHUNK DOWNLOAD
# ==========================================
def download_chunk(start_date: str, end_date: str, label: str,
                   force: bool = False) -> bool:
    out_path = chunk_path(label)

    if os.path.exists(out_path) and not force:
        # Non-empty existing chunk means resume skip. Corrupt 0-byte files
        # from crashed runs are retried.
        if os.path.getsize(out_path) > 1024:
            print(f"  [SKIP] {label} already exists ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
            return True
        else:
            print(f"  [RETRY] {label} exists but is empty - re-downloading")

    print(f"  [FETCH] {label}: {start_date} to {end_date}")
    try:
        df = pyb.statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"  [ERROR] Statcast call failed for {label}: {e}")
        return False

    if df is None or df.empty:
        print(f"  [EMPTY] No data returned for {label} (likely offseason)")
        pd.DataFrame(columns=CORE_COLUMNS).to_parquet(out_path, engine='pyarrow', index=False)
        return True

    available = [c for c in CORE_COLUMNS if c in df.columns]
    missing   = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [!] {label}: columns missing from Statcast response: {missing}")

    df = df[available].copy()

    # Drop pitches with no pitch type or missing physics. Non-batted-ball
    # pitches (strikeouts, walks) are kept because their inning/events are
    # still useful.
    df = df.dropna(subset=['pitch_type', 'release_speed'])

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    df.to_parquet(out_path, engine='pyarrow', index=False)
    print(f"  [SAVED] {label}: {len(df):,} pitches")
    return True


# ==========================================
# STITCH CHUNKS INTO MASTER VAULT
# ==========================================
def consolidate_vault(chunks_dir: str = CHUNKS_DIR,
                      output_path: str = VAULT_PATH) -> None:
    if not os.path.exists(chunks_dir):
        print("[ERROR] No chunks directory found. Run download first.")
        return

    chunk_files = sorted([
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.startswith('statcast_') and f.endswith('.parquet')
    ])

    if not chunk_files:
        print("[ERROR] No chunk files found.")
        return

    print(f"\n[CONSOLIDATE] Stitching {len(chunk_files)} chunks into master vault...")

    dfs = []
    for cf in chunk_files:
        try:
            df = pd.read_parquet(cf, engine='pyarrow')
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [!] Could not read {cf}: {e}")

    if not dfs:
        print("[ERROR] No non-empty chunks to consolidate.")
        return

    master = pd.concat(dfs, ignore_index=True)

    # Deduplicate on the true unique key: game_pk + at_bat_number + pitch_number
    if all(c in master.columns for c in ['game_pk', 'at_bat_number', 'pitch_number']):
        before = len(master)
        master = master.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
        after = len(master)
        if before != after:
            print(f"  [DEDUP] Removed {before-after:,} duplicate pitches")

    master['game_date'] = pd.to_datetime(master['game_date'])
    sort_cols = ['game_date']
    for c in ['game_pk', 'at_bat_number', 'pitch_number']:
        if c in master.columns:
            sort_cols.append(c)
    master = master.sort_values(sort_cols)

    master.to_parquet(output_path, engine='pyarrow', index=False)

    print(f"\n[SUCCESS] Master vault written to {output_path}")
    print(f"  Total pitches: {len(master):,}")
    print(f"  Date range: {master['game_date'].min().date()} to {master['game_date'].max().date()}")
    if 'game_pk' in master.columns:
        print(f"  Unique games: {master['game_pk'].nunique():,}")
    print(f"  File size: {os.path.getsize(output_path)/1024/1024:.1f} MB")


# ==========================================
# DRIVER
# ==========================================
def run_ingestion(start_year: int = 2019, end_year: int = 2026,
                  consolidate: bool = True):
    print("=" * 60)
    print(f"  STATCAST INGESTION v2")
    print(f"  Range: {start_year}-{end_year}")
    print(f"  Columns: {len(CORE_COLUMNS)} fields")
    print("=" * 60)

    start_time = time.time()

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*40}\n  YEAR {year}\n{'='*40}")
        for start_date, end_date, label in month_chunks(year):
            if pd.Timestamp(start_date) > pd.Timestamp.today():
                print(f"  [SKIP] {label}: future date")
                continue
            download_chunk(start_date, end_date, label)

    elapsed = (time.time() - start_time) / 60
    print(f"\n[TIMING] Total download elapsed: {elapsed:.1f} minutes")

    if consolidate:
        consolidate_vault()


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statcast ingestion v2")
    parser.add_argument('--start-year', type=int, default=2019)
    parser.add_argument('--end-year',   type=int, default=2026)
    parser.add_argument('--year',       type=int, default=None,
                        help='Shortcut: download just this one year')
    parser.add_argument('--consolidate-only', action='store_true',
                        help='Skip download; just stitch existing chunks')
    args = parser.parse_args()

    if args.consolidate_only:
        consolidate_vault()
    elif args.year:
        run_ingestion(start_year=args.year, end_year=args.year)
    else:
        run_ingestion(start_year=args.start_year, end_year=args.end_year)