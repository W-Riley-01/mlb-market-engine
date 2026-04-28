"""
Post-ingestion verification.

Runs a series of sanity checks on the master vault and reports anything
that looks off. Catches silent ingestion issues before they poison
downstream analysis:
  - Missing columns from the expected schema
  - Month-level gaps in coverage
  - Chunks that exist but have suspiciously low row counts
  - NaN prevalence in critical columns
  - Duplicate rows slipping past the dedup step

Run once after data_ingestion.py completes. Takes ~30 seconds.
"""

import os
import pandas as pd


VAULT_PATH  = './data/master_physics_vault.parquet'
CHUNKS_DIR  = './data/statcast_chunks'

# These columns are load-bearing for downstream code. If any are missing
# or fully null, the NRFI/F5 work can't proceed.
CRITICAL_COLUMNS = [
    'game_pk', 'game_date', 'inning', 'inning_topbot',
    'pitcher', 'batter', 'events',
    'post_away_score', 'post_home_score',
    'pitch_type', 'release_speed',
]

EXPECTED_SCHEMA = [
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


def verify_chunks():
    """Inspect each monthly chunk individually for basic integrity."""
    print("=" * 70)
    print("  CHUNK-LEVEL INTEGRITY CHECK")
    print("=" * 70)

    if not os.path.isdir(CHUNKS_DIR):
        print(f"[ERROR] No chunks directory at {CHUNKS_DIR}")
        return False

    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR)
                          if f.startswith('statcast_') and f.endswith('.parquet')])

    if not chunk_files:
        print("[ERROR] No chunk files found")
        return False

    # Expected ~8 chunks per season (Mar-Oct). Typical in-season month has
    # 130k-180k pitches. Anything under 10k during an in-season month is
    # almost certainly a truncated download.
    print(f"\nFound {len(chunk_files)} chunk files\n")
    print(f"{'Chunk':<22}{'Size (MB)':<12}{'Rows':<12}{'Status':<30}")
    print("-" * 76)

    issues = []
    for f in chunk_files:
        path = os.path.join(CHUNKS_DIR, f)
        size_mb = os.path.getsize(path) / 1024 / 1024

        try:
            df = pd.read_parquet(path, engine='pyarrow')
            n_rows = len(df)

            # Heuristic: any month in Apr-Sep with < 10k pitches is suspicious
            month = f.replace('statcast_', '').replace('.parquet', '')
            month_num = int(month.split('-')[1]) if '-' in month else 0
            in_season = 4 <= month_num <= 9

            if n_rows == 0:
                status = "EMPTY (offseason OK)"
            elif in_season and n_rows < 10_000:
                status = "SUSPICIOUS (in-season)"
                issues.append(f)
            else:
                status = "OK"
        except Exception as e:
            n_rows = "ERROR"
            status = f"UNREADABLE: {e}"
            issues.append(f)

        print(f"{month:<22}{size_mb:<12.2f}{str(n_rows):<12}{status:<30}")

    if issues:
        print(f"\n[!] {len(issues)} chunks need attention: {issues}")
        print("    Delete the suspect chunk files and re-run data_ingestion.py")
        return False

    print("\n[OK] All chunks look healthy")
    return True


def verify_vault():
    """Deep check on the consolidated master vault."""
    print("\n" + "=" * 70)
    print("  MASTER VAULT VERIFICATION")
    print("=" * 70)

    if not os.path.exists(VAULT_PATH):
        print(f"[ERROR] No master vault at {VAULT_PATH}")
        print("        Run: python data_ingestion.py --consolidate-only")
        return False

    print(f"\nLoading {VAULT_PATH}...")
    vault = pd.read_parquet(VAULT_PATH, engine='pyarrow')
    vault['game_date'] = pd.to_datetime(vault['game_date'])

    # ---- Basic counts ----
    print(f"\n-- Basics --")
    print(f"  Total pitches:  {len(vault):,}")
    print(f"  Unique games:   {vault['game_pk'].nunique():,}" if 'game_pk' in vault.columns else "  [!] game_pk column missing")
    print(f"  Date range:     {vault['game_date'].min().date()} -> {vault['game_date'].max().date()}")
    print(f"  File size:      {os.path.getsize(VAULT_PATH)/1024/1024:.1f} MB")

    # ---- Schema check ----
    print(f"\n-- Schema check ({len(EXPECTED_SCHEMA)} expected columns) --")
    missing = [c for c in EXPECTED_SCHEMA if c not in vault.columns]
    extra   = [c for c in vault.columns if c not in EXPECTED_SCHEMA]
    if missing:
        print(f"  [!] MISSING COLUMNS: {missing}")
    if extra:
        print(f"  [info] Extra columns (OK, just unexpected): {extra}")
    if not missing and not extra:
        print(f"  [OK] Schema matches exactly")
    elif not missing:
        print(f"  [OK] All expected columns present")

    # ---- Critical column NaN check ----
    print(f"\n-- Critical column NaN rates --")
    for c in CRITICAL_COLUMNS:
        if c not in vault.columns:
            print(f"  [!] {c}: column missing entirely")
            continue
        nan_rate = vault[c].isna().mean()
        if nan_rate > 0.05:
            print(f"  [!] {c}: {nan_rate*100:.1f}% NaN (suspicious)")
        else:
            print(f"  [OK] {c}: {nan_rate*100:.2f}% NaN")

    # ---- Games per year ----
    print(f"\n-- Games per season --")
    vault['year'] = vault['game_date'].dt.year
    per_year = vault.groupby('year')['game_pk'].nunique() if 'game_pk' in vault.columns else vault.groupby('year').size() / 290
    for year, n_games in per_year.items():
        # Typical MLB regular season has ~2,430 games. Playoffs add ~35-40.
        # 2020 was COVID-shortened to ~900.
        expected = 900 if year == 2020 else 2430
        gap_pct = abs(n_games - expected) / expected * 100
        status = "OK" if gap_pct < 5 else ("LOW" if n_games < expected else "HIGH")
        print(f"  {year}: {n_games:>5} games  (expected ~{expected})  [{status}]")

    # ---- Inning coverage (NRFI-specific) ----
    if 'inning' in vault.columns:
        print(f"\n-- Inning coverage --")
        inning_counts = vault['inning'].value_counts().sort_index().head(10)
        for inn, count in inning_counts.items():
            print(f"  Inning {int(inn):>2}: {count:>10,} pitches")

    # ---- Top/bottom split sanity ----
    if 'inning_topbot' in vault.columns:
        print(f"\n-- Inning top/bot distribution --")
        tb = vault['inning_topbot'].value_counts()
        for label, count in tb.items():
            print(f"  {label}: {count:,}")

    # ---- Duplicate check ----
    if all(c in vault.columns for c in ['game_pk', 'at_bat_number', 'pitch_number']):
        n_dup = vault.duplicated(subset=['game_pk', 'at_bat_number', 'pitch_number']).sum()
        if n_dup > 0:
            print(f"\n[!] Found {n_dup:,} duplicate pitch rows")
        else:
            print(f"\n[OK] No duplicates")

    return len(missing) == 0


if __name__ == "__main__":
    chunks_ok = verify_chunks()
    vault_ok  = verify_vault()

    print("\n" + "=" * 70)
    if chunks_ok and vault_ok:
        print("  VERIFICATION PASSED - ready to proceed")
    else:
        print("  VERIFICATION FOUND ISSUES - see above")
    print("=" * 70)