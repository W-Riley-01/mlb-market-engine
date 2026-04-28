"""
Early Inning Pitcher Tracker — Historical Record.

Produces a clean stat card for every starting pitcher in the vault:
innings 1-3 performance, split by home/away and day/night.

This is a RECORD of what actually happened. No smoothing, no Bayesian
priors, no predictions. Raw facts only. A pitcher with 12 starts shows
as having 12 starts — the small-sample caveat is visible at a glance
via the n_starts column.

For predictive work (tonight's NRFI probability), see matchup.py and
prediction.py (to be built on top of this).

Output:
    ./data/pitcher_early_inning_record.parquet
    One row per (pitcher, inning, split) with the pitcher's record.

Usage:
    python early_inning_tracker.py                    # full rebuild
    python early_inning_tracker.py --pitcher 554430   # spot check one pitcher
    python early_inning_tracker.py --top 1            # top/bottom 10 in inning 1
"""

import argparse
import os
import time
from typing import Optional

import pandas as pd
import numpy as np
import requests


# ==========================================
# PATHS
# ==========================================
VAULT_PATH     = './data/master_physics_vault.parquet'
GAME_META_PATH = './data/game_metadata.parquet'
OUTPUT_PATH    = './data/pitcher_early_inning_record.parquet'

GAME_META_URL = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"

INNINGS_TRACKED = [1, 2, 3]


# ==========================================
# STEP 1: AGGREGATE PER-START-INNING
# One row per (pitcher, game, inning). Counts pitches, PAs, runs allowed,
# hits, Ks, walks. This is the atomic unit we build everything from.
# ==========================================
def aggregate_per_start_inning(vault: pd.DataFrame) -> pd.DataFrame:
    df = vault[vault['inning'].isin(INNINGS_TRACKED)].copy()

    # Home-team pitcher is on the mound during the TOP of the inning
    # (when the away team bats). Runs "allowed" by that pitcher show
    # up in post_away_score.
    df['opp_score'] = np.where(
        df['inning_topbot'] == 'Top',
        df['post_away_score'],
        df['post_home_score']
    )
    df['pitcher_team'] = np.where(
        df['inning_topbot'] == 'Top',
        df['home_team'],
        df['away_team']
    )
    df['pitcher_is_home'] = (df['inning_topbot'] == 'Top')

    # Terminal events (the pitch that ended each PA). These carry the
    # event label; non-terminal pitches have NaN in events.
    terminal_mask = df['events'].notna() & (df['events'] != '')

    gkeys = ['pitcher', 'game_pk', 'game_date', 'inning',
             'pitcher_team', 'pitcher_is_home']

    # Main aggregation
    agg = df.groupby(gkeys).agg(
        pitches=('pitch_type', 'size'),
        opp_score_start=('opp_score', 'min'),
        opp_score_end=('opp_score', 'max'),
    ).reset_index()
    agg['runs_allowed'] = agg['opp_score_end'] - agg['opp_score_start']

    # Terminal-event counts joined in separately
    terminals = df[terminal_mask].groupby(gkeys).agg(
        pa=('events', 'size'),
        hits=('events', lambda x: x.isin(['single', 'double', 'triple', 'home_run']).sum()),
        ks  =('events', lambda x: (x == 'strikeout').sum()),
        bbs =('events', lambda x: x.isin(['walk', 'intent_walk']).sum()),
    ).reset_index()

    agg = agg.merge(terminals, on=gkeys, how='left').fillna(
        {'pa': 0, 'hits': 0, 'ks': 0, 'bbs': 0}
    )
    agg = agg.drop(columns=['opp_score_start', 'opp_score_end'])

    # Scoreless indicator: the core NRFI/YRFI fact
    agg['scoreless'] = (agg['runs_allowed'] == 0).astype(int)

    return agg


# ==========================================
# STEP 2: GAME METADATA (day/night)
# Not in the vault — fetched once per game from MLB's live-feed API
# and cached locally so we only pay the cost on first run.
# ==========================================
def fetch_game_metadata(game_pk: int) -> Optional[dict]:
    try:
        r = requests.get(GAME_META_URL.format(game_pk), timeout=6)
        r.raise_for_status()
        data = r.json()
        dt_info = data.get('gameData', {}).get('datetime', {})
        return {
            'game_pk':   int(game_pk),
            'day_night': dt_info.get('dayNight'),
            'game_time': dt_info.get('dateTime'),
        }
    except Exception:
        return None


def build_or_update_game_metadata(needed_game_pks: list) -> pd.DataFrame:
    if os.path.exists(GAME_META_PATH):
        cached = pd.read_parquet(GAME_META_PATH, engine='pyarrow')
        have = set(cached['game_pk'].astype(int).tolist())
    else:
        cached = pd.DataFrame(columns=['game_pk', 'day_night', 'game_time'])
        have = set()

    missing = [int(g) for g in needed_game_pks if int(g) not in have]

    if not missing:
        print(f"[GameMeta] All {len(needed_game_pks):,} games already cached.")
        return cached

    print(f"[GameMeta] Fetching day/night for {len(missing):,} new games...")
    fetched = []
    for i, gpk in enumerate(missing):
        meta = fetch_game_metadata(gpk)
        if meta:
            fetched.append(meta)
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(missing)}")
            if fetched:
                partial = pd.concat([cached, pd.DataFrame(fetched)], ignore_index=True)
                partial.to_parquet(GAME_META_PATH, engine='pyarrow', index=False)
        time.sleep(0.05)

    if fetched:
        cached = pd.concat([cached, pd.DataFrame(fetched)], ignore_index=True)
        cached = cached.drop_duplicates(subset=['game_pk'])
        cached.to_parquet(GAME_META_PATH, engine='pyarrow', index=False)
        print(f"[GameMeta] Total cached: {len(cached):,}")

    return cached


# ==========================================
# STEP 3: COMPUTE SPLITS — RAW FACTS ONLY
# For each (pitcher, inning, split), count starts and tally outcomes.
# No regression, no smoothing. Small samples stay small — visible in n_starts.
# ==========================================
SPLITS = [
    ('overall',     lambda d: pd.Series([True] * len(d), index=d.index)),
    ('home',        lambda d: d['pitcher_is_home']),
    ('away',        lambda d: ~d['pitcher_is_home']),
    ('day',         lambda d: d['day_night'] == 'day'),
    ('night',       lambda d: d['day_night'] == 'night'),
    ('home_day',    lambda d: d['pitcher_is_home'] & (d['day_night'] == 'day')),
    ('home_night',  lambda d: d['pitcher_is_home'] & (d['day_night'] == 'night')),
    ('away_day',    lambda d: ~d['pitcher_is_home'] & (d['day_night'] == 'day')),
    ('away_night',  lambda d: ~d['pitcher_is_home'] & (d['day_night'] == 'night')),
]


def compute_splits(starts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print("[Splits] Computing per-pitcher historical records...")

    for inning in INNINGS_TRACKED:
        inn_df = starts[starts['inning'] == inning]

        for pitcher, pdf in inn_df.groupby('pitcher'):
            for split_name, split_fn in SPLITS:
                sub = pdf[split_fn(pdf)]
                n = len(sub)
                if n == 0:
                    continue

                total_pa = int(sub['pa'].sum())
                rows.append({
                    'pitcher':          int(pitcher),
                    'inning':           int(inning),
                    'split':            split_name,

                    # Volume
                    'n_starts':         n,
                    'total_pa':         total_pa,

                    # The primary facts
                    'scoreless_starts': int(sub['scoreless'].sum()),
                    'scoreless_pct':    round(sub['scoreless'].mean(), 3),
                    'total_runs':       int(sub['runs_allowed'].sum()),
                    'runs_per_inn':     round(sub['runs_allowed'].mean(), 3),

                    # Peripherals — "how they got there"
                    'total_hits':       int(sub['hits'].sum()),
                    'total_ks':         int(sub['ks'].sum()),
                    'total_bbs':        int(sub['bbs'].sum()),
                    'hits_per_pa':      round(sub['hits'].sum() / max(total_pa, 1), 3),
                    'k_per_pa':         round(sub['ks'].sum()   / max(total_pa, 1), 3),
                    'bb_per_pa':        round(sub['bbs'].sum()  / max(total_pa, 1), 3),

                    # Workload context
                    'avg_pitches':      round(sub['pitches'].mean(), 1),
                    'avg_pa':           round(sub['pa'].mean(), 1),
                })

    return pd.DataFrame(rows)


# ==========================================
# ORCHESTRATOR
# ==========================================
def build_tracker():
    print("=" * 70)
    print("  PITCHER EARLY-INNING HISTORICAL RECORD")
    print("=" * 70)

    if not os.path.exists(VAULT_PATH):
        print(f"[ERROR] Vault missing at {VAULT_PATH}.")
        return

    print(f"\n[1/4] Loading vault...")
    vault = pd.read_parquet(VAULT_PATH, engine='pyarrow')
    vault['game_date'] = pd.to_datetime(vault['game_date'])
    print(f"      {len(vault):,} pitches")

    print(f"\n[2/4] Aggregating to per-start-inning level...")
    starts = aggregate_per_start_inning(vault)
    print(f"      {len(starts):,} pitcher-start-innings")
    print(f"      {starts['pitcher'].nunique():,} unique pitchers")
    print(f"      {starts['game_pk'].nunique():,} unique games")

    print(f"\n[3/4] Fetching game day/night metadata...")
    meta = build_or_update_game_metadata(starts['game_pk'].unique().tolist())
    starts = starts.merge(meta[['game_pk', 'day_night']], on='game_pk', how='left')
    missing_dn = starts['day_night'].isna().sum()
    if missing_dn:
        print(f"      [!] {missing_dn:,} rows have no day/night (excluded from those splits only)")

    print(f"\n[4/4] Computing splits (historical record, no smoothing)...")
    record = compute_splits(starts)
    print(f"      {len(record):,} (pitcher, inning, split) rows")

    record.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
    print(f"\n[DONE] Record saved to {OUTPUT_PATH}")


# ==========================================
# LOOKUP HELPER — used by app and future matchup/prediction layers
# ==========================================
class PitcherRecord:
    """Fast lookup into the pitcher_early_inning_record parquet."""

    def __init__(self, path: str = OUTPUT_PATH):
        if os.path.exists(path):
            self.df = pd.read_parquet(path, engine='pyarrow')
            self._idx = self.df.set_index(['pitcher', 'inning', 'split']).sort_index()
        else:
            self.df = pd.DataFrame()
            self._idx = None

    def get(self, pitcher_id: int, inning: int = 1,
            split: str = 'overall') -> Optional[dict]:
        """Return one specific row, or None if not found."""
        if self._idx is None:
            return None
        try:
            row = self._idx.loc[(int(pitcher_id), int(inning), split)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        except KeyError:
            return None

    def all_splits(self, pitcher_id: int, inning: int = 1) -> pd.DataFrame:
        """Return every split for a pitcher in a given inning. For UI display."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df[
            (self.df['pitcher'] == int(pitcher_id)) &
            (self.df['inning'] == int(inning))
        ].copy()

    def nrfi_rate(self, pitcher_id: int, split: str = 'overall') -> Optional[float]:
        """Convenience: raw 1st-inning scoreless percentage."""
        row = self.get(pitcher_id, 1, split)
        return row['scoreless_pct'] if row else None


# ==========================================
# NAME LOOKUP (used only for CLI sanity prints)
# ==========================================
def load_name_map() -> dict:
    path = './data/player_dictionary.csv'
    if not os.path.exists(path):
        return {}
    try:
        d = pd.read_csv(path)
        return dict(zip(d['key_mlbam'].astype(int), d['player_name']))
    except Exception:
        return {}


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pitcher', type=int, default=None,
                        help='Print one pitcher\'s full record')
    parser.add_argument('--top', type=int, default=None, metavar='INNING',
                        help='Show top/bottom 10 for a given inning after build')
    parser.add_argument('--split', type=str, default='overall',
                        help='Split to use with --top (default: overall)')
    parser.add_argument('--min-starts', type=int, default=30,
                        help='Minimum starts to qualify for --top ranking (default: 30)')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip rebuild; just run the display queries')
    args = parser.parse_args()

    if not args.skip_build:
        build_tracker()

    names = load_name_map()
    def label(pid):
        return names.get(int(pid), f"ID {int(pid)}")

    rec = PitcherRecord()
    if rec.df.empty:
        print("\n[!] Record is empty.")
        exit(0)

    # Spot check a single pitcher
    if args.pitcher:
        print(f"\n--- {label(args.pitcher)} - full record (all splits, innings 1-3) ---")
        sub = rec.df[rec.df['pitcher'] == args.pitcher]
        if sub.empty:
            print(f"No data for pitcher {args.pitcher}")
        else:
            print(sub.to_string(index=False))

    # Top/bottom ranking for an inning
    if args.top:
        inn = args.top
        q = rec.df[
            (rec.df['inning'] == inn) &
            (rec.df['split'] == args.split) &
            (rec.df['n_starts'] >= args.min_starts)
        ].copy()

        if q.empty:
            print(f"\nNo pitchers match filters (inning={inn}, split={args.split}, min_starts={args.min_starts})")
        else:
            q['name'] = q['pitcher'].map(label)
            print(f"\n--- BEST inning-{inn} starters | split={args.split} | min {args.min_starts} starts ---")
            print(q.nlargest(10, 'scoreless_pct')[
                ['name', 'n_starts', 'scoreless_starts', 'scoreless_pct', 'runs_per_inn', 'hits_per_pa', 'k_per_pa']
            ].to_string(index=False))

            print(f"\n--- WORST inning-{inn} starters | split={args.split} | min {args.min_starts} starts ---")
            print(q.nsmallest(10, 'scoreless_pct')[
                ['name', 'n_starts', 'scoreless_starts', 'scoreless_pct', 'runs_per_inn', 'hits_per_pa', 'k_per_pa']
            ].to_string(index=False))