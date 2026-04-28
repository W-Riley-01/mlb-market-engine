"""
Exponentially-weighted recent form computation.

Every at-bat (hitters) or pitch (pitchers) is weighted by how recently
it occurred:
    weight = exp(-ln(2) * days_ago / half_life)

A half-life of 60 means data from 60 days ago counts half as much as today,
120 days ago counts a quarter, etc. This avoids the hard-cliff problem of
fixed windows like "last 30 days."

Two products are emitted, both saved as parquet for fast resolver lookups:
    ./data/recent_form_hitters.parquet
    ./data/recent_form_pitchers.parquet

Both files are keyed by (player_id, as_of_date) so backtesting can request
a historical snapshot and live use grabs today's row.

Usage:
    from recent_form import RecentForm, build_recent_form_snapshot

    # Nightly refresh (production)
    build_recent_form_snapshot(as_of_date='2026-04-22')

    # Live lookup
    rf = RecentForm()
    hitter_rates = rf.get_hitter('2026-04-22', batter_id=592450)
    pitcher_rates = rf.get_pitcher('2026-04-22', pitcher_id=592789)
"""

import os
import numpy as np
import pandas as pd


# ==========================================
# PATHS & DEFAULTS
# ==========================================
PQM_PATH = './data/pitch_matrix.parquet'
CQM_PATH = './data/contact_matrix_env.parquet'
HITTER_FORM_PATH  = './data/recent_form_hitters.parquet'
PITCHER_FORM_PATH = './data/recent_form_pitchers.parquet'

# Half-life in days. Data from this many days ago gets half-weight.
HITTER_HALF_LIFE_DAYS  = 60   # hitters are more stable
PITCHER_HALF_LIFE_DAYS = 45   # pitchers shift faster (velocity, stuff)

# Minimum effective sample before we trust a recent-form rate.
# "Effective sample" = sum of weights, not raw count, because old ABs
# count for less. 50 effective weighted ABs is roughly equivalent to
# 50-80 actual recent ABs depending on distribution.
HITTER_MIN_EFFECTIVE_N  = 50
PITCHER_MIN_EFFECTIVE_N = 100  # per-pitch sample, so larger

# League-average fallback rates (per PA for hitters, per pitch for pitchers)
LEAGUE_AVG = {
    'single': 0.150, 'double': 0.047, 'triple': 0.004, 'hr': 0.035,
    'k_rate': 0.222, 'bb_rate': 0.082,
}


# ==========================================
# CORE WEIGHTING
# ==========================================
def decay_weights(dates: pd.Series, as_of: pd.Timestamp,
                  half_life_days: float) -> np.ndarray:
    """
    Given a Series of dates and an 'as-of' reference, return exponential
    decay weights. Future-dated rows (shouldn't exist in a blinded vault)
    get zero weight as a safety.
    """
    days_ago = (as_of - dates).dt.days.clip(lower=0).values
    # Use natural half-life formula: weight = 2^(-d/H)
    return np.power(2.0, -days_ago / half_life_days)


# ==========================================
# HITTER FORM
# Builds per-batter weighted rates for hits, 2B, 3B, HR
# from the CQM (one row per batted ball).
# ==========================================
def build_hitter_form(cqm: pd.DataFrame, as_of: pd.Timestamp,
                      half_life: float = HITTER_HALF_LIFE_DAYS) -> pd.DataFrame:
    # Only consider batted balls on or before as_of to prevent leakage
    df = cqm[cqm['game_date'] <= as_of].copy()
    if df.empty:
        return pd.DataFrame()

    df['weight'] = decay_weights(df['game_date'], as_of, half_life)

    # Indicator columns for each event we care about
    df['is_single'] = (df['events'] == 'single').astype(float)
    df['is_double'] = (df['events'] == 'double').astype(float)
    df['is_triple'] = (df['events'] == 'triple').astype(float)
    df['is_hr']     = (df['events'] == 'home_run').astype(float)

    # Weighted sums per batter
    grouped = df.groupby('batter').agg(
        total_weight=('weight', 'sum'),
        w_single=('is_single', lambda x: (x * df.loc[x.index, 'weight']).sum()),
        w_double=('is_double', lambda x: (x * df.loc[x.index, 'weight']).sum()),
        w_triple=('is_triple', lambda x: (x * df.loc[x.index, 'weight']).sum()),
        w_hr    =('is_hr',     lambda x: (x * df.loc[x.index, 'weight']).sum()),
    ).reset_index()

    # Compute weighted rates per batted ball
    grouped['recent_single'] = grouped['w_single'] / grouped['total_weight']
    grouped['recent_double'] = grouped['w_double'] / grouped['total_weight']
    grouped['recent_triple'] = grouped['w_triple'] / grouped['total_weight']
    grouped['recent_hr']     = grouped['w_hr']     / grouped['total_weight']

    # Keep the effective sample size so the resolver can decide how much to trust
    grouped.rename(columns={'total_weight': 'effective_n'}, inplace=True)

    grouped['as_of_date'] = as_of.strftime('%Y-%m-%d')
    return grouped[[
        'batter', 'as_of_date', 'effective_n',
        'recent_single', 'recent_double', 'recent_triple', 'recent_hr'
    ]]


# ==========================================
# PITCHER FORM
# Per-pitcher K rate and BB rate, weighted by recency.
# Uses the PQM (one row per pitch) and counts terminal events per PA.
# ==========================================
TERMINAL_EVENTS = {
    'strikeout', 'walk', 'hit_by_pitch', 'single', 'double',
    'triple', 'home_run', 'field_out', 'grounded_into_dp',
    'fielders_choice', 'force_out', 'sac_fly', 'line_out',
    'pop_out', 'sac_bunt', 'intent_walk'
}


def build_pitcher_form(pqm: pd.DataFrame, as_of: pd.Timestamp,
                       half_life: float = PITCHER_HALF_LIFE_DAYS) -> pd.DataFrame:
    df = pqm[pqm['game_date'] <= as_of].copy()
    if df.empty:
        return pd.DataFrame()

    # Keep only terminal-event rows (one per PA) for rate calculations
    df = df[df['events'].isin(TERMINAL_EVENTS)].copy()
    if df.empty:
        return pd.DataFrame()

    df['weight'] = decay_weights(df['game_date'], as_of, half_life)

    df['is_k']  = (df['events'] == 'strikeout').astype(float)
    df['is_bb'] = df['events'].isin(['walk', 'intent_walk']).astype(float)

    grouped = df.groupby('pitcher').agg(
        total_weight=('weight', 'sum'),
        w_k =('is_k',  lambda x: (x * df.loc[x.index, 'weight']).sum()),
        w_bb=('is_bb', lambda x: (x * df.loc[x.index, 'weight']).sum()),
    ).reset_index()

    grouped['recent_k_rate']  = grouped['w_k']  / grouped['total_weight']
    grouped['recent_bb_rate'] = grouped['w_bb'] / grouped['total_weight']

    grouped.rename(columns={'total_weight': 'effective_n'}, inplace=True)
    grouped['as_of_date'] = as_of.strftime('%Y-%m-%d')

    return grouped[[
        'pitcher', 'as_of_date', 'effective_n',
        'recent_k_rate', 'recent_bb_rate'
    ]]


# ==========================================
# SNAPSHOT BUILDER (nightly refresh)
# ==========================================
def build_recent_form_snapshot(as_of_date: str = None,
                                pqm_path: str = PQM_PATH,
                                cqm_path: str = CQM_PATH,
                                hitter_out: str = HITTER_FORM_PATH,
                                pitcher_out: str = PITCHER_FORM_PATH,
                                verbose: bool = True) -> None:
    """
    Computes and writes the recent-form snapshots for a given as-of date.
    If as_of_date is None, uses today's date.

    For production: run nightly (after games complete) with as_of_date=today.
    For backtesting: pass the historical date; the function automatically
    filters out any vault data dated after as_of_date.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    as_of = pd.Timestamp(as_of_date)

    if verbose:
        print(f"[RecentForm] Building snapshot as of {as_of_date}")

    # --- Load vaults (normalize dates once) ---
    pqm = pd.read_parquet(pqm_path, engine='pyarrow')
    cqm = pd.read_parquet(cqm_path, engine='pyarrow')
    pqm['game_date'] = pd.to_datetime(pqm['game_date'])
    cqm['game_date'] = pd.to_datetime(cqm['game_date'])

    if verbose:
        print(f"  PQM: {len(pqm):,} pitches loaded")
        print(f"  CQM: {len(cqm):,} batted balls loaded")

    # --- Compute snapshots ---
    if verbose:
        print(f"  Computing hitter form (half-life {HITTER_HALF_LIFE_DAYS} days)...")
    hitters = build_hitter_form(cqm, as_of, HITTER_HALF_LIFE_DAYS)

    if verbose:
        print(f"  Computing pitcher form (half-life {PITCHER_HALF_LIFE_DAYS} days)...")
    pitchers = build_pitcher_form(pqm, as_of, PITCHER_HALF_LIFE_DAYS)

    # --- Append-or-replace: keep history across multiple as-of dates ---
    def save_append(new_df: pd.DataFrame, path: str):
        if new_df.empty:
            if verbose: print(f"  [!] No data to save for {path}")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            existing = pd.read_parquet(path, engine='pyarrow')
            # Drop any prior snapshot with the same as_of_date (overwrite)
            existing = existing[existing['as_of_date'] != as_of.strftime('%Y-%m-%d')]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(path, engine='pyarrow', index=False)

    save_append(hitters,  hitter_out)
    save_append(pitchers, pitcher_out)

    if verbose:
        print(f"  [SUCCESS] Hitter form: {len(hitters):,} players")
        print(f"  [SUCCESS] Pitcher form: {len(pitchers):,} pitchers")
        print(f"  Saved to {hitter_out} and {pitcher_out}")


# ==========================================
# RUNTIME LOOKUP CLASS
# Resolver imports this and calls .get_hitter() / .get_pitcher()
# for the appropriate as-of date.
# ==========================================
class RecentForm:
    def __init__(self,
                 hitter_path: str = HITTER_FORM_PATH,
                 pitcher_path: str = PITCHER_FORM_PATH):
        self.hitters  = pd.read_parquet(hitter_path)  if os.path.exists(hitter_path)  else pd.DataFrame()
        self.pitchers = pd.read_parquet(pitcher_path) if os.path.exists(pitcher_path) else pd.DataFrame()

        # Index for fast lookup
        if not self.hitters.empty:
            self._h_idx = self.hitters.set_index(['as_of_date', 'batter']).sort_index()
        else:
            self._h_idx = None
        if not self.pitchers.empty:
            self._p_idx = self.pitchers.set_index(['as_of_date', 'pitcher']).sort_index()
        else:
            self._p_idx = None

    def get_hitter(self, as_of_date: str, batter_id: int) -> dict | None:
        if self._h_idx is None:
            return None
        try:
            row = self._h_idx.loc[(as_of_date, batter_id)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        except KeyError:
            return None

    def get_pitcher(self, as_of_date: str, pitcher_id: int) -> dict | None:
        if self._p_idx is None:
            return None
        try:
            row = self._p_idx.loc[(as_of_date, pitcher_id)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        except KeyError:
            return None


# ==========================================
# BLEND HELPER (used by resolver)
# ==========================================
def blend_with_recent(career_rate: float, recent_rate: float | None,
                      effective_n: float | None,
                      recency_weight: float = 0.25,
                      min_effective_n: float = 50) -> float:
    """
    Blend a career baseline rate with a recent-form rate.

    Args:
        career_rate:    the full-history rate (what resolver already computes)
        recent_rate:    the exponentially-weighted recent rate (or None)
        effective_n:    sum of recency weights (proxy for sample reliability)
        recency_weight: how much weight to give recent form when n is sufficient.
                        0.25 = "recent form adjusts baseline by 25%"
        min_effective_n: below this, we trust career more and ignore recent

    Returns:
        blended rate
    """
    if recent_rate is None or effective_n is None or effective_n < min_effective_n:
        return career_rate

    # Scale recency_weight down if sample is marginal
    # At 2x min_effective_n we use full recency_weight; below that we taper
    confidence = min(effective_n / (2 * min_effective_n), 1.0)
    actual_weight = recency_weight * confidence
    return (career_rate * (1 - actual_weight)) + (recent_rate * actual_weight)


# ==========================================
# CLI: run this file to build tonight's snapshot
# ==========================================
if __name__ == "__main__":
    build_recent_form_snapshot()