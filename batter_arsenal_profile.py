"""
Per-batter, per-pitch-type damage profile.

For each batter we compute shrunken rates against each of the 9 core
Statcast pitch types:
    FF  four-seam fastball
    SI  sinker / two-seam
    FC  cutter
    SL  slider
    ST  sweeper        (only reliably labelled from 2023 on)
    CU  curveball
    KC  knuckle-curve
    CH  changeup
    FS  splitter

Rates computed (per batted ball) for each batter x pitch_type:
    single_rate, double_rate, triple_rate, hr_rate
    xwoba          (avg of either estimated_woba_using_speedangle from
                    Statcast, or xwoba_est from our home-grown lookup;
                    NaN if neither column is present)
    effective_n    (raw batted-ball count in that bucket)

All rates are shrunk toward PITCH-TYPE-SPECIFIC league baselines, not the
global baseline. A .200 HR rate vs sweepers means something very different
from a .200 HR rate vs sinkers; pooling them would blur real signal.

Shrinkage is classic empirical-Bayes:
    shrunk = (n / (n + k)) * batter_rate + (k / (n + k)) * league_rate
where k (SHRINKAGE_K) is the effective sample size at which the batter
earns 50% weight. Default 75 batted balls, roughly the point at which
xwOBA vs a single pitch type starts to stabilize.

Products, both written as parquet for fast backtest-safe lookups:
    ./data/batter_arsenal_profiles.parquet    keyed (batter, as_of_date, pitch_type)
    ./data/pitch_type_baselines.parquet       keyed (as_of_date, pitch_type)

Backtesting: pass historical as_of_date; function filters vault to
strictly-prior dates. No look-ahead.
"""

import os
import numpy as np
import pandas as pd


# ==========================================
# PATHS & CONSTANTS
# ==========================================
CQM_PATH = './data/contact_matrix_env.parquet'
PQM_PATH = './data/pitch_matrix.parquet'
PROFILE_PATH  = './data/batter_arsenal_profiles.parquet'
BASELINE_PATH = './data/pitch_type_baselines.parquet'

# The 9 pitch types we bucket into. Anything outside this list is dropped
# rather than force-mapped, because misclassification is worse than absence.
PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'KC', 'CH', 'FS']

# Empirical-Bayes shrinkage anchor. At n=SHRINKAGE_K batted balls vs a
# pitch type, the batter's observed rate gets 50% weight against the
# league baseline. Tunable.
SHRINKAGE_K = 75

# Minimum batted balls league-wide vs a pitch type for its baseline to
# be considered reliable. Below this we fall back to the global baseline
# (this mostly protects early-season sweeper data).
MIN_BASELINE_N = 500

# Global fallback baselines (per batted ball). Used when a pitch-type
# baseline is too thin.
GLOBAL_BASELINE = {
    'single': 0.230,  # per batted ball, not per PA
    'double': 0.072,
    'triple': 0.007,
    'hr':     0.053,
    'xwoba':  0.370,  # batted-ball xwOBA (ignores Ks/BBs)
}

# xwOBA column candidates, in preference order:
#   1. Statcast's native column if we ever ingest it
#   2. Our home-grown xwoba_lookup.py output
# The first one found in the CQM wins. None -> xwoba falls through to NaN.
XWOBA_COLUMN_CANDIDATES = ['estimated_woba_using_speedangle', 'xwoba_est']


def _find_xwoba_col(df: pd.DataFrame) -> str | None:
    for col in XWOBA_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


# ==========================================
# LEAGUE BASELINES PER PITCH TYPE
# ==========================================
def compute_pitch_type_baselines(cqm: pd.DataFrame,
                                  as_of: pd.Timestamp) -> pd.DataFrame:
    """
    League-average outcome rates and xwOBA per pitch type, computed from
    all batted balls on or before as_of. This is the regression target
    for the batter-level shrinkage.
    """
    df = cqm[cqm['game_date'] <= as_of]
    xwoba_col = _find_xwoba_col(df)
    has_xwoba = xwoba_col is not None

    rows = []
    for pt in PITCH_TYPES:
        subset = df[df['pitch_type'] == pt]
        n = len(subset)

        if n < MIN_BASELINE_N:
            # Too thin (e.g., early Sweeper era) — use global baseline
            row = {'pitch_type': pt, 'n': n, 'reliable': False, **GLOBAL_BASELINE}
        else:
            ev = subset['events'].value_counts()
            row = {
                'pitch_type': pt,
                'n': n,
                'reliable': True,
                'single': ev.get('single',   0) / n,
                'double': ev.get('double',   0) / n,
                'triple': ev.get('triple',   0) / n,
                'hr':     ev.get('home_run', 0) / n,
                'xwoba': (subset[xwoba_col].mean()
                          if has_xwoba else GLOBAL_BASELINE['xwoba']),
            }
        row['as_of_date'] = as_of.strftime('%Y-%m-%d')
        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================
# BATTER PROFILES
# ==========================================
def build_batter_arsenal_profiles(cqm: pd.DataFrame,
                                   as_of: pd.Timestamp,
                                   baselines: pd.DataFrame,
                                   shrinkage_k: int = SHRINKAGE_K) -> pd.DataFrame:
    """
    For each batter with batted-ball data as of the given date, compute
    per-pitch-type rates shrunk toward the pitch-type-specific league
    baseline.

    Only batter x pitch_type combinations with at least 1 batted ball
    are included. Callers looking up a missing combination should fall
    back to the baseline row instead.
    """
    df = cqm[(cqm['game_date'] <= as_of) & (cqm['pitch_type'].isin(PITCH_TYPES))]
    if df.empty:
        return pd.DataFrame()

    xwoba_col = _find_xwoba_col(df)
    has_xwoba = xwoba_col is not None

    # Index baselines for fast lookup inside the loop
    baseline_by_pt = baselines.set_index('pitch_type').to_dict('index')

    # Pre-compute event indicators once (vectorized) — then groupby
    df = df.copy()
    df['_single'] = (df['events'] == 'single').astype(float)
    df['_double'] = (df['events'] == 'double').astype(float)
    df['_triple'] = (df['events'] == 'triple').astype(float)
    df['_hr']     = (df['events'] == 'home_run').astype(float)

    agg_dict = {
        '_single': 'sum',
        '_double': 'sum',
        '_triple': 'sum',
        '_hr':     'sum',
        'events':  'size',   # batted-ball count
    }
    if has_xwoba:
        agg_dict[xwoba_col] = 'mean'

    grouped = df.groupby(['batter', 'pitch_type']).agg(agg_dict).reset_index()
    grouped.rename(columns={'events': 'n'}, inplace=True)

    # Vectorized raw rates
    grouped['raw_single'] = grouped['_single'] / grouped['n']
    grouped['raw_double'] = grouped['_double'] / grouped['n']
    grouped['raw_triple'] = grouped['_triple'] / grouped['n']
    grouped['raw_hr']     = grouped['_hr']     / grouped['n']

    # Apply shrinkage per row
    def shrink_row(row):
        pt = row['pitch_type']
        n  = row['n']
        w  = n / (n + shrinkage_k)
        bl = baseline_by_pt.get(pt, GLOBAL_BASELINE)

        shrunk = {
            'single_rate': w * row['raw_single'] + (1 - w) * bl['single'],
            'double_rate': w * row['raw_double'] + (1 - w) * bl['double'],
            'triple_rate': w * row['raw_triple'] + (1 - w) * bl['triple'],
            'hr_rate':     w * row['raw_hr']     + (1 - w) * bl['hr'],
        }

        if has_xwoba:
            raw_xwoba = row.get(xwoba_col, np.nan)
            if pd.isna(raw_xwoba):
                shrunk['xwoba'] = bl['xwoba']
            else:
                shrunk['xwoba'] = w * raw_xwoba + (1 - w) * bl['xwoba']
        else:
            shrunk['xwoba'] = np.nan

        return pd.Series(shrunk)

    shrunk_cols = grouped.apply(shrink_row, axis=1)
    out = pd.concat([
        grouped[['batter', 'pitch_type', 'n']].rename(columns={'n': 'effective_n'}),
        shrunk_cols,
    ], axis=1)
    out['as_of_date'] = as_of.strftime('%Y-%m-%d')

    return out[[
        'batter', 'as_of_date', 'pitch_type', 'effective_n',
        'single_rate', 'double_rate', 'triple_rate', 'hr_rate', 'xwoba',
    ]]


# ==========================================
# PITCHER ARSENAL MIX
# ==========================================
def compute_pitcher_arsenal_mix(pqm: pd.DataFrame,
                                  pitcher_id: int,
                                  as_of: pd.Timestamp,
                                  lookback_days: int | None = None) -> dict | None:
    """
    Returns {pitch_type: fraction} for the pitcher's usage up to as_of.
    Fractions sum to 1.0 over the 9 tracked pitch types; pitches outside
    the taxonomy are dropped before normalizing.

    If lookback_days is provided, only considers the most recent N days
    of usage — useful if a pitcher has recently added/dropped a pitch.
    None = full career to date.
    """
    df = pqm[(pqm['pitcher'] == pitcher_id) & (pqm['game_date'] <= as_of)]
    if lookback_days is not None:
        cutoff = as_of - pd.Timedelta(days=lookback_days)
        df = df[df['game_date'] >= cutoff]
    if df.empty:
        return None

    df = df[df['pitch_type'].isin(PITCH_TYPES)]
    if df.empty:
        return None

    counts = df['pitch_type'].value_counts()
    total  = counts.sum()
    return {pt: float(counts.get(pt, 0) / total) for pt in PITCH_TYPES}


# ==========================================
# RESOLVER-FACING BLENDER
# Takes a batter profile + pitcher arsenal and produces the
# arsenal-weighted rate for a given outcome key.
# ==========================================
def arsenal_weighted_rate(key: str,
                           batter_profile: dict | None,
                           pitcher_arsenal: dict | None,
                           baselines_by_pt: dict) -> float:
    """
    Args:
        key:              'single', 'double', 'triple', 'hr', or 'xwoba'
        batter_profile:   {pitch_type: {single_rate,...,xwoba,n}} from
                          BatterArsenalProfile.get(). None = no data.
        pitcher_arsenal:  {pitch_type: fraction} from compute_pitcher_arsenal_mix().
                          None = no data.
        baselines_by_pt:  {pitch_type: baseline_dict} for fallback when the
                          batter has never faced this pitch type.

    Returns:
        Arsenal-weighted rate. When the batter lacks data for a pitch type
        the pitcher throws, we substitute the league baseline for that
        pitch type (NOT zero, NOT the batter's overall rate) — because
        the best guess for an unseen matchup is league-average against
        that specific offering.
    """
    # Map our profile keys to the rate keys
    profile_key_map = {
        'single': 'single_rate',
        'double': 'double_rate',
        'triple': 'triple_rate',
        'hr':     'hr_rate',
        'xwoba':  'xwoba',
    }
    profile_key  = profile_key_map[key]
    baseline_key = key  # baseline dict uses raw keys (single/double/triple/hr/xwoba)

    # No pitcher arsenal at all — can't weight
    if pitcher_arsenal is None:
        return _global_fallback(key)

    weighted = 0.0
    total    = 0.0
    for pt, pct in pitcher_arsenal.items():
        if pct == 0:
            continue
        # Prefer batter-specific shrunken rate; fall back to pitch-type baseline
        rate = None
        if batter_profile is not None and pt in batter_profile:
            rate = batter_profile[pt].get(profile_key)
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            bl = baselines_by_pt.get(pt, GLOBAL_BASELINE)
            rate = bl.get(baseline_key, _global_fallback(key))

        weighted += rate * pct
        total    += pct

    if total == 0:
        return _global_fallback(key)
    return weighted / total


def _global_fallback(key: str) -> float:
    if key == 'xwoba':
        return GLOBAL_BASELINE['xwoba']
    return GLOBAL_BASELINE.get(key, 0.0)


# ==========================================
# SNAPSHOT BUILDER (nightly)
# ==========================================
def build_snapshot(as_of_date: str | None = None,
                    cqm_path: str = CQM_PATH,
                    profile_out: str = PROFILE_PATH,
                    baseline_out: str = BASELINE_PATH,
                    verbose: bool = True) -> None:
    """
    Computes profiles + baselines for a given as-of date and appends to
    the parquet store (overwrites any existing snapshot for the same date).
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    as_of = pd.Timestamp(as_of_date)

    if verbose:
        print(f"[ArsenalProfile] Building snapshot as of {as_of_date}")

    cqm = pd.read_parquet(cqm_path, engine='pyarrow')
    cqm['game_date'] = pd.to_datetime(cqm['game_date'])

    xwoba_col = _find_xwoba_col(cqm)
    if xwoba_col is None:
        if verbose:
            print("  [WARN] No xwOBA column found in CQM (neither "
                  "estimated_woba_using_speedangle nor xwoba_est). xwOBA column "
                  "will be NaN. Run xwoba_lookup.py to enrich the CQM with xwoba_est.")
    elif verbose:
        print(f"  xwOBA source: {xwoba_col}")

    if verbose:
        print(f"  CQM: {len(cqm):,} batted balls loaded")

    # Baselines first (profiles need them for shrinkage)
    baselines = compute_pitch_type_baselines(cqm, as_of)
    if verbose:
        reliable = baselines[baselines['reliable']]['pitch_type'].tolist()
        print(f"  Baselines computed. Reliable pitch types: {reliable}")

    profiles = build_batter_arsenal_profiles(cqm, as_of, baselines)
    if verbose:
        n_batters = profiles['batter'].nunique() if not profiles.empty else 0
        print(f"  Profiles: {len(profiles):,} batter x pitch_type rows "
              f"across {n_batters:,} batters")

    _append_snapshot(profiles,  profile_out,  'as_of_date')
    _append_snapshot(baselines, baseline_out, 'as_of_date')

    if verbose:
        print(f"  [SAVED] {profile_out}")
        print(f"  [SAVED] {baseline_out}")


def _append_snapshot(new_df: pd.DataFrame, path: str, date_col: str) -> None:
    if new_df.empty:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine='pyarrow')
        # Overwrite any prior snapshot for this date
        existing = existing[existing[date_col] != new_df[date_col].iloc[0]]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, engine='pyarrow', index=False)


# ==========================================
# RUNTIME LOOKUP
# Imported by resolver for fast per-request lookups.
# ==========================================
class BatterArsenalProfile:
    """
    In-memory lookup for batter arsenal profiles and pitch-type baselines.
    Loads both parquet files at construction; resolver keeps one instance.
    """
    def __init__(self,
                 profile_path: str = PROFILE_PATH,
                 baseline_path: str = BASELINE_PATH):
        self.profiles  = (pd.read_parquet(profile_path)
                          if os.path.exists(profile_path)  else pd.DataFrame())
        self.baselines = (pd.read_parquet(baseline_path)
                          if os.path.exists(baseline_path) else pd.DataFrame())

        # Sorted multi-index for fast .loc lookups
        if not self.profiles.empty:
            self._p_idx = self.profiles.set_index(
                ['as_of_date', 'batter']
            ).sort_index()
        else:
            self._p_idx = None

        if not self.baselines.empty:
            self._b_idx = self.baselines.set_index(
                ['as_of_date', 'pitch_type']
            ).sort_index()
        else:
            self._b_idx = None

    def get(self, as_of_date: str, batter_id: int) -> dict | None:
        """
        Returns {pitch_type: {single_rate, double_rate, triple_rate,
                              hr_rate, xwoba, n}}
        for every pitch type the batter has seen on or before as_of_date.
        None if the batter has no profile for that date.
        """
        if self._p_idx is None:
            return None
        try:
            rows = self._p_idx.loc[(as_of_date, batter_id)]
        except KeyError:
            return None
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        return {
            r['pitch_type']: {
                'single_rate': r['single_rate'],
                'double_rate': r['double_rate'],
                'triple_rate': r['triple_rate'],
                'hr_rate':     r['hr_rate'],
                'xwoba':       r['xwoba'],
                'n':           r['effective_n'],
            }
            for _, r in rows.iterrows()
        }

    def get_baselines(self, as_of_date: str) -> dict:
        """
        Returns {pitch_type: {single, double, triple, hr, xwoba, n, reliable}}
        for the given date. Falls back to GLOBAL_BASELINE keys if unavailable.
        """
        if self._b_idx is None:
            return {pt: dict(GLOBAL_BASELINE) for pt in PITCH_TYPES}
        try:
            rows = self._b_idx.loc[as_of_date]
        except KeyError:
            return {pt: dict(GLOBAL_BASELINE) for pt in PITCH_TYPES}
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        out = {}
        for pt, r in rows.iterrows():
            out[pt] = {
                'single': r['single'],
                'double': r['double'],
                'triple': r['triple'],
                'hr':     r['hr'],
                'xwoba':  r['xwoba'],
            }
        # Fill any missing pitch types with global
        for pt in PITCH_TYPES:
            if pt not in out:
                out[pt] = dict(GLOBAL_BASELINE)
        return out


# ==========================================
# DISPLAY HELPER: LETTER GRADE
# Continuous xwOBA -> A+/A/B+/.../F for the Streamlit app.
# Keep the continuous value for EV math; grade is display only.
# ==========================================
GRADE_CUTS = [
    ('A+', 0.420),
    ('A',  0.390),
    ('B+', 0.370),
    ('B',  0.350),
    ('C+', 0.335),
    ('C',  0.320),
    ('D',  0.300),
    ('F',  0.000),
]


def xwoba_to_grade(xwoba: float) -> str:
    if xwoba is None or (isinstance(xwoba, float) and np.isnan(xwoba)):
        return 'N/A'
    for grade, floor in GRADE_CUTS:
        if xwoba >= floor:
            return grade
    return 'F'


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    build_snapshot()