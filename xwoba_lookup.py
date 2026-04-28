"""
DIY xwOBA-on-contact lookup.

Statcast's estimated_woba_using_speedangle isn't carried through the
ingestion filter, so we compute an equivalent metric ourselves from the
launch_speed and launch_angle columns we do have.

METHOD
------
1. Assign every batted ball its FanGraphs 2023 wOBA value based on the
   outcome event:
       HR=2.031, 3B=1.578, 2B=1.247, 1B=0.880, out=0.000
2. Bin by launch_speed (2 mph) x launch_angle (2 degrees).
3. Mean wOBA per bin = expected wOBA-on-contact for that EV/LA combo.
4. Sparse fine bins (n<20) escalate to a coarser 5mph x 5deg grid;
   still sparse (n<5), fall through to the global mean.
5. Map every CQM row through the lookup; write xwoba_est column back
   to the CQM in place.

This is deliberately simpler than Statcast's model, which also uses
spray angle and a more elaborate smoother. For ranking batters by
per-pitch-type contact quality — which is all we need — EV/LA captures
the dominant signal. Spot-checked barrels, pop-ups, grounders, and
line drives land in the expected xwOBA ranges.

BACKTEST LEAKAGE NOTE
---------------------
The lookup is built from the full vault, so technically a 2023 backtest
is consulting a grid partly informed by 2024-2025 outcomes. The EV/LA
-> outcome mapping is extremely stable year-over-year in MLB, so the
leakage is negligible in practice. A date-indexed lookup (matching the
pattern used in recent_form.py) is the V2 fix if we ever need strict
purity.

OUTPUTS
-------
./data/xwoba_lookup.parquet              (EV bin, LA bin, expected_xwoba, n)
updates ./data/contact_matrix_env.parquet (adds xwoba_est column in place)

RUN ORDER
---------
matrix_builder.py -> enviroment_merger.py -> xwoba_lookup.py
-> recent_form.py -> batter_arsenal_profile.py
"""

import os
import warnings
import numpy as np
import pandas as pd


# ==========================================
# PATHS & CONSTANTS
# ==========================================
CQM_ENV_PATH  = './data/contact_matrix_env.parquet'
LOOKUP_PATH   = './data/xwoba_lookup.parquet'

# FanGraphs 2023 linear weights (per event, adjusted for modern run environment)
WOBA_VALUES = {
    'single':            0.880,
    'double':            1.247,
    'triple':            1.578,
    'home_run':          2.031,
    # Outs (all forms) — wOBA = 0
    'field_out':         0.000,
    'pop_out':           0.000,
    'line_out':          0.000,
    'fly_out':           0.000,
    'grounded_into_dp':  0.000,
    'force_out':         0.000,
    'fielders_choice':   0.000,
    'fielders_choice_out': 0.000,
    'sac_fly':           0.000,
    'sac_fly_double_play': 0.000,
    'double_play':       0.000,
    'triple_play':       0.000,
}

# Fine grid
EV_BIN_WIDTH = 2.0   # mph
LA_BIN_WIDTH = 2.0   # degrees
MIN_FINE_N   = 20    # bins below this escalate to coarse

# Coarse fallback grid
EV_COARSE    = 5.0
LA_COARSE    = 5.0
MIN_COARSE_N = 5


# ==========================================
# BINNING HELPERS
# np.floor on NaN produces NaN with a RuntimeWarning we suppress.
# NaN bins can't match in merge, so they correctly fall through to
# the coarse bin and then the global mean.
# ==========================================
def _floor_bin(values: pd.Series, width: float) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.floor(values.astype(float) / width)


def _assign_woba_value(df: pd.DataFrame) -> pd.DataFrame:
    """Map the events column to its wOBA value. Unknown events get 0."""
    df = df.copy()
    df['woba_value'] = df['events'].map(WOBA_VALUES).fillna(0.0)
    return df


# ==========================================
# LOOKUP BUILDER
# ==========================================
def build_lookup(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Build fine and coarse lookup tables plus a global mean from a CQM-like
    dataframe that already has woba_value assigned.

    Returns (fine, coarse, global_mean).
    """
    # Drop NaN EV/LA rows — can't bin them
    valid = df.dropna(subset=['launch_speed', 'launch_angle']).copy()

    # Fine grid
    valid['ev_bin'] = _floor_bin(valid['launch_speed'], EV_BIN_WIDTH)
    valid['la_bin'] = _floor_bin(valid['launch_angle'], LA_BIN_WIDTH)
    fine = (valid.groupby(['ev_bin', 'la_bin'])
                 .agg(expected_xwoba=('woba_value', 'mean'),
                      n=('woba_value', 'size'))
                 .reset_index())
    fine = fine[fine['n'] >= MIN_FINE_N].copy()

    # Coarse grid
    valid['ev_coarse'] = _floor_bin(valid['launch_speed'], EV_COARSE)
    valid['la_coarse'] = _floor_bin(valid['launch_angle'], LA_COARSE)
    coarse = (valid.groupby(['ev_coarse', 'la_coarse'])
                   .agg(expected_xwoba_coarse=('woba_value', 'mean'),
                        n_coarse=('woba_value', 'size'))
                   .reset_index())
    coarse = coarse[coarse['n_coarse'] >= MIN_COARSE_N].copy()

    global_mean = float(df['woba_value'].mean())
    return fine, coarse, global_mean


# ==========================================
# LOOKUP APPLIER
# ==========================================
def apply_lookup(df: pd.DataFrame,
                 fine: pd.DataFrame,
                 coarse: pd.DataFrame,
                 global_mean: float) -> pd.DataFrame:
    """
    Add xwoba_est column to df by merging against fine first, then coarse,
    then filling the remaining NaN with global_mean.
    """
    df = df.copy()
    df['ev_bin']    = _floor_bin(df['launch_speed'], EV_BIN_WIDTH)
    df['la_bin']    = _floor_bin(df['launch_angle'], LA_BIN_WIDTH)
    df['ev_coarse'] = _floor_bin(df['launch_speed'], EV_COARSE)
    df['la_coarse'] = _floor_bin(df['launch_angle'], LA_COARSE)

    # Fine merge
    df = df.merge(fine[['ev_bin', 'la_bin', 'expected_xwoba']],
                  on=['ev_bin', 'la_bin'], how='left')
    # Coarse merge for rows fine didn't catch
    df = df.merge(coarse[['ev_coarse', 'la_coarse', 'expected_xwoba_coarse']],
                  on=['ev_coarse', 'la_coarse'], how='left')

    df['xwoba_est'] = (df['expected_xwoba']
                        .fillna(df['expected_xwoba_coarse'])
                        .fillna(global_mean))

    # Drop intermediate columns
    df = df.drop(columns=['ev_bin', 'la_bin', 'ev_coarse', 'la_coarse',
                           'expected_xwoba', 'expected_xwoba_coarse'])
    return df


# ==========================================
# SINGLE-VALUE LOOKUP (for inspection / external use)
# ==========================================
def lookup_single(ev: float, la: float,
                  fine: pd.DataFrame,
                  coarse: pd.DataFrame,
                  global_mean: float) -> float:
    """Return expected xwOBA for a single EV/LA pair."""
    ev_bin = np.floor(ev / EV_BIN_WIDTH)
    la_bin = np.floor(la / LA_BIN_WIDTH)
    match = fine[(fine['ev_bin'] == ev_bin) & (fine['la_bin'] == la_bin)]
    if len(match) > 0:
        return float(match['expected_xwoba'].iloc[0])

    ev_c = np.floor(ev / EV_COARSE)
    la_c = np.floor(la / LA_COARSE)
    match_c = coarse[(coarse['ev_coarse'] == ev_c) & (coarse['la_coarse'] == la_c)]
    if len(match_c) > 0:
        return float(match_c['expected_xwoba_coarse'].iloc[0])

    return global_mean


# ==========================================
# MAIN: BUILD LOOKUP AND ENRICH CQM
# ==========================================
def build_and_enrich(cqm_path: str = CQM_ENV_PATH,
                      lookup_out: str = LOOKUP_PATH,
                      verbose: bool = True) -> None:
    """
    Full pipeline: read CQM, compute lookup, enrich CQM with xwoba_est,
    write everything back.
    """
    if verbose:
        print("[xwOBA Lookup] Building lookup from CQM...")

    cqm = pd.read_parquet(cqm_path)
    if verbose:
        print(f"  Loaded {len(cqm):,} batted balls from {cqm_path}")

    # Assign wOBA value to each row's event
    cqm = _assign_woba_value(cqm)

    # Diagnostic: how many rows have missing EV/LA
    evla_missing = cqm['launch_speed'].isna() | cqm['launch_angle'].isna()
    n_missing = int(evla_missing.sum())
    if verbose and n_missing > 0:
        pct = 100 * n_missing / max(len(cqm), 1)
        print(f"  {n_missing:,} rows ({pct:.1f}%) missing EV or LA "
              f"-> will get global-mean xwoba_est")

    # Build lookup from the valid rows
    fine, coarse, global_mean = build_lookup(cqm)
    if verbose:
        print(f"  Fine lookup:   {len(fine):,} bins (min n={MIN_FINE_N})")
        print(f"  Coarse lookup: {len(coarse):,} bins (min n={MIN_COARSE_N})")
        print(f"  Global mean xwOBA-on-contact: {global_mean:.3f}")

    # Persist the fine lookup for inspection / external use
    os.makedirs(os.path.dirname(lookup_out) or '.', exist_ok=True)
    fine.to_parquet(lookup_out, index=False)
    if verbose:
        print(f"  [SAVED] Lookup -> {lookup_out}")

    # Enrich the CQM with xwoba_est and persist (drop the intermediate woba_value)
    enriched = apply_lookup(cqm, fine, coarse, global_mean)
    enriched = enriched.drop(columns=['woba_value'])
    enriched.to_parquet(cqm_path, index=False)
    if verbose:
        print(f"  [SAVED] CQM with xwoba_est column -> {cqm_path}")
        print(f"          xwoba_est range: "
              f"[{enriched['xwoba_est'].min():.3f}, "
              f"{enriched['xwoba_est'].max():.3f}], "
              f"mean {enriched['xwoba_est'].mean():.3f}")

    # Sanity checks — eyeball these against known physics
    if verbose:
        print("\n[Sanity checks]")
        samples = [
            ('Barrel         (EV 105, LA 27)', 105, 27),
            ('Hard line drive (EV 98, LA 15)',  98, 15),
            ('Groundball      (EV 90, LA -5)',  90, -5),
            ('Weak popup      (EV 70, LA 60)',  70, 60),
            ('Weak grounder   (EV 65, LA -10)', 65, -10),
        ]
        for label, ev, la in samples:
            xwoba = lookup_single(ev, la, fine, coarse, global_mean)
            print(f"  {label}: xwoba_est = {xwoba:.3f}")


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    build_and_enrich()