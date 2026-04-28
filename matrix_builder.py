"""
Matrix builder v2.

Reads the master vault (produced by data_ingestion.py v2) and derives the
two working matrices:
  - PQM (pitch_matrix.parquet):    one row per pitch, for pitcher arsenals
  - CQM (contact_matrix.parquet):  one row per batted ball, for hitter damage

Both now carry the new inning/game_pk/score columns so downstream code
(NRFI tracker, F5 analysis, etc.) can access them without re-querying
the vault.

CQM also carries estimated_woba_using_speedangle (Statcast's xwOBA on
contact) so the batter arsenal profile can rank damage by pitch type
using xwOBA rather than relying on noisy outcome rates alone.

Run after data_ingestion.py completes.
"""

import pandas as pd
import os


# ==========================================
# PATHS
# ==========================================
VAULT_PATH = './data/master_physics_vault.parquet'
PQM_PATH   = './data/pitch_matrix.parquet'
CQM_PATH   = './data/contact_matrix.parquet'


# ==========================================
# MATRIX SCHEMAS
# ==========================================
PQM_COLUMNS = [
    # Identity & date
    'game_pk', 'game_date', 'game_year', 'home_team', 'away_team',

    # Inning context (new)
    'inning', 'inning_topbot',
    'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
    'at_bat_number', 'pitch_number',

    # Players
    'pitcher', 'batter', 'stand', 'p_throws',

    # Play result
    'events', 'description', 'type',

    # Running scores (new)
    'post_away_score', 'post_home_score',
    'away_score', 'home_score',

    # Pitch characteristics
    'pitch_type', 'release_speed',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'balls', 'strikes', 'zone',
]

CQM_COLUMNS = [
    # Identity & date
    'game_pk', 'game_date', 'home_team', 'away_team',

    # Inning context (new — needed for "first-inning damage" analysis)
    'inning', 'inning_topbot', 'outs_when_up',

    # Players
    'pitcher', 'batter', 'stand', 'p_throws',

    # Play result
    'events',

    # Pitch type leading to contact
    'pitch_type',

    # Contact physics
    'launch_speed', 'launch_angle', 'hc_x', 'hc_y',

    # Statcast's xwOBA on contact — damage metric for arsenal profiles.
    # Derived from launch_speed + launch_angle, independent of defense.
    'estimated_woba_using_speedangle',
]


# ==========================================
# CONTACT EVENTS
# ==========================================
CONTACT_EVENTS = [
    'single', 'double', 'triple', 'home_run',
    'field_out', 'grounded_into_dp', 'fielders_choice',
    'force_out', 'sac_fly', 'line_out', 'pop_out',
]


# ==========================================
# BUILDERS
# ==========================================
def build_pqm(vault: pd.DataFrame, output_path: str = PQM_PATH) -> None:
    print("Building Pitch Quality Matrix (PQM)...")
    cols = [c for c in PQM_COLUMNS if c in vault.columns]
    missing = [c for c in PQM_COLUMNS if c not in vault.columns]
    if missing:
        print(f"  [!] Missing from vault (will be omitted): {missing}")

    pqm = vault[cols].copy()
    pqm.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"  [SAVED] PQM: {len(pqm):,} pitches -> {output_path}")


def build_cqm(vault: pd.DataFrame, output_path: str = CQM_PATH) -> None:
    print("Building Contact Quality Matrix (CQM)...")
    contact = vault[vault['events'].isin(CONTACT_EVENTS)].copy()

    cols = [c for c in CQM_COLUMNS if c in contact.columns]
    missing = [c for c in CQM_COLUMNS if c not in contact.columns]
    if missing:
        print(f"  [!] Missing from vault (will be omitted): {missing}")

    cqm = contact[cols].copy()
    cqm.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"  [SAVED] CQM: {len(cqm):,} batted balls -> {output_path}")


def build_matrices(vault_path: str = VAULT_PATH) -> None:
    print("=" * 60)
    print("  MATRIX BUILDER v2")
    print("=" * 60)

    if not os.path.exists(vault_path):
        print(f"[ERROR] Master vault not found at {vault_path}")
        print("        Run data_ingestion.py first.")
        return

    print(f"\nLoading master vault from {vault_path}...")
    vault = pd.read_parquet(vault_path, engine='pyarrow')
    print(f"  Loaded {len(vault):,} pitches")
    print(f"  Available columns: {len(vault.columns)}")

    build_pqm(vault)
    build_cqm(vault)

    print("\n[DONE] Matrices rebuilt. Remember to also rerun:")
    print("  1. enviroment_ingestion.py (weather log, if date range grew)")
    print("  2. enviroment_merger.py    (re-attach air density to CQM)")
    print("  3. xwoba_lookup.py         (rebuild xwOBA lookup + enrich CQM)")
    print("  4. recent_form.py          (rebuild recency snapshots)")
    print("  5. batter_arsenal_profile.py (rebuild arsenal profiles)")


if __name__ == "__main__":
    build_matrices()