import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

VAULT_FILE = "my_mlb_database.parquet"
PLAYER_DICT_FILE = "player_dictionary.csv"
HITTER_PROFILES_FILE = "hitter_profiles_advanced.parquet"
PITCHER_PROFILES_FILE = "pitcher_profiles_advanced.parquet"


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def categorize_pitches(vault_df):
    fastballs = ['FF', 'SI', 'FC']
    vault_df['pitch_category'] = vault_df['pitch_type'].apply(
        lambda x: 'Fastball' if x in fastballs else 'Secondary'
    )

    # Identify Swings and Whiffs for plate discipline metrics
    swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip']
    whiffs = ['swinging_strike', 'swinging_strike_blocked']

    vault_df['is_swing'] = vault_df['description'].isin(swings).astype(int)
    vault_df['is_whiff'] = vault_df['description'].isin(whiffs).astype(int)

    return vault_df


# ==========================================
# 2. ADVANCED HITTER PROFILES
# ==========================================
def build_advanced_hitter_profiles(vault_df, player_dict):
    print("Building Advanced Hitter Profiles (with Platoon Splits & Launch Angle)...")

    # Group by Batter, Pitch Category, AND Pitcher Handedness ('p_throws')
    hitter_stats = vault_df.groupby(['batter', 'pitch_category', 'p_throws']).agg(
        pitches_seen=('pitch_type', 'count'),
        avg_xwOBA=('estimated_woba_using_speedangle', 'mean'),
        avg_launch_angle=('launch_angle', 'mean'),
        total_swings=('is_swing', 'sum'),
        total_whiffs=('is_whiff', 'sum')
    ).reset_index()

    # Calculate Whiff Rate (Misses / Swings)
    # Using a safe division to avoid dividing by zero
    hitter_stats['whiff_rate'] = hitter_stats.apply(
        lambda row: row['total_whiffs'] / row['total_swings'] if row['total_swings'] > 0 else 0,
        axis=1
    )

    # Filter for real sample sizes (minimum 50 pitches for specific splits)
    hitter_stats = hitter_stats[hitter_stats['pitches_seen'] > 50]

    # Create a unified column name for the pivot (e.g., 'Fastball_vs_R')
    hitter_stats['split_name'] = hitter_stats['pitch_category'] + '_vs_' + hitter_stats['p_throws']

    # Pivot the table!
    hitter_profiles = hitter_stats.pivot(
        index='batter',
        columns='split_name',
        values=['avg_xwOBA', 'avg_launch_angle', 'whiff_rate']
    ).reset_index()

    # Flatten the columns
    hitter_profiles.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in hitter_profiles.columns]

    # Merge names
    hitter_profiles = hitter_profiles.merge(player_dict, left_on='batter', right_on='key_mlbam', how='inner')
    hitter_profiles.to_parquet(HITTER_PROFILES_FILE, index=False)
    print(f"-> Saved {len(hitter_profiles)} Advanced Hitter Profiles.")


# ==========================================
# 3. ADVANCED PITCHER PROFILES
# ==========================================
def build_advanced_pitcher_profiles(vault_df, player_dict):
    print("Building Advanced Pitcher Profiles (with Platoon Splits & Whiff Rates)...")

    # Group by Pitcher, Pitch Category, AND Batter Stance ('stand')
    pitcher_stats = vault_df.groupby(['pitcher', 'pitch_category', 'stand']).agg(
        pitches_thrown=('pitch_type', 'count'),
        avg_velocity=('release_speed', 'mean'),
        avg_launch_angle_allowed=('launch_angle', 'mean'),
        total_swings=('is_swing', 'sum'),
        total_whiffs=('is_whiff', 'sum')
    ).reset_index()

    # Calculate Pitcher Whiff Rate (How often they make batters miss)
    pitcher_stats['whiff_rate_generated'] = pitcher_stats.apply(
        lambda row: row['total_whiffs'] / row['total_swings'] if row['total_swings'] > 0 else 0,
        axis=1
    )

    # Calculate Total Pitches just to filter out non-pitchers
    total_pitches = pitcher_stats.groupby('pitcher')['pitches_thrown'].transform('sum')
    pitcher_stats = pitcher_stats[total_pitches > 200]

    # Create the split name (e.g., 'Secondary_vs_L')
    pitcher_stats['split_name'] = pitcher_stats['pitch_category'] + '_vs_' + pitcher_stats['stand']

    # Pivot!
    pitcher_profiles = pitcher_stats.pivot(
        index='pitcher',
        columns='split_name',
        values=['avg_velocity', 'whiff_rate_generated', 'avg_launch_angle_allowed']
    ).reset_index()

    # Flatten columns
    pitcher_profiles.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pitcher_profiles.columns]

    # Merge names
    pitcher_profiles = pitcher_profiles.merge(player_dict, left_on='pitcher', right_on='key_mlbam', how='inner')
    pitcher_profiles.to_parquet(PITCHER_PROFILES_FILE, index=False)
    print(f"-> Saved {len(pitcher_profiles)} Advanced Pitcher Profiles.")


# ==========================================
# 4. RUNNING THE SCRIPT
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(VAULT_FILE) or not os.path.exists(PLAYER_DICT_FILE):
        print("Missing Vault or Player Dictionary!")
    else:
        vault = pd.read_parquet(VAULT_FILE)
        dictionary = pd.read_csv(PLAYER_DICT_FILE)

        vault = categorize_pitches(vault)

        build_advanced_hitter_profiles(vault, dictionary)
        build_advanced_pitcher_profiles(vault, dictionary)
        print("\n*** ADVANCED DATASET COMPLETE! ***")