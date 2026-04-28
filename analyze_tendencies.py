import pandas as pd
import pybaseball as pyb
import warnings
import os

warnings.filterwarnings('ignore')

VAULT_FILE = "my_mlb_database.parquet"
PLAYER_DICT_FILE = "player_dictionary.csv"


# ==========================================
# 1. THE PLAYER DICTIONARY BUILDER
# ==========================================
def get_player_dictionary(vault_df):
    """
    Creates or loads a permanent mapping of MLB ID numbers to actual player names.
    """
    if os.path.exists(PLAYER_DICT_FILE):
        print("Loading Player Dictionary...")
        return pd.read_csv(PLAYER_DICT_FILE)

    print("Building Player Dictionary for the first time... (This takes a minute)")

    # Get a list of every unique batter ID in our massive vault
    unique_ids = vault_df['batter'].dropna().unique().tolist()

    # Use pybaseball to look up all the names at once
    names_df = pyb.playerid_reverse_lookup(unique_ids, key_type='mlbam')

    # Format the names nicely
    names_df['player_name'] = names_df['name_first'].str.title() + ' ' + names_df['name_last'].str.title()

    # Keep only the ID and the Name
    player_dict = names_df[['key_mlbam', 'player_name']].copy()

    # Save it so we never have to download it again!
    player_dict.to_csv(PLAYER_DICT_FILE, index=False)
    print("Player Dictionary saved!")

    return player_dict


# ==========================================
# 2. PITCH TYPE PREFERENCE ENGINE
# ==========================================
def analyze_hitter_preferences(vault_df, player_dict):
    """
    Finds out which hitters crush fastballs vs. secondary pitches.
    """
    print("Analyzing hitter pitch preferences...")

    # Create broad pitch categories
    fastballs = ['FF', 'SI', 'FC']  # Four-seam, Sinker, Cutter

    # Create a new column categorizing the pitch
    # Using a lambda function to check if the pitch_type is in our fastball list
    vault_df['pitch_category'] = vault_df['pitch_type'].apply(
        lambda x: 'Fastball' if x in fastballs else 'Secondary'
    )

    # Group by Batter AND Pitch Category
    preferences = vault_df.groupby(['batter', 'pitch_category']).agg(
        pitches_seen=('pitch_type', 'count'),
        avg_exit_velo=('launch_speed', 'mean'),
        avg_xwOBA=('estimated_woba_using_speedangle', 'mean')
    ).reset_index()

    # Filter out small sample sizes (need to have seen at least 100 of that pitch type)
    preferences = preferences[preferences['pitches_seen'] > 100]

    # Merge our dictionary to get the actual names!
    preferences = preferences.merge(player_dict, left_on='batter', right_on='key_mlbam', how='inner')

    return preferences


# ==========================================
# 3. RUNNING THE SCRIPT
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(VAULT_FILE):
        print("Vault not found! Please run your downloader script first.")
    else:
        print("Loading massive data vault... (This might take a few seconds)")
        vault = pd.read_parquet(VAULT_FILE)

        # 1. Get our names
        dictionary = get_player_dictionary(vault)

        # 2. Analyze the hitters
        hitter_tendencies = analyze_hitter_preferences(vault, dictionary)

        # 3. Let's look at the best Fastball hitters!
        fastball_hitters = hitter_tendencies[hitter_tendencies['pitch_category'] == 'Fastball']

        print("\n--- TOP 10 FASTBALL HITTERS (By Expected wOBA) ---")
        # Sort by xwOBA and show the best
        top_fb = fastball_hitters.sort_values(by='avg_xwOBA', ascending=False).head(10)
        print(top_fb[['player_name', 'pitches_seen', 'avg_exit_velo', 'avg_xwOBA']].to_string(index=False))