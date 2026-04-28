import pandas as pd
import pybaseball as pyb
import warnings

# ==========================================
# 1. SETUP AND CACHING
# ==========================================
# Hide those annoying pandas warnings!
warnings.filterwarnings('ignore')
pyb.cache.enable()


# ==========================================
# 2. FETCHING STATCAST DATA
# ==========================================
def load_statcast_data(start_date, end_date):
    """Pulls ALL pitch-by-pitch data for a date range."""
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    pitches = pyb.statcast(start_dt=start_date, end_dt=end_date)
    return pitches


# ==========================================
# 3. SYNTHESIZING HITTERS AND PITCHERS
# ==========================================
def synthesize_hitters(pitches_df):
    """Groups the raw pitch data by batter ID and looks up their names."""
    print("Synthesizing Hitter Data...")

    # Group by 'batter' (which is their MLB ID number)
    hitter_stats = pitches_df.groupby('batter').agg(
        total_pitches_seen=('pitch_type', 'count'),
        avg_exit_velocity=('launch_speed', 'mean'),
        avg_xwOBA=('estimated_woba_using_speedangle', 'mean')
    ).reset_index()

    # Filter for batters who saw > 30 pitches
    hitter_stats = hitter_stats[hitter_stats['total_pitches_seen'] > 30]

    # --- NEW: Name Lookup ---
    # Get a list of the batter IDs
    batter_ids = hitter_stats['batter'].tolist()

    # Look up their names using pybaseball
    print("Looking up hitter names...")
    names_df = pyb.playerid_reverse_lookup(batter_ids, key_type='mlbam')

    # Combine first and last name into one column
    names_df['batter_name'] = names_df['name_first'].str.title() + ' ' + names_df['name_last'].str.title()

    # Merge the names back into our stats table based on their ID
    hitter_stats = hitter_stats.merge(names_df[['key_mlbam', 'batter_name']],
                                      left_on='batter',
                                      right_on='key_mlbam',
                                      how='left')

    return hitter_stats


def synthesize_pitchers(pitches_df):
    """Groups the raw pitch data by pitcher."""
    print("Synthesizing Pitcher Data...")

    fastballs = pitches_df[pitches_df['pitch_type'] == 'FF']

    pitcher_stats = fastballs.groupby('player_name').agg(
        fastballs_thrown=('pitch_type', 'count'),
        avg_fastball_velo=('release_speed', 'mean')
    ).reset_index()

    pitcher_stats = pitcher_stats[pitcher_stats['fastballs_thrown'] > 10]
    return pitcher_stats


# ==========================================
# 4. RUNNING THE SCRIPT
# ==========================================
if __name__ == "__main__":
    raw_data = load_statcast_data('2023-03-30', '2023-04-07')

    hitters = synthesize_hitters(raw_data)
    pitchers = synthesize_pitchers(raw_data)

    print("\n--- Top 5 ACTUAL Hitters by Average Exit Velocity ---")
    # Notice we print 'batter_name' now instead of 'player_name'
    print(hitters[['batter_name', 'total_pitches_seen', 'avg_exit_velocity', 'avg_xwOBA']].sort_values(
        by='avg_exit_velocity', ascending=False).head(5))

    print("\n--- Top 5 Hardest Throwing Pitchers ---")
    print(pitchers.sort_values(by='avg_fastball_velo', ascending=False).head(5))