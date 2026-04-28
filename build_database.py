import pandas as pd
import pybaseball as pyb
import warnings
import os

# ==========================================
# 1. SETUP
# ==========================================
warnings.filterwarnings('ignore')
pyb.cache.enable()

DATABASE_FILE = "my_mlb_database.parquet"


# ==========================================
# 2. THE AUTOMATED LOOPER
# ==========================================
def fetch_season_data(year):
    """
    Breaks a baseball season into monthly chunks and downloads them.
    This prevents memory crashes and server timeouts.
    """
    print(f"\n{'=' * 40}")
    print(f" STARTING DOWNLOAD FOR {year} SEASON ")
    print(f"{'=' * 40}")

    # Standard regular season + postseason months
    months = [
        (f"{year}-03-20", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-11-05")  # Catches October baseball!
    ]

    for start_date, end_date in months:
        print(f"Fetching: {start_date} to {end_date}...")

        # We use a try/except block. If MLB servers glitch on one month,
        # it won't crash our entire multi-year download!
        try:
            new_data = pyb.statcast(start_dt=start_date, end_dt=end_date)

            if new_data is None or new_data.empty:
                print("  -> No data found (likely the offseason).")
                continue

            # Append to vault
            if os.path.exists(DATABASE_FILE):
                existing_data = pd.read_parquet(DATABASE_FILE)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data.to_parquet(DATABASE_FILE, index=False)
            else:
                new_data.to_parquet(DATABASE_FILE, index=False)

            print(f"  -> Success! Added {len(new_data)} pitches.")

        except Exception as e:
            print(f"  -> ERROR fetching {start_date}: {e}")


# ==========================================
# 3. RUNNING THE PIPELINE
# ==========================================
if __name__ == "__main__":
    # IMPORTANT: If you want a perfectly clean database, it's best to delete
    # your old 'my_mlb_database.parquet' file before running this so we don't
    # duplicate the 2 weeks you already downloaded.

    # Loop through our target years
    target_years = [2023, 2024, 2025]

    for y in target_years:
        fetch_season_data(y)

    # Final check
    if os.path.exists(DATABASE_FILE):
        final_db = pd.read_parquet(DATABASE_FILE)
        print(f"\n*** VAULT COMPLETE! ***")
        print(f"Total Pitches in Database: {len(final_db)}")