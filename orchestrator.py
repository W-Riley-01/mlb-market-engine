import os
import pandas as pd
from datetime import datetime
# Import the function from our previous script
from data_ingestion import fetch_physics_data, DATA_DIR

MASTER_FILE = os.path.join(DATA_DIR, "master_physics_vault.parquet")


def build_multi_year_vault(start_year: int, end_year: int):
    """
    Orchestrates the downloading of multi-year Statcast data in safe, monthly chunks.
    Combines all cached chunks into a single master Parquet vault.
    """
    print(f"--- INITIALIZING MULTI-YEAR VAULT BUILD ({start_year}-{end_year}) ---")

    # Baseball season typically spans March (Spring Training/Opening Day) to November (World Series)
    season_months = ['03', '04', '05', '06', '07', '08', '09', '10', '11']
    all_chunks = []

    for year in range(start_year, end_year + 1):
        for month in season_months:
            # Create start and end dates for the month
            start_date = f"{year}-{month}-01"

            # Simple logic to find the last day of the month
            if month in ['04', '06', '09', '11']:
                end_date = f"{year}-{month}-30"
            elif month == '02':  # Just in case we expand to February later
                end_date = f"{year}-{month}-28"
            else:
                end_date = f"{year}-{month}-31"

            # Stop if the requested date is in the future (e.g., later in 2026)
            if datetime.strptime(start_date, "%Y-%m-%d") > datetime.now():
                print(f"[INFO] {start_date} is in the future. Stopping fetch loop.")
                break

            # Fetch the chunk using our cached pipeline
            chunk_df = fetch_physics_data(start_date, end_date)

            if not chunk_df.empty:
                all_chunks.append(chunk_df)

    if not all_chunks:
        print("[ERROR] No data collected.")
        return

    print("\n--- COMPILING MASTER VAULT ---")
    # Concatenate all monthly chunks into one massive DataFrame
    master_df = pd.concat(all_chunks, ignore_index=True)

    # Save to a single master Parquet file
    master_df.to_parquet(MASTER_FILE, engine='pyarrow', index=False)
    print(f"[SUCCESS] Master Vault built with {len(master_df)} pitches!")
    print(f"[SUCCESS] Saved to {MASTER_FILE}")


if __name__ == "__main__":
    # We pull 2023, 2024, 2025, and the active 2026 season
    build_multi_year_vault(2023, 2026)

    # Verification read
    if os.path.exists(MASTER_FILE):
        vault = pd.read_parquet(MASTER_FILE, engine='pyarrow')
        print(f"\nVerification: Vault spans from {vault['game_date'].min()} to {vault['game_date'].max()}")