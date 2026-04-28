import pandas as pd
import numpy as np
import os


def rebuild_weather_log(vault_path='./data/master_physics_vault.parquet',
                        output_path='./data/game_weather_log.parquet'):
    print("--- INITIATING WEATHER & STADIUM REBUILDER ---")

    # 1. Load the MASTER VAULT to find every unique game we have data for
    # (The master vault contains the 'home_team' column we need)
    print(f"Scanning {vault_path} for unique games...")
    try:
        df = pd.read_parquet(vault_path, columns=['game_date', 'home_team'])
    except Exception as e:
        print(f"[ERROR] Could not load Master Vault: {e}")
        return

    # Drop duplicates to get a clean list of every game played
    unique_games = df.drop_duplicates().copy()
    print(f"Found {len(unique_games)} unique game days in the vault.")

    # 2. Stadium Elevation Map (Feet above sea level)
    stadium_elevations = {
        'COL': 5200, 'ARI': 1082, 'ATL': 975, 'MIN': 812, 'KC': 750,
        'LAD': 267, 'MIL': 593, 'STL': 455, 'PIT': 743, 'CIN': 683,
        'CWS': 587, 'CHC': 598, 'CLE': 682, 'DET': 593, 'TOR': 24,
        'BOS': 20, 'NYY': 53, 'NYM': 13, 'PHI': 39, 'BAL': 130,
        'WSH': 25, 'MIA': 9, 'TB': 42, 'TEX': 603, 'HOU': 38,
        'OAK': 42, 'SF': 15, 'SD': 13, 'LAA': 160, 'SEA': 15
    }

    # 3. Build the Environment Data
    print("Mapping stadium elevations and calculating historical thermodynamics...")

    # Map elevation (Default to 50ft if team string doesn't perfectly match)
    unique_games['elevation'] = unique_games['home_team'].map(stadium_elevations).fillna(50)

    # Extract the month to approximate the temperature
    unique_games['game_date'] = pd.to_datetime(unique_games['game_date'])
    unique_games['month'] = unique_games['game_date'].dt.month

    def approximate_temp(month):
        if month in [3, 4]:
            return np.random.normal(60, 5)  # Spring
        elif month in [5, 6]:
            return np.random.normal(72, 5)  # Early Summer
        elif month in [7, 8]:
            return np.random.normal(85, 5)  # High Summer
        elif month in [9, 10]:
            return np.random.normal(68, 5)  # Fall
        else:
            return 65

    unique_games['temperature'] = unique_games['month'].apply(approximate_temp)

    # Drop the temporary month column
    unique_games = unique_games.drop(columns=['month'])

    # Ensure date is string format to match the CQM later
    unique_games['game_date'] = unique_games['game_date'].dt.strftime('%Y-%m-%d')

    # 4. Save the new Weather Log
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_games.to_parquet(output_path, engine='pyarrow', index=False)

    print(f"[SUCCESS] Rebuilt Weather Log saved to {output_path}")


if __name__ == "__main__":
    rebuild_weather_log()