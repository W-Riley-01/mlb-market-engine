"""
Weather ingestion v2.

Fetches historical daily weather for every MLB stadium covering the date
range present in the master vault. Designed to survive rate limits and
crashes:

  - Saves each team immediately after fetching (not batched at the end)
  - Supports resume: skips teams already fully covered in the existing log
  - Honors HTTP 429 responses with exponential backoff
  - Paces requests with a deliberate delay between stadiums
  - Splits long date ranges into smaller year-by-year requests so any
    single failure loses at most one year of one team, not everything

Run after: data_ingestion.py (needs the master vault to know date ranges)

Usage:
    python enviroment_ingestion.py              # resume/refresh normally
    python enviroment_ingestion.py --teams MIN NYM NYY   # just these teams
    python enviroment_ingestion.py --force      # re-fetch everything
"""

import argparse
import os
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings('ignore')

DATA_DIR     = "./data"
MASTER_FILE  = os.path.join(DATA_DIR, "master_physics_vault.parquet")
WEATHER_FILE = os.path.join(DATA_DIR, "game_weather_log.parquet")

# Base delay between successful requests (seconds). Open-Meteo's free tier
# allows ~10k calls/day. At 2 calls per team-year over 7 years x 30 teams =
# ~420 calls total, so we're well under the quota with a 2-second pace.
REQUEST_DELAY     = 2.0
RETRY_BASE_DELAY  = 60     # on 429, wait this long before first retry
MAX_RETRIES       = 3      # stop after this many 429s on the same request


# ==========================================
# STADIUM DATA (unchanged from v1)
# ==========================================
STADIUM_DATA = {
    'ARI': {'lat': 33.4455, 'lon': -112.0667, 'elevation': 331, 'dome': True},
    'ATL': {'lat': 33.8908, 'lon': -84.4678,  'elevation': 305, 'dome': False},
    'BAL': {'lat': 39.2840, 'lon': -76.6215,  'elevation': 13,  'dome': False},
    'BOS': {'lat': 42.3466, 'lon': -71.0972,  'elevation': 6,   'dome': False},
    'CHC': {'lat': 41.9484, 'lon': -87.6553,  'elevation': 182, 'dome': False},
    'CWS': {'lat': 41.8299, 'lon': -87.6338,  'elevation': 181, 'dome': False},
    'CIN': {'lat': 39.0979, 'lon': -84.5072,  'elevation': 148, 'dome': False},
    'CLE': {'lat': 41.4962, 'lon': -81.6852,  'elevation': 200, 'dome': False},
    'COL': {'lat': 39.7559, 'lon': -104.9942, 'elevation': 1581,'dome': False},
    'DET': {'lat': 42.3390, 'lon': -83.0485,  'elevation': 183, 'dome': False},
    'HOU': {'lat': 29.7573, 'lon': -95.3555,  'elevation': 14,  'dome': True},
    'KC':  {'lat': 39.0517, 'lon': -94.4803,  'elevation': 268, 'dome': False},
    'LAA': {'lat': 33.8003, 'lon': -117.8827, 'elevation': 48,  'dome': False},
    'LAD': {'lat': 34.0739, 'lon': -118.2400, 'elevation': 142, 'dome': False},
    'MIA': {'lat': 25.7783, 'lon': -80.2195,  'elevation': 4,   'dome': True},
    'MIL': {'lat': 43.0282, 'lon': -87.9712,  'elevation': 181, 'dome': True},
    'MIN': {'lat': 44.9817, 'lon': -93.2776,  'elevation': 255, 'dome': False},
    'NYM': {'lat': 40.7571, 'lon': -73.8458,  'elevation': 4,   'dome': False},
    'NYY': {'lat': 40.8296, 'lon': -73.9262,  'elevation': 9,   'dome': False},
    'OAK': {'lat': 37.7516, 'lon': -122.2005, 'elevation': 6,   'dome': False},
    'PHI': {'lat': 39.9061, 'lon': -75.1665,  'elevation': 9,   'dome': False},
    'PIT': {'lat': 40.4469, 'lon': -80.0057,  'elevation': 223, 'dome': False},
    'SD':  {'lat': 32.7076, 'lon': -117.1570, 'elevation': 10,  'dome': False},
    'SF':  {'lat': 37.7786, 'lon': -122.3893, 'elevation': 5,   'dome': False},
    'SEA': {'lat': 47.5914, 'lon': -122.3325, 'elevation': 6,   'dome': True},
    'STL': {'lat': 38.6226, 'lon': -90.1928,  'elevation': 136, 'dome': False},
    'TB':  {'lat': 27.7682, 'lon': -82.6534,  'elevation': 13,  'dome': True},
    'TEX': {'lat': 32.7373, 'lon': -97.0845,  'elevation': 171, 'dome': True},
    'TOR': {'lat': 43.6414, 'lon': -79.3894,  'elevation': 78,  'dome': True},
    'WSH': {'lat': 38.8730, 'lon': -77.0074,  'elevation': 5,   'dome': False},
}


# ==========================================
# OPEN-METEO REQUEST WITH RETRY
# ==========================================
def open_meteo_request(params: dict) -> dict | None:
    """
    Makes one Open-Meteo request with 429 backoff. Returns parsed JSON or None.
    On 429, waits RETRY_BASE_DELAY * 2^n seconds before each retry, up to MAX_RETRIES.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                if attempt == MAX_RETRIES:
                    print(f"  [!] Still rate-limited after {MAX_RETRIES} retries. Giving up on this request.")
                    return None
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            # Other network errors: short backoff, still retry
            if attempt == MAX_RETRIES:
                print(f"  [!] Network error after retries: {e}")
                return None
            time.sleep(5)
    return None


# ==========================================
# DOME GENERATOR
# Domes don't need API calls — we synthesize the fixed baseline.
# ==========================================
def build_dome_data(team: str, start_date: str, end_date: str) -> pd.DataFrame:
    coords = STADIUM_DATA[team]
    dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')
    return pd.DataFrame({
        'game_date':            dates,
        'home_team':            team,
        'temperature_f':        72.0,
        'wind_speed_mph':       0.0,
        'surface_pressure_hpa': 1013.25,
        'elevation_m':          coords['elevation'],
    })


# ==========================================
# OUTDOOR FETCHER (chunked by year)
# Splits a multi-year range into year chunks so a single failure only loses
# one year, not everything. Each chunk is one Open-Meteo call.
# ==========================================
def fetch_outdoor_weather(team: str, start_date: str, end_date: str) -> pd.DataFrame:
    coords = STADIUM_DATA[team]
    start_yr = int(start_date[:4])
    end_yr   = int(end_date[:4])

    all_chunks = []

    for year in range(start_yr, end_yr + 1):
        # Bound this year's window by the overall start/end
        y_start = f"{year}-01-01" if year != start_yr else start_date
        y_end   = f"{year}-12-31" if year != end_yr   else end_date

        params = {
            "latitude":         coords['lat'],
            "longitude":        coords['lon'],
            "start_date":       y_start,
            "end_date":         y_end,
            "daily":            "temperature_2m_max,wind_speed_10m_max",
            "hourly":           "surface_pressure",
            "timezone":         "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
        }

        print(f"  [{team}] {y_start} to {y_end}...")
        data = open_meteo_request(params)
        if data is None:
            print(f"  [{team}] Year {year} failed. Skipping this year.")
            continue

        try:
            daily = data['daily']
            df = pd.DataFrame({
                'game_date':      daily['time'],
                'home_team':      team,
                'temperature_f':  daily['temperature_2m_max'],
                'wind_speed_mph': daily['wind_speed_10m_max'],
                'elevation_m':    coords['elevation'],
            })

            hourly = pd.DataFrame({
                'time':     pd.to_datetime(data['hourly']['time']),
                'pressure': data['hourly']['surface_pressure'],
            })
            hourly['game_date'] = hourly['time'].dt.strftime('%Y-%m-%d')
            daily_pressure = hourly.groupby('game_date')['pressure'].mean().reset_index()

            df = df.merge(daily_pressure, on='game_date', how='left')
            df.rename(columns={'pressure': 'surface_pressure_hpa'}, inplace=True)

            all_chunks.append(df)
        except (KeyError, ValueError) as e:
            print(f"  [{team}] Parse error on year {year}: {e}")
            continue

        time.sleep(REQUEST_DELAY)  # be polite between chunks

    if not all_chunks:
        return pd.DataFrame()
    return pd.concat(all_chunks, ignore_index=True)


# ==========================================
# PER-TEAM SAVE HELPER
# Appends new rows to the weather log parquet without clobbering existing data.
# ==========================================
def save_team_weather(team_df: pd.DataFrame, weather_path: str):
    if team_df.empty:
        return

    team_df['game_date'] = pd.to_datetime(team_df['game_date'])

    if os.path.exists(weather_path):
        existing = pd.read_parquet(weather_path, engine='pyarrow')
        # Drop any rows for this team that overlap — we're refreshing those
        team = team_df['home_team'].iloc[0]
        existing = existing[existing['home_team'] != team]
        combined = pd.concat([existing, team_df], ignore_index=True)
    else:
        combined = team_df

    os.makedirs(os.path.dirname(weather_path), exist_ok=True)
    combined.to_parquet(weather_path, engine='pyarrow', index=False)


# ==========================================
# RESUME SUPPORT
# Checks which teams already have complete coverage for their required range.
# ==========================================
def teams_needing_fetch(team_date_ranges: pd.DataFrame, weather_path: str,
                        force: bool = False) -> list[str]:
    """Returns list of team codes that still need weather data."""
    if force or not os.path.exists(weather_path):
        return team_date_ranges['home_team'].tolist()

    existing = pd.read_parquet(weather_path, engine='pyarrow')
    existing['game_date'] = pd.to_datetime(existing['game_date'])
    needed = []

    for _, row in team_date_ranges.iterrows():
        team = row['home_team']
        req_start = pd.Timestamp(row['min'])
        req_end   = pd.Timestamp(row['max'])

        team_rows = existing[existing['home_team'] == team]
        if team_rows.empty:
            needed.append(team)
            continue

        have_start = team_rows['game_date'].min()
        have_end   = team_rows['game_date'].max()

        # If existing coverage doesn't span the required range, re-fetch
        if have_start > req_start or have_end < req_end:
            needed.append(team)

    return needed


# ==========================================
# ORCHESTRATOR
# ==========================================
def build_weather_log(force: bool = False, only_teams: list[str] | None = None):
    print("=" * 60)
    print("  WEATHER INGESTION v2")
    print("=" * 60)

    if not os.path.exists(MASTER_FILE):
        print(f"[ERROR] Master vault not found at {MASTER_FILE}")
        return

    print(f"\n[1/3] Scanning vault for date ranges...")
    df = pd.read_parquet(MASTER_FILE, columns=['game_date', 'home_team'])
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    team_date_ranges = df.groupby('home_team')['game_date'].agg(['min', 'max']).reset_index()

    if only_teams:
        team_date_ranges = team_date_ranges[team_date_ranges['home_team'].isin(only_teams)]

    print(f"      {len(team_date_ranges)} teams in vault")

    print(f"\n[2/3] Checking existing coverage...")
    to_fetch = teams_needing_fetch(team_date_ranges, WEATHER_FILE, force=force)
    already_have = [t for t in team_date_ranges['home_team'].tolist() if t not in to_fetch]

    if already_have:
        print(f"      Already complete: {', '.join(sorted(already_have))}")
    if not to_fetch:
        print("\n[DONE] All teams fully covered. Nothing to do.")
        return

    print(f"      Need to fetch: {', '.join(sorted(to_fetch))}")

    print(f"\n[3/3] Fetching weather for {len(to_fetch)} teams...")
    for _, row in team_date_ranges.iterrows():
        team = row['home_team']
        if team not in to_fetch:
            continue

        start_d = row['min']
        end_d   = row['max']

        if team not in STADIUM_DATA:
            print(f"  [!] No stadium data for {team}; skipping.")
            continue

        if STADIUM_DATA[team]['dome']:
            print(f"[{team}] Dome — using indoor baseline")
            team_df = build_dome_data(team, start_d, end_d)
        else:
            print(f"[{team}] Fetching outdoor weather {start_d} to {end_d}")
            team_df = fetch_outdoor_weather(team, start_d, end_d)

        if team_df.empty:
            print(f"  [!] No data saved for {team}")
            continue

        save_team_weather(team_df, WEATHER_FILE)
        print(f"  [SAVED] {team}: {len(team_df):,} daily records")

    print(f"\n[DONE] Weather log updated at {WEATHER_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teams', nargs='+', default=None,
                        help='Only fetch these specific teams')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch all teams even if already cached')
    args = parser.parse_args()

    build_weather_log(force=args.force, only_teams=args.teams)