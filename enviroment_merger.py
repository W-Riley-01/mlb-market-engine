"""
Merge weather log into the Contact Quality Matrix (CQM) and compute
per-batted-ball air density ratios.

Run after:
    matrix_builder.py          (produces contact_matrix.parquet)
    enviroment_ingestion.py    (produces game_weather_log.parquet)
"""
import os
import pandas as pd


def merge_and_calculate_physics():
    print("--- LOADING MATRICES ---")
    try:
        cqm = pd.read_parquet('./data/contact_matrix.parquet', engine='pyarrow')
        weather = pd.read_parquet('./data/game_weather_log.parquet', engine='pyarrow')
        print(f"Loaded CQM: {len(cqm):,} batted balls.")
        print(f"Loaded Weather: {len(weather):,} game days.")
    except Exception as e:
        print(f"[ERROR] Missing CQM or Weather Log. {e}")
        return

    # The weather ingestor writes columns as `temperature_f` and `elevation_m`.
    # Downstream code expects `temperature` and `elevation`. Normalize here
    # so both scripts can stay independent.
    weather = weather.rename(columns={
        'temperature_f': 'temperature',
        'elevation_m':   'elevation',
    })

    # Safety: some older weather files may have the short names already.
    # Require both columns to exist before proceeding.
    needed = {'temperature', 'elevation', 'game_date', 'home_team'}
    missing = needed - set(weather.columns)
    if missing:
        print(f"[ERROR] Weather log missing columns: {missing}")
        print(f"        Available: {list(weather.columns)}")
        return

    print("\n--- MERGING ENVIRONMENTAL DATA ---")

    # Format dates identically to guarantee a clean merge
    cqm['game_date']     = pd.to_datetime(cqm['game_date']).dt.strftime('%Y-%m-%d')
    weather['game_date'] = pd.to_datetime(weather['game_date']).dt.strftime('%Y-%m-%d')

    env_cqm = cqm.merge(
        weather[['game_date', 'home_team', 'temperature', 'elevation']],
        on=['game_date', 'home_team'],
        how='inner'
    )

    print(f"Mapped weather to {len(env_cqm):,} batted balls.")

    if len(env_cqm) == 0:
        print("[ERROR] Merge produced zero rows. Check date formats and team codes.")
        return

    print("\n--- CALCULATING AIR DENSITY (ρ) ---")
    # Base density at sea level, 60°F is represented as 1.0.
    # Elevation drops density roughly ~3.5% per 1,000 feet.
    # Heat drops density roughly ~1% per 10°F above 60°F.
    env_cqm['density_ratio'] = (
        1.0
        - (env_cqm['elevation'] * 0.000035)
        - ((env_cqm['temperature'] - 60) * 0.001)
    )

    # Sanity check on the physics using the two most extreme parks
    print("\n--- BASELINE PHYSICS CHECK ---")
    coors_data  = env_cqm[env_cqm['home_team'] == 'COL']
    oracle_data = env_cqm[env_cqm['home_team'] == 'SF']

    if not coors_data.empty:
        print(f"Average Density Ratio in Colorado (Coors): {coors_data['density_ratio'].mean():.3f} (lower = thinner)")
    if not oracle_data.empty:
        print(f"Average Density Ratio in San Fran (Oracle): {oracle_data['density_ratio'].mean():.3f} (higher = thicker)")

    output_path = './data/contact_matrix_env.parquet'
    env_cqm.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n[SUCCESS] Saved to {output_path}")


if __name__ == "__main__":
    merge_and_calculate_physics()