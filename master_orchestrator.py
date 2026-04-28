import numpy as np
from datetime import datetime

# Import our custom modules
from resolver import MatchupResolver
from advanced_simulator import run_advanced_monte_carlo
from daily_scraper import fetch_todays_schedule, fetch_game_rosters


def run_daily_slate(iterations=2000):
    print("==================================================")
    print(f"  QUANTITATIVE MLB SYNDICATE ENGINE - {datetime.today().strftime('%Y-%m-%d')}  ")
    print("==================================================")

    # 1. Boot up the Physics Engine
    print("\n[SYSTEM] Booting Matchup Resolver...")
    try:
        resolver = MatchupResolver(
            pqm_path='./data/pitch_matrix.parquet',
            cqm_path='./data/contact_matrix_env.parquet'
        )
    except Exception as e:
        print(f"[FATAL ERROR] Could not load data matrices. Run Phase 1 & 2. Error: {e}")
        return

    # 2. Get Today's Slate
    slate = fetch_todays_schedule()
    if not slate:
        print("[SYSTEM] Slate is empty. Shutting down.")
        return

    # 3. Process Each Game
    for game in slate:
        if game['status'] not in ['Scheduled', 'Pre-Game']:
            continue

        print(f"\n==================================================")
        print(f" ANALYZING: {game['matchup']}")
        print(f"==================================================")

        rosters = fetch_game_rosters(game['game_id'])

        if not rosters or not rosters['Away']['lineup'] or not rosters['Home']['lineup']:
            print("[INFO] Official lineups not yet posted. Skipping simulation for now.")
            continue

        away = rosters['Away']
        home = rosters['Home']

        # 4. Execute the Advanced Monte Carlo
        print(f"[SYSTEM] Lineups locked. Running {iterations} physics simulations...")

        results = run_advanced_monte_carlo(
            resolver=resolver,
            # THE FIX: Pass the 'lineup_details' list of dictionaries instead of raw IDs
            away_lineup=away['lineup_details'],
            home_lineup=home['lineup_details'],
            away_starter=away['starter_id'],
            home_starter=home['starter_id'],
            away_bullpen=away['bullpen_ids'],
            home_bullpen=home['bullpen_ids'],
            density_ratio=1.0,
            iterations=iterations
        )

        # 5. Print the Daily Market Action Card
        print("\n--- THE MARKET ACTION CARD ---")

        # Game Markets
        home_ml_prob = 1.0 - results['away_win_prob']
        print(
            f"Moneyline -> {away['team_name']}: {results['away_win_prob'] * 100:.1f}% | {home['team_name']}: {home_ml_prob * 100:.1f}%")

        # F5 Markets
        home_f5_prob = 1.0 - results['f5_away_win_prob'] - results['f5_tie_prob']
        print(
            f"First 5 (F5) -> {away['team_name']}: {results['f5_away_win_prob'] * 100:.1f}% | {home['team_name']}: {home_f5_prob * 100:.1f}% | Tie: {results['f5_tie_prob'] * 100:.1f}%")

        # Inning & Total Markets
        print(f"NRFI (No Runs 1st Inning) Prob: {results['nrfi_prob'] * 100:.1f}%")
        print(f"Median Game Total: {results['median_total']} Runs")

        # Pitcher Prop Markets
        print(
            f"Pitcher Ks -> {away['starter_name']}: {results['away_pitcher_median_k']} | {home['starter_name']}: {results['home_pitcher_median_k']}")

        # Player Prop Highlights (Printing just the Leadoff hitters to keep console clean, but all 18 are in memory!)
        print("\n--- TOP PLAYER PROPS (Leadoff Hitters) ---")
        away_leadoff = away['lineup_details'][0]['name']
        home_leadoff = home['lineup_details'][0]['name']

        print(
            f"{away_leadoff} -> 1+ Hits: {results['player_props'][away_leadoff]['1+ Hits'] * 100:.1f}% | 2+ TB: {results['player_props'][away_leadoff]['2+ TB'] * 100:.1f}% | 1+ HR: {results['player_props'][away_leadoff]['1+ HR'] * 100:.1f}%")
        print(
            f"{home_leadoff} -> 1+ Hits: {results['player_props'][home_leadoff]['1+ Hits'] * 100:.1f}% | 2+ TB: {results['player_props'][home_leadoff]['2+ TB'] * 100:.1f}% | 1+ HR: {results['player_props'][home_leadoff]['1+ HR'] * 100:.1f}%")


if __name__ == "__main__":
    run_daily_slate(iterations=3500)