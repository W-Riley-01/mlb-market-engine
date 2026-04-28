"""
Walk-forward backtester with rich logging.

Design principle: the simulation is expensive, the analysis is cheap.
So we run the simulation ONCE and log every raw probability and every
actual outcome to a CSV. Then analysis.py can slice the data many
different ways without ever re-running the sim.

The ledger captures per-game:
  - All model probabilities (ML, F5, NRFI, median total, K distributions)
  - Full player prop distributions
  - All actual outcomes (final scores, F5 scores, NRFI, per-pitcher Ks,
    per-batter hits/HR/RBI/TB/SB)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import requests

from resolver import MatchupResolver
from advanced_simulator import run_advanced_monte_carlo
from daily_scraper import fetch_todays_schedule, fetch_game_rosters
from odds_ingestion import OddsBook


LEDGER_PATH = './data/backtest_ledger.csv'
PQM_PATH    = './data/pitch_matrix.parquet'
CQM_PATH    = './data/contact_matrix_env.parquet'
ODDS_PATH   = './data/MLB_Basic.csv'


# MLB Stats API returns full team names ("Atlanta Braves") but the odds CSV uses
# abbreviations ("ATL"). This map bridges the two. It is intentionally verbose
# to be self-documenting — no fuzzy matching, no silent failures.
TEAM_NAME_TO_ABBR = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',  # pre-2022 name
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Florida Marlins': 'MIA',  # pre-2012 name
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM', 'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK', 'Athletics': 'OAK',  # 2025+ rebrand
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}


def team_abbr(full_name: str) -> str:
    """Converts 'Atlanta Braves' -> 'ATL'. Returns None with a warning if unknown."""
    abbr = TEAM_NAME_TO_ABBR.get(full_name)
    if abbr is None:
        print(f"  [!] Unknown team name: {full_name!r}")
    return abbr


# ==========================================
# HISTORICAL OUTCOME FETCHING
# Pulls both linescore and boxscore to capture every gradable market.
# ==========================================
def fetch_historical_outcomes(game_id: int) -> dict:
    out = {
        'final_away': None, 'final_home': None,
        'f5_away': 0, 'f5_home': 0,
        'nrfi': None,
        'pitcher_ks': {},    # {pitcher_id (str): strikeouts}
        'pitcher_bf': {},    # {pitcher_id (str): batters faced}
        'batter_stats': {},  # {batter_id (str): {hits, hr, rbi, tb, sb}}
    }

    # --- Linescore: inning-by-inning runs ---
    try:
        ls = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_id}/linescore",
            timeout=8
        ).json()
        innings = ls.get('innings', [])

        if innings:
            inn1 = innings[0]
            r1a = inn1.get('away', {}).get('runs', 0)
            r1h = inn1.get('home', {}).get('runs', 0)
            out['nrfi'] = (r1a == 0 and r1h == 0)

        for inn in innings[:5]:
            out['f5_away'] += inn.get('away', {}).get('runs', 0)
            out['f5_home'] += inn.get('home', {}).get('runs', 0)

        teams = ls.get('teams', {})
        out['final_away'] = teams.get('away', {}).get('runs', 0)
        out['final_home'] = teams.get('home', {}).get('runs', 0)
    except Exception:
        return None

    # --- Boxscore: player-level stats ---
    try:
        bs = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore",
            timeout=8
        ).json()

        for side in ['away', 'home']:
            players = bs['teams'][side].get('players', {})
            for _, pdata in players.items():
                pid = str(pdata['person']['id'])
                stats = pdata.get('stats', {})

                pstat = stats.get('pitching', {})
                if pstat and 'strikeOuts' in pstat:
                    out['pitcher_ks'][pid] = pstat.get('strikeOuts', 0)
                    out['pitcher_bf'][pid] = pstat.get('battersFaced', 0)

                bstat = stats.get('batting', {})
                if bstat and bstat.get('atBats', 0) > 0:
                    out['batter_stats'][pid] = {
                        'hits': bstat.get('hits', 0),
                        'hr':   bstat.get('homeRuns', 0),
                        'rbi':  bstat.get('rbi', 0),
                        'tb':   bstat.get('totalBases', 0),
                        'sb':   bstat.get('stolenBases', 0),
                    }
    except Exception:
        pass  # partial data is still usable

    return out


# ==========================================
# LEDGER I/O — incremental append with resume
# ==========================================
def load_processed_game_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        existing = pd.read_csv(path, usecols=['game_id'])
        return set(existing['game_id'].astype(str).tolist())
    except Exception:
        return set()


def append_rows(rows: list, path: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode='a', index=False, header=header)


# ==========================================
# MAIN BACKTEST LOOP
# ==========================================
def run_backtest(start_date: str, end_date: str,
                 iterations: int = 1000, output_path: str = LEDGER_PATH):

    print("=" * 60)
    print(f"  WALK-FORWARD BACKTEST  {start_date} to {end_date}")
    print(f"  Iterations per game: {iterations}")
    print("=" * 60)

    # --- Load master vaults into memory once ---
    print("\n[SYSTEM] Loading master vaults...")
    master_pqm = pd.read_parquet(PQM_PATH, engine='pyarrow')
    master_cqm = pd.read_parquet(CQM_PATH, engine='pyarrow')
    master_pqm['game_date'] = pd.to_datetime(master_pqm['game_date'])
    master_cqm['game_date'] = pd.to_datetime(master_cqm['game_date'])

    resolver = MatchupResolver(pqm_path=PQM_PATH, cqm_path=CQM_PATH)

    # --- Load historical odds ---
    try:
        oddsbook = OddsBook(ODDS_PATH)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Continuing without odds. Ledger will log model predictions "
              "but EV analysis will not be possible.")
        oddsbook = None

    # --- Resume support ---
    already_done = load_processed_game_ids(output_path)
    if already_done:
        print(f"[RESUME] Skipping {len(already_done)} previously processed games.")

    current_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt     = datetime.strptime(end_date,   '%Y-%m-%d')

    pending_rows = []
    games_processed = 0

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')
        cutoff   = pd.Timestamp(date_str)

        # --- Data blinding: resolver can only see strictly prior games ---
        resolver.pqm = master_pqm[master_pqm['game_date'] < cutoff]
        resolver.cqm = master_cqm[master_cqm['game_date'] < cutoff]

        slate = fetch_todays_schedule(date_str)
        if not slate:
            current_dt += timedelta(days=1)
            continue

        finals = [g for g in slate if g['status'] == 'Final'
                  and str(g['game_id']) not in already_done]

        if finals:
            print(f"\n[{date_str}] Processing {len(finals)} final games...")

        for game in finals:
            rosters = fetch_game_rosters(game['game_id'])
            if not rosters or not rosters['Away']['lineup'] or not rosters['Home']['lineup']:
                continue

            away, home = rosters['Away'], rosters['Home']

            # --- Run the sim ---
            try:
                results = run_advanced_monte_carlo(
                    resolver=resolver,
                    away_lineup=away['lineup_details'],
                    home_lineup=home['lineup_details'],
                    away_starter=away['starter_id'],
                    home_starter=home['starter_id'],
                    away_bullpen=away['bullpen_ids'],
                    home_bullpen=home['bullpen_ids'],
                    density_ratio=1.0,
                    iterations=iterations,
                )
            except Exception as e:
                print(f"  [!] Sim failed on {game['matchup']}: {e}")
                continue

            # --- Grade against historical reality ---
            outcomes = fetch_historical_outcomes(game['game_id'])
            if outcomes is None:
                continue

            # --- Look up historical closing odds ---
            odds_row = None
            if oddsbook is not None:
                away_abbr = team_abbr(away['team_name'])
                home_abbr = team_abbr(home['team_name'])
                if away_abbr and home_abbr:
                    odds_row = oddsbook.get(date_str, away_abbr, home_abbr)

            # --- Build the rich log row ---
            row = {
                'date':               date_str,
                'game_id':            game['game_id'],
                'matchup':            game['matchup'],
                'away_team':          away['team_name'],
                'home_team':          home['team_name'],
                'away_starter_id':    away['starter_id'],
                'home_starter_id':    home['starter_id'],
                'away_starter_name':  away['starter_name'],
                'home_starter_name':  home['starter_name'],

                # Model predictions
                'model_away_win':     results['away_win_prob'],
                'model_f5_away':      results['f5_away_win_prob'],
                'model_f5_tie':       results['f5_tie_prob'],
                'model_nrfi':         results['nrfi_prob'],
                'model_median_total': results['median_total'],
                'model_away_k_med':   results['away_pitcher_median_k'],
                'model_home_k_med':   results['home_pitcher_median_k'],

                # Full distributions (JSON-encoded for later flexibility)
                'model_away_k_dist':  json.dumps(results.get('away_k_dist', [])),
                'model_home_k_dist':  json.dumps(results.get('home_k_dist', [])),
                'model_player_props': json.dumps(results.get('player_props', {})),

                # Actual outcomes
                'actual_away_score':  outcomes['final_away'],
                'actual_home_score':  outcomes['final_home'],
                'actual_f5_away':     outcomes['f5_away'],
                'actual_f5_home':     outcomes['f5_home'],
                'actual_nrfi':        outcomes['nrfi'],
                'actual_pitcher_ks':  json.dumps(outcomes['pitcher_ks']),
                'actual_pitcher_bf':  json.dumps(outcomes['pitcher_bf']),
                'actual_batter_stats': json.dumps(outcomes['batter_stats']),

                # Historical closing odds (None if not found in odds CSV)
                'odds_away_ml_close':     odds_row.get('away_ml_close') if odds_row else None,
                'odds_home_ml_close':     odds_row.get('home_ml_close') if odds_row else None,
                'odds_away_ml_open':      odds_row.get('away_ml_open')  if odds_row else None,
                'odds_home_ml_open':      odds_row.get('home_ml_open')  if odds_row else None,
                'odds_total_close':       odds_row.get('over_close')    if odds_row else None,
                'odds_over_close_juice':  odds_row.get('over_close_odds')  if odds_row else None,
                'odds_under_close_juice': odds_row.get('under_close_odds') if odds_row else None,
            }
            pending_rows.append(row)
            games_processed += 1

            # Flush every 10 games so a crash doesn't lose work
            if len(pending_rows) >= 10:
                append_rows(pending_rows, output_path)
                pending_rows = []
                print(f"  [checkpoint] {games_processed} new games saved")

        current_dt += timedelta(days=1)

    # Final flush
    append_rows(pending_rows, output_path)

    print(f"\n[DONE] {games_processed} new games added to ledger.")
    print(f"[DONE] Ledger at: {output_path}")


if __name__ == "__main__":
    # 2025 regular season
    run_backtest(start_date='2023-03-30', end_date='2025-11-01', iterations=1000)