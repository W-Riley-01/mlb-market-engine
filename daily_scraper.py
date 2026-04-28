import requests
from datetime import datetime

# ==========================================
# 1. MLB API ENDPOINTS
# ==========================================
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{}/boxscore"


def fetch_todays_schedule(date_str=None):
    """Fetches all MLB games scheduled for a given date."""
    if not date_str:
        # Defaults to today's date (YYYY-MM-DD)
        date_str = datetime.today().strftime('%Y-%m-%d')

    print(f"--- FETCHING MLB SLATE FOR {date_str} ---")

    params = {
        'sportId': 1,
        'date': date_str,
        # Hydrate with linescore so we get live inning/score data in one call
        # instead of hitting the boxscore endpoint per-game just for state.
        'hydrate': 'linescore',
    }
    try:
        response = requests.get(SCHEDULE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data['totalGames'] == 0:
            print("No games scheduled for today.")
            return []

        games = data['dates'][0]['games']
        slate = []

        for game in games:
            game_id = game['gamePk']
            away_team = game['teams']['away']['team']['name']
            home_team = game['teams']['home']['team']['name']

            # Status (Scheduled, In Progress, Final)
            status = game['status']['detailedState']

            # Live score + inning state (present on in-progress and final games).
            # Scheduled games have 'score' absent from the teams payload, so
            # we use .get() with a None default and let the UI decide how to
            # handle missing values.
            away_score = game['teams']['away'].get('score')
            home_score = game['teams']['home'].get('score')

            # Linescore block carries the current inning + top/bot indicator.
            # Only present once the game is actually underway.
            linescore = game.get('linescore', {}) or {}
            current_inning      = linescore.get('currentInning')
            inning_half         = linescore.get('inningHalf')          # 'Top' or 'Bottom'
            inning_ordinal      = linescore.get('currentInningOrdinal')  # '3rd', '7th', etc.
            outs                = linescore.get('outs')

            # First-pitch time in UTC ISO format, e.g. '2026-04-23T23:05:00Z'.
            # Needed by weather.py to snap the forecast to the right hour.
            game_datetime = game.get('gameDate')

            slate.append({
                'game_id':        game_id,
                'matchup':        f"{away_team} @ {home_team}",
                'status':         status,
                'away_team':      away_team,
                'home_team':      home_team,
                'game_datetime':  game_datetime,
                'away_score':     away_score,
                'home_score':     home_score,
                'inning':         current_inning,
                'inning_half':    inning_half,
                'inning_ordinal': inning_ordinal,
                'outs':           outs,
            })

        print(f"Found {len(slate)} games.")
        return slate

    except Exception as e:
        print(f"[ERROR] Failed to fetch schedule: {e}")
        return []


# ==========================================
# 2. ROSTER & LINEUP EXTRACTOR
# ==========================================
def fetch_game_rosters(game_id):
    """Pulls the starting pitchers, batting orders, bullpen, AND player names."""
    url = BOXSCORE_URL.format(game_id)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['teams']

        rosters = {'Away': {}, 'Home': {}}

        for side, key in [('Away', 'away'), ('Home', 'home')]:
            team_data = data[key]
            players_dict = team_data.get('players', {})

            # Helper to get name from ID
            def get_name(pid):
                player_str = f"ID{pid}"
                if player_str in players_dict:
                    return players_dict[player_str]['person']['fullName']
                return str(pid)

            pitchers = team_data.get('pitchers', [])
            starter_id = pitchers[0] if pitchers else None
            bullpen_ids = pitchers[1:] if len(pitchers) > 1 else []
            lineup_ids = team_data.get('battingOrder', [])

            # Create a lineup list that contains both ID and Name
            lineup_with_names = [{'id': pid, 'name': get_name(pid)} for pid in lineup_ids]

            rosters[side] = {
                'team_name': team_data['team']['name'],
                'starter_id': starter_id,
                'starter_name': get_name(starter_id) if starter_id else "TBD",
                'bullpen_ids': bullpen_ids,
                'lineup': lineup_ids,  # Keep raw IDs for the math engine
                'lineup_details': lineup_with_names  # Use this for the Action Card printout
            }

        return rosters

    except Exception as e:
        print(f"[ERROR] Failed to fetch rosters for Game {game_id}: {e}")
        return None


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Grab Today's Games
    todays_games = fetch_todays_schedule()

    if todays_games:
        # Let's just test the very first game on the slate
        test_game = todays_games[0]
        print(f"\n--- SCRAPING GAME: {test_game['matchup']} ---")

        rosters = fetch_game_rosters(test_game['game_id'])

        if rosters:
            away = rosters['Away']
            home = rosters['Home']

            print(f"\n{away['team_name']} (Away)")
            print(f"  Starter ID: {away['starter_id']}")
            print(f"  Lineup Posted: {'Yes' if away['lineup'] else 'No (TBD)'}")
            if away['lineup']:
                print(f"  Batting Order: {away['lineup']}")

            print(f"\n{home['team_name']} (Home)")
            print(f"  Starter ID: {home['starter_id']}")
            print(f"  Lineup Posted: {'Yes' if home['lineup'] else 'No (TBD)'}")
            if home['lineup']:
                print(f"  Batting Order: {home['lineup']}")