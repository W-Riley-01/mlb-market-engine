import pandas as pd


# ==========================================
# 1. THE SPORTSBOOK API MOCK
# ==========================================
def fetch_live_odds(game_id="NYY_BOS_TODAY"):
    """Mock dictionary of American Odds."""
    print("[API CALL] Fetching live odds from DraftKings/FanDuel...")
    return {
        'Away_ML': -110,  # Yankees Moneyline
        'Home_ML': -110,  # Red Sox Moneyline
        'Over_8_5': +140,  # Game goes over 8.5 runs
        'Judge_HR': +250,  # Aaron Judge hits a Home Run
        'SGP_JudgeHR_Over85': +650  # Book's SGP Payout
    }


# ==========================================
# 2. THE QUANTITATIVE MATHEMATICS
# ==========================================
def american_to_implied(odds: int) -> float:
    """Converts American odds to Implied Probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def calculate_ev(true_prob: float, odds: int, wager: float = 100.0) -> float:
    """Calculates Expected Value (EV) in dollars based on a standard wager."""
    if odds < 0:
        profit = wager * (100 / abs(odds))
    else:
        profit = wager * (odds / 100)

    prob_loss = 1.0 - true_prob
    ev = (true_prob * profit) - (prob_loss * wager)
    return ev


# ==========================================
# 3. THE ALPHA ENGINE
# ==========================================
def analyze_edges(monte_carlo_results: dict, live_odds: dict):
    print("\n--- ALPHA ENGINE: HUNTING +EV ---")

    analysis = []

    # Normalize ML probabilities to remove the tie rate from the basic Phase 3 engine
    total_win_prob = monte_carlo_results['away_win_prob'] + monte_carlo_results['home_win_prob']
    true_away_ml = monte_carlo_results['away_win_prob'] / total_win_prob

    # 1. Check Away ML
    away_implied = american_to_implied(live_odds['Away_ML'])
    away_ev = calculate_ev(true_away_ml, live_odds['Away_ML'])
    analysis.append({'Bet': 'Away ML', 'True_Prob': true_away_ml, 'Implied': away_implied, 'Odds': live_odds['Away_ML'],
                     'EV_$100': away_ev})

    # 2. Check Game Total (Over 8.5)
    over_implied = american_to_implied(live_odds['Over_8_5'])
    over_ev = calculate_ev(monte_carlo_results['over_8_5_prob'], live_odds['Over_8_5'])
    analysis.append({'Bet': 'Over 8.5 Runs', 'True_Prob': monte_carlo_results['over_8_5_prob'], 'Implied': over_implied,
                     'Odds': live_odds['Over_8_5'], 'EV_$100': over_ev})

    # 3. Analyze the SGP (Same Game Parlay)
    sgp_implied = american_to_implied(live_odds['SGP_JudgeHR_Over85'])
    sgp_true_prob = monte_carlo_results['sgp_judge_hr_and_over']
    sgp_ev = calculate_ev(sgp_true_prob, live_odds['SGP_JudgeHR_Over85'])
    analysis.append({'Bet': 'SGP: Judge HR + Over 8.5', 'True_Prob': sgp_true_prob, 'Implied': sgp_implied,
                     'Odds': live_odds['SGP_JudgeHR_Over85'], 'EV_$100': sgp_ev})

    df = pd.DataFrame(analysis)

    # --- ACTIONABLE ALERTS (Run while math is still pure floats) ---
    print("\n--- ACTIONABLE ALERTS ---")
    for index, row in df.iterrows():
        ev = row['EV_$100']
        if ev > 5.00:
            print(
                f"🔥 GREEN LIGHT: {row['Bet']} yields high +EV (+${ev:.2f}). (Model: {row['True_Prob'] * 100:.1f}% vs Book: {row['Implied'] * 100:.1f}%)")
        elif ev < -20.00:
            print(f"🛑 RED LIGHT: {row['Bet']} is a heavily negative EV trap (-${abs(ev):.2f}). Avoid.")

    # --- FORMAT FOR DISPLAY ---
    df['True_Prob'] = (df['True_Prob'] * 100).map("{:.1f}%".format)
    df['Implied'] = (df['Implied'] * 100).map("{:.1f}%".format)
    df['EV_$100'] = df['EV_$100'].map(lambda x: f"+${x:.2f}" if x > 0 else f"-${abs(x):.2f}")

    print("\n--- FULL ODDS TABLE ---")
    print(df.to_string(index=False))


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    odds = fetch_live_odds()

    my_model_output = {
        'away_win_prob': 0.513,
        'home_win_prob': 0.360,
        'over_8_5_prob': 0.219,
        'sgp_judge_hr_and_over': 0.160
    }

    analyze_edges(my_model_output, odds)