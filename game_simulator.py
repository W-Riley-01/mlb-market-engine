import numpy as np
import pandas as pd
from resolver import MatchupResolver  # Imports your Phase 3, Task 2 script


# ==========================================
# 1. THE ADVANCED STATE MACHINE
# ==========================================
class SimulatedGame:
    def __init__(self, away_prob_card, home_prob_card):
        self.inning = 1
        self.top_of_inning = True
        self.outs = 0
        self.bases = [False, False, False]
        self.away_score = 0
        self.home_score = 0

        # Lineup trackers (0 through 8)
        self.away_batter_index = 0
        self.home_batter_index = 0

        self.away_prob_card = away_prob_card
        self.home_prob_card = home_prob_card

    def reset_inning(self):
        self.outs = 0
        self.bases = [False, False, False]
        if self.top_of_inning:
            self.top_of_inning = False
        else:
            self.top_of_inning = True
            self.inning += 1

    def process_event(self, event: str):
        runs = 0
        if event == 'Out':
            self.outs += 1
        elif event == 'HR':
            runs = sum(self.bases) + 1
            self.bases = [False, False, False]
        elif event == '3B':
            runs = sum(self.bases)
            self.bases = [False, False, True]
        elif event == '2B':
            runs = self.bases[1] + self.bases[2]
            self.bases[2], self.bases[1], self.bases[0] = self.bases[0], True, False
        elif event == '1B':
            runs = self.bases[2]
            self.bases[2], self.bases[1], self.bases[0] = self.bases[1], self.bases[0], True
        elif event == 'BB':
            if self.bases[0] and self.bases[1] and self.bases[2]:
                runs = 1
            elif self.bases[0] and self.bases[1]:
                self.bases[2] = True
            elif self.bases[0]:
                self.bases[1] = True
            self.bases[0] = True

        if self.top_of_inning:
            self.away_score += runs
        else:
            self.home_score += runs

        if self.outs >= 3:
            self.reset_inning()

    def play_game(self):
        outcomes = ['Out', '1B', '2B', '3B', 'HR', 'BB']

        # Play until 9 innings are done (Simplified: ignores bottom of 9th walk-offs for speed)
        while self.inning <= 9:
            if self.top_of_inning:
                # Get current batter's probabilities
                probs_dict = self.away_prob_card[self.away_batter_index]
                probs = [probs_dict[o] for o in outcomes]

                # Roll the dice
                event = np.random.choice(outcomes, p=probs)
                self.process_event(event)

                # Move to next batter in lineup
                self.away_batter_index = (self.away_batter_index + 1) % 9
            else:
                probs_dict = self.home_prob_card[self.home_batter_index]
                probs = [probs_dict[o] for o in outcomes]

                event = np.random.choice(outcomes, p=probs)
                self.process_event(event)
                self.home_batter_index = (self.home_batter_index + 1) % 9

        return self.away_score, self.home_score


# ==========================================
# 2. THE MONTE CARLO ORCHESTRATOR
# ==========================================
def run_monte_carlo(resolver, away_lineup, home_lineup, away_pitcher, home_pitcher, density_ratio, iterations=5000):
    print("\n--- PRE-CALCULATING MATCHUP PROBABILITIES ---")
    # Generate the Probability Cards for the specific game
    away_prob_card = [resolver.generate_probabilities(batter, home_pitcher, density_ratio) for batter in away_lineup]
    home_prob_card = [resolver.generate_probabilities(batter, away_pitcher, density_ratio) for batter in home_lineup]

    print(f"--- RUNNING {iterations} SIMULATIONS ---")
    away_wins = 0
    home_wins = 0
    game_totals = []

    for _ in range(iterations):
        game = SimulatedGame(away_prob_card, home_prob_card)
        away_score, home_score = game.play_game()

        game_totals.append(away_score + home_score)
        if away_score > home_score:
            away_wins += 1
        elif home_score > away_score:
            home_wins += 1

    # Return the probability distributions
    return {
        'away_win_prob': away_wins / iterations,
        'home_win_prob': home_wins / iterations,
        'median_total': np.median(game_totals),
        'over_8_5_prob': sum(1 for x in game_totals if x > 8.5) / iterations
    }


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    resolver_engine = MatchupResolver(
        pqm_path='./data/pitch_matrix.parquet',
        cqm_path='./data/contact_matrix_env.parquet'
    )

    # Example IDs (We will use a mix of real IDs for a mock game)
    # NYY (Away) vs BOS (Home)
    away_lineup = [592450, 518692, 650402, 624413, 640449, 592122, 650333, 643396,
                   683011]  # Judge, LeMahieu, Torres, etc.
    home_lineup = [646240, 678882, 608700, 672386, 656061, 598265, 605141, 628329, 668800]  # Devers, Yoshida, etc.

    away_pitcher = 543037  # Gerrit Cole
    home_pitcher = 669456  # Brayan Bello

    # Assuming Fenway Park on a neutral night
    fenway_density = 1.0

    results = run_monte_carlo(
        resolver=resolver_engine,
        away_lineup=away_lineup,
        home_lineup=home_lineup,
        away_pitcher=away_pitcher,
        home_pitcher=home_pitcher,
        density_ratio=fenway_density,
        iterations=5000
    )

    print("\n*** FINAL MONTE CARLO DISTRIBUTIONS ***")
    print(f"Away Team Win Probability: {results['away_win_prob'] * 100:.1f}%")
    print(f"Home Team Win Probability: {results['home_win_prob'] * 100:.1f}%")
    print(f"Median Game Total: {results['median_total']}")
    print(f"Probability over 8.5 Runs: {results['over_8_5_prob'] * 100:.1f}%")