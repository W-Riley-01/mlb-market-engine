import numpy as np


# ==========================================
# 1. THE BASEBALL STATE MACHINE
# ==========================================
class GameState:
    def __init__(self):
        self.inning = 1
        self.top_of_inning = True
        self.outs = 0
        # Bases: [1st, 2nd, 3rd] represented as booleans
        self.bases = [False, False, False]
        self.away_score = 0
        self.home_score = 0

    def reset_inning(self):
        """Clears bases and outs for the next half-inning."""
        self.outs = 0
        self.bases = [False, False, False]

        if self.top_of_inning:
            self.top_of_inning = False
        else:
            self.top_of_inning = True
            self.inning += 1

    def process_event(self, event: str):
        """
        Takes an event (Out, 1B, 2B, 3B, HR, BB) and updates the base state.
        This is a simplified base-advancement model.
        """
        runs_scored_on_play = 0

        if event == 'Out' or event == 'Strikeout':
            self.outs += 1

        elif event == 'HR':
            # Everyone on base scores, plus the batter
            runs_scored_on_play = sum(self.bases) + 1
            self.bases = [False, False, False]

        elif event == '3B':
            runs_scored_on_play = sum(self.bases)
            self.bases = [False, False, True]

        elif event == '2B':
            # Assuming runners on 2nd and 3rd score, runner on 1st goes to 3rd
            runs_scored_on_play = self.bases[1] + self.bases[2]
            self.bases[2] = self.bases[0]  # 1st goes to 3rd
            self.bases[1] = True  # Batter on 2nd
            self.bases[0] = False

        elif event == '1B':
            # Assuming runner on 3rd scores, 2nd goes to 3rd, 1st goes to 2nd
            runs_scored_on_play = self.bases[2]
            self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
            self.bases[0] = True

        elif event == 'BB':  # Walk
            if self.bases[0] and self.bases[1] and self.bases[2]:
                runs_scored_on_play = 1  # Bases loaded walk
            elif self.bases[0] and self.bases[1]:
                self.bases[2] = True
            elif self.bases[0]:
                self.bases[1] = True
            self.bases[0] = True

        # Add runs to the correct team
        if self.top_of_inning:
            self.away_score += runs_scored_on_play
        else:
            self.home_score += runs_scored_on_play

        # Check for inning end
        if self.outs >= 3:
            self.reset_inning()


# ==========================================
# 2. THE SIMULATION ENGINE
# ==========================================
def simulate_game(away_probs: dict, home_probs: dict) -> dict:
    """
    Simulates a standard 9-inning game using given probabilities.
    """
    game = GameState()

    # Standard outcomes we care about for the base state
    outcomes = ['Out', '1B', '2B', '3B', 'HR', 'BB']

    # Keep playing until 9 innings are complete
    # (Simplified: doesn't account for home team winning in bottom of 9th yet)
    while game.inning <= 9:

        # Determine who is batting
        if game.top_of_inning:
            probs = [away_probs[outcome] for outcome in outcomes]
        else:
            probs = [home_probs[outcome] for outcome in outcomes]

        # 1. THE RESOLVER: Roll the virtual dice!
        event = np.random.choice(outcomes, p=probs)

        # 2. THE STATE MACHINE: Update the game
        game.process_event(event)

    return {
        'away_score': game.away_score,
        'home_score': game.home_score,
        'total_runs': game.away_score + game.home_score
    }


# ==========================================
# 3. RUNNING A 5,000 ITERATION MONTE CARLO
# ==========================================
if __name__ == "__main__":
    print("--- INITIATING MONTE CARLO ENGINE ---")

    # DUMMY DATA: In our next script, we will calculate these dynamically from our PQM/CQM
    # Order: ['Out', '1B', '2B', '3B', 'HR', 'BB']
    # Total probability MUST equal 1.0
    team_a_probs = {'Out': 0.68, '1B': 0.15, '2B': 0.05, '3B': 0.01, 'HR': 0.03, 'BB': 0.08}
    team_b_probs = {'Out': 0.72, '1B': 0.14, '2B': 0.04, '3B': 0.00, 'HR': 0.02, 'BB': 0.08}

    iterations = 5000
    team_a_wins = 0
    total_runs_list = []

    print(f"Running game simulations...")

    for _ in range(iterations):
        result = simulate_game(team_a_probs, team_b_probs)
        total_runs_list.append(result['total_runs'])

        if result['away_score'] > result['home_score']:
            team_a_wins += 1

    # Analytics Output
    win_prob_a = (team_a_wins / iterations) * 100
    median_total = np.median(total_runs_list)
    over_8_5_prob = (sum(1 for runs in total_runs_list if runs > 8.5) / iterations) * 100

    print("\n--- SIMULATION RESULTS ---")
    print(f"Team A Win Probability: {win_prob_a:.1f}%")
    print(f"Team B Win Probability: {100 - win_prob_a:.1f}%")
    print(f"Median Game Total: {median_total}")
    print(f"Probability of Game going Over 8.5 runs: {over_8_5_prob:.1f}%")