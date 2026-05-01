import numpy as np


# ==========================================
# 1. THE FULL MARKET STATE MACHINE
# ==========================================
class AdvancedSimulatedGame:
    def __init__(self, away_lineup, home_lineup, away_starter_card, home_starter_card, away_bullpen_card,
                 home_bullpen_card,
                 away_starter_card_1st=None, home_starter_card_1st=None):
        self.inning = 1
        self.top_of_inning = True
        self.outs = 0
        self.bases = [None, None, None]
        self.away_score = 0
        self.home_score = 0

        # Lineup Trackers
        self.away_batter_index = 0
        self.home_batter_index = 0

        # Pitching Trackers
        self.away_pitch_count = 0
        self.home_pitch_count = 0

        # Starter cards: main (innings 2-9) and 1st-inning blended.
        # Fall back to the main card if no 1st-inning variant was provided —
        # keeps old callers working unchanged.
        self.away_starter_card     = away_starter_card
        self.home_starter_card     = home_starter_card
        self.away_starter_card_1st = away_starter_card_1st or away_starter_card
        self.home_starter_card_1st = home_starter_card_1st or home_starter_card

        # Active cards start on the 1st-inning variant. `check_pitching_changes`
        # swaps them to the main card at the top of inning 2, and to the bullpen
        # once the starter exhausts.
        self.away_pitcher_card = self.away_starter_card_1st
        self.home_pitcher_card = self.home_starter_card_1st

        self.away_bullpen_card = away_bullpen_card
        self.home_bullpen_card = home_bullpen_card

        # MARKET UPGRADE 1: Pitcher Strikeouts
        self.away_starter_ks = 0
        self.home_starter_ks = 0
        self.away_starter_active = True
        self.home_starter_active = True

        # MARKET UPGRADE 2: Inning Snapshots (NRFI & F5)
        self.nrfi = False
        self.f5_away_score = 0
        self.f5_home_score = 0

        # MARKET UPGRADE 3: Full Box Score with Names
        # Expects lineup format: [{'id': 123, 'name': 'Aaron Judge'}, ...]
        self.away_box_score = {
            i: {'name': away_lineup[i]['name'], 'id': away_lineup[i]['id'], 'Hits': 0, '2B': 0, 'HR': 0, 'TB': 0,
                'RBI': 0, 'R': 0, 'SB': 0}
            for i in range(9)
        }
        self.home_box_score = {
            i: {'name': home_lineup[i]['name'], 'id': home_lineup[i]['id'], 'Hits': 0, '2B': 0, 'HR': 0, 'TB': 0,
                'RBI': 0, 'R': 0, 'SB': 0}
            for i in range(9)
        }

    def reset_inning(self):
        # Snapshot: First 5 Innings (F5)
        if self.inning == 5 and not self.top_of_inning:
            self.f5_away_score = self.away_score
            self.f5_home_score = self.home_score

        # Snapshot: NRFI (No Runs First Inning)
        if self.inning == 1 and not self.top_of_inning:
            self.nrfi = (self.away_score == 0 and self.home_score == 0)

        self.outs = 0
        self.bases = [None, None, None]

        if self.top_of_inning:
            self.top_of_inning = False
        else:
            self.top_of_inning = True
            self.inning += 1

    def score_run(self, runner_index, batter_index):
        if self.top_of_inning:
            self.away_score += 1
            if runner_index is not None: self.away_box_score[runner_index]['R'] += 1
            self.away_box_score[batter_index]['RBI'] += 1
        else:
            self.home_score += 1
            if runner_index is not None: self.home_box_score[runner_index]['R'] += 1
            self.home_box_score[batter_index]['RBI'] += 1

    def attempt_steal(self, runner_index, box):
        """Temporary baseline steal logic. Uses 6% average attempt/success rate."""
        # Only steal if 2nd base is open
        if self.bases[1] is None and np.random.rand() < 0.06:
            self.bases[1] = runner_index
            self.bases[0] = None
            box[runner_index]['SB'] += 1

    def process_event(self, event: str, batter_index: int):
        box = self.away_box_score if self.top_of_inning else self.home_box_score

        # Pitch Count Estimation
        pitch_increment = np.random.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1])
        if self.top_of_inning:
            self.home_pitch_count += pitch_increment
        else:
            self.away_pitch_count += pitch_increment

        # Resolving the play
        if event == 'Out':
            # Sac fly / productive out: with R3 and < 2 outs, the runner
            # on third scores ~35% of the time. The simulator can't
            # distinguish fly outs from grounders, so this is the blended
            # league-wide rate at which a generic "Out" event with R3
            # and <2 outs converts to a run. RBI is credited via score_run.
            # Calibration target: real MLB sac-fly + productive-out RBI
            # rate is ~30-40% of qualifying outs depending on outs and
            # runner speed; 0.35 is a reasonable midpoint to start.
            SAC_FLY_RATE_R3_LT2OUTS = 0.35
            if self.bases[2] is not None and self.outs < 2:
                if np.random.rand() < SAC_FLY_RATE_R3_LT2OUTS:
                    self.score_run(self.bases[2], batter_index)
                    self.bases[2] = None
            self.outs += 1

        elif event == 'K':  # NEW: Explicit Strikeout logic
            self.outs += 1
            if self.top_of_inning and self.home_starter_active:
                self.home_starter_ks += 1
            elif not self.top_of_inning and self.away_starter_active:
                self.away_starter_ks += 1

        elif event == 'HR':
            box[batter_index]['TB'] += 4
            box[batter_index]['HR'] += 1
            box[batter_index]['Hits'] += 1
            for runner in self.bases:
                if runner is not None: self.score_run(runner, batter_index)
            self.score_run(batter_index, batter_index)
            self.bases = [None, None, None]

        elif event == '3B':
            box[batter_index]['TB'] += 3
            box[batter_index]['Hits'] += 1
            for runner in self.bases:
                if runner is not None: self.score_run(runner, batter_index)
            self.bases = [None, None, batter_index]  # Batter on 3rd, all others score

        elif event == '2B':
            box[batter_index]['TB'] += 2
            box[batter_index]['2B'] += 1
            box[batter_index]['Hits'] += 1
            # R3 and R2 always score on a double (R2 scores ~95% in reality;
            # we round to 100% for simplicity since the small miss is
            # systematically the same direction across all teams).
            if self.bases[2] is not None:
                self.score_run(self.bases[2], batter_index)
            if self.bases[1] is not None:
                self.score_run(self.bases[1], batter_index)
            # R1 scores ~55% of the time on a double. When held, ends up on 3B.
            R1_SCORE_ON_2B = 0.55
            r1_scored = False
            if self.bases[0] is not None and np.random.rand() < R1_SCORE_ON_2B:
                self.score_run(self.bases[0], batter_index)
                r1_scored = True
            self.bases[2] = None if r1_scored else self.bases[0]
            self.bases[1] = batter_index
            self.bases[0] = None

        elif event == '1B':
            box[batter_index]['TB'] += 1
            box[batter_index]['Hits'] += 1
            # R3 always scores on a single
            if self.bases[2] is not None:
                self.score_run(self.bases[2], batter_index)
            # R2 scores ~40% of the time on a single. When held, advances
            # to 3B in the standard rotation. The 40% is a season-long
            # average across all R2/single situations — the actual rate
            # depends on hit type (line drive vs. ground ball), runner
            # speed, and number of outs, but those second-order effects
            # aren't modeled here.
            R2_SCORE_ON_1B = 0.40
            r2_scored = False
            if self.bases[1] is not None and np.random.rand() < R2_SCORE_ON_1B:
                self.score_run(self.bases[1], batter_index)
                r2_scored = True
            # Standard advancement: R2 (if held) → 3B, R1 → 2B, batter → 1B
            self.bases[2] = None if r2_scored else self.bases[1]
            self.bases[1] = self.bases[0]
            self.bases[0] = batter_index
            self.attempt_steal(batter_index, box)  # Check for steal

        elif event == 'BB':
            if self.bases[0] is not None and self.bases[1] is not None and self.bases[2] is not None:
                self.score_run(self.bases[2], batter_index)
            elif self.bases[0] is not None and self.bases[1] is not None:
                self.bases[2] = self.bases[1]
            elif self.bases[0] is not None:
                self.bases[1] = self.bases[0]
            self.bases[0] = batter_index
            self.attempt_steal(batter_index, box)  # Check for steal

        if self.outs >= 3:
            self.reset_inning()

    def check_pitching_changes(self):
        # Transition from the 1st-inning blended card to the main starter card
        # once we're past inning 1. Only applies while the starter is still in;
        # if the bullpen already took over we leave that alone.
        if self.inning >= 2 and self.home_starter_active and \
                self.home_pitcher_card is self.home_starter_card_1st:
            self.home_pitcher_card = self.home_starter_card
        if self.inning >= 2 and self.away_starter_active and \
                self.away_pitcher_card is self.away_starter_card_1st:
            self.away_pitcher_card = self.away_starter_card

        if self.top_of_inning and self.home_pitch_count > 85 and self.home_starter_active:
            self.home_pitcher_card = self.home_bullpen_card
            self.home_starter_active = False

        if not self.top_of_inning and self.away_pitch_count > 85 and self.away_starter_active:
            self.away_pitcher_card = self.away_bullpen_card
            self.away_starter_active = False

    def play_game(self):
        # Notice we added 'K' to the outcomes
        outcomes = ['Out', 'K', '1B', '2B', '3B', 'HR', 'BB']

        while self.inning <= 9:
            # Walk-off check: if it's the bottom of the 9th (or later) and
            # the home team is already winning, stop — they don't need to bat.
            if not self.top_of_inning and self.inning >= 9 and self.home_score > self.away_score:
                break

            self.check_pitching_changes()

            if self.top_of_inning:
                probs_dict = self.home_pitcher_card[self.away_batter_index]
                probs = [probs_dict.get(o, 0) for o in outcomes]
                # Normalize to guard against float drift
                total = sum(probs)
                probs = [p / total for p in probs]
                event = np.random.choice(outcomes, p=probs)
                self.process_event(event, self.away_batter_index)
                self.away_batter_index = (self.away_batter_index + 1) % 9
            else:
                probs_dict = self.away_pitcher_card[self.home_batter_index]
                probs = [probs_dict.get(o, 0) for o in outcomes]
                total = sum(probs)
                probs = [p / total for p in probs]
                # Walk-off mid-inning: stop as soon as home team takes the lead in 9th+
                if self.inning >= 9 and self.home_score > self.away_score:
                    break
                event = np.random.choice(outcomes, p=probs)
                self.process_event(event, self.home_batter_index)
                self.home_batter_index = (self.home_batter_index + 1) % 9

        return {
            'away_score': self.away_score,
            'home_score': self.home_score,
            'nrfi': self.nrfi,
            'f5_away': self.f5_away_score,
            'f5_home': self.f5_home_score,
            'away_starter_ks': self.away_starter_ks,
            'home_starter_ks': self.home_starter_ks,
            'away_box': self.away_box_score,
            'home_box': self.home_box_score
        }


# ==========================================
# 2. THE MARKET ORCHESTRATOR
# ==========================================
def run_advanced_monte_carlo(resolver, away_lineup, home_lineup, away_starter, home_starter, away_bullpen, home_bullpen,
                             density_ratio, iterations=5000, as_of_date=None,
                             first_inning_weight=0.20):
    arsenal_status = as_of_date if as_of_date else "DISABLED (no as_of_date passed)"
    print(f"\n--- RUNNING {iterations} MARKET SIMULATIONS (arsenal={arsenal_status}) ---")

    # We pass the ID explicitly for the resolver, but the simulator takes the whole dict.
    # as_of_date is threaded through so the resolver can activate arsenal-profile + recent-form blending.
    # Cards are built ONCE here (not per iteration) because they're invariant within a game.
    #
    # We build TWO starter cards per team:
    #   - the main card (used innings 2-9)
    #   - a 1st-inning-blended card (used only in inning 1)
    # The 1st-inning variant shifts pitcher K% and BB% toward his career
    # 1st-inning splits at `first_inning_weight`, sample-size-scaled.
    away_starter_card = [
        resolver.generate_probabilities(batter['id'], home_starter, density_ratio, as_of_date=as_of_date)
        for batter in away_lineup
    ]
    home_starter_card = [
        resolver.generate_probabilities(batter['id'], away_starter, density_ratio, as_of_date=as_of_date)
        for batter in home_lineup
    ]

    away_starter_card_1st = [
        resolver.generate_probabilities(
            batter['id'], home_starter, density_ratio, as_of_date=as_of_date,
            first_inning=True, first_inning_weight=first_inning_weight)
        for batter in away_lineup
    ]
    home_starter_card_1st = [
        resolver.generate_probabilities(
            batter['id'], away_starter, density_ratio, as_of_date=as_of_date,
            first_inning=True, first_inning_weight=first_inning_weight)
        for batter in home_lineup
    ]

    away_bullpen_card = [
        resolver.generate_bullpen_probabilities(batter['id'], home_bullpen, density_ratio, as_of_date=as_of_date)
        for batter in away_lineup
    ]
    home_bullpen_card = [
        resolver.generate_bullpen_probabilities(batter['id'], away_bullpen, density_ratio, as_of_date=as_of_date)
        for batter in home_lineup
    ]

    # Pull both starters' 1st-inning stats for display on the card.
    # These are historical / retrospective and feed the UI, not the sim.
    home_starter_fi_stats = resolver.get_first_inning_stats(home_starter, as_of_date=as_of_date)
    away_starter_fi_stats = resolver.get_first_inning_stats(away_starter, as_of_date=as_of_date)

    # Global Trackers
    away_wins, f5_away_wins, f5_ties, nrfi_hits = 0, 0, 0, 0
    home_wins_by_2_plus = 0   # for run line: home -1.5
    away_wins_by_2_plus = 0   # for run line: away -1.5
    game_totals = []
    away_k_dist, home_k_dist = [], []

    # Total-runs threshold ladder. We compute P(total >= X) for each X so
    # downstream consumers (Supabase, the calibration notebook, the UI)
    # can read the over/under for any standard market line without going
    # back to the raw distribution. 6.5 to 11.5 covers ~99% of MLB totals.
    TOTAL_THRESHOLDS_X = (6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5)

    # Prop Trackers (We track 1+ Hits, 2+ TB, 1+ HR, 1+ RBI for every batter)
    player_props = {}
    for batter in away_lineup + home_lineup:
        player_props[batter['name']] = {'1+ Hits': 0, '2+ TB': 0, '1+ HR': 0, '1+ RBI': 0, '1+ SB': 0}

    for _ in range(iterations):
        game = AdvancedSimulatedGame(
            away_lineup, home_lineup,
            away_starter_card, home_starter_card,
            away_bullpen_card, home_bullpen_card,
            away_starter_card_1st=away_starter_card_1st,
            home_starter_card_1st=home_starter_card_1st,
        )
        res = game.play_game()

        # 1. Game & F5 Trackers
        game_totals.append(res['away_score'] + res['home_score'])
        if res['away_score'] > res['home_score']: away_wins += 1

        # Run-line trackers (margin >= 2 in either direction)
        margin = res['home_score'] - res['away_score']
        if margin >=  2: home_wins_by_2_plus += 1
        if margin <= -2: away_wins_by_2_plus += 1

        if res['f5_away'] > res['f5_home']:
            f5_away_wins += 1
        elif res['f5_away'] == res['f5_home']:
            f5_ties += 1

        if res['nrfi']: nrfi_hits += 1

        # 2. Pitcher K Trackers
        away_k_dist.append(res['away_starter_ks'])
        home_k_dist.append(res['home_starter_ks'])

        # 3. Player Prop Aggregation
        for team_box in [res['away_box'], res['home_box']]:
            for i in range(9):
                p_name = team_box[i]['name']
                if team_box[i]['Hits'] >= 1: player_props[p_name]['1+ Hits'] += 1
                if team_box[i]['TB'] >= 2:   player_props[p_name]['2+ TB'] += 1
                if team_box[i]['HR'] >= 1:   player_props[p_name]['1+ HR'] += 1
                if team_box[i]['RBI'] >= 1:  player_props[p_name]['1+ RBI'] += 1
                if team_box[i]['SB'] >= 1:   player_props[p_name]['1+ SB'] += 1

    # Compute totals statistics from the full distribution. The raw list
    # is also passed through so engine_runner can apply weather adjustments
    # (carry_delta) and recompute the threshold ladder consistently.
    totals_arr = np.asarray(game_totals)
    total_thresholds = {
        x: float(np.mean(totals_arr >= x)) for x in TOTAL_THRESHOLDS_X
    }

    # Format the return dictionary
    final_results = {
        'away_win_prob': away_wins / iterations,
        'f5_away_win_prob': f5_away_wins / iterations,
        'f5_tie_prob': f5_ties / iterations,
        'nrfi_prob': nrfi_hits / iterations,
        'median_total': float(np.median(totals_arr)),
        'total_mean':   float(np.mean(totals_arr)),
        'total_std':    float(np.std(totals_arr)),
        # Raw distribution kept on the return dict so weather adjustments
        # can be applied downstream without re-running the simulation.
        # Each consumer that wants weather-aware totals shifts this list,
        # then re-derives median/mean/thresholds from the shifted version.
        'game_totals_dist': game_totals,
        # P(total >= X) for the standard market ladder. Pre-weather; the
        # engine_runner re-derives a weather-adjusted version when carry
        # delta is non-zero.
        'total_thresholds': total_thresholds,
        # Run lines: margin >= 2 in either direction.
        'run_line_home_minus_1_5': home_wins_by_2_plus / iterations,
        'run_line_away_minus_1_5': away_wins_by_2_plus / iterations,
        'away_pitcher_median_k': float(np.median(away_k_dist)),
        'home_pitcher_median_k': float(np.median(home_k_dist)),
        # Full K distributions — the app uses these to compute over-X thresholds
        'away_k_dist': away_k_dist,
        'home_k_dist': home_k_dist,
        # Retrospective 1st-inning stats for the pitcher card (may be None if
        # the pitcher lacks enough starts; the renderer handles that).
        'away_starter_first_inn': away_starter_fi_stats,
        'home_starter_first_inn': home_starter_fi_stats,
    }

    # Convert prop counts to probabilities
    final_results['player_props'] = {}
    for name, props in player_props.items():
        final_results['player_props'][name] = {k: v / iterations for k, v in props.items()}

    return final_results