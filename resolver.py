import pandas as pd
import numpy as np

from recent_form import RecentForm, blend_with_recent
from batter_arsenal_profile import (
    BatterArsenalProfile, arsenal_weighted_rate, PITCH_TYPES
)


class MatchupResolver:
    def __init__(self,
                 pqm_path='./data/pitch_matrix.parquet',
                 cqm_path='./data/contact_matrix_env.parquet',
                 load_recent_form: bool = True,
                 load_arsenal_profile: bool = True):
        print("[SYSTEM] Initializing Advanced Resolver Engine (Platoon & Arsenal Active)...")
        self.pqm = pd.read_parquet(pqm_path, engine='pyarrow')
        self.cqm = pd.read_parquet(cqm_path, engine='pyarrow')

        self.fastballs = ['FF', 'SI', 'FC']

        # League averages — used as the regression anchor for all players
        self.league_avg = {
            'k_rate':  0.222,
            'bb_rate': 0.082,
            'single':  0.150,
            'double':  0.047,
            'triple':  0.004,
            'hr':      0.035,
        }

        # How many contact events before we trust the player's data fully (regression anchor)
        self.CONTACT_THRESHOLD = 200
        self.PLATOON_THRESHOLD  = 100
        self.PITCHER_THRESHOLD  = 150  # terminal events, not pitches

        # 1st-inning-specific thresholds. A full-time starter accrues ~4 PA per
        # start in the 1st, so 80 PA ≈ 20 starts. Lower than PITCHER_THRESHOLD
        # because 1st-inning samples are inherently thinner; we shrink harder
        # toward league average to compensate.
        self.FIRST_INN_PA_THRESHOLD = 80
        self.FIRST_INN_MIN_STARTS    = 5   # below this, don't report stats at all

        # Recent form: loaded lazily since backtester may disable it
        self.recent_form = None
        if load_recent_form:
            try:
                self.recent_form = RecentForm()
                print("[SYSTEM] Recent-form snapshots loaded.")
            except Exception as e:
                print(f"[SYSTEM] Recent-form unavailable ({e}). Using career-only rates.")

        # Batter arsenal profiles: pre-computed per-pitch-type shrunken rates.
        # Resolver falls back to the batter's overall contact rate if unavailable,
        # so this is safe to disable from the backtester during cold starts.
        self.arsenal_profile = None
        if load_arsenal_profile:
            try:
                self.arsenal_profile = BatterArsenalProfile()
                print("[SYSTEM] Batter arsenal profiles loaded.")
            except Exception as e:
                print(f"[SYSTEM] Arsenal profile unavailable ({e}). Falling back to overall rates.")

    # ==========================================
    # UTILITY: SAFE NORMALIZER
    # Always call this before returning a card.
    # Guarantees probs sum to exactly 1.0 so np.random.choice never crashes.
    # ==========================================
    @staticmethod
    def _normalize(card: dict) -> dict:
        keys = ['K', 'BB', 'Out', '1B', '2B', '3B', 'HR']
        total = sum(card.get(k, 0.0) for k in keys)
        if total <= 0:
            # Catastrophic fallback — should never happen
            return {'K': 0.22, 'BB': 0.08, 'Out': 0.48, '1B': 0.14, '2B': 0.04, '3B': 0.004, 'HR': 0.036}
        return {k: max(card.get(k, 0.0) / total, 0.0) for k in keys}

    # ==========================================
    # UTILITY: SMOOTH BLEND (Regression to Mean)
    # Replaces the hard sample-size cliff with a
    # continuous weight that scales 0→1 as n grows.
    # ==========================================
    @staticmethod
    def _blend(player_val: float, league_val: float, sample_n: int, threshold: int) -> float:
        weight = min(sample_n / threshold, 1.0)
        return (player_val * weight) + (league_val * (1.0 - weight))

    # ==========================================
    # UTILITY: TERMINAL EVENT PA COUNTER
    # Counts actual plate appearances from the PQM
    # instead of estimating via pitches / 3.9
    # ==========================================
    @staticmethod
    def _count_terminal_pa(p_data: pd.DataFrame) -> int:
        """Counts rows where an actual PA result is recorded."""
        terminal_events = [
            'strikeout', 'walk', 'hit_by_pitch', 'single', 'double',
            'triple', 'home_run', 'field_out', 'grounded_into_dp',
            'fielders_choice', 'force_out', 'sac_fly', 'line_out',
            'pop_out', 'sac_bunt', 'intent_walk'
        ]
        return p_data['events'].isin(terminal_events).sum()

    # ==========================================
    # 1ST-INNING PITCHER PROFILE
    # Computed directly from the PQM by filtering to inning == 1.
    # Used for:
    #   (a) display — the pitcher's NRFI record & 1st-inning ERA on the card
    #   (b) model  — low-weight blend into the per-PA card during inning 1
    #                via the `first_inning` flag on generate_probabilities()
    # ==========================================
    def get_first_inning_stats(self, pitcher_id: int, as_of_date: str = None) -> dict:
        """
        Returns 1st-inning-only stats for a starting pitcher. All counts are
        computed from terminal events (actual PA results), not pitches.

        Returns None if the pitcher has fewer than FIRST_INN_MIN_STARTS
        games — in that case the caller should fall back silently and not
        display a record.

        Returned dict:
            games_started     : # of 1st-inning appearances (sample size)
            n_pa              : # of terminal-event PAs in the 1st
            era_1st           : 1st-inning ERA (R/IP * 9), None if IP == 0
            runs_allowed_1st  : total runs allowed across all 1st innings
            k_rate_1st        : 1st-inning strikeout rate
            bb_rate_1st       : 1st-inning walk rate (incl. intentional)
            nrfi_starts       : # of starts where his half of the 1st stayed scoreless
            yrfi_starts       : # of starts where a run scored in his half
            nrfi_rate         : nrfi_starts / games_started
            opp_obp_1st       : opponents' on-base rate in his 1st innings
        """
        p_data = self.pqm[
            (self.pqm['pitcher'] == pitcher_id) &
            (self.pqm['inning'] == 1)
        ]

        # Respect as-of filtering for backtester look-ahead safety. The
        # backtester already slices self.pqm by date, so this is a no-op there;
        # but a live caller that just passes today's date gets free safety.
        if as_of_date is not None and 'game_date' in p_data.columns and len(p_data) > 0:
            cutoff = pd.Timestamp(as_of_date)
            p_data = p_data[p_data['game_date'] < cutoff]

        if len(p_data) == 0:
            return None

        game_groups = p_data.groupby('game_pk')
        games_started = game_groups.ngroups

        if games_started < self.FIRST_INN_MIN_STARTS:
            return None

        # Terminal-event PA counts
        terminal_events = [
            'strikeout', 'walk', 'hit_by_pitch', 'single', 'double',
            'triple', 'home_run', 'field_out', 'grounded_into_dp',
            'fielders_choice', 'force_out', 'sac_fly', 'line_out',
            'pop_out', 'sac_bunt', 'intent_walk'
        ]
        term = p_data[p_data['events'].isin(terminal_events)]
        n_pa = len(term)
        if n_pa == 0:
            return None

        ev = term['events'].value_counts()
        k_rate_1st  = ev.get('strikeout', 0) / n_pa
        bb_rate_1st = (ev.get('walk', 0) + ev.get('intent_walk', 0)) / n_pa

        # On-base rate for display context ("opp OBP in 1st")
        on_base = ['single', 'double', 'triple', 'home_run',
                   'walk', 'intent_walk', 'hit_by_pitch']
        opp_obp_1st = term['events'].isin(on_base).sum() / n_pa

        # Per-start runs allowed. Compute from the post_*_score delta during
        # this pitcher's half of the 1st. Only the *opposing* team scores
        # against him, so we look at whichever side is batting.
        runs_per_start = []
        nrfi_starts = 0
        for _, grp in game_groups:
            runs_this_start = 0
            for half, half_grp in grp.groupby('inning_topbot'):
                half_grp = half_grp.sort_values('pitch_number')
                if len(half_grp) == 0:
                    continue
                if half == 'Top':
                    # Away is batting → away team accrues runs against him
                    runs = (half_grp['post_away_score'].iloc[-1]
                            - half_grp['away_score'].iloc[0])
                else:
                    runs = (half_grp['post_home_score'].iloc[-1]
                            - half_grp['home_score'].iloc[0])
                runs_this_start += max(int(runs), 0)
            runs_per_start.append(runs_this_start)
            if runs_this_start == 0:
                nrfi_starts += 1

        yrfi_starts = games_started - nrfi_starts
        total_runs = int(sum(runs_per_start))

        # Approximate IP from recorded outs. A double play records 2 outs but
        # we count it as 1 — the error is small and consistent across all
        # pitchers, so relative comparisons stay clean.
        out_events = ['strikeout', 'field_out', 'grounded_into_dp',
                      'fielders_choice', 'force_out', 'sac_fly',
                      'line_out', 'pop_out', 'sac_bunt']
        total_outs = int(term['events'].isin(out_events).sum())
        ip_1st = total_outs / 3.0
        era_1st = (total_runs / ip_1st * 9.0) if ip_1st > 0 else None

        return {
            'games_started':    int(games_started),
            'n_pa':             int(n_pa),
            'era_1st':          era_1st,
            'runs_allowed_1st': total_runs,
            'k_rate_1st':       float(k_rate_1st),
            'bb_rate_1st':      float(bb_rate_1st),
            'nrfi_starts':      int(nrfi_starts),
            'yrfi_starts':      int(yrfi_starts),
            'nrfi_rate':        float(nrfi_starts / games_started),
            'opp_obp_1st':      float(opp_obp_1st),
        }

    # ==========================================
    # CORE: GENERATE STARTER MATCHUP CARD
    # ==========================================
    def generate_probabilities(self, batter_id: int, pitcher_id: int,
                               density_ratio: float = 1.0,
                               as_of_date: str = None,
                               recency_weight: float = 0.25,
                               first_inning: bool = False,
                               first_inning_weight: float = 0.20) -> dict:
        """
        Generate event probabilities for a batter vs starting pitcher matchup.

        Args:
            batter_id, pitcher_id: MLB Stats API IDs
            density_ratio:  ballpark/weather air density (1.0 = neutral)
            as_of_date:     YYYY-MM-DD. If provided AND recent_form is loaded,
                            blends recent form into the career baseline.
                            Required for backtesting to avoid look-ahead.
            recency_weight: 0.0-1.0, how much to weight recent form. 0.25 = 25%.
            first_inning:   If True, blend the pitcher's 1st-inning-specific
                            K% / BB% into the standard rates. Captures the
                            "fresh off the mound" effect — early-game stuff,
                            first trip through the order, warmup carryover.
                            Only affects K/BB; hit-rate splits by inning are
                            too noisy per-pitcher to be useful.
            first_inning_weight: 0.0-1.0, max blend weight when the pitcher
                            has a full 1st-inning sample. The effective weight
                            scales with sample size. Default 0.20 — deliberately
                            modest, since aggregate rates remain the primary signal.
        """
        # Pull recent-form snapshots if we have them and a date was given
        recent_pitcher = None
        recent_hitter  = None
        if self.recent_form is not None and as_of_date is not None:
            recent_pitcher = self.recent_form.get_pitcher(as_of_date, pitcher_id)
            recent_hitter  = self.recent_form.get_hitter(as_of_date, batter_id)

        # ------------------------------------------
        # STEP 1: PITCHER PROFILE
        # ------------------------------------------
        p_data = self.pqm[self.pqm['pitcher'] == pitcher_id]
        terminal_pa = self._count_terminal_pa(p_data)

        if terminal_pa < self.PITCHER_THRESHOLD:
            p_throws = 'R'
            k_rate  = self.league_avg['k_rate']
            bb_rate = self.league_avg['bb_rate']
            fb_pct  = 0.60
        else:
            p_throws = p_data['p_throws'].iloc[0]
            ev = p_data['events'].value_counts()

            raw_k  = ev.get('strikeout', 0) / terminal_pa
            raw_bb = (ev.get('walk', 0) + ev.get('intent_walk', 0)) / terminal_pa

            # Blend pitcher's career rates toward league average (regression to mean)
            k_rate  = self._blend(raw_k,  self.league_avg['k_rate'],  terminal_pa, self.PITCHER_THRESHOLD)
            bb_rate = self._blend(raw_bb, self.league_avg['bb_rate'], terminal_pa, self.PITCHER_THRESHOLD)

            # Apply recent-form blend if available
            if recent_pitcher:
                k_rate = blend_with_recent(
                    career_rate=k_rate,
                    recent_rate=recent_pitcher.get('recent_k_rate'),
                    effective_n=recent_pitcher.get('effective_n'),
                    recency_weight=recency_weight,
                    min_effective_n=100,
                )
                bb_rate = blend_with_recent(
                    career_rate=bb_rate,
                    recent_rate=recent_pitcher.get('recent_bb_rate'),
                    effective_n=recent_pitcher.get('effective_n'),
                    recency_weight=recency_weight,
                    min_effective_n=100,
                )

            # 1st-inning blend: shift K% / BB% toward the pitcher's own 1st-inning
            # history. The effective weight scales linearly with PA sample size,
            # capping at first_inning_weight when the pitcher has a full sample
            # (FIRST_INN_PA_THRESHOLD). This makes "the pitcher with 40 1st-inning
            # PAs gets half the nudge a 100-PA pitcher gets" fall out naturally.
            if first_inning:
                fi = self.get_first_inning_stats(pitcher_id, as_of_date=as_of_date)
                if fi is not None and fi['n_pa'] > 0:
                    conf = min(fi['n_pa'] / self.FIRST_INN_PA_THRESHOLD, 1.0)
                    w = first_inning_weight * conf
                    k_rate  = (1.0 - w) * k_rate  + w * fi['k_rate_1st']
                    bb_rate = (1.0 - w) * bb_rate + w * fi['bb_rate_1st']

            # Hard caps — no pitcher walks 25% or strikes out 45%
            k_rate  = min(k_rate,  0.40)
            bb_rate = min(bb_rate, 0.20)

            fb_count = p_data['pitch_type'].isin(self.fastballs).sum()
            fb_pct   = fb_count / max(len(p_data), 1)

        os_pct = 1.0 - fb_pct

        # ------------------------------------------
        # STEP 2: BATTER PROFILE (Platoon-aware)
        # ------------------------------------------
        # Try platoon-split data first
        b_platoon = self.cqm[
            (self.cqm['batter'] == batter_id) &
            (self.cqm['p_throws'] == p_throws)
        ]
        b_all = self.cqm[self.cqm['batter'] == batter_id]

        platoon_n = len(b_platoon)
        overall_n = len(b_all)

        if overall_n < 10:
            # Truly no data — rookie or missing — return league card
            return self._normalize(self._get_league_average_card())

        # Calculate both platoon and overall contact rates then blend based on sample size
        def calc_contact_rates(subset: pd.DataFrame) -> dict:
            if len(subset) == 0:
                return {k: self.league_avg[k] for k in ['single', 'double', 'triple', 'hr']}
            ev = subset['events'].value_counts()
            n  = len(subset)
            return {
                'single': ev.get('single', 0) / n,
                'double': ev.get('double', 0) / n,
                'triple': ev.get('triple', 0) / n,
                'hr':     ev.get('home_run', 0) / n,
            }

        platoon_rates = calc_contact_rates(b_platoon)
        overall_rates = calc_contact_rates(b_all)

        # Blend platoon rates toward overall as platoon n grows; both toward league avg
        def blended_rate(key):
            platoon_val = self._blend(platoon_rates[key], self.league_avg[key], platoon_n, self.PLATOON_THRESHOLD)
            overall_val = self._blend(overall_rates[key], self.league_avg[key], overall_n, self.CONTACT_THRESHOLD)
            # If we have decent platoon data, weight it higher
            platoon_confidence = min(platoon_n / self.PLATOON_THRESHOLD, 1.0)
            return (platoon_val * platoon_confidence) + (overall_val * (1.0 - platoon_confidence))

        # ------------------------------------------
        # STEP 3: ARSENAL BLEND (full 9-pitch-type)
        # Weight the batter's per-pitch-type shrunken rates by the pitcher's
        # actual arsenal mix (FF, SI, FC, SL, ST, CU, KC, CH, FS).
        # Reuses p_data from STEP 1 to avoid a second 4.5M-row scan.
        #
        # If arsenal data is unavailable or the pitcher's sample is too thin
        # to trust the pitch mix, we fall back to the batter's overall
        # contact rate — the same signal that feeds platoon blending.
        # ------------------------------------------
        MIN_ARSENAL_PITCHES = 50  # below this, don't trust the pitch mix

        pitcher_arsenal = None
        if self.arsenal_profile is not None and len(p_data) >= MIN_ARSENAL_PITCHES:
            pt_frame = p_data[p_data['pitch_type'].isin(PITCH_TYPES)]
            if len(pt_frame) >= MIN_ARSENAL_PITCHES:
                counts = pt_frame['pitch_type'].value_counts()
                total  = counts.sum()
                pitcher_arsenal = {pt: float(counts.get(pt, 0) / total) for pt in PITCH_TYPES}

        batter_profile  = None
        baselines_by_pt = {}
        if self.arsenal_profile is not None and as_of_date is not None:
            batter_profile  = self.arsenal_profile.get(as_of_date, batter_id)
            baselines_by_pt = self.arsenal_profile.get_baselines(as_of_date)

        def arsenal_blended(key):
            # Bail to the batter's overall rate if either half of the
            # arsenal signal is missing (no pitcher sample, or no batter
            # profile because as_of_date wasn't passed / snapshot missing).
            # Without this guard, a missing batter_profile would fall
            # through to flat league baselines, which is strictly worse
            # than the batter's own aggregate contact rate.
            if pitcher_arsenal is None or batter_profile is None:
                return overall_rates[key]
            return arsenal_weighted_rate(key, batter_profile, pitcher_arsenal, baselines_by_pt)

        # ------------------------------------------
        # STEP 4: MERGE PLATOON + ARSENAL
        # Average the platoon-aware rate and the
        # arsenal-weighted rate equally
        # ------------------------------------------
        merged = {}
        for key in ['single', 'double', 'triple', 'hr']:
            merged[key] = (blended_rate(key) + arsenal_blended(key)) / 2.0

        # ------------------------------------------
        # STEP 4.5: RECENT FORM BLEND (if available)
        # Recent-form rates are per-batted-ball, same as merged[] above,
        # so they blend on the same scale.
        # ------------------------------------------
        if recent_hitter:
            eff_n = recent_hitter.get('effective_n')
            for key, recent_key in [
                ('single', 'recent_single'),
                ('double', 'recent_double'),
                ('triple', 'recent_triple'),
                ('hr',     'recent_hr'),
            ]:
                merged[key] = blend_with_recent(
                    career_rate=merged[key],
                    recent_rate=recent_hitter.get(recent_key),
                    effective_n=eff_n,
                    recency_weight=recency_weight,
                    min_effective_n=50,
                )

        # ------------------------------------------
        # STEP 5: ENVIRONMENTAL PHYSICS
        # Thin air boosts HR AND doubles (not just HR).
        # Density > 1.0 = thin air (Coors), < 1.0 = thick (SF).
        # ------------------------------------------
        merged['hr']     *= density_ratio
        merged['double'] *= 1.0 + (density_ratio - 1.0) * 0.30   # ~30% of the HR boost applies to 2B

        # ------------------------------------------
        # STEP 6: ASSEMBLE FINAL CARD
        # ------------------------------------------
        p_in_play = max(1.0 - k_rate - bb_rate, 0.0)

        hit_sum = merged['single'] + merged['double'] + merged['triple'] + merged['hr']
        # Protect against hit rates summing > 1 from noisy data
        if hit_sum > 0.98:
            scale = 0.98 / hit_sum
            for k in merged:
                merged[k] *= scale
            hit_sum = 0.98

        out_contact = max(1.0 - hit_sum, 0.02)  # Always at least 2% contact out

        raw_card = {
            'K':   k_rate,
            'BB':  bb_rate,
            'Out': p_in_play * out_contact,
            '1B':  p_in_play * merged['single'],
            '2B':  p_in_play * merged['double'],
            '3B':  p_in_play * merged['triple'],
            'HR':  p_in_play * merged['hr'],
        }

        # Always normalize before returning — guarantees sum == 1.0
        return self._normalize(raw_card)

    # ==========================================
    # CORE: GENERATE BULLPEN CARD
    # Now actually uses bullpen pitcher data.
    # Falls back to batter overall if no BP data.
    # ==========================================
    def generate_bullpen_probabilities(self, batter_id: int, bullpen_ids: list,
                                       density_ratio: float = 1.0,
                                       as_of_date: str = None,
                                       recency_weight: float = 0.25) -> dict:

        # Pull recent hitter form if available (bullpen aggregate has no single recent snapshot)
        recent_hitter = None
        if self.recent_form is not None and as_of_date is not None:
            recent_hitter = self.recent_form.get_hitter(as_of_date, batter_id)

        # Aggregate all bullpen pitchers' data into one pool
        if bullpen_ids:
            bp_data = self.pqm[self.pqm['pitcher'].isin(bullpen_ids)]
        else:
            bp_data = pd.DataFrame()

        # Build a blended bullpen K/BB rate from actual BP arms
        if len(bp_data) > 50:
            terminal_pa = self._count_terminal_pa(bp_data)
            if terminal_pa > 0:
                ev      = bp_data['events'].value_counts()
                raw_k   = ev.get('strikeout', 0) / terminal_pa
                raw_bb  = (ev.get('walk', 0) + ev.get('intent_walk', 0)) / terminal_pa
                k_rate  = self._blend(raw_k,  self.league_avg['k_rate'],  terminal_pa, self.PITCHER_THRESHOLD)
                bb_rate = self._blend(raw_bb, self.league_avg['bb_rate'], terminal_pa, self.PITCHER_THRESHOLD)
                # Bullpen arms generally throw more strikes — slight K boost is realistic
                k_rate  = min(k_rate * 1.05, 0.40)
                bb_rate = min(bb_rate, 0.20)
            else:
                k_rate  = self.league_avg['k_rate']
                bb_rate = self.league_avg['bb_rate']
        else:
            # No bullpen data — use league avg with a slight K boost (relievers K more)
            k_rate  = self.league_avg['k_rate'] * 1.05
            bb_rate = self.league_avg['bb_rate']

        # Batter contact profile vs. bullpen (use overall, no platoon split needed here)
        b_data = self.cqm[self.cqm['batter'] == batter_id]
        if len(b_data) < 10:
            return self._normalize(self._get_league_average_card())

        ev = b_data['events'].value_counts()
        n  = len(b_data)

        contact = {
            'single': self._blend(ev.get('single', 0) / n,    self.league_avg['single'], n, self.CONTACT_THRESHOLD),
            'double': self._blend(ev.get('double', 0) / n,    self.league_avg['double'], n, self.CONTACT_THRESHOLD),
            'triple': self._blend(ev.get('triple', 0) / n,    self.league_avg['triple'], n, self.CONTACT_THRESHOLD),
            'hr':     self._blend(ev.get('home_run', 0) / n,  self.league_avg['hr'],     n, self.CONTACT_THRESHOLD),
        }

        # Apply recent-form hitter blend (bullpen pitchers are varied, so we
        # trust the hitter's own recent performance as the primary signal)
        if recent_hitter:
            eff_n = recent_hitter.get('effective_n')
            for key, recent_key in [
                ('single', 'recent_single'),
                ('double', 'recent_double'),
                ('triple', 'recent_triple'),
                ('hr',     'recent_hr'),
            ]:
                contact[key] = blend_with_recent(
                    career_rate=contact[key],
                    recent_rate=recent_hitter.get(recent_key),
                    effective_n=eff_n,
                    recency_weight=recency_weight,
                    min_effective_n=50,
                )

        # Environmental physics
        contact['hr']     *= density_ratio
        contact['double'] *= 1.0 + (density_ratio - 1.0) * 0.30

        p_in_play = max(1.0 - k_rate - bb_rate, 0.0)
        hit_sum   = sum(contact.values())
        if hit_sum > 0.98:
            scale = 0.98 / hit_sum
            for k in contact:
                contact[k] *= scale

        out_contact = max(1.0 - sum(contact.values()), 0.02)

        raw_card = {
            'K':   k_rate,
            'BB':  bb_rate,
            'Out': p_in_play * out_contact,
            '1B':  p_in_play * contact['single'],
            '2B':  p_in_play * contact['double'],
            '3B':  p_in_play * contact['triple'],
            'HR':  p_in_play * contact['hr'],
        }

        return self._normalize(raw_card)

    # ==========================================
    # FALLBACK: LEAGUE AVERAGE CARD
    # ==========================================
    def _get_league_average_card(self) -> dict:
        """Emergency fallback for players with no historical data."""
        return {
            'K':   self.league_avg['k_rate'],
            'BB':  self.league_avg['bb_rate'],
            'Out': 0.477,
            '1B':  self.league_avg['single'],
            '2B':  self.league_avg['double'],
            '3B':  self.league_avg['triple'],
            'HR':  self.league_avg['hr'],
        }