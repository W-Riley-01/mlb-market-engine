# Full v4 backtest — addendum, 2026-10-23: lineup defect

**Status.** Written 2026-10-23, after the results in `BACKTEST_V4_RESULTS.md` were seen.
- It is post hoc and does not edit the frozen plan (`BACKTEST_V4_ANALYSIS_PLAN.md`, sha256 `4a57fcaa…`).
- Its own SHA-256 is recorded in `analysis/out/PLAN_HASH.txt`.
- No re-simulation was run.
- Evidence: `analysis/v4_lineup_check.py` (log `analysis/out/v4_lineup_check_log.txt`) and EXPLORATORY E0 in
  `BACKTEST_V4_RESULTS.md`.

## 1. The defect

`analysis/backtest_v4.py` (line 170) builds each team's lineup from the box score's `teams.{side}.battingOrder`.
- **What that list holds:** the **last** player in each of the nine slots at the end of the game, not the starting
  nine.
- **Example:** in game 747060, the Angels' list has Mickey Moniak (code `301`, entered for Mike Trout, `300`) in
  slot 3, Luis Rengifo (`501`) in slot 5 and Jo Adell (`701`) in slot 7.
- **Lineup check is ineffective:** the "full lineup" check (`len(battingOrder) == 9`) always passes, so it never
  tested for a starting nine.

**Counts** (all 9,704 ledger games, 19,408 team-games, 174,672 lineup entries):
- **Substitutes:** 24,251 lineup entries (13.9%) are substitutes, meaning their box `battingOrder` code does not end
  in `00`. By season: 2022 5,757; 2023 6,170; 2024 6,143; 2025 6,181.
- **Team-games:** only 5,221 (26.9%) have the true starting nine; 14,187 (73.1%) contain at least one substitute.
- **Substitutes per game** (both lineups, 9,696 analysis-set games): mean 2.50.

| substitutes in the game | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| games | 858 | 1,942 | 2,363 | 2,138 | 1,318 | 708 | 269 | 82 | 16 | 2 |

- **Substitutes' plate appearances in the game:**

| PA in the game | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| substitute entries | 6,956 | 11,752 | 4,512 | 846 | 164 | 21 |

- **Slots:** every entry's slot is correct (the first digit of its code matches the ledger slot); only the player is
  wrong.

The lineups therefore contain **post-game information**: who came off the bench, which depends on injuries,
blowouts, pinch-hitting matchups and extra innings. They also omit starters who were replaced.

## 2. Plan statements this contradicts

| document | statement | what is actually true |
|---|---|---|
| `BACKTEST_V4_PLAN.md:13` | Lineups = "box-score starting nine (`battingOrder`)", "the posted lineup, known to the market before first pitch" | `battingOrder` is the end-of-game nine, not the posted lineup. |
| `BACKTEST_V4_PLAN.md:25` | Exclude "games without a full starting nine on both sides" | The check tested the length of the end-of-game list, so it could not detect this. |
| `BACKTEST_V4_ANALYSIS_PLAN.md:140` (test 3 pool) | "All starting-lineup batters (the ledger `batters` field, slots 1–9)" | The `batters` field is not the starting lineup; the pool holds 23,928 substitutes among 172,502 batter-games. |
| `BACKTEST_V4_ANALYSIS_PLAN.md:214` (section 3b) | "all starting-lineup batter-games" | Same. |
| `BACKTEST_V4_ANALYSIS_PLAN.md:270` (limitation 4) | "The starting lineup is taken from the box score…" | It is the end-of-game lineup, which is post-game information. |

## 3. Tests affected, and how

**Every simulation input lineup is affected**, so every model output is affected: win probabilities, totals,
strikeout distributions and props.

**Not affected:**
- the baselines B1, B2 and the strikeout baseline (built from Statcast `game_date < D`);
- the outcomes (box-score runs, hits, Ks);
- the starter choice (probable pitchers).

**Test 1, sides.**
- *What is wrong:* the model's P(home) was simulated with partly post-game lineups. The test therefore is not a
  leak-free pre-game comparison with the closing line.
- *Exploratory diagnostic:* the home-minus-away substitute count correlates −0.090 with (model logit − market
  logit) and −0.092 with (home win − p_market). That means some outcome-related information may have reached the
  model through the lineups.
- *Direction:* the effect on the blend-vs-market statistic is not known without re-running.

**Test 2, totals.**
- *What is wrong:* `model_total_mean` was simulated with the same lineups.
- *Exploratory diagnostic:* the substitute count correlates −0.110 with (model − line) and +0.197 with (actual −
  line). Adding it as a covariate leaves the model slope at −0.106 (−0.202 to −0.005).

**Test 3, Beat the Streak.**
- *Pool:* the candidate pool contains substitutes, contrary to the plan.
- *Picks:* the simulator's top-1 picks include 53 substitutes, against 12 for B1 and 5 for B2. All 26 simulator
  passes are substitutes, 24 of them with 0 PA.
- *Teammates:* every batter's P(1+ hit) was also simulated with partly wrong teammates. That changes how many PAs
  the batter gets and the run environment around him.
- *Consequence:* the paired comparison against B2 is not the comparison the plan intended.

**Test 4, starter strikeouts.**
- *Pitcher side:* the starter is not affected.
- *Hitter side:* each starter's simulated strikeouts came from facing the opponent's end-of-game nine. Late
  substitutes in reality faced mostly relievers.
- *Size:* the effect on ΔMAE is not known.

**Test 5, descriptive.**
- *Home-win AUC:* the same issue as test 1.
- *Home-run AUC:* the population includes substitute batter-games with PA ≥ 1 (17,295 across all ledger games),
  whose props were simulated as if they played the whole game.

**Section 3b.** The same issue as test 3.

**Conclusion of this addendum.** The confirmatory outcomes in `BACKTEST_V4_RESULTS.md` were computed exactly as the
plan specified. However, the ledger they rest on does not meet the plan's stated input assumption, and the
outcomes cannot be read as tests of a leak-free, pre-game v4 model.

## 4. The correct field

The true starting nine is `teams.{side}.players[ID*].battingOrder == slot × 100` (codes `100` … `900`).

| check | team-games | exactly nine starters in slots 1–9 | match Statcast first-nine order |
|---|---|---|---|
| verified games: 8 double-switch, 10 doubleheader (5 pairs), 8 random | 52 | 52 | 52 |
| all ledger games | 19,408 | **19,408 (100%)** | 19,275 (99.3%) |

- **Double-switch games** were confirmed in the live feed: an incoming pitcher took a position player's slot.
- **The 133 Statcast disagreements** are attributable to the Statcast proxy, not the box field:
  - 68: the same nine players in a different first-appearance order;
  - 57: a box starter with 0 PA, so he never appears in Statcast;
  - 6: a box starter absent from Statcast's first nine despite having PA;
  - 2: no Statcast rows.

## 5. Same defect elsewhere

**Pilot ledger.**
- `analysis/out/ledger_v4_pilot.csv`: 717 of 5,400 lineup entries (13.3%) are substitutes.
- `ledger_v4_pilot_azath_fixed.csv`: 88 of 702 (12.5%).
- **Consequence:** the pilot results in `BACKTEST_V4_PLAN.md` and `REPORT.md` share the defect.

**Old ledger, `data/backtest_ledger.csv`.**
- *Code path:* produced by `backtester.py` via `daily_scraper.fetch_game_rosters`, which uses the same
  `battingOrder` list (`daily_scraper.py:115`) on final games. The same function already supplies the known
  post-game bullpen (`pitchers[1:]`) and starter (`pitchers[0]`).
- *Data check:* 7,264 of its 8,354 rows have a cached box score. Of their 130,379 lineup names matched to a box
  code (373 unmatched), 18,359 (14.1%) are substitutes.

**Live engine (`engine_runner.py` → `fetch_game_rosters`).**
- *Code:* the same field is used. Before first pitch, `battingOrder` is the posted lineup, and predictions are
  logged only for statuses in `PRE_FIRST_PITCH_STATUSES` (Scheduled, Pre-Game, Warmup, Delayed Start).
- *Data check* (read-only DB, `player_predictions`, 60 random 2026 games, 1,080 rows):
  - 1,077 rows are box starters (`x00`); slot matches in 1,071.
  - **Game 824785:** one substitute (Sean Keys, `502`). The prediction is time-stamped 2026-09-23 15:01 UTC, about
    16 h after the scheduled start of 2026-09-22 22:35 UTC.
  - **Games 823063 and 823184:** the lineup changed after a pre-game prediction (players moved or scratched). These
    are not this defect.
- *Conclusion:* the live path is correct when the game has not started. The single exception was logged after the
  scheduled start time, and its cause was not established.
