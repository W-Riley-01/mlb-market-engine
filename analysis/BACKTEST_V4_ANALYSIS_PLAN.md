# Full v4 backtest — pre-registered analysis plan

**Written and frozen:** 2026-10-22, before any `ledger_v4_full_*.csv` was opened, read, merged or summarized.
Only the run logs (dates, gamePks, sim seconds) had been looked at. SHA-256 of this file and the freeze time are
in `analysis/out/PLAN_HASH.txt`.
- **Amended 2026-10-22, still before any ledger was opened.** The changes are made in place and listed in
  "Amendments before results"; the new hash is added beside the old one.
- Any change made after results are seen goes in a dated **Addendum** at the end and is labelled post hoc. Text
  above the addendum is never edited after results are seen.

## 0. Data and common rules

**Inputs**
- **Predictions:** the union of `analysis/out/ledger_v4_full_{A,B,C,D,E,F}_*.csv`. The runs used
  `analysis/backtest_v4.py`:
  - regular season only (gameType R), 1,000 iterations;
  - date ranges 2024-03-01..2025-10-01 (A–C) and 2022-03-01..2023-10-02 (D–F);
  - four seasons in total: 2022, 2023, 2024, 2025;
  - `--contact-matrix data/contact_matrix_env_fixed.parquet`;
  - `MODEL_VERSION` v4 (sides wiring fixed).
- **Outcomes:** taken from the cached box scores `analysis/cache/api/box/{gamePk}.json`, never from the ledger's own
  outcome fields (`*_runs`, `*_starter_actual_k`, the box-line fields inside `batters`, `actual_*`).
  - Batters are joined by (gamePk, player id) to `teams.{side}.players.ID{id}.stats.batting`, using
    `atBats`, `plateAppearances`, `hits` and `homeRuns`.
  - Starters are joined by (gamePk, player id) to `stats.pitching.strikeOuts`.
  - Runs come from `teams.{side}.teamStats.batting.runs`.
  - Any id that does not join is dropped and counted.
- **Baselines:** use `data/pitch_matrix.parquet` (Statcast, 2019-03-20..2025-11-01) restricted to `game_date < D`.
  The file has no game-type column, so the trailing windows include any postseason games it holds.

**Integrity gate.** Runs before any test; it only counts rows and checks keys.
1. All six logs end with a `skipped:` line and contain no `Traceback`. If not, resume the chunk and wait.
2. No gamePk appears twice across the files. If one does, keep the first row and report the duplicate.
3. Report games per season against the schedule's count of final regular-season games, with the skip reasons from
   the logs.
4. Report the counts of `odds_match` values, `starter_source` values and `name_collision`.

**Analysis set.** Ledger rows where **both** `*_starter_source == "probable"`.
- Rows with `actual_fallback` used the post-game starter, so they are excluded from every confirmatory test.
  Their count is reported.
- **Season** = year of `date`. `date` is the schedule's official (ET) date.

**Orientation check (stop rule).** P(home win) = `1 − model_away_win`; no swapping is allowed.
If the home-win AUC on the analysis set is < 0.5, stop and report a possible wiring regression before any test.

**Intervals and seeds**
- Percentile bootstrap, 2,000 resamples. Seed = 20261022 + test number.
- Wilson 95% intervals for proportions.
- Every number is reported with its n.
- Calibration bins with n < 30 are shown but not interpreted.

**Pooling.** Tests 1–4 pool all four seasons. Every result is also reported per season. 2022 is reported
separately and flagged: it is the one season outside 2023–25, the window in which the simulator constants were
tuned.

**Multiplicity.** Tests 1–4 are the confirmatory family. No correction is applied to the rules below.
- Each rule fires only when a 95% interval lies entirely on the **favourable** side of zero. For an effect that is
  exactly zero, that is one tail of a two-sided 95% interval, so the per-test false-positive rate is about 0.025.
- Test 1 adds the 3-of-4-seasons condition, so its rate is at most 0.025.
- Test 4's "baseline better" outcome is not a claim for the model and is not counted.
- If all four nulls are true:
  - **independent tests:** 1 − 0.975⁴ = 1 − 0.9037 = **0.096 (about 10%)**;
  - **any dependence:** the Bonferroni bound gives 4 × 0.025 = **0.10** at most.
- The earlier "about 19%" (1 − 0.95⁴ = 0.185) wrongly counted both tails.
- Percentile-bootstrap coverage is approximate, so these figures are too.
- The report will state this next to any positive result.

**Exploratory.** Everything in Section 5 and anything not listed in this plan is exploratory and labelled as such.
The secondary analyses in Section 3b sit outside the confirmatory family and do not change the figure above.

## 1. Sides (moneyline) — confirmatory

**Data.** Analysis-set games with `odds_match ∈ {unique, score_disambiguated}` and both `odds_away_ml_close` and
`odds_home_ml_close` present.
- Market probability: p_mkt = de-vigged home probability, using multiplicative normalisation of the American-odds
  implied probabilities (`implied()` as in `ledger_market.py`).
- Games with overround < 0 or > 0.10 are dropped as data errors and counted.
- y = home won (box-score runs).

**Model.** As in `analysis/ledger_market.py`: logistic regression y ~ 1 + logit(p_mkt) + logit(p_model).
- p_model = 1 − model_away_win; probabilities are clipped to [1e-6, 1 − 1e-6].
- Cross-fitted by season, leave one season out: for each of 2022–2025, fit on the other three seasons and predict
  the held-out season. This gives p_blend.

**Statistic (primary).** Mean per-game log loss of p_blend minus log loss of p_mkt, pooled over all four held-out
seasons. Negative means the blend beats the market.

**Interval.** Cluster bootstrap by game, which is one row per game. The 2,000 resamples redraw the held-out
per-game differences with the fold coefficients held fixed, as in `ledger_market.py`.

**Decision rule.**
- **"Information confirmed"** only if the pooled 95% interval is entirely below 0 **and** the held-out point
  estimate is negative in at least 3 of the 4 seasons.
- Otherwise **"not confirmed"**.

**Also reported**
- The full-sample coefficient on logit(p_model), with the coefficient on logit(p_market). Each gets a 2,000-resample
  game bootstrap interval. These are descriptive.
- Per-fold coefficients and log loss / Brier score for market, raw model and blend.
- **ROI, descriptive only:**
  - Edges are measured against the de-vigged market: edge_home = p_blend − p_mkt and
    edge_away = (1 − p_blend) − (1 − p_mkt).
  - For each game, take the side with the larger edge and bet it only if that edge is ≥ **0.04**. The threshold is
    fixed now and is the only one used.
  - Stake 1 unit at that side's closing American price.
  - Report the number of bets, the win rate (Wilson interval), and ROI = profit ÷ staked with a game-bootstrap
    interval, pooled and per season.
  - ROI carries no decision rule. Closing prices may not have been obtainable.

## 2. Totals — confirmatory

**Data.** Analysis-set games with `odds_over_close` present.

`over close` is the closing total line. This was checked on the odds file `data/MLB_Basic.csv` via `OddsBook`,
not on a ledger, on 2026-10-22:
- n = 41,214 games, 2009–2026, no missing values;
- min 3.5, max 19.5;
- most common values: 8.5 (8,500), 7.5 (6,733), 9.0 (6,663), 8.0 (6,166), 9.5 (3,922), 7.0 (3,614);
- `under close` equals `over close` in every row.

**Model value.** `model_total_mean`, the simulated mean total runs. This ledger has the mean, so the median used
previously is not used.

**Statistic.** OLS of (box-score total runs − `odds_over_close`) on (`model_total_mean` − `odds_over_close`).
The statistic is the slope.

**Interval.** Cluster bootstrap by game, 2,000 resamples.

**Decision rule.**
- **"Confirmed"** only if the slope's 95% interval lies entirely above 0.
- Otherwise **"not confirmed"**. A negative slope is "not confirmed".

**Also reported.** Intercept, slope and interval per season, plus the mean and SD of (model − line), as descriptive
results.

**Caveat.** density_ratio = 1.0, so the model's totals have no park or weather. The market's do.

## 3. Beat the Streak — confirmatory

**Candidate pool.** All starting-lineup batters (the ledger `batters` field, slots 1–9) in analysis-set games on that
date.
- A batter in both games of a doubleheader counts only in `game_number` 1.
- The simulator's score is `p_1h`.
- Ties are broken by lower gamePk, then lower player id. This applies to every method.

**Outcome per pick** (box score):

| outcome | condition |
|---|---|
| **hit** | hits ≥ 1 |
| **loss** | atBats ≥ 1 and hits = 0 |
| **pass** | atBats = 0, excluded |

**Picks.** Top-1 and top-2 per date, for each method.

**Baselines.** Built only from pitch_matrix rows with game_date < D, over the trailing 365 days [D − 365, D).

*Definitions*
- PA = pitch rows with a non-null `events` value. This applies to every baseline, so the baselines do not inherit
  the resolver's event-list problem (limitation 7).
- H = single, double, triple, home_run.
- AB = PA minus walk, hit_by_pitch, sac_fly, sac_bunt, sac_fly_double_play, sac_bunt_double_play and
  catcher_interf.

*B1*
- Highest trailing-365-day batting average, H/AB.
- Only batters with **≥ 200 PA** in the window are eligible. Others in the pool are skipped by B1.

*B2*
- **Batter rate:** b = (H + 200·L) / (PA + 200).
- **Starter rate:** p = (H allowed + 400·L) / (batters faced + 400), for the opposing **predicted** starter
  (`*_starter_pred`). A starter with no window data gets p = L.
- L is the league H/PA in the same window. The shrinkage constants 200 and 400 are fixed now and are not tuned.
- **Per-PA rate against the starter (log5):** x = (b·p/L) / (b·p/L + (1−b)(1−p)/(1−L)).
- **Per-PA rate for the remaining PAs:** log5 with the pitcher rate set to L, which reduces to b.
- **Expected PA:** E = mean PA in the window for a starting batter in that lineup slot and side (home/away).
  - The starting batters are the first 9 distinct batters of each team-game, ordered by `at_bat_number`.
  - Only their own PAs are counted.
- **Starter share:** s = the share of those PAs that came against the opposing team's starting pitcher, by lineup
  slot and side, over the same window. The starting pitcher is the team's first pitcher of the game.
- **P(1+ hit)** = 1 − (1 − x)^(s·E) · (1 − b)^((1 − s)·E).

**Statistic (primary).** Success rate of the top-1 pick, hits ÷ (hits + losses), for the simulator, B1 and B2.

**Paired comparison.** By date, over dates where both picks are decided (not a pass). Per date, d = 1[sim hit] −
1[baseline hit]; the statistic is the mean of d. Bootstrap clustered by date, 2,000 resamples.

**Decision rule.**
- The simulator **"adds value"** only if its paired difference against **B2** has a 95% interval entirely above 0.
- Otherwise **"does not add value"**.
- The comparison against B1 is reported with its interval but carries no rule.

**Also reported**
- **Calibration** of `p_1h`:
  - Population: all pool batters with atBats ≥ 1 (passes excluded, as in BTS).
  - Bins: <0.50, 0.50–0.60, 0.60–0.65, 0.65–0.70, 0.70–0.75, 0.75–0.80, 0.80–0.85, ≥0.85.
  - Each bin shows n, mean predicted, realized and a Wilson interval. Bins at and above 0.75 are the ones of
    interest.
- Mean predicted vs realized for the **top-1 picks** themselves. These are expected to read high: picking the
  maximum of noisy p_1h (Monte Carlo SE about 0.012) inflates the prediction (winner's curse).
- **Top-2 both-hit rate:** dates where both of a method's top-2 picks have atBats ≥ 1. Shown for all three methods
  with Wilson intervals, with dates excluded by passes counted.
- **57-game streak:** the probability of at least one run of ≥ 57 consecutive hits in 180 independent daily top-1
  picks. Computed exactly by dynamic programming over the current run length.
  - Evaluated at the observed top-1 rate and at each end of its 95% interval, for each method.
  - Passes, streak savers and double-downs are ignored.
- Per-season rates.

## 3b. Beat the Streak — secondary

These analyses sit outside the confirmatory family and carry no decision rule.

**Log loss of P(1+ hit), simulator vs B2.**
- Population: all starting-lineup batter-games that join a box score, both games of a doubleheader included.
- y = hits ≥ 1.
- Statistic: mean per-row log loss (simulator − B2), with probabilities clipped as in test 1.
- Bootstrap clustered by date, 2,000 resamples.

**Top-5 success rate, simulator vs B2.**
- Each method takes its top 5 picks per date from the Section 3 pool, with the same tie-breaks.
- Pooled success rate = hits ÷ (hits + losses) over all picks; passes are excluded.
- Statistic: simulator rate − B2 rate, paired by date (the same dates for both methods).
- Bootstrap clustered by date, 2,000 resamples.

## 4. Starter strikeouts — confirmatory

**Data.** Analysis-set starters where `*_starter_pred == *_starter_actual` (the predicted starter started).
- Model point forecast = `model_*_k_med`.
- Outcome = box-score `strikeOuts` of that pitcher.

**Baseline.** The pitcher's trailing-365-day mean strikeouts per start (window [D − 365, D)).
- A start = a game in which he threw his team's first pitch (pitch_matrix, ordered by at_bat_number and
  pitch_number).
- Ks = his events in strikeout or strikeout_double_play.
- He needs **≥ 3 starts** in the window. Otherwise the baseline is the league mean Ks per start in the same window.

**Statistics.** For model and baseline:
- bias = mean(predicted − actual);
- MAE;
- Pearson r with actual.

**Primary statistic.** ΔMAE = MAE(model) − MAE(baseline), paired on the same starts. Bootstrap clustered by game
(both starters of a game together), 2,000 resamples.

**Decision rule.**

| result | condition |
|---|---|
| **"Model better"** | 95% interval entirely below 0 |
| **"Baseline better"** | interval entirely above 0 |
| **"No difference shown"** | otherwise |

**Also reported.** r of the model's K with the *other* starter's actual Ks, as a wiring check. Per-season splits.

## 5. Descriptive only (no decision rules)

- **Home-win AUC.** Game bootstrap interval, pooled and per season, on the analysis set and on all ledger rows.
- **HR props.** For starters with plateAppearances ≥ 1, `p_1hr` vs homeRuns ≥ 1:
  - AUC with a game-cluster bootstrap interval;
  - calibration in 10 equal-count bins with Wilson intervals;
  - pooled and per season.
- **Per-season splits** of every statistic above.

## 6. Known limitations, stated before results

1. **No weather or park.** density_ratio = 1.0 in every game.
2. **Arsenal off.** The log says "arsenal=DISABLED (no as_of_date passed)".
3. **Tuned on the test years.** Model constants were tuned on 2023–25 data, overlapping the test years, so this is
   not a fully out-of-time test of the constants.
4. **Actual lineups.** The starting lineup is taken from the box score, which is known only about 1–3 h before first
   pitch. That is fine for closing lines, but it is not a morning-of forecast.
5. **Monte Carlo noise.** 1,000 iterations put about 0.012 of noise on P(hit) near 0.8 and add noise to the K
   medians.
6. **Unverified odds.** The closing-odds source (`data/MLB_Basic.csv`) has not been verified against an independent
   source. Team codes switch to AZ/ATH in 2025 and are aliased.
7. **Event-name mismatch (noticed while writing this plan).**
   - The resolver's PA event lists (`resolver.py:172`, `:231`, `:278`) use `grounded_into_dp`.
   - pitch_matrix uses `grounded_into_double_play` (22,154 rows).
   - Events such as `double_play`, `fielders_choice_out`, `strikeout_double_play` and `field_error` are absent from
     those lists.
   - The model's PA denominators may therefore be undercounted. This is a property of the model under test. It is
     not corrected here.
8. **Postseason in baselines.** The baselines' trailing windows may include postseason games, because pitch_matrix
   has no game type.
9. **Pitchers batting in league baselines.** The model's expanding-window league baselines start in 2019, so they
   include pitcher plate appearances from 2019 and 2021, before the universal DH. The 2022 trailing-365-day
   baselines also reach into 2021.
10. **The model is tested as it stands.** It is the post-fix v4 simulator, including the known plate-appearance
    event-list problem (limitation 7). Later fixes will be tested as paired re-runs on the same games, not by
    changing this analysis.

## Amendments before results (2026-10-22)

Made before any `ledger_v4_full_*.csv` was opened. The original hash is kept in `PLAN_HASH.txt`.

a. **Sample.**
   - The 2022 and 2023 regular seasons are added: 2022-03-01..2023-10-02, chunks D–F, with the same filters and the
     same fixed contact matrix.
   - Tests 1–4 pool all four seasons.
   - Test 1: leave-one-season-out cross-fitting. The rule is now: pooled 95% interval below 0 and a negative
     held-out estimate in at least 3 of 4 seasons.
   - 2022 is reported separately as the season outside the 2023–25 tuning window.
   - Integrity gate: six logs instead of three.

b. **B2.**
   - The starter's rate is weighted by the starter's share of PAs (trailing 365 days, by slot and side, from
     Statcast). The league pitcher rate is used for the remainder.
   - A PA is now any pitch row with a non-null `events` value. This replaces "excluding `truncated_pa`" and also
     applies to B1's 200-PA minimum and to the AB definition.
   - The constants stay at 200 PA and 400 batters faced.

c. **Secondary analyses.** Section 3b is new: log loss of P(1+ hit), simulator vs B2; and the top-5 success rate,
   simulator vs B2, paired by date. Both are outside the confirmatory family.

d. **Totals line.** `odds_over_close` was confirmed as the closing total line from the odds file. The figures are in
   test 2.

e. **Limitations.** Added limitations 9 (pitcher plate appearances in league baselines) and 10 (the model is tested
   as it stands; later fixes become paired re-runs).

f. **Multiplicity.** "About 19%" is replaced by the favourable-tail figure, with the arithmetic: 1 − 0.975⁴ = 0.096
   if the tests are independent, and at most 4 × 0.025 = 0.10 under any dependence.

## Addendum

(none)
