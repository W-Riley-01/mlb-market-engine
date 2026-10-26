# Full v4 backtest, corrected re-run (R2) — results

Plan: `analysis/BACKTEST_V4_ANALYSIS_PLAN_R2.md`, SHA-256 `ec6b21042fcb34bee58c51e0acb62b641d684514a5088c2076ea44a7afe4a86c` (matches the last `r2 amended sha256:` line of `analysis/out/PLAN_HASH.txt`). Run by `analysis/v4r2_analysis.py` (reusing `v4_analysis.py`, `v4_integrity.py`) (logs: `analysis/out/v4r2_integrity_log.txt`, `analysis/out/v4r2_analysis_log.txt`). Outcomes from cached box scores; baselines from `data/pitch_matrix.parquet` with game_date < D.

## Integrity gate and analysis set

- Ledger rows 9704 (2022 2429, 2023 2423, 2024 2426, 2025 2426); no duplicate gamePks; every eligible scheduled game present; every gamePk has a cached box score; all batter and starter ids joined (0 unjoined). Integrity: PASS.
- Excluded for `actual_fallback`: 8 games → analysis set 9696 games.
- Lineup entries in analysis-set games whose box `battingOrder` is not slot × 100, excluded from the pool: 0.
- Orientation check: home-win AUC on the analysis set = 0.5867 (n = 9696) ≥ 0.5 → continue; nothing swapped.

## R2 headline next to R1

| test | n | R2 (95% interval) | R2 outcome | R1 |
|---|---|---|---|---|
| 1 Sides: dLL blend − market | 9659 | 0.00027 (-0.00008 to 0.00061) | not confirmed | -0.00053 (-0.00143 to 0.00038); not confirmed |
| 2 Totals: slope | 9685 | 0.075 (-0.023 to 0.171) | not confirmed | -0.201 (-0.301 to -0.105); not confirmed |
| 3 BTS: top-1 simulator − B2 (paired dates) | 728 | -0.010 (-0.047 to 0.032) | does not add value | -0.020 (-0.063 to 0.021); does not add value |
| 3b LL P(1+ hit) simulator − B2 | 174528 | 0.00627 (0.00566 to 0.00688) | no rule | 0.01138 (0.01077 to 0.01203) |
| 3b top-5 rate simulator − B2 (dates) | 730 | -0.008 (-0.026 to 0.009) | no rule | 0.001 (-0.015 to 0.018) |
| 4 Starter K: ΔMAE model − baseline | 19372 | 0.025 (0.011 to 0.038) | Baseline better | 0.043 (0.029 to 0.057); Baseline better |
| 5 HR AUC `p_1hr` (pooled) | 174470 | 0.608 (0.604 to 0.612) | descriptive | 0.613 (0.609 to 0.617) |

R2 governs tests 1, 2, 3 and 4 and Section 3b (plan, "Which run governs" and amendment 1). R1 outcomes stay reported in `BACKTEST_V4_RESULTS.md`.

Integrity extras: `starter_source` values (team-games) {'probable': 19400, 'actual_fallback': 8}; `odds_match` {'unique': 9379, 'score_disambiguated': 314, 'none': 9, 'ambiguous': 2}; `name_collision` games 0. Full gate output: `analysis/out/v4r2_integrity_log.txt`.

## Confirmatory outcomes (plan's words)

| test | outcome |
|---|---|
| 1 Sides | not confirmed |
| 2 Totals | not confirmed |
| 3 Beat the Streak | does not add value |
| 4 Starter strikeouts | Baseline better |

Multiplicity (plan): if all four nulls are true, the chance of at least one false favourable outcome is about 10% (1 − 0.975⁴ = 0.096 if independent; at most 0.10 under any dependence).

## Test 1 — Sides (moneyline) — confirmatory

- **Data:** analysis-set games with a matched closing moneyline on both sides. Dropped for overround outside 0–0.10: 26. n = 9659 games (2022: 2399, 2023: 2416, 2024: 2422, 2025: 2422). Mean overround 0.044.
- **Statistic:** mean per-game log loss of p_blend minus log loss of p_mkt, pooled over the four held-out seasons (leave-one-season-out cross-fit).
- **Result:** **0.00027**, 95% interval **-0.00008 to 0.00061** (game bootstrap, 2,000 resamples, seed 20261023).

| held-out season | n | dLL (blend − market) | 95% interval |
|---|---|---|---|
| 2022 | 2399 | -0.00005 | -0.00040 to 0.00028 |
| 2023 | 2416 | 0.00047 | -0.00044 to 0.00139 |
| 2024 | 2422 | -0.00000 | -0.00052 to 0.00054 |
| 2025 | 2422 | 0.00064 | -0.00022 to 0.00157 |

Seasons with a negative point estimate: 2 of 4.

**Rule (plan, verbatim):** "**"Information confirmed"** only if the pooled 95% interval is entirely below 0 **and** the held-out point estimate is negative in at least 3 of the 4 seasons. Otherwise **"not confirmed"**."

**Outcome: not confirmed.**

**Also reported (descriptive).**

- Full-sample coefficients (n = 9659, game bootstrap 2,000, seed 20261023): logit(p_model) 0.132 (-0.057 to 0.339); logit(p_market) 0.886 (0.727 to 1.060); intercept -0.022.

Per-fold coefficients, log loss and Brier score:

| held-out season | n | b_market | b_model | LL market | Brier market | LL model raw | Brier model raw | LL blend | Brier blend |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2399 | 0.967 | 0.008 | 0.66970 | 0.23840 | 0.67394 | 0.24049 | 0.66964 | 0.23838 |
| 2023 | 2416 | 0.837 | 0.212 | 0.67560 | 0.24142 | 0.68509 | 0.24598 | 0.67607 | 0.24164 |
| 2024 | 2422 | 0.878 | 0.108 | 0.67359 | 0.24044 | 0.67980 | 0.24345 | 0.67359 | 0.24043 |
| 2025 | 2422 | 0.872 | 0.186 | 0.67621 | 0.24179 | 0.68278 | 0.24486 | 0.67685 | 0.24210 |

ROI at closing prices, larger cross-fitted edge side, edge ≥ 0.04 (descriptive only, no rule):

| set | games | bets | win rate (Wilson) | ROI (bootstrap) |
|---|---|---|---|---|
| pooled | 9659 | 8 | 0.250 (0.071–0.591) | -0.335 (-1.000 to 0.698) |
| 2022 | 2399 | 0 | – | nan (nan to nan) |
| 2023 | 2416 | 7 | 0.286 (0.082–0.641) | -0.240 (-1.000 to 0.773) |
| 2024 | 2422 | 0 | – | nan (nan to nan) |
| 2025 | 2422 | 1 | 0.000 (0.000–0.793) | -1.000 (-1.000 to 0.000) |

## Test 2 — Totals — confirmatory

- **Data:** analysis-set games with `odds_over_close`; n = 9685. Model value `model_total_mean`; outcome = box-score total runs.
- **Statistic:** OLS slope of (actual total − closing line) on (`model_total_mean` − closing line).
- **Result:** slope **0.075**, 95% interval **-0.023 to 0.171** (game bootstrap, 2,000, seed 20261024).

| set | n | intercept | slope | slope 95% interval | mean (model − line) | SD (model − line) | mean (actual − line) |
|---|---|---|---|---|---|---|---|
| pooled | 9685 | 0.439 | 0.075 | -0.023 to 0.171 | 0.265 | 0.963 | 0.459 |
| 2022 | 2421 | 0.279 | 0.113 | -0.080 to 0.297 | 0.690 | 0.937 | 0.357 |
| 2023 | 2419 | 0.518 | 0.154 | -0.066 to 0.355 | 0.032 | 0.969 | 0.523 |
| 2024 | 2423 | 0.439 | 0.212 | 0.006 to 0.410 | 0.259 | 0.876 | 0.494 |
| 2025 | 2422 | 0.469 | -0.081 | -0.276 to 0.122 | 0.078 | 0.926 | 0.462 |

**Rule (plan, verbatim):** "**"Confirmed"** only if the slope's 95% interval lies entirely above 0. Otherwise **"not confirmed"**. A negative slope is "not confirmed"."

**Outcome: not confirmed.**

## Test 3 — Beat the Streak — confirmatory

- **Pool:** 172679 starting-lineup batter-games in analysis-set games (doubleheader rule removed 1849), 730 dates. B1-eligible (≥200 PA): 142579.
- **Statistic:** top-1 success rate = hits ÷ (hits + losses), with Wilson intervals.

| method | set | dates | passes | decided | hits | success rate (Wilson) |
|---|---|---|---|---|---|---|
| simulator | pooled | 730 | 1 | 729 | 539 | 0.739 (0.706–0.770) |
| simulator | 2022 | 179 | 0 | 179 | 134 | 0.749 (0.680–0.806) |
| simulator | 2023 | 182 | 0 | 182 | 135 | 0.742 (0.674–0.800) |
| simulator | 2024 | 185 | 1 | 184 | 140 | 0.761 (0.694–0.817) |
| simulator | 2025 | 184 | 0 | 184 | 130 | 0.707 (0.637–0.768) |
| B1 | pooled | 730 | 0 | 730 | 532 | 0.729 (0.695–0.760) |
| B1 | 2022 | 179 | 0 | 179 | 127 | 0.709 (0.639–0.771) |
| B1 | 2023 | 182 | 0 | 182 | 144 | 0.791 (0.726–0.844) |
| B1 | 2024 | 185 | 0 | 185 | 139 | 0.751 (0.684–0.808) |
| B1 | 2025 | 184 | 0 | 184 | 122 | 0.663 (0.592–0.727) |
| B2 | pooled | 730 | 1 | 729 | 546 | 0.749 (0.716–0.779) |
| B2 | 2022 | 179 | 1 | 178 | 128 | 0.719 (0.649–0.780) |
| B2 | 2023 | 182 | 0 | 182 | 138 | 0.758 (0.691–0.815) |
| B2 | 2024 | 185 | 0 | 185 | 143 | 0.773 (0.707–0.827) |
| B2 | 2025 | 184 | 0 | 184 | 137 | 0.745 (0.677–0.802) |

- **Paired comparison** (dates where both picks were decided; date bootstrap, 2,000, seed 20261025):

| comparison | set | paired dates | mean difference | 95% interval |
|---|---|---|---|---|
| simulator − B2 | pooled | 728 | -0.010 | -0.047 to 0.032 |
| simulator − B2 | 2022 | 178 | 0.028 | -0.051 to 0.107 |
| simulator − B2 | 2023 | 182 | -0.016 | -0.093 to 0.066 |
| simulator − B2 | 2024 | 184 | -0.011 | -0.076 to 0.054 |
| simulator − B2 | 2025 | 184 | -0.038 | -0.120 to 0.049 |
| simulator − B1 | pooled | 729 | 0.011 | -0.032 to 0.052 |
| simulator − B1 | 2022 | 179 | 0.039 | -0.050 to 0.134 |
| simulator − B1 | 2023 | 182 | -0.049 | -0.126 to 0.027 |
| simulator − B1 | 2024 | 184 | 0.011 | -0.065 to 0.087 |
| simulator − B1 | 2025 | 184 | 0.043 | -0.049 to 0.141 |

**Primary result:** simulator − B2 = **-0.010**, 95% interval **-0.047 to 0.032**, n = 728 paired dates.

**Rule (plan, verbatim):** "The simulator **"adds value"** only if its paired difference against **B2** has a 95% interval entirely above 0. Otherwise **"does not add value"**."

**Outcome: does not add value.**

**Also reported.**

Calibration of `p_1h` (pool batters with atBats ≥ 1, n = 172451):

| bin | n | mean predicted | realized (Wilson) |
|---|---|---|---|
| <0.50 | 1966 | 0.472 | 0.461 (0.439–0.483) |
| 0.50–0.60 | 26908 | 0.566 | 0.535 (0.529–0.541) |
| 0.60–0.65 | 39295 | 0.626 | 0.581 (0.576–0.586) |
| 0.65–0.70 | 49057 | 0.675 | 0.623 (0.618–0.627) |
| 0.70–0.75 | 39481 | 0.722 | 0.664 (0.659–0.668) |
| 0.75–0.80 | 14327 | 0.768 | 0.689 (0.682–0.697) |
| 0.80–0.85 | 1407 | 0.813 | 0.729 (0.705–0.751) |
| ≥0.85 | 10 | 0.857 | 0.900 (0.596–0.982) (n<30, not interpreted) |

Top-1 picks, mean predicted vs realized (selection inflates the prediction, as the plan notes):

| method | decided picks | mean predicted P(1+ hit) | realized |
|---|---|---|---|
| simulator | 729 | 0.815 | 0.739 (0.706–0.770) |
| B2 | 729 | 0.772 | 0.749 (0.716–0.779) |

Top-2 both-hit rate:

| method | dates | excluded (a pick with 0 AB) | both-hit rate (Wilson) |
|---|---|---|---|
| simulator | 730 | 3 | 0.520 (0.484–0.556) |
| B1 | 730 | 2 | 0.523 (0.487–0.559) |
| B2 | 730 | 3 | 0.531 (0.495–0.567) |

Chance of a 57-game streak in 180 independent daily top-1 picks (rate and Wilson interval ends):

| method | top-1 rate | P(57) at rate | at lower end | at upper end |
|---|---|---|---|---|
| simulator | 0.739 | 1.11e-06 | 0.706 → 9.17e-08 | 0.770 → 9.87e-06 |
| B1 | 0.729 | 5.05e-07 | 0.695 → 3.91e-08 | 0.760 → 4.83e-06 |
| B2 | 0.749 | 2.23e-06 | 0.716 → 1.97e-07 | 0.779 → 1.86e-05 |

## Section 3b — Beat the Streak — secondary (no decision rule)

- **Log loss of P(1+ hit), simulator − B2**, all 174528 lineup batter-games (both doubleheader games): **0.00627**, 95% interval 0.00566 to 0.00688 (date bootstrap, 2,000, seed 20261025). Mean log loss: simulator 0.66798, B2 0.66171.
- **Top-5 pooled success rate**, 730 dates: simulator 0.717 (2612/3643), B2 0.725 (2641/3643); difference **-0.008**, 95% interval -0.026 to 0.009 (paired by date, date bootstrap, 2,000, seed 20261025).

## Test 4 — Starter strikeouts — confirmatory

- **Data:** analysis-set starts where the predicted starter started; n = 19372 starts in 9692 games. Baseline fell back to the league mean for 1474 starts (fewer than 3 starts in the window).
- **Statistic:** ΔMAE = MAE(model `model_*_k_med`) − MAE(baseline, trailing-365-day K per start), paired on the same starts.
- **Result:** ΔMAE **0.025**, 95% interval **0.011 to 0.038** (game bootstrap, 2,000, seed 20261026).

All statistics with 95% intervals (bias = predicted − actual; r_other = model K vs the other starter's actual Ks):

| set | starts | bias_model | bias_base | mae_model | mae_base | dmae | r_model | r_base | r_other |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 19372 | 0.444 (0.409 to 0.478) | 0.091 (0.055 to 0.124) | 1.893 (1.873 to 1.914) | 1.868 (1.849 to 1.889) | 0.025 (0.011 to 0.038) | 0.355 (0.343 to 0.367) | 0.374 (0.361 to 0.386) | -0.035 (-0.050 to -0.020) |
| 2022 | 4843 | 0.466 (0.402 to 0.529) | 0.035 (-0.032 to 0.097) | 1.888 (1.846 to 1.929) | 1.842 (1.804 to 1.880) | 0.046 (0.018 to 0.075) | 0.379 (0.354 to 0.402) | 0.409 (0.386 to 0.431) | -0.043 (-0.076 to -0.012) |
| 2023 | 4840 | 0.461 (0.391 to 0.531) | 0.134 (0.064 to 0.202) | 1.920 (1.876 to 1.961) | 1.904 (1.865 to 1.944) | 0.016 (-0.010 to 0.043) | 0.360 (0.335 to 0.386) | 0.376 (0.352 to 0.400) | -0.013 (-0.044 to 0.019) |
| 2024 | 4848 | 0.426 (0.361 to 0.492) | 0.113 (0.051 to 0.179) | 1.876 (1.836 to 1.918) | 1.877 (1.840 to 1.915) | -0.000 (-0.027 to 0.027) | 0.345 (0.320 to 0.370) | 0.356 (0.332 to 0.379) | -0.065 (-0.096 to -0.035) |
| 2025 | 4841 | 0.425 (0.358 to 0.491) | 0.080 (0.013 to 0.145) | 1.888 (1.848 to 1.927) | 1.851 (1.812 to 1.890) | 0.037 (0.011 to 0.062) | 0.333 (0.308 to 0.358) | 0.349 (0.324 to 0.374) | -0.022 (-0.051 to 0.008) |

**Rule (plan's table, cells verbatim):** **"Model better"** | 95% interval entirely below 0; **"Baseline better"** | interval entirely above 0; **"No difference shown"** | otherwise.

**Outcome: Baseline better.**

## Section 5 — Descriptive only (no decision rules)

Home-win AUC, P(home) = 1 − `model_away_win`; game bootstrap, 2,000, seed 20261027:

| games | set | n | home-win AUC |
|---|---|---|---|
| analysis set | pooled | 9696 | 0.587 (0.575 to 0.598) |
| analysis set | 2022 | 2428 | 0.611 (0.588 to 0.634) |
| analysis set | 2023 | 2422 | 0.576 (0.554 to 0.598) |
| analysis set | 2024 | 2424 | 0.590 (0.567 to 0.612) |
| analysis set | 2025 | 2422 | 0.567 (0.544 to 0.590) |
| all ledger rows | pooled | 9704 | 0.587 (0.575 to 0.598) |
| all ledger rows | 2022 | 2429 | 0.610 (0.587 to 0.633) |
| all ledger rows | 2023 | 2423 | 0.576 (0.552 to 0.598) |
| all ledger rows | 2024 | 2426 | 0.590 (0.567 to 0.612) |
| all ledger rows | 2025 | 2426 | 0.567 (0.544 to 0.591) |

HR props (starters with plateAppearances ≥ 1), `p_1hr` vs homeRuns ≥ 1; game-cluster bootstrap, 2,000, seed 20261027:

| set | batter-games | HRs | HR AUC |
|---|---|---|---|
| pooled | 174470 | 20182 | 0.608 (0.604 to 0.612) |
| 2022 | 43690 | 4763 | 0.609 (0.601 to 0.617) |
| 2023 | 43582 | 5332 | 0.606 (0.598 to 0.614) |
| 2024 | 43620 | 4978 | 0.604 (0.596 to 0.612) |
| 2025 | 43578 | 5109 | 0.614 (0.606 to 0.622) |

HR calibration, 10 equal-count bins (pooled):

| decile | n | predicted range | mean predicted | realized (Wilson) |
|---|---|---|---|---|
| 1 | 17447 | 0.000–0.068 | 0.047 | 0.054 (0.050–0.057) |
| 2 | 17447 | 0.068–0.090 | 0.079 | 0.076 (0.072–0.080) |
| 3 | 17447 | 0.090–0.106 | 0.098 | 0.089 (0.084–0.093) |
| 4 | 17447 | 0.106–0.121 | 0.114 | 0.097 (0.093–0.102) |
| 5 | 17447 | 0.121–0.136 | 0.129 | 0.105 (0.101–0.110) |
| 6 | 17447 | 0.136–0.152 | 0.144 | 0.115 (0.111–0.120) |
| 7 | 17447 | 0.152–0.170 | 0.161 | 0.132 (0.127–0.137) |
| 8 | 17447 | 0.170–0.191 | 0.180 | 0.145 (0.140–0.151) |
| 9 | 17447 | 0.191–0.219 | 0.204 | 0.154 (0.148–0.159) |
| 10 | 17447 | 0.219–0.415 | 0.252 | 0.190 (0.184–0.196) |

## Implementation notes

- Seeds: 20261022 + test number (1–5). Section 3b sits under test 3 and uses seed 20261022 + 3; the plan gives no separate number.
- Per-season intervals use the same seed and settings as the pooled interval.
- Test 3: the 57-game streak inputs use each method's pooled top-1 rate and its Wilson interval ends.
- Test 4's rule wording is quoted from the plan's table.

## Pre-registered expectations (a)–(e), R2 vs R1

No decision rule; they do not change any test outcome (plan).

| expectation | rule | R2 | R1 | result |
|---|---|---|---|---|
| (a) totals slope | −0.201 < R2 slope < 0 | 0.075 | -0.201 | **not held** (not met) |
| (b) model coefficient | R2 logit(p_model) coefficient < 0 | 0.132 (-0.057 to 0.339) | -0.391 | **not held** (not met) |
| (c) passes | simulator passes ≤ max(B1, B2 passes) + 3 | sim 1, B1 0, B2 1 | sim 26, B1 1, B2 1 | **held** (met) |
| (c) substitute picks | simulator top-1 substitute picks = 0 | 0 | 53 | **held** (met) |
| (d) top-1 rate, simulator | abs(R2 − R1) ≤ 0.02 | 0.739 (539/729) | 0.727 | **held** (met) |
| (d) top-1 rate, B1 | abs(R2 − R1) ≤ 0.02 | 0.729 (532/730) | 0.726 | **held** (met) |
| (d) top-1 rate, B2 | abs(R2 − R1) ≤ 0.02 | 0.749 (546/729) | 0.749 | **held** (met) |
| (e) primary: HR AUC simulator − baseline | abs(difference) ≤ 0.01 | -0.0015 (-0.0042 to 0.0010) | 0.0025 | **held** (met) |
| (e) second check: test 5 pooled HR AUC | abs(R2 − 0.613) ≤ 0.01 | 0.6079 (Δ -0.0051) | 0.613 | **held** (met) |

(e) primary: 174470 true-starting-nine batter-games with plateAppearances ≥ 1 (20182 with a HR); AUC simulator p_1hr 0.6079 (0.6037 to 0.6119); HR baseline 0.6094 (0.6053 to 0.6134). Baseline and bootstrap as in `v4_exploratory.py` `q5` (paired game bootstrap, 2,000, seed 20261023, the seed q5 used for R1; the amendment names no seed).

(c) substitute picks are counted on the simulator's top-1 picks by box `battingOrder` ≠ slot × 100.

## EXPLORATORY

None. Nothing outside the plan was computed.
