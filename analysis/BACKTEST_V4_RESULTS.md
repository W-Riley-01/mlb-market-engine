# Full v4 backtest — results

Plan: `analysis/BACKTEST_V4_ANALYSIS_PLAN.md`, SHA-256 `4a57fcaab100b14b714dd4bdb04b96f16316a527d8758845080b2d909b27152a` (matches the last `sha256:` line of `analysis/out/PLAN_HASH.txt`). Run by `analysis/v4_analysis.py` after `analysis/v4_integrity.py` (log: `analysis/out/v4_integrity_log.txt`). Outcomes from cached box scores; baselines from `data/pitch_matrix.parquet` with game_date < D.

## Integrity gate and analysis set

- Ledger rows 9704 (2022 2429, 2023 2423, 2024 2426, 2025 2426); no duplicate gamePks; every eligible scheduled game present; every gamePk has a cached box score; all batter and starter ids joined (0 unjoined). Integrity: PASS.
- Excluded for `actual_fallback`: 8 games → analysis set 9696 games.
- Orientation check: home-win AUC on the analysis set = 0.5651 (n = 9696) ≥ 0.5 → continue; nothing swapped.

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
- **Result:** **-0.00053**, 95% interval **-0.00143 to 0.00038** (game bootstrap, 2,000 resamples, seed 20261023).

| held-out season | n | dLL (blend − market) | 95% interval |
|---|---|---|---|
| 2022 | 2399 | -0.00009 | -0.00235 to 0.00211 |
| 2023 | 2416 | -0.00182 | -0.00304 to -0.00055 |
| 2024 | 2422 | 0.00057 | -0.00121 to 0.00243 |
| 2025 | 2422 | -0.00077 | -0.00234 to 0.00078 |

Seasons with a negative point estimate: 3 of 4.

**Rule (plan, verbatim):** "**"Information confirmed"** only if the pooled 95% interval is entirely below 0 **and** the held-out point estimate is negative in at least 3 of the 4 seasons. Otherwise **"not confirmed"**."

**Outcome: not confirmed.**

**Also reported (descriptive).**

- Full-sample coefficients (n = 9659, game bootstrap 2,000, seed 20261023): logit(p_model) -0.391 (-0.587 to -0.203); logit(p_market) 1.224 (1.073 to 1.389); intercept 0.042.

Per-fold coefficients, log loss and Brier score:

| held-out season | n | b_market | b_model | LL market | Brier market | LL model raw | Brier model raw | LL blend | Brier blend |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2399 | 1.268 | -0.456 | 0.66970 | 0.23840 | 0.68349 | 0.24511 | 0.66961 | 0.23831 |
| 2023 | 2416 | 1.161 | -0.293 | 0.67560 | 0.24142 | 0.69240 | 0.24950 | 0.67378 | 0.24053 |
| 2024 | 2422 | 1.237 | -0.459 | 0.67359 | 0.24044 | 0.68403 | 0.24549 | 0.67416 | 0.24070 |
| 2025 | 2422 | 1.236 | -0.362 | 0.67621 | 0.24179 | 0.68838 | 0.24758 | 0.67545 | 0.24140 |

ROI at closing prices, larger cross-fitted edge side, edge ≥ 0.04 (descriptive only, no rule):

| set | games | bets | win rate (Wilson) | ROI (bootstrap) |
|---|---|---|---|---|
| pooled | 9659 | 590 | 0.529 (0.488–0.569) | 0.074 (-0.015 to 0.162) |
| 2022 | 2399 | 267 | 0.539 (0.479–0.598) | 0.105 (-0.021 to 0.235) |
| 2023 | 2416 | 28 | 0.500 (0.326–0.674) | -0.107 (-0.435 to 0.228) |
| 2024 | 2422 | 193 | 0.492 (0.423–0.562) | 0.030 (-0.119 to 0.180) |
| 2025 | 2422 | 102 | 0.578 (0.481–0.670) | 0.127 (-0.073 to 0.318) |

## Test 2 — Totals — confirmatory

- **Data:** analysis-set games with `odds_over_close`; n = 9685. Model value `model_total_mean`; outcome = box-score total runs.
- **Statistic:** OLS slope of (actual total − closing line) on (`model_total_mean` − closing line).
- **Result:** slope **-0.201**, 95% interval **-0.301 to -0.105** (game bootstrap, 2,000, seed 20261024).

| set | n | intercept | slope | slope 95% interval | mean (model − line) | SD (model − line) | mean (actual − line) |
|---|---|---|---|---|---|---|---|
| pooled | 9685 | 0.485 | -0.201 | -0.301 to -0.105 | 0.129 | 0.986 | 0.459 |
| 2022 | 2421 | 0.470 | -0.196 | -0.400 to -0.007 | 0.573 | 0.946 | 0.357 |
| 2023 | 2419 | 0.507 | -0.147 | -0.366 to 0.059 | -0.109 | 0.994 | 0.523 |
| 2024 | 2423 | 0.503 | -0.081 | -0.287 to 0.124 | 0.114 | 0.901 | 0.494 |
| 2025 | 2422 | 0.439 | -0.371 | -0.572 to -0.173 | -0.063 | 0.949 | 0.462 |

**Rule (plan, verbatim):** "**"Confirmed"** only if the slope's 95% interval lies entirely above 0. Otherwise **"not confirmed"**. A negative slope is "not confirmed"."

**Outcome: not confirmed.**

## Test 3 — Beat the Streak — confirmatory

- **Pool:** 172502 starting-lineup batter-games in analysis-set games (doubleheader rule removed 2026), 730 dates. B1-eligible (≥200 PA): 139487.
- **Statistic:** top-1 success rate = hits ÷ (hits + losses), with Wilson intervals.

| method | set | dates | passes | decided | hits | success rate (Wilson) |
|---|---|---|---|---|---|---|
| simulator | pooled | 730 | 26 | 704 | 512 | 0.727 (0.693–0.759) |
| simulator | 2022 | 179 | 7 | 172 | 117 | 0.680 (0.607–0.745) |
| simulator | 2023 | 182 | 4 | 178 | 130 | 0.730 (0.661–0.790) |
| simulator | 2024 | 185 | 12 | 173 | 129 | 0.746 (0.676–0.805) |
| simulator | 2025 | 184 | 3 | 181 | 136 | 0.751 (0.684–0.809) |
| B1 | pooled | 730 | 1 | 729 | 529 | 0.726 (0.692–0.757) |
| B1 | 2022 | 179 | 1 | 178 | 126 | 0.708 (0.637–0.770) |
| B1 | 2023 | 182 | 0 | 182 | 141 | 0.775 (0.709–0.829) |
| B1 | 2024 | 185 | 0 | 185 | 140 | 0.757 (0.690–0.813) |
| B1 | 2025 | 184 | 0 | 184 | 122 | 0.663 (0.592–0.727) |
| B2 | pooled | 730 | 1 | 729 | 546 | 0.749 (0.716–0.779) |
| B2 | 2022 | 179 | 1 | 178 | 127 | 0.713 (0.643–0.775) |
| B2 | 2023 | 182 | 0 | 182 | 136 | 0.747 (0.679–0.805) |
| B2 | 2024 | 185 | 0 | 185 | 144 | 0.778 (0.713–0.832) |
| B2 | 2025 | 184 | 0 | 184 | 139 | 0.755 (0.689–0.812) |

- **Paired comparison** (dates where both picks were decided; date bootstrap, 2,000, seed 20261025):

| comparison | set | paired dates | mean difference | 95% interval |
|---|---|---|---|---|
| simulator − B2 | pooled | 703 | -0.020 | -0.063 to 0.021 |
| simulator − B2 | 2022 | 171 | -0.029 | -0.111 to 0.058 |
| simulator − B2 | 2023 | 178 | -0.011 | -0.084 to 0.073 |
| simulator − B2 | 2024 | 173 | -0.035 | -0.110 to 0.035 |
| simulator − B2 | 2025 | 181 | -0.006 | -0.088 to 0.077 |
| simulator − B1 | pooled | 703 | 0.001 | -0.041 to 0.046 |
| simulator − B1 | 2022 | 171 | -0.018 | -0.105 to 0.076 |
| simulator − B1 | 2023 | 178 | -0.045 | -0.129 to 0.034 |
| simulator − B1 | 2024 | 173 | -0.023 | -0.098 to 0.052 |
| simulator − B1 | 2025 | 181 | 0.088 | 0.000 to 0.177 |

**Primary result:** simulator − B2 = **-0.020**, 95% interval **-0.063 to 0.021**, n = 703 paired dates.

**Rule (plan, verbatim):** "The simulator **"adds value"** only if its paired difference against **B2** has a 95% interval entirely above 0. Otherwise **"does not add value"**."

**Outcome: does not add value.**

**Also reported.**

Calibration of `p_1h` (pool batters with atBats ≥ 1, n = 164232):

| bin | n | mean predicted | realized (Wilson) |
|---|---|---|---|
| <0.50 | 1711 | 0.472 | 0.470 (0.447–0.494) |
| 0.50–0.60 | 26076 | 0.567 | 0.518 (0.512–0.524) |
| 0.60–0.65 | 38364 | 0.626 | 0.557 (0.552–0.562) |
| 0.65–0.70 | 47136 | 0.675 | 0.605 (0.601–0.610) |
| 0.70–0.75 | 36671 | 0.722 | 0.653 (0.648–0.658) |
| 0.75–0.80 | 12943 | 0.768 | 0.687 (0.679–0.695) |
| 0.80–0.85 | 1324 | 0.812 | 0.721 (0.697–0.745) |
| ≥0.85 | 7 | 0.857 | 0.857 (0.487–0.974) (n<30, not interpreted) |

Top-1 picks, mean predicted vs realized (selection inflates the prediction, as the plan notes):

| method | decided picks | mean predicted P(1+ hit) | realized |
|---|---|---|---|
| simulator | 704 | 0.814 | 0.727 (0.693–0.759) |
| B2 | 729 | 0.770 | 0.749 (0.716–0.779) |

Top-2 both-hit rate:

| method | dates | excluded (a pick with 0 AB) | both-hit rate (Wilson) |
|---|---|---|---|
| simulator | 730 | 59 | 0.511 (0.473–0.549) |
| B1 | 730 | 3 | 0.513 (0.477–0.549) |
| B2 | 730 | 4 | 0.528 (0.491–0.564) |

Chance of a 57-game streak in 180 independent daily top-1 picks (rate and Wilson interval ends):

| method | top-1 rate | P(57) at rate | at lower end | at upper end |
|---|---|---|---|---|
| simulator | 0.727 | 4.52e-07 | 0.693 → 3.29e-08 | 0.759 → 4.53e-06 |
| B1 | 0.726 | 4.00e-07 | 0.692 → 3.02e-08 | 0.757 → 3.91e-06 |
| B2 | 0.749 | 2.23e-06 | 0.716 → 1.97e-07 | 0.779 → 1.86e-05 |

## Section 3b — Beat the Streak — secondary (no decision rule)

- **Log loss of P(1+ hit), simulator − B2**, all 174528 lineup batter-games (both doubleheader games): **0.01138**, 95% interval 0.01077 to 0.01203 (date bootstrap, 2,000, seed 20261025). Mean log loss: simulator 0.69485, B2 0.68347.
- **Top-5 pooled success rate**, 730 dates: simulator 0.720 (2515/3494), B2 0.718 (2613/3637); difference **0.001**, 95% interval -0.015 to 0.018 (paired by date, date bootstrap, 2,000, seed 20261025).

## Test 4 — Starter strikeouts — confirmatory

- **Data:** analysis-set starts where the predicted starter started; n = 19372 starts in 9692 games. Baseline fell back to the league mean for 1474 starts (fewer than 3 starts in the window).
- **Statistic:** ΔMAE = MAE(model `model_*_k_med`) − MAE(baseline, trailing-365-day K per start), paired on the same starts.
- **Result:** ΔMAE **0.043**, 95% interval **0.029 to 0.057** (game bootstrap, 2,000, seed 20261026).

All statistics with 95% intervals (bias = predicted − actual; r_other = model K vs the other starter's actual Ks):

| set | starts | bias_model | bias_base | mae_model | mae_base | dmae | r_model | r_base | r_other |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 19372 | 0.464 (0.428 to 0.498) | 0.091 (0.055 to 0.124) | 1.911 (1.892 to 1.932) | 1.868 (1.849 to 1.889) | 0.043 (0.029 to 0.057) | 0.339 (0.327 to 0.352) | 0.374 (0.361 to 0.386) | -0.040 (-0.055 to -0.025) |
| 2022 | 4843 | 0.490 (0.425 to 0.555) | 0.035 (-0.032 to 0.097) | 1.905 (1.862 to 1.947) | 1.842 (1.804 to 1.880) | 0.063 (0.033 to 0.093) | 0.366 (0.341 to 0.391) | 0.409 (0.386 to 0.431) | -0.049 (-0.078 to -0.019) |
| 2023 | 4840 | 0.480 (0.408 to 0.549) | 0.134 (0.064 to 0.202) | 1.943 (1.897 to 1.985) | 1.904 (1.865 to 1.944) | 0.039 (0.011 to 0.066) | 0.333 (0.308 to 0.359) | 0.376 (0.352 to 0.400) | -0.014 (-0.045 to 0.017) |
| 2024 | 4848 | 0.451 (0.387 to 0.517) | 0.113 (0.051 to 0.179) | 1.898 (1.857 to 1.939) | 1.877 (1.840 to 1.915) | 0.021 (-0.005 to 0.049) | 0.329 (0.305 to 0.355) | 0.356 (0.332 to 0.379) | -0.067 (-0.098 to -0.036) |
| 2025 | 4841 | 0.434 (0.365 to 0.499) | 0.080 (0.013 to 0.145) | 1.898 (1.858 to 1.937) | 1.851 (1.812 to 1.890) | 0.047 (0.021 to 0.073) | 0.325 (0.300 to 0.350) | 0.349 (0.324 to 0.374) | -0.035 (-0.064 to -0.004) |

**Rule (plan's table, cells verbatim):** **"Model better"** | 95% interval entirely below 0; **"Baseline better"** | interval entirely above 0; **"No difference shown"** | otherwise.

**Outcome: Baseline better.**

## Section 5 — Descriptive only (no decision rules)

Home-win AUC, P(home) = 1 − `model_away_win`; game bootstrap, 2,000, seed 20261027:

| games | set | n | home-win AUC |
|---|---|---|---|
| analysis set | pooled | 9696 | 0.565 (0.554 to 0.577) |
| analysis set | 2022 | 2428 | 0.582 (0.559 to 0.606) |
| analysis set | 2023 | 2422 | 0.552 (0.528 to 0.575) |
| analysis set | 2024 | 2424 | 0.575 (0.553 to 0.597) |
| analysis set | 2025 | 2422 | 0.549 (0.527 to 0.572) |
| all ledger rows | pooled | 9704 | 0.565 (0.554 to 0.576) |
| all ledger rows | 2022 | 2429 | 0.582 (0.558 to 0.605) |
| all ledger rows | 2023 | 2423 | 0.551 (0.529 to 0.574) |
| all ledger rows | 2024 | 2426 | 0.575 (0.552 to 0.598) |
| all ledger rows | 2025 | 2426 | 0.549 (0.526 to 0.573) |

HR props (starters with plateAppearances ≥ 1), `p_1hr` vs homeRuns ≥ 1; game-cluster bootstrap, 2,000, seed 20261027:

| set | batter-games | HRs | HR AUC |
|---|---|---|---|
| pooled | 167578 | 18741 | 0.613 (0.609 to 0.617) |
| 2022 | 41986 | 4441 | 0.611 (0.603 to 0.620) |
| 2023 | 41862 | 4963 | 0.611 (0.603 to 0.620) |
| 2024 | 41908 | 4619 | 0.609 (0.600 to 0.617) |
| 2025 | 41822 | 4718 | 0.622 (0.614 to 0.631) |

HR calibration, 10 equal-count bins (pooled):

| decile | n | predicted range | mean predicted | realized (Wilson) |
|---|---|---|---|---|
| 1 | 16758 | 0.000–0.066 | 0.046 | 0.050 (0.047–0.054) |
| 2 | 16758 | 0.066–0.088 | 0.078 | 0.074 (0.070–0.078) |
| 3 | 16758 | 0.088–0.104 | 0.096 | 0.082 (0.078–0.086) |
| 4 | 16757 | 0.104–0.119 | 0.112 | 0.092 (0.088–0.096) |
| 5 | 16758 | 0.119–0.134 | 0.126 | 0.098 (0.094–0.103) |
| 6 | 16758 | 0.134–0.150 | 0.142 | 0.115 (0.110–0.120) |
| 7 | 16757 | 0.150–0.167 | 0.158 | 0.127 (0.122–0.132) |
| 8 | 16758 | 0.167–0.188 | 0.177 | 0.137 (0.132–0.142) |
| 9 | 16758 | 0.188–0.216 | 0.201 | 0.158 (0.152–0.163) |
| 10 | 16758 | 0.216–0.423 | 0.250 | 0.187 (0.181–0.193) |

## Implementation notes

- Seeds: 20261022 + test number (1–5). Section 3b sits under test 3 and uses seed 20261022 + 3; the plan gives no separate number.
- Per-season intervals use the same seed and settings as the pooled interval.
- Test 3: the 57-game streak inputs use each method's pooled top-1 rate and its Wilson interval ends.
- Test 4's rule is the plan's three-row table, quoted cell by cell.
- Spot-check: one B2 value (2023-07-14) and one K-per-start baseline were recomputed independently from raw pitch_matrix rows and matched to 5 and 4 decimals.

## EXPLORATORY

Nothing outside the plan was computed when the confirmatory results above were produced.

### EXPLORATORY — 2026-10-23 (post hoc, after the results above were seen; not part of the frozen plan)

**Sources.**
- Script: `analysis/v4_exploratory.py`; log: `analysis/out/v4_exploratory_log.txt`. Bootstrap seed 20261023.
- The substitute counts in E0 come from one-off checks of the box-score `battingOrder` codes that are not saved
  in that script.
- The analysis set and helpers are the same as in `v4_analysis.py`.

#### E0. The ledger lineups are not starting lineups (found while answering Q3)

`analysis/backtest_v4.py` builds each lineup from the box score's `battingOrder` list. That list holds the **last**
player in each lineup slot at the end of the game, not the starter.

**Size of the problem**
- Lineup entries: 24,251 of 174,672 (13.9%) are substitutes. A substitute's box `battingOrder` code does not end in
  `00`; every entry's slot digit matches the ledger slot.
  - By season: 2022 5,757; 2023 6,170; 2024 6,143; 2025 6,181.
- Team-games: only 5,221 of 19,408 (26.9%) have all nine starters.
- Plate appearances of those substitutes in the game: 0 PA 6,956; 1 PA 11,752; 2 PA 4,512; 3 PA 846; 4 PA 164; 5 PA 21.
- Every lineup entry with 0 PA is a substitute.

**What this means for the plan**
- The simulator was given lineups that include post-game information: substitutes who were not known before first
  pitch.
- The Section 3 pool contains 23,928 substitutes among its 172,502 batter-games.
- The plan's limitation 4 and Section 3 assumed the `batters` field holds the starting lineup. That assumption is
  false for these rows.
- Tests 1–5 were run as frozen, on these lineups. Nothing has been re-run.

#### Q1. Test 1 (moneyline), same 9,659 games

**Fit on these games**

| | AUC | log loss |
|---|---|---|
| market (de-vigged closing odds) | 0.6050 | 0.67378 |
| model, raw (1 − `model_away_win`, uncalibrated) | 0.5649 | 0.68708 |

- **Regression behind −0.391** (full sample, y ~ 1 + logit(p_mkt) + logit(p_model)): intercept 0.0422,
  **b_market 1.2245, b_model −0.3914**.
- For reference, the model alone (y ~ 1 + logit(p_model)) has slope 0.692.
- SD of logit(p_model) is 0.333 and of logit(p_mkt) 0.397; the correlation between the two logits is 0.753.

**Was the model logit recalibrated before blending?** No. Raw logit(p_model) enters the blend regression directly,
next to logit(p_mkt). The cross-fitted regression coefficients are the only calibration applied.

**ROI bets:** 582 of the 590 bets **fade** the model: the bet side is opposite to the side where p_model exceeds the
market. 8 follow it.

#### Q2. Test 2 (totals), n = 9,685

- **mean(`model_total_mean` − line):** 0.129 (SD 0.986).
- **Correlation of (model − line) with the line:** −0.772.
- **(actual − line) on the line alone:** slope −0.102 (−0.193 to −0.012, game bootstrap 2,000), intercept 1.320.
- **Spread:** mean line 8.42, SD 1.012; mean `model_total_mean` 8.55, SD 0.676; r(`model_total_mean`, line) 0.373.

**Sign convention:** −0.201 is the slope of y = (actual total − line) on x = (`model_total_mean` − line). Positive x
means the model is above the line; positive y means the game went over the line.

#### Q3. Test 3 (Beat the Streak), 730 dates

| method | dates | decided picks | passes (no official AB) | top-1 success rate, Wilson 95% |
|---|---|---|---|---|
| simulator | 730 | 704 | 26 | 0.727 (0.693–0.759) |
| B1 | 730 | 729 | 1 | 0.726 (0.692–0.757) |
| B2 | 730 | 729 | 1 | 0.749 (0.716–0.779) |

**Why 704 vs 729: all 26 simulator passes are lineup entries that are substitutes (E0).**
- **What happened in those 26 games:** 24 had 0 PA; 1 had one PA that was a walk; 1 had one PA that was a hit by
  pitch.
- **Simulator picks:** 53 of its 730 top-1 picks are substitutes. Their median trailing-365-day PA is 4.
- **Baseline picks:** B1 picked 12 substitutes and B2 picked 5. Their median trailing PA is 647.5 and 611, and each
  method had 1 pass among them.
- **Spread of picks:** the simulator's top-1 picks span 138 distinct players, against 50 for B1 and 87 for B2.

#### Q4. Test 4 (starter strikeouts), 19,372 starts

**Workload per start**

| | batters faced | innings | Ks | K per batter faced |
|---|---|---|---|---|
| actual (box score) | 21.95 (SD 4.67) | 5.19 | 4.801 | 0.2187 |
| model | 22.87, implied | not stored | 5.265 (median), 5.368 (distribution mean) | 0.2347, implied |
| baseline | — | — | 4.891 | — |

- **Predicted batters faced and innings are not stored** in the ledger, and `run_advanced_monte_carlo` does not
  return them.
- **The implied 22.87 is exact, not simulated:** the simulator pulls every starter once his pitch count exceeds 85
  (`advanced_simulator.py:260–266`). It draws pitches per PA from a fixed distribution with mean 3.84, so the
  expected batters faced is the same for every starter.
- **Predicted innings** depend on outs per PA, so they cannot be derived without re-simulating.
- **Correlation of the model's predicted Ks with the baseline's:** r = 0.606 using the median; 0.628 using the
  distribution mean.

#### Q5. Home-run AUC vs a trailing-365-day HR baseline

**Baseline:** built exactly like B2, with home runs in place of hits:
- batter HR/PA shrunk with 200 PA;
- predicted opposing starter's HR allowed per batter faced, shrunk with 400 batters faced;
- league HR/PA;
- log5, with the starter's share of expected PA by slot and side.

**Population:** 167,578 batter-games with PA ≥ 1, of which 18,741 had a home run. Intervals are a paired game
bootstrap, 2,000 resamples.

| | AUC (95% interval) | mean predicted | log loss |
|---|---|---|---|
| simulator `p_1hr` | 0.6128 (0.6083–0.6171) | 0.1386 | 0.34621 |
| HR baseline | 0.6103 (0.6060–0.6144) | 0.1199 | 0.34290 |

- AUC difference, simulator − baseline: 0.0025 (−0.0004 to 0.0053).
- Realized HR rate: 0.1118.

#### E1. EXPLORATORY (2026-10-23): substitute count vs tests 1 and 2

**Sources.**
- Script: `analysis/v4_lineup_check.py`; log: `analysis/out/v4_lineup_check_log.txt`. Seed 20261023, 2,000 game
  resamples.
- Subs = lineup entries per game, both sides, that are not box starters (code ≠ slot × 100). Analysis set: mean
  2.50 per game.
- The defect is written up in `BACKTEST_V4_ADDENDUM_2026-10-23.md`.

**(a) Totals, n = 9,685**
- r(subs, `model_total_mean` − line) = −0.110.
- r(subs, actual − line) = +0.197.

**(b) Moneyline, n = 9,659**

| | r with (model logit − market logit) | r with (home win − p_market) |
|---|---|---|
| subs | 0.012 | −0.034 |
| home-minus-away subs | −0.090 | −0.092 |

**(c) Test 2 with subs as a covariate:** (actual − line) ~ 1 + (model − line) + subs

| term | coefficient | 95% interval |
|---|---|---|
| intercept | −0.885 | −1.044 to −0.729 |
| slope on (model − line) | −0.106 | −0.202 to −0.005 |
| subs | 0.543 | 0.485 to 0.601 |

**(c) Test 1 with subs as a covariate:** y ~ 1 + logit(p_mkt) + logit(p_model) + subs

| term | coefficient | 95% interval |
|---|---|---|
| intercept | 0.1547 | 0.0742 to 0.2397 |
| b_market | 1.2216 | 1.0691 to 1.3863 |
| b_model | −0.3937 | −0.5906 to −0.2038 |
| subs | −0.0446 | −0.0700 to −0.0199 |

**(c) Test 1 with home-minus-away subs instead**

| term | coefficient | 95% interval |
|---|---|---|
| intercept | 0.0548 | 0.0038 to 0.1052 |
| b_market | 1.2980 | 1.1413 to 1.4677 |
| b_model | −0.5056 | −0.7097 to −0.3138 |
| home-minus-away subs | −0.1508 | −0.1817 to −0.1209 |

The substitute count is itself post-game information, so these regressions are diagnostics only.

#### Q6. Credentials and untracked files

- **Credentials:** none. `v4_integrity.py`, `v4_analysis.py` and `v4_exploratory.py` contain no password, token,
  connection string or `PGPASSWORD` reference (grep for password, pgpass, secret, token, api_key and
  postgresql:// found nothing). They do not connect to the database; they read parquet, CSV and cached JSON only.
- **Untracked files under `analysis/`:** `git status --porcelain -uall analysis/` lists 10,938 files, all untracked
  (`??`). Of these, 10,890 are cached API JSON: `analysis/cache/api/box/` 9,704, `analysis/cache/api/schedule/`
  1,161, `analysis/cache/box/` 25. These are **not** git-ignored; only `*.csv` and `.venv` are. The other 48 are
  listed in the report to the user.
