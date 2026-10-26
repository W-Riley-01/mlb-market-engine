# MLB Market Engine: prediction vs outcome review (2026 season)

Prepared 2026-10-22, revised the same day after a verification pass (section 0A) and corrections (marked
**[rev]**). Data: local PostgreSQL restore (`mlb_engine`), read-only. Scripts and CSVs are in `analysis/`;
figures are in `analysis/out/figures/`. The slice family was frozen in `SLICES.md` before any slice was
computed; everything outside it is marked **post-hoc**.

> **There are no 2026 odds or market lines anywhere in the database or on disk.** The only odds are the 2023
> backtest ledger and the 2009+ historical file. Nothing in this report speaks to profitability or to edge
> against the market.

---

## 0. Home/away wiring is crossed inside the simulator (pipeline bug, all versions, high confidence)

`advanced_simulator.run_advanced_monte_carlo` builds `away_starter_card` as *away batters vs the home starter*.
`AdvancedSimulatedGame.__init__` takes that argument as the **away pitcher's** card and uses it in the bottom
half, indexed by `home_batter_index` (`advanced_simulator.py:37–38, 281, 290, 386–392`). The bullpen and
1st-inning cards are wired the same way. Each team's runs are therefore produced by the **other** team's
hitters facing its own starter. Git history shows the code was present on 2026-04-28 (`ac85dc5`), so v1, v2
and v3 are all affected.

### 0A. Verification [rev]

**Tracer test** (`tracer_wiring.py`). This calls `run_advanced_monte_carlo` with a stub resolver: no database,
no network, no writes. The only import needed is `advanced_simulator`, which imports only numpy.

| case | away_win_prob | away K dist mean | home K dist mean | P(1+ hit), players named away / home |
|---|---|---|---|---|
| (i) elite **home** offense vs terrible **away** starter | **1.000** | 5.00 | 2.31 | **0.999** / 0.639 |
| (ii) mirror: elite **away** offense vs terrible **home** starter | **0.000** | 2.33 | 5.03 | 0.641 / **0.997** |
| (iii) high-K **away** starter vs low-K **home** starter | 0.447 | 1.80 | **9.18** | 0.642 / 0.629 |
| control, all average | 0.453 | 4.95 | 5.05 | 0.642 / 0.631 |

In every case the opposite output side responds: the home offense's quality shows up as away runs and on the
away batters' names, and the away starter's K rate shows up in `home_k_dist`. The wiring error is in the
simulator. In the control case the team batting second wins 0.547; that is the last-at-bat edge.

**API spot check** (`api_spotcheck.py`, `out/api_spotcheck.csv`). 25 random analysis games (13 v2, 12 v3)
compared with MLB API box scores:

| field | comparisons | matches |
|---|---|---|
| team names by side | 50 | 50 |
| final score by side | 50 | 50 |
| starter IDs | 50 | 50 |
| starter Ks | 50 | 50 |
| batter hits (3 per team) | 150 | **149** |

The one mismatch is Wilyer Abreu (gp 1070): DB 2 hits, API 1. The DB row was recorded the next morning, so
this is probably a later official scoring change. It does not change the 1+ hit outcome. DB team labels match
the schedule for all 1,460 analysis games. **The outcomes are labeled correctly.**

**How the "actual season win %" check was computed.** In-sample, over the 1,460 analysis games. Each team's win
% comes from its games in that set, with labels taken from `game_predictions.away_team/home_team` and results
from `game_outcomes.away_won`, not from the schedule. Those labels have since been verified against the
schedule (1,460/1,460) and the outcomes against the API (above). The check is in-sample: each game contributes
to its own teams' win %, but each team has about 97 games, so the effect is small.

The correlations were corr(home_win_prob, home − away win %) = −0.27 (v2, n=802) and −0.31 (v3, n=658).

### 0B. Swapped-back re-scoring and recalibration (post-hoc; `posthoc_recal.py`, `out/posthoc_recal.csv`) [rev]

Swapping the stored sides back gives the sim's estimate for the right team. It is approximate: the swapped
offense bats first, so the real home team loses the last-at-bat edge (about +0.05 in the control above) and
any home-field term. That is the likely reason swapped home-win is under-predicted.

Recalibration is cross-fitted (5 folds grouped by game; never scored on fitted games). The skill columns are
Brier skill vs the in-cell base rate.

| target | version | n | mean pred / rate | bias | cal. intercept | cal. slope | AUC | skill raw | skill, intercept refit | skill, intercept + slope refit |
|---|---|---|---|---|---|---|---|---|---|---|
| home win | v2 | 802 | 0.445 / 0.512 | −0.067 (−0.102 to −0.033) | 0.28 (0.14–0.42) | 0.37 (−0.10 to 0.87) | 0.524 (0.485–0.567) | −0.024 (−0.055 to 0.001) | −0.008 (−0.030 to 0.010) | −0.003 (−0.014 to 0.004) |
| home win | v3 | 658 | 0.450 / 0.538 | −0.088 (−0.123 to −0.049) | 0.36 (0.20–0.51) | 0.95 (0.43–1.45) | **0.579 (0.535–0.617)** | −0.012 (−0.051 to 0.019) | +0.016 (−0.007 to 0.037) | +0.013 (−0.010 to 0.032) |
| K ≥ 5 away | v2 | 801 | 0.578 / 0.482 | +0.096 (0.063–0.130) | −0.43 | 0.72 (0.51–0.95) | 0.631 (0.590–0.668) | +0.013 (−0.035 to 0.057) | +0.045 (0.004–0.084) | **+0.051 (0.018–0.080)** |
| K ≥ 5 away | v3 | 658 | 0.610 / 0.436 | +0.174 (0.135–0.213) | −0.78 | 0.60 (0.36–0.88) | 0.621 (0.577–0.667) | −0.090 (−0.171 to −0.023) | +0.029 (−0.019 to 0.077) | +0.030 (−0.003 to 0.061) |
| K ≥ 5 home | v3 | 656 | 0.612 / 0.512 | +0.100 (0.063–0.135) | −0.45 | 0.71 (0.48–0.98) | 0.634 (0.592–0.677) | +0.011 (−0.043 to 0.059) | +0.048 (0.004–0.092) | **+0.048 (0.013–0.082)** |
| K ≥ 7 home | v3 | 656 | 0.277 / 0.245 | +0.031 (−0.001 to 0.063) | −0.18 | 1.01 (0.74–1.31) | 0.700 (0.652–0.744) | +0.085 (0.032–0.131) | +0.085 (0.037–0.130) | **+0.082 (0.033–0.128)** |
| 1+ hit | v2 | 14,405 | 0.650 / 0.609 | +0.042 (0.032–0.051) | −0.18 | 0.50 (0.41–0.61) | 0.547 (0.538–0.557) | −0.006 (−0.012 to −0.000) | −0.000 (−0.005 to 0.005) | **+0.006 (0.003–0.008)** |
| 1+ hit | v3 | 11,818 | 0.648 / 0.607 | +0.041 (0.031–0.053) | −0.18 | 0.57 (0.44–0.69) | 0.552 (0.540–0.562) | −0.003 (−0.010 to 0.002) | +0.003 (−0.003 to 0.009) | **+0.007 (0.004–0.010)** |
| 1+ HR | v2 | 14,405 | 0.139 / 0.123 | +0.016 (0.010–0.022) | −0.15 | 0.58 (0.49–0.67) | 0.595 (0.582–0.608) | +0.003 (−0.005 to 0.010) | +0.008 (0.002–0.013) | **+0.012 (0.009–0.015)** |
| 1+ HR | v3 | 11,818 | 0.135 / 0.113 | +0.023 (0.017–0.028) | −0.22 | 0.63 (0.52–0.74) | 0.598 (0.581–0.613) | −0.000 (−0.010 to 0.009) | +0.008 (0.002–0.015) | **+0.012 (0.008–0.015)** |

All 10 K thresholds × 2 versions and all 5 props are in `out/posthoc_recal.csv`. After the intercept + slope
refit, the K thresholds have positive skill whose CI excludes 0 in 10 of 20 cells, with point estimates of
+0.008 to +0.082; none is negative. The prop calibration slopes of 0.50–0.63 mean the prop probabilities are
spread too wide: the predictions are overconfident about which batters will vs won't.

**Props vs baseline (c)** (player's own past rate, past games only, 20-game prior):

| prop | version | AUC sim (swapped) | AUC baseline (c) | skill vs (c): raw swapped | after intercept refit | after intercept + slope refit |
|---|---|---|---|---|---|---|
| 1+ hit | v2 | 0.547 (0.538–0.557) | 0.539 (0.530–0.548) | −0.008 (−0.014 to −0.001) | −0.001 (−0.007 to 0.004) | +0.004 (0.000–0.008) |
| 1+ hit | v3 | 0.552 (0.540–0.562) | 0.545 (0.533–0.555) | −0.008 (−0.014 to −0.001) | −0.002 (−0.008 to 0.005) | +0.003 (−0.002 to 0.008) |
| 1+ HR | v2 | 0.595 (0.582–0.608) | 0.568 (0.553–0.581) | −0.002 (−0.010 to 0.004) | +0.003 (−0.003 to 0.009) | **+0.007 (0.003–0.011)** |
| 1+ HR | v3 | 0.598 (0.581–0.613) | 0.590 (0.576–0.605) | −0.008 (−0.016 to −0.001) | +0.001 (−0.006 to 0.007) | +0.004 (−0.001 to 0.009) |

Swapped back and recalibrated, the sim ties baseline (c) on 1+ hit and slightly beats it on HR in v2. As
stored and uncalibrated, it loses to (c) in all four cells.

---

## 1. Data, exclusions, methods

Analysis set: **1,460 games (v2 802, v3 658)**, **26,223 player rows** (final, PA > 0, minus the 2 name-collision
rows). Exclusion waterfall: `out/exclusion_waterfall.csv`. Step 0 reproduced `misc/calibration_report.txt`
exactly (285 games, 33,910 rows, every metric within 4-decimal rounding) once gp 233 is removed. That game was
rained out on 05-09 and still unresolved when the report ran.

- **Dates:** ET schedule date everywhere.
- **Regimes (ET):** v2 05-01–07-19 · v3a 07-20–08-04 · v3b 08-10–09-27. v1 is excluded from headline numbers.
- **Bias** = mean predicted − realized rate, so positive means over-predicting.
- **Intervals:** 95%, cluster bootstrap by game (B = 2,000; B = 1,000 for AUC and calibration slope/intercept;
  B = 300 for swapped props). Log loss is clipped at ε = 1e-6.
- **Minimums:** ≥ 50 events and ≥ 50 non-events per cell; ≥ 30 of each per equal-count reliability bin.
- **Multiplicity:** BH at q = 0.05 over 1,624 tests (812 cells × bias and skill). 451 of 1,263 cells were "insufficient".

**Slice types that cannot qualify:** all park cells for game-level targets, SB by park and by team, about
half of HR by park/team, and about half the weather-label and |shift| cells for game targets.

**Deviations from SLICES.md:** B = 1,000 for row-level refits; the 2 Max Muncy rows were dropped. Everything
in 0A, 0B and the [rev] items is post-hoc.

---

## 2. Pipeline and scoring problems

| # | problem | evidence (n) | effect |
|---|---|---|---|
| P1 | **Home/away card wiring crossed in the simulator** (section 0) | tracer test, API check, 3 data checks; all versions | Every sided prediction belongs to the other team |
| P2 | `calibration.py:487` scores rung X as `total > X`, but the writer stores P(total ≥ X) | n=284 | +0.045 to +0.085 false bias on whole-number rungs |
| P3 | Rained-out attempts scored against the makeup game | 9 rows | Graded against games played weeks later |
| P4 | `game_date` is the UTC date | 308/1,544 (20%) | Night games stored a day late |
| P5 | Coverage depends on start time | 73% overall; 44% for 9 pm+ ET vs 90% for 7–9 pm; teams 57–86% | West Coast and day games under-represented |
| P6 | 74 rows "final" with 0 PA scored as misses | 54 in analysis set | Small bias in old reports |
| P7 | No unique constraint on (game_id, game_date, model_version) | schema | Upsert is app-only |
| P8 | Weather-shifted totals ladder inconsistent (whole-number vs half-run rung of the same event) | 19% (v2) / 32% (v3) of rung pairs | Whole-number rung has worse Brier in all 5 pairs in v3: +0.004 to +0.006, CIs > 0, n=658 |
| P9 | Props keyed by player name | gp 876 (two Max Muncys), p_1h = 1.08 | Same-name pairs merge |
| P10 | **[rev] p_1hr exactly 0, explained.** 72 of the 74 rows are **Chandler Simpson's** sim slot (the other 2 are Michael Stefanic). His career Statcast vault has 0 HR, the resolver keeps that at exactly 0 with no floor, and P1 attaches it to the opposing same-slot batter's name (Seager, Henderson, …; 4 of those named batters homered). Exactly 1 such row per game; slots 1 (20) and 5 (30) mostly | `p1hr_zero.py`, `out/p1hr_zero_rows.csv` | Unbounded log loss; HR probability floor missing |
| P11 | `prediction_logger.py:64` reads `os.environ["RDS_SECRET_ARN"]` at import | — | `calibration.py` cannot be imported without AWS env vars |
| P12 | Docs/code drift. The v3 commit says "log5 K resolution", but v2 already applied log5 to K; v3 actually **removed** log5 from BB and changed the pitch-count model. **[rev]** The v1→v2 step of "~+1 run" in predicted totals rests on **41 v1 games** (04-28 to 05-01, 15 of them at 2,000 iterations with no ladder) and is **not established**. The documented +0.3–0.5 cannot be confirmed or refuted from this data | git `3b94d41`; daily series | Version notes understate what changed |
| P13 | **[rev]** Local model inputs are stale: `data/batter_arsenal_profiles.parquet` has a single as_of_date (2026-04-23) and the vault ends 2025-11-01. Production fetched its inputs at runtime, so the profiles used in production cannot be reconstructed locally | `p1hr_zero.py` | Profile-existence checks cannot be done for production dates |

---

## 3. Model problems (confidence labels)

Confidence: **high** = BH-significant, replicates in both halves and across versions where applicable;
**medium** = significant, one version or partial replication; **low** = lead or descriptive.

**M1. Starter strikeouts are over-predicted, v3 more than v2, and the distribution is too narrow. Confidence: high.**
- **Every threshold, both sides, both versions** (`out/s4_k_thresholds.csv`). The bias level is unchanged by the swap.
- **Full distribution:** expected K − actual K is +0.27 (0.13–0.40) in v2 (1,603 starts) and **+0.72 (0.58–0.86)** in v3 (1,314 starts).
- **Pre-specified test (a), pitcher_4k_away:**

  | | n | events | bias (95% CI) |
  |---|---|---|---|
  | v2 | 801 | 518 | +0.100 (0.067–0.135) |
  | v3 | 658 | 404 | +0.165 (0.129–0.202) |
  | **v3 − v2** | | | **+0.065 (0.012–0.116)** |

  Swapped back: +0.109 in v2 and +0.164 in v3.
- **Direction of the v3 change:** v3's pitch-count change raised expected K as intended (5.02 → 5.27), but v2
  already over-predicted, so calibration moved the wrong way.
- **[rev] Dispersion after removing the mean bias** (`k_extra.py`): residual SD / RMS predicted SD is
  1.36 (1.32–1.41) in v2 and 1.28 (1.23–1.32) in v3 as stored. Swapped back it is **1.26 (1.22–1.31)** and
  **1.16 (1.11–1.20)**. The predicted K distribution is too narrow even for the right pitcher.
- **[rev] PIT histogram, swapped, 10 bins** (expected 160 per bin in v2, 131 in v3):

  | | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | v2 | 318 | 160 | 155 | 127 | 118 | 122 | 140 | 135 | 127 | 201 |
  | v3 | 299 | 155 | 127 | 128 | 105 | 90 | 104 | 92 | 117 | 97 |

  The low tail is about 2× expected in both versions (more starts with far fewer Ks than predicted). v2 also
  has a heavy high tail. As-stored counts are in `out/k_pit_hist.csv`.
- **[rev] Trend over months:** the slope of K ≥ 5 bias on month is −0.016 per month (−0.049 to 0.017) in v2 and
  +0.023 (−0.013 to 0.059) in v3. **There is no evidence that the K bias grows within a version.** The
  earlier "grows through v3" wording is withdrawn.

**M2. Run totals are under-predicted and too narrow. Confidence: high.**
- **Mean:** total_mean − actual is −0.77 (−1.10 to −0.46) in v2 (n=802) and −0.62 (−0.98 to −0.28) in v3 (n=658).
- **Spread:** SD of (actual − mean)/std is 1.14 (1.07–1.21) and 1.13 (1.06–1.20).
- **Over 8.5:** bias −0.056 (−0.092 to −0.019) and −0.051 (−0.089 to −0.014).
- Totals are a sum across both sides, so P1 changes them only through the batting order of innings.

**M3. Weather shift: the wind term is oversized; the density term is not. Confidence: medium. [rev]**

The shift is 0.05 runs/ft × carry. Carry = density term + 3.8 ft × wind component along the CF axis (mph). The
shift adjusts only the totals (`median_total`, `total_mean`, `total_std`, ladder). Props get a separate HR / 2+TB
multiplier clamped at ±30 ft; the totals shift is unclamped. Win, run line, F5 and NRFI are untouched.

Carry decomposition, outdoor games (n=760, runs):

| | median | 5% | 95% | min | max |
|---|---|---|---|---|---|
| density term | 0.25 | −0.11 | 0.67 | −0.39 | 1.70 (Coors) |
| wind term | 0.15 | −1.54 | 2.07 | −3.25 | **6.12** (WSH 05-19, 32 mph out) |
| total shift | 0.45 | −1.57 | 2.47 | −3.47 | 6.20 |

- The density term is reconstructed as carry − 3.8 × wind_out from stored rounded values, so it carries about ±0.012 runs of rounding error.
- The extreme values are essentially all wind.

**Regression of (actual − unshifted mean) on the two terms** (one row per game, so the game bootstrap is the
cluster bootstrap; a coefficient of 1 means the term is right-sized):

| set | n | density coef | wind coef |
|---|---|---|---|
| all games with weather | 1,153 | **1.49 (0.57–2.41)** | **0.19 (−0.09 to 0.48)** |
| outdoor non-Coors | 726 | 0.82 (−0.96 to 2.61) | 0.25 (−0.05 to 0.56) |
| Coors | 34 | 6.66 (−2.22 to 20.9) | −0.60 (−1.82 to 0.83) |
| v2 | 542 | 1.71 (0.25–3.10) | −0.01 (−0.45 to 0.38) |
| v3 | 611 | 1.41 (0.14–2.62) | 0.35 (−0.03 to 0.74) |

- **Wind:** the coefficient's CI excludes 1 everywhere it is estimable, and no wind effect is detectable.
- **Density:** consistent with right-sized or too small. The pooled estimate leans on Coors, where
  under-prediction is −3.1 runs unshifted and −1.9 shifted (n=34).

**Totals MSE vs the current shift (wind scale 1)**, density term kept:

| set | n | wind × 0 | wind × 0.33 | no shift at all |
|---|---|---|---|---|
| all with weather | 1,153 | −0.37 (−0.91 to 0.16) | **−0.45 (−0.82 to −0.10)** | +0.13 (−0.45 to 0.68) |
| outdoor | 760 | −0.55 (−1.32 to 0.26) | **−0.69 (−1.19 to −0.14)** | +0.19 (−0.69 to 1.07) |

The 0.33 scale was suggested by the earlier β = 0.32, so the improvement at 0.33 is optimistic (chosen on the
same data).

**Why test (b) used n=759 and the regression n=1,153.** The regression uses every game with weather fields:
393 domes (shift 0 by design) plus 760 outdoor games. Test (b)'s paired ΔMSE uses only games where the shift
is non-zero: 760 outdoor games minus 1 with carry exactly 0 = 759. The 307 games with missing weather are in
neither. Test (b) itself is unchanged: ΔMSE −0.19 (−1.03 to 0.63), n=759; β = 0.32 (0.06–0.58), n=1,153.

**M4. Prop levels: hits and total bases over-predicted, SB and RBI under-predicted. Confidence: high.**

| prop | v2 bias (n=14,405) | v3 bias (n=11,818) |
|---|---|---|
| 1+ hit | +0.042 (0.032–0.051) | +0.041 (0.031–0.052) |
| 2+ TB | +0.044 (0.034–0.053) | +0.050 (0.041–0.060) |
| 1+ HR | +0.016 (0.010–0.022) | +0.023 (0.017–0.029) |
| 1+ RBI | −0.017 (−0.026 to −0.009) | −0.012 (−0.021 to −0.002) |
| 1+ SB | −0.029 (−0.034 to −0.025) | −0.019 (−0.023 to −0.015) |

These levels are unchanged by the swap. The calibration slopes are 0.50–0.63 after the swap, so prop
probabilities are spread too wide (section 0B).

**M5. Props vs baselines. Confidence: high as stored; medium after swap + recalibration. [rev]**
- **As stored:** the sim loses to (a) league rate, (b) slot rate and (c) the player's own past rate for every
  prop. For example, 1+ hit vs (c) is −0.018 (−0.024 to −0.012) in v2.
- **Swapped back and recalibrated (intercept + slope, cross-fitted):**
  - It **ties (c) on 1+ hit**: +0.004 (0.000–0.008) in v2; +0.003 (−0.002 to 0.008) in v3.
  - It **beats (c) slightly on HR in v2**: +0.007 (0.003–0.011). In v3: +0.004 (−0.001 to 0.009).
  - Its AUC is above (c)'s in all four cells; for HR in v2 it is 0.595 vs 0.568.

**M6. Game-level markets. Revised: K markets have skill once fixed; win probability does not detectably. Confidence: medium. [rev]**
- **Home win, swapped, v3:** the sim **discriminates** (AUC 0.579, 0.535–0.617, n=658), but after recalibration
  its skill is +0.013 (−0.010 to 0.032), not distinguishable from 0. v2: AUC 0.524 (0.485–0.567); recalibrated
  skill −0.003 (−0.014 to 0.004).
  - The swapped probabilities under-predict home wins (bias −0.088 in v3), most likely because of the batting-order artifact in section 0B.
- **Starter K, swapped and recalibrated:** positive skill in 10 of 20 threshold cells with CIs above 0, including
  K ≥ 5 away in v2 +0.051 (0.018–0.080), K ≥ 5 home in v3 +0.048 (0.013–0.082) and K ≥ 7 home in v3 +0.082
  (0.033–0.128). AUC is 0.59–0.70.
- **NRFI** is unaffected by the swap: AUC 0.519 (0.480–0.559) in v2 and 0.489 (0.444–0.532) in v3, with bias CIs
  including 0. **No detectable discrimination.**
- **F5 and run line, swapped:** raw skill CIs include 0. Recalibration was not run for these.

**M7. v3 regimes. Confidence: low.** v3a (n=148) vs v3b (n=510):
- K ≥ 5 away: +0.146 (0.068–0.225) vs +0.185 (0.140–0.228).
- Over 8.5: +0.013 (−0.066 to 0.091) vs −0.070 (−0.113 to −0.025).

v3a is too small to separate weather completeness from the calendar.

---

## 4. Slices

### 4a. Frozen family, absolute bias (as originally run)
- 812 cells qualified. BH flagged 325 bias cells (283 "finding", 42 "lead") and 213 skill cells.
- Almost all of them restate M1, M2 and M4 within segments. Two BH-significant cells showed positive skill:
  - KC batters, 2+ TB, v3: +0.035 (0.003–0.059), n=441.
  - Cubs park, HR, v3: +0.031 (0.003–0.051), n=431; a lead.

### 4b. [rev] Post-hoc re-analysis: relative bias (`slices_diff.py`, `out/slices_diff.csv`)
Each cell's bias was re-tested as (cell bias − bias of all other rows of that target in the same version), with
a joint cluster bootstrap over the version's games and BH over the same family (803 cells; 9 regime cells have
no complement). Replication rule as before.

**Of the 283 earlier "findings", 4 survive as relative findings. 2 earlier leads become findings, and 1 new lead appears.**

| cell | n | cell bias | rest of version | difference (95% CI) | q | status |
|---|---|---|---|---|---|---|
| Athletics park, 1+ hit, v2 | 446 | −0.097 | +0.046 | **−0.143 (−0.188 to −0.096)** | <0.001 | finding (was lead) |
| Athletics park, 2+ TB, v2 | 446 | −0.090 | +0.048 | **−0.138 (−0.192 to −0.087)** | <0.001 | finding (was lead) |
| Athletics park, 1+ RBI, v2 | 446 | −0.113 | −0.014 | −0.099 (−0.152 to −0.047) | 0.040 | finding |
| San Diego batters, 1+ hit, v2 | 504 | +0.132 | +0.038 | +0.094 (0.048–0.138) | 0.009 | finding (exploratory: team; the batter identity is swapped by P1) |
| 1+ HR, \|shift\| 1.5–2.5 runs, v2 | 1,242 | +0.050 | +0.013 | +0.037 (0.016–0.057) | 0.042 | finding |
| 1+ HR, "Very HR Friendly" label, v3b | 2,350 | +0.047 | +0.017 | +0.031 (0.015–0.045) | 0.016 | finding |
| 1+ HR, May, v2 | 5,999 | +0.031 | +0.005 | +0.026 (0.015–0.037) | 0.002 | lead (single month) |

- **The two weather-HR cells point the same way as M3:** when the forecast says the ball will carry, HR props are
  over-predicted relative to other games. The prop HR multiplier (1% per ft, clamped at ±30 ft) is too strong.
  Confidence: medium.
- **The Athletics park cells (Sacramento)** show hits, TB and RBI under-predicted relative to other parks, with
  the same sign in v3 (−0.036 for hits). Park effects are game-level, so P1 does not move them.
- **The other 279 absolute "findings"** were the global biases seen again within a segment.

---

## 5. Coverage sensitivity (post-stratified by month × start time)

| target | version | unweighted bias | post-stratified bias |
|---|---|---|---|
| NRFI | v3 | +0.038 (−0.003 to 0.075) | +0.077 (0.023–0.129) |
| Home win | v3 | +0.012 | −0.027 (−0.092 to 0.032) |
| Over 8.5 | v3 | −0.051 | −0.074 (−0.129 to −0.019) |
| K, hits, HR | both | | within ±0.006 of unweighted |

Selection on when lineups were posted cannot be corrected.

## 6. Monthly series (`monthly_bias.png`, `out/monthly.csv`)

No month reverses the sign of M1, M2 or M4. **[rev]** The month-to-month K movement is not a significant
trend (M1). September has the most negative over 8.5 bias, −0.100 (−0.157 to −0.039), n=279. It is confounded
with call-ups and resting starters.

## 7. What is uncertain

- **The swap re-scoring is approximate:** the real home team bats first in the swapped view. A true fixed-model
  evaluation needs a re-simulation on archived inputs, and P13 means those inputs are not available locally.
- **Recalibration skill** is cross-fitted, but the recalibration form (intercept, slope) was chosen after seeing
  the slopes.
- **v3a** has 148 games.
- **Coverage selection** (P5) is only partly correctable.
- **The wind-scale 0.33 MSE gain** is optimistic, because the scale was chosen on the same data.

## Figures
`out/figures/`:
- `reliability.png`
- `monthly_bias.png`
- `k_threshold_bias.png`, `k_pit.png`
- `totals_shift_vs_residual.png`, `totals_whole_vs_half.png`

## Reproduce
From `analysis/` with `.venv` (pandas, numpy, scipy, psycopg2-binary, sqlalchemy, matplotlib, pyarrow), run in
this order:
1. `fetch_schedule.py <gamePks>`
2. `step0_acceptance.py`
3. `phase2a_integrity.py`
4. `totals_weather.py`
5. `k_check.py`
6. `headline.py`
7. `slices.py`

Then the verification and post-hoc scripts:
- `tracer_wiring.py`
- `api_spotcheck.py`
- `swapped.py`
- `posthoc_recal.py`
- `weather_decomp.py`
- `slices_diff.py`
- `k_extra.py`
- `p1hr_zero.py`

The DB connection is read-only, and the password comes from PGPASSWORD via libpq.
