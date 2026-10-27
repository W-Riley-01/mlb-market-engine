# Beat the Streak lab — Addendum A2 (amendments before results)

**Dated 2026-10-26, written before any real outcome was read.**
- It amends `analysis/BEAT_THE_STREAK_LAB_PLAN.md` (`f10661f`) as amended by
  `analysis/BEAT_THE_STREAK_LAB_ADDENDUM_A1.md` (`6d9cb3c`). Neither file is changed.
- Basis: the Step 2a self-tests on fake outcomes. Logs:
  - `analysis/out/btslab_selftest_null_log.txt`;
  - `analysis/out/btslab_selftest_null_seed1777_log.txt`;
  - `analysis/out/btslab_selftest_planted_log.txt`;
  - `analysis/out/btslab_analysis_notes.txt`.
- The real hit column had not been read, fitted on or summarised by anything in Step 2a.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `btslab A2 sha256:`.
- Where this addendum differs from the plan or A1, this addendum governs.

## Summary in plain language

We tested the analysis on made-up outcomes before letting it see any real ones. Two parts of the plan misbehaved,
so we fix them now, before any real result exists.

- **The "correction" step could turn the model upside down.** After the model scores each day's best pick, a small
  correction makes those scores honest. When the scores barely differ from day to day, that correction can come out
  absurd: on random data it reversed the order of the days. From now on the correction may only keep the order or
  flatten it, never reverse it.
- **The "best days" were not a fixed share of the season.** The cut-off for the best 20% of days was learned on
  earlier seasons. Applied to a new season, it picked anywhere from 5% to 17% of the days, so the two methods were
  compared on very different numbers of days.
  - The main comparison now takes exactly the best 20% (or 50%, or N days) of each season for each method.
  - That needs the whole season's scores at once, which a real player would not have. So the old,
    usable-in-real-life version is kept and reported next to it.
- **What else changed in Step 2a.** It also fixed four bugs in the analysis code and made one modelling choice.
  These are listed below so nothing is hidden.

## Changes

### a. Recalibration slope constrained to [0, 1]

**Amends:** plan Section 7, T3 (cross-fitted recalibration), and therefore T4, which uses recalibrated probabilities.

**Evidence.**
- In the null self-tests, M1's top-pick predictions barely varied, so the logistic map's slope was not identified.
- Fitted slopes were −1.7 and −6.8 (seed 777) and −0.4 and −3.6 (seed 1777). A negative slope reverses the order of
  the days.

**Change**
- The map a + b·logit(p) is fitted with b constrained to [0, 1].
- If the unconstrained maximum-likelihood b lies outside [0, 1], b is set to the nearer bound and the intercept a is
  refitted given that b.
- b = 0 means a constant: every pick gets the training picks' realised rate.
- Everything else in T3 is unchanged: fitted on the training seasons' out-of-fold daily top-1 picks, applied to the
  test season, per method.

### b. Day tiers and season policies: within-season rank (primary) and training-year thresholds (secondary)

**Amends:** plan Section 7, T2 (thresholds) and T4 (policy family and Double Down tier).

**Evidence.** Training-year thresholds selected 5–17% of test-season days as the "top 20%" in the self-tests
(planted walk-forward: 24 of 365 days for M1 against 73 for B2). The two methods were then compared on very
different numbers of days.

**Change**

*Primary: within-season rank.*
- In each test season, each method's days are ranked by the top pick's calibrated predicted probability.
  - Ties, including all days when the calibration slope is 0, are broken by the raw predicted probability, then by
    date.
  - The ranking is therefore the raw-probability ranking whenever the slope is positive.
- **T2:** each method's top 20% and top 50% of days are exactly round(0.20·D) and round(0.50·D), where D = that
  season's days with a pick.
  - The statistic and the bootstrap are as in the plan.
  - **The T2 decision rule applies to the primary version.**
- **T4:** each method plays exactly N days per season, N ∈ {60, 80, 100, 130, all}; primary N = 100.
  - The Double Down sensitivity doubles on the top round(0.10·D) days by the same ranking.
- **This ranking uses predictions only, never outcomes.** But it needs the whole season's predictions at once,
  which a real player would not have.

*Secondary ("deployable"): training-year thresholds, as in the plan.*
- T2 thresholds are the 80th and 50th percentiles of the training seasons' daily top-pick predicted probability
  (out-of-fold for M1).
- T4 thresholds are the (1 − N/D_train) quantile of the training seasons' recalibrated top-pick probabilities, and
  the 90th percentile for Double Down.
- All are applied unchanged to the test season.
- The realised share of days is reported for T2 and the realised N for T4. A realised N below 57, or below 57
  picks with Double Down, is scored as zero chance by construction.
- The secondary version carries no decision rule.

Both versions are reported side by side.

### c. Disclosure: changes made in Step 2a, before this addendum

None of these used the real hit column. They were found and checked on fake outcomes. The shrinkage estimates use
past-game counts already in the feature table (hits, at-bats, plate appearances and batters faced before the game
date), never the outcome of the game being predicted. Choices are listed in `analysis/out/btslab_analysis_notes.txt`.

**Bugs fixed**
1. **Platoon and recent form, moment estimate** (`btslab_features.estimate_m`).
   - These rates are subsets of the batter's 365-day window, so their sampling variance is μ(1−μ)(1/n − 1/N), not
     μ(1−μ)/n.
   - Ignoring the nesting gave a negative between-player variance and full shrinkage.
   - Fixed with the nested-variance moment estimate.
2. **Expected batters faced, moment estimate** (`btslab_analysis.m_bf()`).
   - A regression on 1/n was confounded: pitchers with few starts are a different kind of pitcher, such as openers.
   - Replaced by a consecutive-start estimate. The per-start variance comes from changes between consecutive full
     10-start windows; the between-pitcher variance comes from 10-start means.
3. **"Shrink fully" fallback.** It returned m = 10⁶, which standardisation turned back into noise. It is now
   m = ∞, which returns the shrinkage target exactly.
4. **Double Down branch of the T4 simulation.** Rewritten before the first run.
   - A Double Down day with one 0-AB pick plays the other pick alone.
   - Self-checked, including the 57-pick boundary.

**Modelling choice.** Platoon H/AB and recent-form H/AB (14 and 30 days) enter M1 as deviations from the batter's
own shrunk H/AB.
- With H/AB in the model this spans the same model.
- Without it the near-copies split H/AB's coefficient, and W7 and expectation (d) could not be read.

**Software.** scikit-learn 1.6.1 was installed into `analysis/.venv` for the exploratory gradient-boosting model
(plan Section 5).

### d. Unchanged

Everything else in the plan and addendum A1 stays as written, including:
- M1, the comparators, the folds and seeds;
- T1, T3's reporting, T5;
- the expectations, the hypotheses and the stop rules.

## Addendum

(none)
