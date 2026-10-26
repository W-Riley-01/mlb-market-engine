# Beat the Streak lab (v1) — pre-registered plan

**Written and frozen:** 2026-10-26, before any database query, ledger read, feature build or model fit for this lab.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `btslab sha256:`.
- Any change made after results are seen goes in a dated **Addendum** at the end and is labelled post hoc.
- Any change made before results goes in a dated **Amendments before results** section and is re-hashed.

## Summary in plain language

Beat the Streak is a game where you pick one MLB hitter a day. If that hitter gets a hit, your streak grows; to win you need
57 in a row. You may skip any day for free.

Our earlier check (R2) found that the big game simulator picked hitters no better than a simple formula ("B2") that
combines the hitter's batting average with the opposing starter's hit rate. Over 2022–2025 the daily top pick got a
hit 73.9% of the time for the simulator, 72.9% for a plain batting-average pick (B1) and 74.9% for B2.

This lab asks two questions:
1. Can a model built only for this job, using only information available before the game, pick a better daily
   hitter than B2?
2. Can it tell which days are unusually good, so that playing only those days gives a real chance at 57?

We decide now, before seeing anything, exactly what data, features, models and tests we will use, and what result
would count as "yes". We also write down what we expect, so we cannot later pretend we expected whatever happened.

One fact up front: the sample is small. About 360 test days and about 72 "best" days (Section 6). A real but modest
improvement of 1–3 percentage points will usually be **too small to prove** with this much data. A "not confirmed"
result means "not shown", not "shown to be useless".

Revised before first commit on 2026-10-26 after Claude Code found the T4 flaw; no results existed.

## 0. Step 0 — data audit (before any feature)

Step 0 checks coverage only. It counts rows and keys and looks at no outcome beyond the counts.
- For each season 2022–2026, report counts of:
  - final regular-season games;
  - games with a cached box score;
  - games with a full true starting nine on both sides;
  - games with a recorded pre-game probable starter on both sides;
  - games with closing total and moneyline, and the odds snapshot time if the source records one;
  - games with weather at scheduled first pitch;
  - games with a known venue id.
- The sources and tables used are named in the Step 0 report. Credentials come from the environment and are never
  written to any file or log.
- **2026** is used as a final untouched test (Section 6) **only if** Step 0 shows complete pre-game inputs for it.
  Otherwise it is not used, and the report says so.
- The Step 0 report is saved before features are built. It does not change this plan; if it forces a change, that
  change is a dated amendment, re-hashed, before any model is fit.

## 1. Question

Using only pre-game information:
- **(a)** can a purpose-built model (M1) pick a better top-1 daily hitter than the log5 baseline B2; and
- **(b)** can it identify a subset of days on which the top pick succeeds often enough to matter for a 57-game
  streak?

Context:
- Skipping a day costs nothing in Beat the Streak, so question (b) is a real policy, not a curiosity.
- R2 (`BACKTEST_V4_RESULTS_R2.md`), top-1 success 2022–2025: simulator 0.739 (539/729), B1 0.729 (532/730),
  B2 0.749 (546/729). The simulator added nothing over B2 (paired difference −0.010, 95% interval −0.047 to 0.032).

## 2. Data and unit

- **Unit:** batter-game.
- **Population:** the true starting nine only, by the rule in `analysis/backtest_v4.py` `starting_nine()`:
  `teams.{side}.players[ID].battingOrder == slot × 100`.
- **Seasons:** regular season (gameType R) 2022–2025, by the schedule's official date.
- **Games used:** final games with a cached box score and a recorded pre-game probable starter on both sides.
  - Games where a side has no probable starter are excluded and counted.
  - Suspended or resumed games are excluded, as in the v4 backtest, and counted.
- **Outcome:** hit = H ≥ 1, on rows with AB ≥ 1, from the box score's `stats.batting`.
- **Passes:** rows with AB = 0. They are excluded from every success rate, and their count is always reported.
- **Doubleheaders:** for daily picks a batter counts only in `game_number` 1, as in R2. Model fitting uses both games.
- **Daily pool:** every eligible batter-game on that date.

## 3. As-of rule

- Every feature for a game on date D uses only games strictly before D (`game_date < D`). No same-day information
  enters, except:
  - the lineup slot and the true starting nine (known at lineup posting; see Section 11);
  - the closing odds (Section 4B);
  - the scheduled venue and the weather at scheduled first pitch (Section 4F).
- Shrinkage priors and league rates are themselves computed as of D.
- **Leak check:**
  - Draw 20 random batter-game rows with seed 20261026.
  - For each, recompute every feature by hand from the raw game records with `game_date < D`, and confirm that it
    matches the built value.
  - Confirm that no game on or after D contributed.
  - **Any mismatch or leak stops everything** (Section 10). It is reported, fixed, and the check is rerun on 20 new
    rows (seed 20261026 + 100) before anything else.

## 4. Features (frozen list)

Common rules:
- **Windows:** trailing 365 days [D − 365, D) unless stated.
- **Plate appearance:** PA = a pitch row with non-null `events`; H and AB as in the R2 plan Section 3.
- **Shrinkage (empirical Bayes):** each rate is a beta-binomial posterior mean, (x + m·μ) / (n + m).
  - μ = the league rate in the same window.
  - m = the prior strength, estimated once per training fold by method of moments on training-season players with
    n ≥ 100, then frozen for that fold.
- **Missing values:** filled with the training-fold mean, with a 0/1 missing flag per group.
- **Scaling:** features are standardised with the training-fold mean and SD.

**A. Batter**
- Shrunk H/AB, K% (per PA), BB% (per PA) and ISO ((TB − H)/AB).
- Platoon H/AB against the probable starter's throwing hand, shrunk toward the batter's own overall H/AB, with m
  estimated as above.

**B. Opportunity**
- Lineup slot (1–9, as indicators).
- **Team implied runs** from the closing total T and the closing moneyline:
  - De-vig the moneyline multiplicatively to get p_home, as in the R2 plan test 1.
  - Split T so that R_home/R_away = (p_home / (1 − p_home))^(1/1.83) and R_home + R_away = T. The exponent 1.83
    is fixed now.
  - Missing odds: fill with the league-average implied runs and set a missing flag.
  - **Pre-game check:** if Step 0 cannot show that the odds snapshot precedes first pitch, the implied-runs feature
    is **dropped** from M1, and the report says so. The model "M1 without implied runs" then becomes M1.

**C. Opposing starter (the probable starter)**
- Shrunk H/AB allowed, K% and BB% (per batter faced), and throwing hand.
- **Expected batters faced:** the mean BF over the starter's last 10 starts before D, shrunk toward the league mean BF per
  start, with m estimated as above. A starter with no prior start gets the league mean.

**D. Expected PA vs starter and vs bullpen**
- E_PA(slot, side) = mean PA in the window for starting batters in that slot and side, as in B2.
- PA vs starter = min(E_PA, max(0, (BF_exp − slot)/9 + 1)), where BF_exp is from C.
- PA vs bullpen = E_PA − PA vs starter.

**E. Bullpen (opposing team)**
- Hit rate allowed by relievers over the 30 days before D, weighted by relief pitches thrown. This is the v4
  weighting used in `analysis/backtest_v4.py` and the resolver's bullpen option.
- **Fatigue, previous 3 days:** total relief pitches thrown, and the number of distinct relievers used.

**F. Park and weather**
- **Hits park factor, as of D, by the actual venue id of each game:** home H/PA ÷ road H/PA for games at that venue
  over the previous three seasons plus the current season before D, shrunk toward 1.
- **Athletics 2025 (Sutter Health Park, Sacramento):** it has its own venue id. Its factor uses only games there
  before D, so it starts at 1 and moves as games accrue. It is never borrowed from Oakland.
- **Arizona (Chase Field) and every other retractable-roof or domed park:**
  - roof status before the game is not known, so temperature is set to the league mean with an "indoor/unknown
    roof" flag;
  - the park factor applies as usual.
  - The list of roof types by venue is written into the Step 0 report.
- **Temperature at scheduled first pitch, open-air parks only.** Observed hourly weather stands in for the forecast
  (Section 11).
- **Wind:** exploratory only; not in M1.

**G. Recent form**
- Batter H/AB over the last 14 days and the last 30 days, each shrunk toward the batter's own 365-day H/AB, with m
  estimated as above.

**H. Exploratory only, never in M1**
- B2's P(1+ hit), as defined in the R2 plan.
- The R2 simulator's `p_1h`.

**Not tested in v1**
- Pitch-type and arsenal matchups.
- Anything not in the data (superstition, "feel").

## 5. Models

- **M1 (primary):** L2-penalised logistic regression of hit on groups A–G.
  - The penalty is chosen by inner cross-validation within the training seasons only: 5 folds of contiguous date
    blocks, 12 log-spaced values of C from 10⁻³ to 10², selected by mean log loss.
  - It is refit on all training seasons with the chosen penalty.
- **Comparators**
  - **B1:** the batter's trailing-365-day H/AB, requiring ≥ 200 PA, exactly as in R2 Section 3, so that R2's
    figures are comparable.
  - **B2:** log5 with the starter-share split, exactly as in R2 Section 3.
- **Exploratory (labelled, no rules)**
  - Gradient boosting with fixed settings, not tuned: `HistGradientBoostingClassifier(max_iter=300,
    learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=20261026)`.
  - M1 without B (implied runs).
  - M1 plus H.
- **Picks:** each method's top-1 per date is the pool batter with the highest predicted P(hit). Ties are broken by
  lower gamePk, then lower player id.

## 6. Design

- **Primary: walk-forward.**
  - Train 2022–23 → test 2024.
  - Train 2022–24 → test 2025.
  - Results are pooled over the two test seasons and also reported per season.
- **Secondary: leave-one-season-out (LOSO)** over 2022–2025. Fit on three seasons and test the fourth. This uses
  future seasons to predict past ones, so it is secondary.
- **2026:** if Step 0 allows, train 2022–25 → test 2026, once, after everything else is reported. Same tests, no
  re-tuning.
- **Day counts.** The real numbers are stated in the results from the data. Expected:
  - walk-forward: about 360 test days and about 72 top-20% days;
  - LOSO: about 730 test days and about 146 top-20% days.

**What can and cannot be resolved.** These are approximations from R2's paired interval width (±0.040 on 728 days)
and binomial SEs at a success rate near 0.80:

| comparison | design | n days | approx. 95% half-width | difference needed for ~80% power |
|---|---|---|---|---|
| T1: M1 − B2, all days | walk-forward | ~360 | ±0.056 | ~0.08 |
| T1: M1 − B2, all days | LOSO | ~730 | ±0.040 | ~0.056 |
| one method's rate, top-20% days | walk-forward | ~72 | ±0.092 | — |
| T2: M1 − B2, each on its own top-20% days | walk-forward | ~72 each | up to ±0.13 | ~0.18 |
| T2: M1 − B2, each on its own top-20% days | LOSO | ~146 each | up to ±0.092 | ~0.13 |

Stated plainly:
- An overall lift of 1–3 points **cannot be resolved** in either design.
- About 5 points is borderline under LOSO.
- On top-20% days, only very large differences, above 10 points, could be confirmed.
- T4's season odds depend steeply on the success rate, so their ranges will span orders of magnitude.

## 7. Tests

Common rules:
- **Bootstrap:** clustered by date, 2,000 resamples; seed = 20261026 + test number.
- **Intervals:** percentile intervals for differences, Wilson intervals for single proportions.
- **Confirmatory family:** T1 and T2. Each fires only if its 95% interval excludes 0 in favour of M1, about 0.025
  per test under the null. Then 1 − 0.975² = 0.049 if the tests are independent, and at most 0.05 under any
  dependence.
- Everything else is descriptive or exploratory.

**T1 — Overall (confirmatory).**
- Statistic: M1 top-1 daily success rate minus B2's, paired by date over dates where both picks are decided (not a
  pass), pooled over the walk-forward test seasons. LOSO is reported alongside.
- **Rule:** **"Confirmed"** only if the 95% interval lies entirely above 0. Otherwise **"not confirmed"**.
- Also reported:
  - each method's top-1 rate (Wilson), passes per method, and the difference with B1;
  - per season.

**T2 — Day tiers (confirmatory).**
- **Thresholds:** for each method and each test season, set from the training seasons only.
  - Take the 80th and 50th percentiles of that method's daily top-pick predicted probability over the training
    seasons.
  - For M1 these use out-of-fold predictions from the inner CV. B2 needs no fit; its own training-season values are
    used.
  - Each threshold is applied unchanged to the test season. The realised share of days above it is reported; it
    need not be exactly 20% or 50%.
- **Statistic:** M1's success rate on its own top-20% days minus B2's success rate on its own top-20% days.
  - Date bootstrap: resample dates, then recompute each method's rate on its own qualifying days within the
    resample.
  - The day sets differ and overlap; the overlap count is reported.
- **Rule:** **"Confirmed"** only if the 95% interval lies entirely above 0. Otherwise **"not confirmed"**.
- Also reported, descriptive:
  - the same comparison at the top-50% tier;
  - each method's rate on its top-20% and top-50% days vs its overall rate.

**T3 — Calibration (descriptive).**
- **Cross-fitted recalibration:** a logistic map (intercept + slope on logit p) is fit on the training seasons'
  out-of-fold daily top-1 picks (predicted vs hit), then applied to the test season. Done per method (M1, B2).
- Reported for all top-1 picks, top-50% days and top-20% days:
  - n, mean raw predicted, mean recalibrated predicted, realised (Wilson);
  - batter-game calibration of M1 in the R2 bins (Section 3 of the R2 plan).

**T4 — Season odds (descriptive).**
- **Why the policy family.** A streak needs 57 successful picks, and skipped days add nothing. A policy that plays
  fewer than 57 days a season (fewer than 57 picks, with Double Down) therefore has **zero chance by construction**.
  The top-20% and top-50% tiers belong to T2 only, not to T4.
- **Policy family:** play the N best days of each season, ranked by the top pick's recalibrated probability (T3).
  - N ∈ {60, 80, 100, 130, all days}. **Primary: N = 100.**
  - **Thresholds, fixed from the training years:**
    - D_train = the mean number of days per training season.
    - The threshold for N is the (1 − N/D_train) quantile of the training seasons' daily top-pick recalibrated
      probabilities, from out-of-fold predictions for M1 and from B2's own values.
    - Each threshold is applied unchanged to the test season. "All days" has no threshold.
  - The realised number of played days per test season is reported for each N. Any policy whose realised count is
    below 57 is reported as zero by construction.
- **Simulation:**
  - Monte Carlo, 100,000 seasons per method (M1, B2) and per N, seed 20261030.
  - Each season is the test season's calendar of played days. Each played day succeeds with that day's
    recalibrated probability, and a failure resets the streak to 0.
  - A 0-AB pick is a pass and does not break the streak.
- **Reported, for M1 and B2 at every N:**
  - the chance of a streak of ≥ 57 within a season, as "1 in N";
  - the mean recalibrated probability and the realised top-1 rate (Wilson) on the played days;
  - a range from success-rate uncertainty: rerun with all recalibrated probabilities shifted on the logit scale so
    that their mean equals each end of that Wilson interval.
  - One table across N shows the tradeoff: more days give more attempts but weaker picks.
- **Streak Saver** is ignored throughout.
- **Double Down sensitivity, at every N:**
  - On played days in the top 10% tier, pick two (the top 2); both must hit, and the streak gains +2.
  - Other played days take one pick.
  - The top-10% threshold is the 90th percentile of the training seasons' daily top-pick recalibrated probability,
    fixed as above.

**T5 — Drop-one-group value (EXPLORATORY).**
- Refit M1 without each of groups A–G in turn.
- Report the change in T1's statistic and in batter-game log loss on the test seasons, with date-bootstrap
  intervals (seed 20261031).
- No rule.

## 8. Hypotheses from William (recorded as stated)

These are read off M1's standardised coefficients and T5, plus the descriptive check named for each. None carries a
decision rule.

- **W1.** Lineup slots 1–4 have a higher hit chance (more at-bats and better hitters).
  - Check: the slot coefficients, and the realised hit rate by slot.
- **W2.** Higher OBP / more walks helps (hitter-friendly counts).
  - **Claude expects** the effect to be small or negative: walks take away at-bats.
  - Check: the BB% coefficient.
- **W3.** Power/SLG is valuable, but strikeout-heavy power hitters are worse; weather and park can limit it.
  - **Claude expects** contact and low K% to dominate, with ISO adding little.
  - Check: the ISO and K% coefficients; T5 for group A.
- **W4.** Hitter-friendly parks raise hit chance.
  - Check: the park-factor coefficient.
- **W5.** The days to find are the ones with a bad starter AND a bad bullpen.
  - M1 is additive in the logit, so this is checked descriptively on the test seasons: the realised hit rate for
    batter-games facing a top-quartile (as-of) starter H/AB allowed **and** a top-quartile bullpen hit rate,
    vs all others, with Wilson intervals.
- **W6.** Bullpen usage patterns matter for who pitches.
  - Check: the fatigue coefficients; T5 for group E.
- **W7.** A hot hitter seeing the ball well is more likely to get a hit.
  - **Claude expects** a small effect.
  - Check: the group G coefficients; T5 for group G.

## 9. Claude's pre-registered expectations

Each is reported as **met** or **not met**. They carry no decision rule.

- **(a)** M1's overall top-1 rate is within −1.5 to +2.0 points of B2's (point estimate, walk-forward), and T1 is
  not confirmed.
- **(b)** M1's top-20% days succeed at least 5 points above M1's overall rate. M1 minus B2 on top-20% days is
  between 0 and +4 points, with about a 15% chance that the interval excludes 0.
  - The chance is low because power is low: there are only about 72 top-20% days in walk-forward, so the interval
    is up to ±0.13 wide (Section 6).
- **(c)** K%, H/AB and slot carry most of the weight, as the three largest absolute standardised coefficients in
  M1. ISO adds nothing given K% (its 95% interval includes 0). BB% is small.
- **(d)** The recent-form effect is small: each group G coefficient is under 0.05 in absolute value, in logit per
  SD.
- **(e)** Bullpen fatigue: no detectable effect. Both fatigue coefficients' intervals include 0.
- **(f)** Before recalibration, raw top-tier probabilities are overconfident by at least 3 points: on top-20% days,
  mean raw predicted minus realised ≥ 0.03, for M1.

Coefficient intervals are date-bootstrap intervals on the full walk-forward training fit (2022–24), seed 20261032.

## 10. Stop rules

1. **Leak check fails** (Section 3): stop everything, report, fix, recheck.
2. **M1 must beat B1** on batter-game AUC (AB ≥ 1 rows) in the training-season inner CV for each walk-forward fold.
   If it does not, stop and report before any test-season result is computed.
3. **Wiring check:** in the training seasons, mean M1 predicted P(hit) for slots 1–4 minus slot 9 must be > 0.
   If the sign is wrong, investigate the wiring before anything else, and do not proceed until it is explained.

## 11. Limits and exposure

1. **Whole-slate pool is optimistic.** Real picks lock 5 minutes before each game, and lineups post late. A real
   player often cannot see every lineup, or wait for it. The pool here assumes every confirmed lineup is known.
2. **Lineups are actual starters.** The true starting nine reflects late scratches known only about 1–3 h before
   first pitch (as in R2, limitation 4).
3. **Weather** is observed at first pitch, not forecast.
4. **Odds:** closing odds are near first pitch. If their snapshot time cannot be shown to be pre-game, the
   implied-runs feature is dropped (Section 4B).
5. **Exposure:**
   - The 2022–2025 B2, B1 and simulator top-1 rates were seen in R1 and R2. B2's definition is unchanged from those
     runs.
   - The feature list was chosen with knowledge of R1/R2 results and of the hypotheses in Section 8.
   - Only 2026, if usable, is fully untouched.
6. **Multiple looks:** T5 and the exploratory models are many comparisons. Nothing in them is a finding.
7. **Changes:** any change after results goes in a dated addendum, labelled post hoc.

## Addendum

(none)
