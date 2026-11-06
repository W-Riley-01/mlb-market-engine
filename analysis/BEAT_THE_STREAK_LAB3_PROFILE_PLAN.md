# Beat the Streak lab 3 (batter profile v2) — pre-registered plan

**Written:** 2026-11-06, before any lab 3 feature, model fit or outcome read. No lab 3 code exists.
- It builds on lab 1 and lab 2. None of these files is changed:
  - lab 1 plan `analysis/BEAT_THE_STREAK_LAB_PLAN.md` (`f10661f`), addenda A1 (`6d9cb3c`) and A2 (`bb6c331`);
  - lab 1 code `analysis/btslab_features.py` and `analysis/btslab_analysis.py`;
  - lab 1 results `analysis/out/btslab_results_log.txt`;
  - lab 2 plan `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`), addenda A1 (`074878e`) and A2 (`478847a`);
  - the lab 3 data log `analysis/out/lab3_data_log.txt`.
- **Freezing.** The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab3 sha256:`.
  Section 8 (hypotheses) and the "Resolved choices" section were settled before it was hashed.
- Changes made before results go in a dated **Amendments before results** section and are re-hashed.
- Changes made after results go in a dated **Addendum**, labelled post hoc.

## Summary in plain language

Lab 1's model (M1) scores each batter's chance of at least one hit from their batting average, strikeout and walk
rates, lineup slot, the opposing starter, the bullpen, the park and recent form.

Lab 3 asks whether a richer picture of the batter adds anything. It adds five groups:
- **How hard and at what angle they hit the ball.** Hard contact, expected hits from each ball's speed and angle, and
  ground-ball, line-drive and pop-up shares.
- **Plate discipline.** Swinging misses, chasing pitches outside the zone, and contact on pitches inside it.
- **Handedness.** Whether the batter faces a same-handed starter, and how the park plays for their side.
- **Speed.** Last season's sprint speed.
- **Age and trend.** The batter's and starter's ages, and whether the batter's last year was better or worse than
  the year before.

We score every batter-game with and without the new groups and compare the **log loss**: how close the predicted
chances are to what happened. As in lab 2, this can detect much smaller improvements than the one-pick-a-day test.

Safeguards:
- **M1 is first rebuilt** and must reproduce lab 1's published numbers exactly, or we stop.
- **A placebo** shuffles the new features among batters playing on the same day. If the "improvement" survives the
  shuffle, the method is fooling itself, and we stop.
- **Everything is tested on made-up outcomes first.** The real hit column stays unread until then.

One fact up front: 2024–25 has already been used by labs 1 and 2. A "confirmed" result here is one more look at the
same seasons. The clean test is a later combined run on 2026, under its own plan.

## 1. Question

Using only pre-game information: does adding contact quality, plate discipline, handedness, speed and age/trend
(group H) to lab 1's M1 improve the predicted probability of at least one hit for a batter-game?
- **Primary:** paired batter-game log loss, M1 against M3 = M1 + H.
- **Secondary:** leave-one-season-out log loss; top-1 daily success (descriptive).

Context:
- **Lab 1 results** (walk-forward, test 2024–25): M1 top-1 0.783, B2 0.762, M1 − B2 +0.022 (95% −0.022 to +0.069),
  not confirmed.
- **Batter-game log loss:** M1 0.65947, B2 0.66210.
- **Lab 1's drop-one-group log-loss intervals** (T5) had half-widths from ±0.00002 to ±0.0004 (Section 12).

## 2. Data and unit

**Rows**
- Exactly lab 1's table `analysis/cache/btslab_features.parquet`, built by `analysis/btslab_features.py`. It has
  174,348 rows:
  - true starting nine (`battingOrder == slot × 100`);
  - regular season 2022–2025;
  - the 18 neutral-site games excluded (lab 1 A1 change 2).
- **Content hash:** `e1fb5d01cf19f18eab46b561575a5975867348cca1f16f29656d203bf7600f9a`, recorded in
  `analysis/out/lab2_step1b_log.txt`.
  - The method is the SHA-256 of `pandas.util.hash_pandas_object(df, index=False)`.
  - The builder recomputes it before anything else and stops on a mismatch (Section 10, rule 2).
- **Countable rows** (AB ≥ 1): 174,117 in all seasons; 87,021 in the walk-forward test seasons 2024–25. Passes
  (AB = 0, 231 rows) are excluded from log loss and success rates, and counted.
- **Group H joins on the row key** (`game_pk`, batter `id`). Lab 1's columns are never modified.

**Pitch-level inputs**
- Source: `data/master_physics_vault.parquet` (pulled 2026-04-22), 2019–2025.
- It is restricted to regular-season games of the schedule (`gameType == "R"`, lab 1 A1 change 2), and each pitch
  is dated by its game's effective date from `btslab_features.load_schedule()`, as in `raw_tables()`.
- **2026 is not opened in this lab.** The 2026 final test is a later combined test with its own plan. When it
  happens, 2025–26 windows use the single current-vintage file `data/statcast_current_2025_2026.parquet` (lab 2 A2
  change 2).
- Pitch-type labels are not used by any lab 3 feature.
- **Exceptions:** two features use lab 1's own source `data/pitch_matrix.parquet`, so that they match their lab 1
  counterparts:
  - trend_hab's H/AB windows;
  - h_park_hand's H/PA counts.

  See Resolved choice 7.

**Other inputs**
- **Bio:** `analysis/cache/player_bio.parquet` (sha256 `3299f3db…a0c`, lab 3 data log): `bat_side`, `pitch_hand`,
  `birth_date`. These are static attributes fetched on 2026-11-06. Every batter and starter in the rows has a birth
  date and a bat side (lab 3 data log: 100% of 2022–25 PAs).
- **Speed:** `analysis/cache/sprint_speed_by_season.parquet` (sha256 `f39f2591…1c8`): Savant sprint speed per player
  and season, 2019–2025, minimum 5 competitive runs (Savant's floor).
- **Contact table:** `analysis/lab2_contact_table.csv` (lab 2 A1, commit `074878e`): P(hit | launch-speed bin,
  launch-angle bin).
  - It was fitted on 2020–2021 regular-season balls in play only (max game_date 2021-10-03), so it predates every
    2022+ game.
  - The `p_hit` column is applied with the table's left-closed bins.
  - **Disclosure.** Some H windows include 2020–21 balls in play, which were in the table's fit. From the windows of
    Sections 3–4 and the regular-season dates (2020 ends 2020-09-27, 2021 ends 2021-10-03):
    - `h_xhit` ([D − 365, D)), and trend_xhit's first term, which is the same window, include 2021 balls for 2022
      games dated through 2022-10-03, i.e. all but the season's last two days.
    - trend_xhit's older window ([D − 730, D − 365)) includes 2021 balls for every 2022 game, 2020 balls for 2022
      games dated through 2022-09-27, and 2021 balls for every 2023 game.
    - No 2024 or 2025 row's features include a 2020–21 ball, so both walk-forward test seasons (T1) are clean. They
      are reached only through what is estimated on training rows (m and the model coefficients).
    - Touched: 2022–23 training rows in both walk-forward folds, and the held-out 2022 and 2023 seasons in their
      leave-one-season-out folds (T2, secondary only).
    - Those balls precede the game date, so no future information enters. Any one batter's balls are a negligible
      share of each of the table's cells, so the effect is at most a slight optimism in the 2022–23 feature values.

## 3. As-of rule

- Every H feature for a game on date D uses only games strictly before D (`date < D`), doubleheaders included, as in
  lab 1.
- League rates and shrinkage targets are computed in the same window.
- **Speed for a game in season s uses season s − 1 only.** An assert in the builder checks this for every row.
- **Static attributes** (bat side, pitch hand, birth date) are same-day-known facts, not history. Age is computed
  on D.
- Each sub-group carries `H1_max_date` … `H5_max_date`: the latest pitch date it used.
  - H3 uses `H3_max_date` for h_park_hand.
  - H4's is the end of season s − 1.

## 4. Features: group H (frozen list)

**Common rules** (lab 1 Section 4, plus A2 change c)
- **Window:** trailing 365 days [D − 365, D) unless stated.
- **Shrinkage:** each rate is (x + m·μ) / (n + m), `btslab_features.shrink`.
  - μ = the league rate in the same window, over all batters' pitches in the vault rows of Section 2.
  - m is estimated once per training fold by `btslab_features.estimate_m`. It uses one row per batter-season (the
    last training row of each batter-season, as `derive()` does), rows with n ≥ 100 in that feature's own trial
    unit.
  - m = ∞ (no between-batter spread) gives μ exactly.
- **No history:** a batter with n = 0 gets μ exactly, plus the sub-group's missing flag.
- **Standardisation and missing fills:** all H columns go through `btslab_analysis.design()`, i.e. training-fold
  mean fill and training-fold mean and SD, as all M1 inputs.
- The feature table stores raw window counts. Shrinkage is applied at fit time per fold, as lab 1 does.

**Definitions** (vault columns; regular-season games only)
- **Launch-data ball in play (BIP):** `type == "X"` with non-null `launch_speed` and `launch_angle`.
- **Swing:** `description` in {swinging_strike, swinging_strike_blocked, foul_tip, foul, foul_bunt, missed_bunt,
  bunt_foul_tip, hit_into_play, foul_pitchout, swinging_pitchout}. This is the swing definition of the 2026-11-06
  audit.
- **Miss:** `description` in {swinging_strike, swinging_strike_blocked, foul_tip}.
- **Zone coding,** confirmed in the vault 2022–25 regular season:
  - `zone` 1–9 is the strike-zone 3×3 grid: 1–3 top row, 7–9 bottom row; plate_x within ±0.83 ft.
  - 11–14 are the four out-of-zone quadrants.
  - There is no zone 10 and no null zone.
  - "In zone" = 1–9; "out of zone" = 11–14.

**H1 — contact quality** (trial unit: launch-data BIP)
- `h_hard`: share of BIP with launch speed ≥ 95 mph. Beta-binomial shrinkage.
- `h_xhit`: mean of the contact table's `p_hit` over the batter's BIP.
  - **Shrinkage (normal variance-components analogue of `estimate_m`):** the stored window sums are n (BIP), S = Σp
    and Q = Σp².
  - **Within-batter variance**, pooled over training unit-season rows with n ≥ 100:
    s²_w = Σ(Q − S²/n) / Σ(n − 1).
  - **Between-batter variance:** τ² = (Σ n (S/n − μ)² − K·s²_w) / Σ n, over the same K rows, with μ = the league mean
    `p_hit` in each row's window. This is `estimate_m`'s formula with μ(1 − μ) replaced by the constant s²_w and c = 1.
  - **Prior strength:** m = s²_w / τ², or ∞ if τ² ≤ 0.
  - **Shrunk value:** (S + m·μ) / (n + m).
- **Ball-type shares** by launch angle:
  - `h_gb`: < 10°;
  - `h_ld`: 10° to < 25°;
  - `h_pu`: > 50°;
  - fly balls (25° to 50°, both ends inclusive) are the omitted remainder.

  Each share is shrunk on its own as a binomial share of BIP, with its own m. They are not constrained to sum with
  the omitted share.
- `H1_miss` = 1 if the batter has no launch-data BIP in the window.

**H2 — plate discipline** (trial unit: pitches seen, as below)
- `h_whiff` = misses / swings. Trial unit: swings.
- `h_chase` = swings at zone 11–14 / pitches at zone 11–14. Trial unit: out-of-zone pitches.
- `h_zcon` = (swings at zone 1–9 − misses at zone 1–9) / swings at zone 1–9. Trial unit: in-zone swings.
- Each is beta-binomial shrunk with its own m.
- `H2_miss` = 1 if the batter saw no pitch in the window.

**H3 — handedness**
- **Batter side s:**
  - bio `bat_side` (R or L);
  - a switch hitter (S) bats opposite the starter's hand, so s = L against a right-handed starter and R against a
    left-handed one.
- **Starter hand:** bio `pitch_hand` of the actual starter `opp_sp`.
- **`h_same`** = 1 if s equals the starter's hand, else 0. A switch hitter therefore always has h_same = 0.
  - If the starter's bio hand is S (switch pitcher) or missing: h_same = 0 and `h_same_miss` = 1.
- **`h_park_hand`:** the venue's H/PA for batters on side s, shrunk toward the home team's road H/PA for side s, then
  divided by that road H/PA.
  - Built exactly as lab 1's `f_park` (`raw_tables()` `park_venue` / `park_road`, the same exclusions, the same window
    "previous three seasons plus the current season before D"), with PAs split by the PA-ending pitch's `stand` in
    `pitch_matrix`.
  - **m:** one m for both sides, estimated on venue-season-side rows by `estimate_m` with target = that side's road
    rate, as `derive()` does for `park`.
  - It equals 1.0 when the road PA for that side is 0, as f_park.
  - **Side of a switch hitter facing a switch pitcher:** the opposite of lab 1's `c_hand` (last `p_throws` before D).
    If `c_hand` is also missing, h_park_hand = f_park.

**H4 — speed**
- `h_speed` = the batter's season s − 1 sprint speed (ft/s), not shrunk (Resolved choice 4).
- `h_speed_miss` = 1 if there is no s − 1 value. About 10.6% of 2022–25 PAs (lab 3 data log: 89.4% covered); the
  fill is the training mean.

**H5 — age and trend**
- `h_age` = (D − batter birth date) / 365.2425 − 28. `h_age2` = h_age².
- `sp_age` and `sp_age2`: the same for the starter.
- Centring at 28 years before squaring is fixed now. The builder asserts that every row has both birth dates.
- **`trend_hab`** = shrunk H/AB over [D − 365, D) minus shrunk H/AB over [D − 730, D − 365).
  - Both use lab 1's H/AB counts (`raw_tables()` `bat`), lab 1's league H/AB in each window as target, and the fold's
    m["hab"].
  - The first term equals `a_hab` exactly.
- **`trend_xhit`** = shrunk h_xhit over [D − 365, D) minus over [D − 730, D − 365), each with the fold's h_xhit m.
- `H5_trend_miss` = 1 if the batter has no AB in [D − 730, D − 365).
- **Age enters as regressors.** It does not change any shrinkage target. M1's features stay exactly as lab 1 built
  them.

**Column list (M3 adds these to M1_COLS, as `GROUPS["H"]`, sub-grouped)**

| sub-group | columns |
|---|---|
| H1 | h_hard, h_xhit, h_gb, h_ld, h_pu, H1_miss |
| H2 | h_whiff, h_chase, h_zcon, H2_miss |
| H3 | h_same, h_same_miss, h_park_hand |
| H4 | h_speed, h_speed_miss |
| H5 | h_age, h_age2, sp_age, sp_age2, trend_hab, trend_xhit, H5_trend_miss |

**Exploratory only:** `x_speed_gb` = z(h_speed) × z(h_gb), the product of the two standardised columns. It is never
in M3.

## 5. Models

- **M1:** lab 1's M1 exactly as committed: `M1_COLS`, `derive()`, `design()`, `fit_l2()`.
  - Refit on the same folds.
  - It must reproduce lab 1's reported numbers (Section 10, rule 3).
- **M3 (primary)** = M1 + group H (Section 4 column list).
- **Fitting, as lab 1:**
  - the same Newton L2 solver (sklearn's objective, intercept unpenalised);
  - training-fold standardisation;
  - inner CV: 5 contiguous date blocks of the training dates, 12 log-spaced C from 10⁻³ to 10², mean log loss;
  - refit on all training rows at the chosen C.
- **M3's penalty** is chosen by its own inner CV, on the same grid. The grid is not extended, even though lab 1
  chose C = 0.001 (its edge) in every fold; a boundary choice is reported.
- **Comparators:** B2, as in lab 1 (top-1 only).
- **Exploratory models** (labelled, no rules):
  - **M1 + Hk** for k = 1..5: one sub-group added. Fitted at M1's chosen C for the fold.
  - **M3 − Hk** for k = 1..5: one sub-group dropped. Fitted at M3's chosen C, as lab 1's T5 used the full model's C.
  - **M3 + x_speed_gb**, at M3's C.
- **Log loss** uses `v4_analysis.ll`, probabilities clipped to [10⁻⁶, 1 − 10⁻⁶], on countable rows.

## 6. Design

- **Primary: walk-forward.**
  - Train 2022–23 → test 2024.
  - Train 2022–24 → test 2025.
  - Results are pooled over 2024–25 (87,021 countable rows) and also reported per season.
- **Secondary: leave-one-season-out** over 2022–2025 (174,117 countable rows).
- **Diagnostic (T6): 2022 dropped from training.**
  - Train 2023 → test 2024.
  - Train 2023–24 → test 2025.
- **Bootstrap:** clustered by date, 2,000 resamples (`btslab_analysis.boot_counts`); percentile intervals.
- **Seeds:** 20261201 + test number. Placebo permutations use 20261250 + k: k = 0..49 in the real run's T4,
  k = 0..9 in the null study. Fake outcomes:
  - null study 4001–4200;
  - planted 4300;
  - planted-null 4301.

## 7. Tests

**Confirmatory family:** T1 alone. Everything else is reported, descriptive or exploratory.

**T1 — PRIMARY: walk-forward log loss.**
- **Statistic:** the mean over walk-forward test rows (2024 and 2025 pooled) of log loss(M1) − log loss(M3), paired
  by row. Positive = M3 better.
- Bootstrap clustered by date, seed 20261201.
- **Rule:** **"Confirmed"** only if the 95% interval lies entirely above 0 **and** the point estimate is positive in
  both test seasons (2024 and 2025 separately). Otherwise **"not confirmed"**.
- Also reported: each season's statistic and interval; M1 and M3 batter-game AUC.

**T2 — LOSO log loss.**
- The same statistic over the four held-out seasons, with seed 20261202.
- T1's rule is applied and reported separately. It is secondary and not confirmatory.

**T3 — Top-1 daily hit rate (descriptive, no rule).**
- M3 − M1 and M3 − B2 top-1 daily success, paired by date over dates where both picks are decided.
- The pool and tie rules are lab 1's: game 1 only for doubleheaders; ties to lower gamePk, then lower id.
- Walk-forward, date bootstrap, seed 20261203. Intervals reported.

**T4 — PLACEBO CONTROL.**
- **Permutation:** within each game date, the full vector of H columns (all sub-groups and flags together) is
  permuted across that date's batter-games. All M1 columns stay with their rows.
- **Repeats:** 50 permutations in the real run, seeds 20261250 + k. Inside the null study (Section 11, step 4) each
  fake-outcome seed uses 10 placebo permutations, k = 0..9 (Resolved choice 2).
- **Refit:** for each permutation, M3 is refitted on the permuted training rows of each walk-forward fold at that
  fold's real M3 C, with no new inner CV. It is scored on the permuted test rows.
- **Placebo gain** = the T1 statistic with M3 replaced by the permuted M3:
  - per-row differences are averaged over the 50 permutations;
  - the mean is bootstrapped by date (2,000, seed 20261204).
  - The 2.5–97.5 percentile range of the 50 per-permutation statistics is also reported.
- **Stop rule:** if the placebo gain's 95% bootstrap interval lies entirely above 0, stop and investigate before
  reading any other result (Section 10, rule 5). It is computed before T1 is read.

**T5 — EXPLORATORY: ablations and coefficients.**
- Log-loss change and top-1 change for M1 + Hk against M1 and M3 − Hk against M3, k = 1..5. Walk-forward, date
  bootstrap, seed 20261205.
- **Coefficient table:** M3's standardised (per-SD) coefficients, fit on 2022–24, with date-bootstrap intervals,
  seed 20261207.
  - The bootstrap is lab 1's: refit at the fold's C with date-resample weights, warm-started at the full fit.

**T6 — DIAGNOSTICS (reported, no rule).**
- T1's statistic retested with 2022 dropped from training (Section 6), seed 20261206.
- T1's statistic split by batter side (bio `bat_side`: R, L, S) on the walk-forward test rows, seed 20261206.

**T7 — HYPOTHESIS CHECKS** (coefficients from T5's table; no confirmatory claim beyond T1)

| hypothesis | pre-specified quantity | direction read as support |
|---|---|---|
| V1 | h_xhit and h_hard (a_hab in the model) | each > 0 |
| V2 | h_zcon and h_whiff (a_k in the model) | h_zcon > 0, h_whiff < 0 |
| V3 | h_same | < 0 |
| V4 | h_speed; exploratory x_speed_gb from M3 + x_speed_gb | each > 0 |
| V5 | age contrast and trend_hab (see below) | age contrast < 0; trend_hab > 0 |
| V6 | h_chase | < 0 |

- **The V5 age contrast** = the logit difference implied by M3's h_age and h_age2 coefficients between batter ages 34
  and 29, holding everything else fixed.
  - It is computed on the original scale: undo standardisation with the 2022–24 training mean and SD.
  - Bootstrap interval from the same resamples.
- Each quantity is reported with its sign and interval.

## 8. Hypotheses

> These six were proposed by Claude. They are not William's words.
> William's keep / cut / change for each: KEEP (2026-11-06; William's reply: "Keep all is probably the right call").

- **V1** (proposed by Claude) — Hitters who hit the ball hard at line-drive angles get more hits than their batting
  average suggests. William's keep / cut / change: KEEP.
- **V2** (proposed by Claude) — For getting at least one hit, making contact (few whiffs) matters more than power.
  William's keep / cut / change: KEEP.
- **V3** (proposed by Claude) — A batter facing a same-handed pitcher does worse. William's keep / cut / change: KEEP.
- **V4** (proposed by Claude) — Fast runners get extra hits on grounders and bunts. William's keep / cut / change:
  KEEP.
- **V5** (proposed by Claude) — Older hitters decline, and last year's trend matters. William's keep / cut / change:
  KEEP.
- **V6** (proposed by Claude) — Hitters who chase pitches outside the zone get fewer hits. William's keep / cut /
  change: KEEP.

Each is checked by T7 only, and carries no decision rule.

## 9. Claude's pre-registered expectations

Each is reported as **met** or **not met**. None carries a decision rule.
- **(a)** T1 gain (M1 loss − M3 loss) is between +0.0002 and +0.0010. There is about a 60% chance the interval
  excludes zero.
- **(b)** Contact quality is the largest single-group gain: in T5, M1 + H1 has the largest log-loss gain over M1 among
  the five M1 + Hk models.
- **(c)** h_same has a negative coefficient whose interval excludes zero: 70%.
- **(d)** Whiff, chase and zone contact add little beyond K%: each of the h_whiff, h_chase and h_zcon coefficient
  intervals includes zero: 60%.
- **(e)** Speed is small, |h_speed coefficient| < 0.02 per SD. The age contrast's interval (T7, V5) excludes zero:
  50%. Trend is about zero: the trend_hab interval includes zero.
- **(f)** Top-1 daily rate M3 − M1 is between −1.5 and +2.5 points. This cannot be resolved (Section 12).
- **(g)** The placebo shows no gain: its interval includes zero.
- **(h)** T6's 2022-dropped retest has the same sign as T1: 75%.
- **(i)** M3's top-20% days are no more than 3 points above M3's overall top-1 rate: 80%.
  - The top-20% days are within-season rank by raw top-pick probability, round(0.20·D) days per season, as lab 1 A2
    change b.

## 10. Stop rules

1. **Leak-check failure** stops everything: report, fix, rerun the checks on new rows (seed + 100).
2. **Base table mismatch:** the lab 1 table's content hash differs from `e1fb5d01…00f9a`.
3. **M1 reproduction fails.** The real run first refits M1 and compares, at the precision printed in
   `analysis/out/btslab_results_log.txt`:
   - for each walk-forward and LOSO fold: C, inner-CV log loss (5 decimals), AUC M1 and B1 in CV (4 decimals),
     wiring (4 decimals);
   - walk-forward M1 test log loss 0.65947 and AUC 0.5740;
   - M1 top-1 0.783 (pooled), 0.765 (2024), 0.801 (2025);
   - M1 − B2 +0.0220 (95% −0.0220 to +0.0687, seed 20261027).

   Any mismatch stops before M3 is fitted.
4. **Self-tests fail** (Section 11): null false confirmation above 8% over 200 seeds; the planted effect (+0.10 per
   SD on h_xhit) not recovered; the planted-null confirms; the placebo false-stops above 8% (10 permutations per
   null seed).
5. **Placebo shows a gain** (T4 interval entirely above 0): stop before any other result is read.
6. **Inherited from lab 1 (Section 10, rules 2–3)**, checked for M1 and M3 in each walk-forward fold:
   - M1 beats B1 on batter-game AUC in the training inner CV;
   - the training slot 1–4 minus slot 9 predicted P(hit) is > 0.

## 11. Required steps before any real run

In order. Each step's log is committed before the next step that depends on it.

1. **Feature builder** `analysis/lab3_features.py` → `analysis/cache/lab3_features.parquet`.
   - It holds the raw window counts and sums for H1–H5, keyed by (`game_pk`, `id`), plus `H1_max_date` …
     `H5_max_date`.
   - Coverage report: missing-flag shares by season.
   - No outcome is read.
2. **Leak checks** `analysis/lab3_leakcheck.py`:
   - **Structural:** every sub-group's max source date < D on every row. Speed season = game season − 1 on every
     row. The contact table's training max date < 2022-01-01, read from the CSV header and asserted.
   - **Future deletion:** for 300 random rows (seed 20261201) plus 150 doubleheader rows, delete every source row
     dated on or after D, recompute every H feature, and confirm it is unchanged.
   - **Plain-path recompute:** 5 rows per sub-group (seed 20261202), recomputed from raw vault, `pitch_matrix`,
     bio and speed rows by a separate simple loop, and matched.
3. **Analysis script** `analysis/lab3_analysis.py`, reusing `btslab_analysis` (`derive`, `design`, `fit_l2`,
   `boot_counts`, `top`, `pool`).
   - Without `--real` the hit column is never loaded, by the same guard as lab 1.
   - `--real` refuses to run if `analysis/out/lab3_results_log.txt` already exists (run once).
4. **Self-tests on fake outcomes** (AB ≥ 1 rows; the real hit column never loaded):
   - **Runtime projection, first:** before the null study starts, time one model fit and report the projected
     null-study runtime. Report only: the 200 seeds and the 10 permutations do not change.
   - **Null study:** 200 seeds (4001–4200), Bernoulli(0.75), 500 bootstrap resamples, parallel.
     - Each seed runs T1 and T4. T4 uses 10 placebo permutations per seed (seeds 20261250 + k, k = 0..9), not the
       real run's 50, with the same refit and stop rule (Resolved choice 2).
     - **Pass:** T1 false confirmation ≤ 8%; T4 false stop ≤ 8%; the T1 statistic centred at 0 (|mean| < 0.25 SD
       across seeds).
   - **Planted H:** logit p = logit(0.75) + 0.30·[slot ≤ 4] − 0.40·z(K%) + 0.50·z(H/AB) + 0.10·z(raw h_xhit)
     (lab 1's planted formula plus the H term; Resolved choice 3).
     - z is over all rows; missing → 0. Raw h_xhit = unshrunk window mean. Seed 4300.
     - **Pass:** M3's h_xhit coefficient is within ±50% of 0.10 (0.05 to 0.15) **and** T1 is Confirmed; the placebo
       is not Confirmed.
   - **Planted-null:** lab 1's planted formula with no H term, seed 4301. **Pass:** T1 not Confirmed.
5. **Commit and push** the plan (hashed), the features, the leak-check logs, the scripts and the self-test logs.
6. **Real run:** `python analysis/lab3_analysis.py --real`, once.

## 12. Resolving power

- **Source.** M1's out-of-fold or test predictions were not saved by lab 1, so a per-row log-loss spread is not
  available. This estimate is the lab 2 plan's (Section 13, item 6), scaled from lab 1's T5 intervals.
- **Lab 1's paired drop-one-group half-widths** in log loss: D ±0.00002, G ±0.00004, B ±0.00007, E ±0.00008,
  F ±0.00028, C ±0.00033, A ±0.00039. The width grows with how much a group moves predictions.
- **For lab 3:**
  - group H as a whole should move predictions about as much as groups A, C or F: half-width about ±0.0003 to
    ±0.0004;
  - single sub-groups (T5) about ±0.00005 to ±0.0003.
- **Resolvable:** a true T1 gain of about 0.0006–0.0008 or more is detectable with high probability.
- **Borderline:** gains near 0.0003–0.0004.
- **Not resolvable:** gains below about 0.0002.
- T1's added requirement (positive in both seasons) lowers power a little.
- **Top-1 success** cannot resolve differences under about 4–5 points (lab 1). T3 is descriptive for that reason.

## 13. Limits and exposure

1. **Seen seasons.**
   - 2022–2025 outcomes were seen in lab 1 and will be in lab 2.
   - 2024–25 is now used by a third analysis, so each lab is another look at the same test seasons.
   - With lab 1's two confirmatory tests, lab 2's one and lab 3's one, the chance of at least one false confirmation
     is about 1 − 0.975⁴ ≈ 0.096 if independent.
   - The clean test is the later 2026 combined run.
2. **Intentional walks.** Automatic intentional walks have no pitch row, so they are missing from pitch-level counts
   (H1, H2). Lab 1's PA counts share this.
3. **Neutral sites.** The 18 neutral-site games are excluded from the rows and from park factors, as in lab 1.
4. **Static bio.** Bat side, pitch hand and birth date were fetched in 2026 and treated as fixed. A player who
   changed his batting side is classified by his 2026 record.
5. **Sprint speed.** Savant values for past seasons are assumed stable since the pull; the file's hash is recorded.
   Values exist only for players with ≥ 5 competitive runs.
6. **Pitch-type labels** are not used by any lab 3 feature, so lab 2 A2's label-vintage issue does not apply.
7. **Penalty at the grid edge.** Lab 1 chose C = 0.001 (the strongest penalty on the grid) in every fold. M3 uses the
   same grid. Strong shrinkage of coefficients can understate H's value.
8. **Whole slate.** The pool is the whole slate, which is optimistic, as in lab 1.
9. **Exploratory comparisons** (T5, T6, T7) are many. Nothing in them is a finding.
10. **Thin early trend windows.** trend_hab's older window (days 366–730 before the game) for 2022 games reaches into
    2020–21, which includes the shortened 2020 season, so early-2022 trend values are thin and lean on shrinkage.
    There is no special handling; T6 (2022 dropped from training) covers it. `data/pitch_matrix.parquet` begins on
    2019-03-20 (dates checked 2026-11-06, no outcomes read), so the days 366–730 windows of every 2022 game are fully
    covered by the data.

## Resolved choices (2026-11-06, settled in Claude chat before any code or data run)

Each choice is recorded with its ruling. "Accepted" means the default stated in the plan stands as written.

1. **Placebo mechanics (T4).** Accepted.
   - Refit at the real M3 C, with no inner CV per permutation.
   - Per-row differences averaged over the permutations, then date-bootstrapped.
   - The stop fires on that interval.
2. **Null-study cost.** CHANGED.
   - Inside the null study each seed uses 10 placebo permutations, not 50. The real run's T4 stays at 50
     permutations.
   - The placebo false-stop rate in the null study is held to the same 8% limit.
   - **Reason:** fewer permutations make the averaged placebo gain noisier while the date bootstrap does not see that
     noise, so passing at 10 should imply passing at 50.
   - Added builder step (Section 11, step 4): time one model fit and report the projected null-study runtime before
     the null study starts. Report only; the 200 seeds and the 10 permutations do not change.
3. **Planted effect.** CHANGED.
   - Plant +0.10 log-odds per SD on h_xhit, not +0.05.
   - **Pass:** the recovered coefficient within ±50% of 0.10 (0.05 to 0.15) **and** T1 confirms.
   - **Reason:** at +0.05 the expected gain (about +0.0002 in log loss) is below the resolving power (about ±0.0003),
     so the check would likely fail for lack of power rather than because of a bug; changed before any code or data
     was run, so it is not result-driven.
4. **h_speed is not shrunk.** Accepted: raw season s − 1 value plus a missing flag.
5. **Age terms.** Accepted: centred at 28.0 years before squaring; V5 and expectation (e) are read off the age
   contrast between 34 and 29.
6. **missed_bunt.** Accepted: literal definitions, so it counts as a swing and as contact, not as a miss (809 pitches
   in 2022–25).
7. **Sources for trend_hab and h_park_hand.** Accepted: `pitch_matrix`, lab 1's source, so that the first term of
   trend_hab equals a_hab exactly and h_park_hand matches f_park.
8. **Contact-table disclosure.** Accepted as worded in Section 2: 2020–21 balls reach h_xhit and trend_xhit for 2022
   and 2023 games only, so they touch 2022–23 training rows in both walk-forward folds and the held-out 2022 and 2023
   seasons in their LOSO folds (secondary), not the T1 test seasons 2024–25 (corrected 2026-11-06 before
   publication: the first version named only 2021 balls in 2022 windows).
9. **M3's penalty.** Accepted: its own inner CV on the same grid.
10. **Expectation thresholds.** Accepted:
    - "speed is small" = |coefficient| < 0.02 per SD;
    - "trend about zero" = the trend_hab interval includes 0;
    - (b) is read from M1 + Hk gains.
11. **Switch hitters.** Accepted: h_same is always 0 for switch hitters; no switch-hitter indicator.
12. **Seeds.** Accepted: base 20261201; fake seeds 4001–4200, 4300 and 4301.

## Amendments before results

(none)

## Addendum

(none)
