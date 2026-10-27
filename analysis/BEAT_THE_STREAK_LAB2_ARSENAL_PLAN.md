# Beat the Streak lab 2 (pitch-type matchups) — pre-registered plan

**Written and frozen:** 2026-10-26, before any lab 2 feature, contact-quality table or model existed. No query, fetch
or install was made to write it.
- It builds on lab 1:
  - `analysis/BEAT_THE_STREAK_LAB_PLAN.md` (`f10661f`);
  - addenda A1 (`6d9cb3c`) and A2 (`0d3a7647…`);
  - results `analysis/out/btslab_results_log.txt`;
  - the lab 2 data inventory `analysis/out/lab2_step0_log.txt`.

  None of those files is changed.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 sha256:`.
- Changes made before results go in a dated **Amendments before results** section and are re-hashed.
- Changes made after results go in a dated **Addendum**, labelled post hoc.

## Summary in plain language

Lab 1 built a model (M1) that scores each batter's chance of a hit from things like their batting average,
strikeout rate, lineup slot, the opposing starter and the ballpark. It could not prove that M1 picks better daily
hitters than a simple formula (B2). That test could only detect differences larger than about 4.5 percentage points,
because a season has few days.

Lab 2 asks a narrower question: **does knowing which kinds of pitches the starter throws, and how this batter does
against those kinds, add anything?** A batter who struggles against breaking balls, facing a pitcher who throws
mostly breaking balls, might be a worse pick than their overall numbers suggest.

To get a sensitive test we score every batter-game, not just one pick a day. The main yardstick is **log loss**:
how close the predicted chances are to what happened, averaged over about 87,000 batter-games in 2024–25. It can
detect much smaller improvements than the daily-pick test.

Safeguards:
- We measure the quality of contact with a table built only from 2019–2021, so later seasons cannot leak in.
- A **placebo** shuffles each pitcher's pitch mix between games. If the "improvement" survives the shuffle, the
  method is fooling itself, and we stop.
- A **diagnostic** asks whether batters and pitchers truly differ by pitch type, beyond their overall skill. If
  they barely do, the matchup score has nothing to find, and we say so.
- Everything is tested on made-up outcomes first, and the real hit column stays unread until then.

If 2026 pitch data can be pulled and checked, the frozen models get one final test on that untouched season.

## 1. Question

Using only pre-game information:
- **Primary:** does adding a pitch-type matchup score (I1) to lab 1's M1 improve the predicted probability of a
  hit for a batter-game? It is measured by paired batter-game log loss.
- **Secondary:** does it change top-1 daily success?

Context (lab 1 results):
- Walk-forward top-1 success was M1 0.783 and B2 0.762, M1 − B2 +0.022 (95% −0.022 to +0.069), not confirmed. The
  interval half-width was about ±4.5 points.
- Batter-game log loss resolves far smaller differences: lab 1's drop-one-group intervals had half-widths of
  0.00002–0.0004 (Section 13). So log loss is lab 2's primary metric.

## 2. Data and unit

**Rows**
- Exactly lab 1's table `analysis/cache/btslab_features.parquet`, built by `analysis/btslab_features.py` (`94fc0c1`
  as amended in `bb6c331`):
  - one row per true starter per game (`battingOrder == slot × 100`);
  - regular season 2022–2025;
  - the 18 neutral-site games excluded;
  - 174,348 rows.
- **Countable rows** (AB ≥ 1): 174,117 in all seasons; the walk-forward test seasons 2024–25 have 87,021.
- Passes (AB = 0) are excluded from log loss and from success rates, and counted.

**As-of rule.** As in lab 1:
- every feature uses only games dated strictly before the game date, including for doubleheaders;
- each game is dated by its last Final schedule date;
- regular season is the schedule's `gameType == "R"` (A1 change 2).

**Pitch data**
- Source: `data/master_physics_vault.parquet`, restricted to regular-season games of the schedule (2019–2025 for
  history).
- It holds `pitch_type`, `events`, `description`, `type`, `stand`, `p_throws`, `launch_speed` and `launch_angle`.
- The existing ingest dropped pitches with no `pitch_type` or `release_speed`; that is accepted as is (Step 0).

**PA-ending pitch** = the last pitch of a PA, i.e. the row with a non-null `events`. Step 0's alignment rules:
- **Automatic intentional walks** have no pitch row. They are not AB-ending and do not enter any pitch-type measure.
- **`truncated_pa` rows** (an inning ended by a baserunning out mid-PA) are not PAs. They are excluded from every
  lab 2 measure.
- **Games with dropped pitches** are flagged. These are games where box-score PA ≠ pitch-row PA − truncated_pa
  − IBB: 276 of 9,704 in 2022–25, plus any found in 2019–21. Their remaining pitch rows are used.
  - The flag `I_gap` is carried for the row's own game but is never a model input. A game's own pitches are
    same-day data and never enter its features.
  - The count of history pitches lost is reported.

**Windows.** Trailing 365 days, [D − 365, D), as in lab 1.

## 3. Contact-quality measure (expected hit)

**Purpose.** Replace each ball in play's actual result with its expected-hit value, to remove fielding and luck noise.

**Table**
- P(hit | launch-speed bin, launch-angle bin), fitted on **2019–2021 regular-season balls in play only**
  (`type == "X"`).
- Hit = single, double, triple or home run.
- Bins:
  - launch speed: < 60 mph, 60–65, …, 110–115, ≥ 115 (13 bins);
  - launch angle: < −30°, −30–−20, …, 50–60, ≥ 60 (11 bins).
- A cell with fewer than 100 balls takes the pooled rate of its whole angle bin.
- Balls in play with no launch speed or angle (0.2–0.9% by season) use their actual result.

**Freezing**
- The table is written to `analysis/lab2_contact_table.csv` and committed (hash recorded in `PLAN_HASH.txt`) before
  any lab 2 feature is built.
- It is then applied unchanged to every season. 2022 history windows use 2021 balls in play, which are inside the
  table's fitting years; those balls all precede every pool row, so no future information enters.

**Not used:** Savant's `estimated_ba_using_speedangle` / `estimated_woba_using_speedangle`. They come from a model
trained on all years, which would put future information into older seasons; they are also not in the vault.

**Sanity checks** (stop rule 4 if either fails):
- **Barrels clearly above weak contact.** Mean P(hit) for speed ≥ 98 mph and angle 20–35° is at least 0.25 above the
  mean for speed < 80 mph, all angles.
- **Monotone in speed.** Within each angle bin from −10° to 40°, P(hit) does not fall by more than 0.03 from one speed
  bin to the next, among cells with ≥ 500 balls.

## 4. Pitch families

| family | codes |
|---|---|
| fastball | FF, SI, FA, FC |
| breaking | SL, ST, SV, CU, KC, CS |
| offspeed | CH, FS, FO, SC |

- Other codes (KN, EP, PO, UN) are dropped, and usage is renormalised over the three families.
- Individual codes are not used in the primary.
- Pooling SL/ST/SV makes the sweeper relabelling drift seen in Step 0 irrelevant within the breaking family.

## 5. Per-AB measure

**AB-ending pitch.** A PA-ending pitch whose PA is an official at-bat: `events` not in walk, hit_by_pitch, sac_fly,
sac_bunt, sac_fly_double_play, sac_bunt_double_play, catcher_interf, truncated_pa. This is lab 1's AB list plus
truncated_pa (Section 2).

**Value x of an AB-ending pitch**
- **Ball in play** (`type == "X"`, including home runs and reaching on error): the expected-hit value from
  Section 3.
- **Strikeout:** 0 (strikeout, strikeout_double_play).
- Any other AB-ending event without a ball in play is 0.

**The measure.** Per-AB measure = sum of x / number of AB-ending pitches, for a batter (or pitcher), a family, and,
for pitchers, a batter side.

**League levels**
- L_f = the league per-AB measure for family f in the same as-of window.
- L = the league per-AB measure over all three families.

## 6. Features

**Side s.** The batter's side against this starter.
- s = the batter's usual side (the side of most pitches seen, 2019 to D).
- For switch hitters (at least 20% of pitches seen from each side), s is the side opposite the starter's hand, as
  in A1 change 7 (a switch hitter's `stand` depends on the pitcher's hand).
- Starter hand as in lab 1 (most recent `p_throws` before D).

**Pitcher usage u(f|s).** The starter's share of pitches in family f to batters on side s, in the window. Two levels
of shrinkage:
- the starter's overall usage u(f), all sides, is shrunk toward the league usage for pitchers of that hand;
- u(f|s) is shrunk toward u(f).

Each level is a convex combination of distributions, so the shares sum to 1 (stop rule 3).

**Batter relative skill r_b(f)**
- The batter's per-AB measure against family f is shrunk toward R_b · L_f, where R_b = the batter's shrunk overall
  per-AB measure / L.
- r_b(f) = shrunk family measure / L_f.
- With no family data, r_b(f) = R_b.

**Pitcher relative vulnerability r_p(f|s)**
- Likewise for the starter, on AB-ending pitches to side s, shrunk toward R_p(s) · L_f.
- R_p(s) = the starter's shrunk overall per-AB measure allowed to side s / L.

**Shrinkage**
- Empirical Bayes by method of moments, per training fold, on one row per unit-season (lab 1 notes item 5).
- Family measures are subsets of the overall window, so they use lab 1's nested-variance estimator
  (`btslab_features.estimate_m` with N = all AB-ending pitches; A2 change c).
- The per-AB measure is not 0/1 for balls in play. The binomial variance μ(1 − μ) is used as an upper bound for
  the per-trial variance, which can only add shrinkage.
- For usage, one m per level: the n-weighted mean of the three families' binomial moment estimates.
- m = ∞ (no detectable spread) gives the target exactly, as in lab 1.

**I1, "arsenal fit" (primary feature)**

I1 = Σ_f u(f|s) · L_f · r_b(f) · r_p(f|s) − Σ_f u(f|s) · L_f · R_b · R_p(s)

- The second sum uses each side's overall relative skill in place of the family-specific values. So I1 is zero
  unless batter and pitcher have family-specific deviations that line up with the pitcher's mix.
- **Thin history:** if the batter has no AB-ending pitches, or the starter no pitches to side s, in the window, then
  I1 = 0 and the flag `I1_miss` = 1.
- I1 is standardised with the training-fold mean and SD, as all M1 inputs.

**Exploratory only**
- **I2** = Σ_f u(f|s) · r_b(f): usage-weighted batter skill.
- **I3** = Σ_f u(f|s) · r_p(f|s): usage-weighted pitcher vulnerability.
- **I4**, "weak-pitch exposure" = max_f u(f|s) · (r_p(f|s) − 1) · r_b(f). This is William's U1 (Section 9).
- **Alternative grouping:** FC with breaking, FS with fastball; I1 recomputed with it.

**Bullpens** are unchanged: there are no arsenal terms for relievers.

## 7. Models

- **M1:** lab 1's M1 exactly.
  - Groups A–G, market excluded.
  - Platoon and recent form as deviations (A2 change c).
  - Same shrinkage, missing flags and standardisation.
- **M2 (primary)** = M1 + I1 + I1_miss.
- **Exploratory** (labelled, no rules):
  - M1 + I1..I4;
  - M1 + I1 (alternative grouping).
- **Fitting:** the same Newton L2 solver, penalty search and folds as lab 1.
  - Inner CV: 5 contiguous date blocks, 12 C values from 10⁻³ to 10², mean log loss.
  - Walk-forward: train 2022–23 → test 2024; train 2022–24 → test 2025.
  - Secondary: leave-one-season-out.
- **Recalibration** of daily top picks: as in A2 change a (slope constrained to [0, 1]).

## 8. Tests

Common rules:
- **Bootstrap:** clustered by date, 2,000 resamples; seed = 20261101 + test number.
- **Intervals:** percentile.
- **Log loss** is on countable rows (AB ≥ 1), probabilities clipped as in lab 1.
- **Confirmatory family:** T1 alone. Its rule fires only if the 95% interval lies entirely on the side of M2
  improving, about 0.025 under the null.

**T1 — PRIMARY: walk-forward log loss.**
- Statistic: mean over walk-forward test rows (2024 and 2025 pooled, 87,021 rows) of log loss(M2) − log loss(M1),
  paired by row.
- Bootstrap clustered by date.
- **Rule:** **"Confirmed"** only if the 95% interval lies entirely below 0 (M2 better). Otherwise **"not confirmed"**.

**T2 — LOSO log loss.** The same statistic over the four held-out seasons (174,117 rows). The same rule is applied
and reported separately. It is secondary and not part of the confirmatory family.

**T3 — Top-1 success and AUC (reported, no rule).**
- M2 − M1 and M2 − B2 top-1 daily success, paired by date as in lab 1 T1.
- Batter-game AUC for M1, M2 and B2.

**T4 — PLACEBO CONTROL.**
- Repeat T1 with each starter-game's usage vector u(·|s) permuted across starter-games within the same season.
  - Permutation seed 20261150.
  - The batter's and pitcher's family skills stay with their own rows; only the pitch mix is shuffled.
- I1 and M2 are refitted on the placebo feature.
- **Expected:** no gain.
- **Stop rule:** if the placebo's interval lies entirely below 0 (M2 better), stop and investigate before reading any
  other result (Section 11, rule 2).

**T5 — DIAGNOSTIC: how much family-specific skill exists.**
- For batters (per family) and pitchers (per family and side), the method-of-moments estimate of the between-unit
  variance of true family-specific deviations beyond overall skill.
- Data: one row per unit-season, walk-forward training data 2022–24, as-of windows.
- Each is reported as variance and as a share of the overall-skill spread (SD ratio).
- Point estimates; date-bootstrap intervals, seed 20261106.
- **Stated plainly:** if the shares are near zero, there is nothing for the fit score to find.

**T6 — EXPLORATORY comparisons.**
- Log loss and top-1 success for the exploratory models in Section 7 against M1.
- Date bootstrap, seed 20261107.
- Labelled exploratory.

**Coefficients.** M2's standardised I1 coefficient, with a date-bootstrap interval on the 2022–24 fit, seed 20261120.

**FINAL TEST on 2026** (only if Section 10's data step passes):
- **Freezing:** M1, M2 and B2 are frozen after training on 2022–2025:
  - penalty from inner CV on 2022–25;
  - shrinkage m on 2022–25;
  - recalibration maps from 2022–25 out-of-fold picks.
- **F1:** M1 − B2 top-1 success on 2026. This is lab 1's question on fresh data. It is a new test under this plan; it
  does not amend lab 1, whose A1 change 1 dropped 2026.
  - Date bootstrap, seed 20261111.
  - Rule: "Confirmed" only if the interval lies entirely above 0.
- **F2:** M2 − M1 log loss on 2026.
  - Confirmatory (same rule as T1) only if T1 was confirmed; otherwise descriptive.
  - Seed 20261112.
- **Run once.**

## 9. Hypotheses and expectations

**Hypotheses from William (recorded as stated)**
- **U1.** "A pitcher's worst pitch, relative to what the league allows on that pitch, thrown often, combined with a
  batter who hits that pitch type well, raises the batter's chance of a hit."
  - Checked by I4 (exploratory, T6) and by the sign of I1 (T1, coefficient).
- **U2.** Pitchers differ meaningfully in how much damage each pitch family allows. Tested by T5.
- **U3.** Batters differ meaningfully in how well they hit each pitch family. Tested by T5.

**Claude's pre-registered expectations** (each reported as met or not met; no decision rule)
- **(a)** T1 improvement small: log loss(M1) − log loss(M2) between 0 and 0.0004, with about a 35% chance the
  interval excludes 0.
- **(b)** The I1 coefficient is positive and small: under 0.03 per standard deviation.
- **(c)** Top-1 success M2 − M1 within ±2 points, and no confirmation.
- **(d)** The placebo (T4) is within noise of 0: its interval includes 0.
- **(e)** I4 adds nothing beyond I1. In T6, M1 + I1..I4 vs M2: the log-loss difference interval includes 0.
- **(f)** T5: family-specific spread is real but modest — positive, under half of the overall-skill spread (SD
  ratio < 0.5) — and larger for pitchers than for batters.
- **(g)** Lab 2 is unlikely to turn the pooled edge over log5 into a confirmed one. M2 − B2 top-1 (T3) is not
  confirmed by the T1-style rule.

## 10. Data contingency for 2026

**Step 1b** pulls the 2026 regular season. **The final test is off until all of this passes; no other test depends
on it.**

1. **Pitch data from Baseball Savant.** The tool is chosen by a validation recorded in the Step 1b log.
   - Install into `analysis/.venv` with pinned versions. The venv's Python is 3.13.0.
   - Order of tools:
     1. **pybaseball via `data_ingestion.py`**, the project's existing path, run with `--year 2026` (monthly chunks
        Mar 15 – Nov 5).
     2. **pybaseballstats v0.6.1** (needs Python ≥ 3.11, satisfied).
     3. **A direct Baseball Savant CSV pull.**
   - **Validation, before use.** Re-pull one 2025 regular-season week with the candidate tool and compare it with the
     same week in the vault. It must match nearly exactly:
     - row count within 0.5%;
     - pitch-type counts per family within 0.5%;
     - release speed and launch speed equal for matched pitches (by game_pk, at_bat_number, pitch_number) in ≥ 99.5%
       of rows.

     If it fails, try the next tool.
   - **Ingest.** The 2026 pull goes through the existing ingest rules: the same columns; pitches with no
     `pitch_type` or `release_speed` dropped. The 2025 comparison week is not written into the vault.
2. **Schedules and box scores for 2026.** Lab 1's feature builder needs these too: true starters, venue, game type,
   batters faced, relief pitches, weather text. They come from the MLB Stats API with the existing per-date fetch
   (`backtest_v4.get_json`), into the existing caches.
3. **2019–2025 is not re-pulled.** Savant revises labels, and mixing pulls from different dates risks inconsistent
   labels. The families already pool SL/ST/SV.
4. **Build and check.** Lab 1's feature builder, then the lab 2 features. Then the full leak checks of Section 11
   on 2026 rows: structural; future deletion on 300 random rows plus 150 doubleheader rows; the plain-path spot
   check.

## 11. Stop rules

1. **Leak check failure** stops everything. The checks:
   - structural: every group's max source date before the game date, including the lab 2 group;
   - future deletion: 300 random rows plus 150 doubleheader rows, all columns;
   - plain-path spot check of 20 rows, including u(f|s), r_b(f) and r_p(f|s).
2. **Placebo confirmation** (T4 interval entirely below 0) stops before any other result is read.
3. **Usage shares must sum to 1** (to 1e-9) on every row with history, or stop.
4. **The contact-quality table must pass both sanity checks** (Section 3), or stop before features are built.
5. **Inherited from lab 1 (Section 10, rules 2–3)**, checked for M1 and M2 in each walk-forward fold:
   - M1 must beat B1 on batter-game AUC in the training inner CV;
   - the training slot 1–4 minus slot 9 predicted P(hit) must be > 0.

## 12. Self-tests before any real run

Lab 1's process is reused with fake outcomes on AB ≥ 1 rows; the real hit column is never loaded.
- **Null study:**
  - 200 seeds (fake seeds 3001–3200), Bernoulli(0.75), 500 bootstrap resamples, parallel.
  - **Pass:** T1 false-confirmation ≤ 8%; T4 false stop ≤ 8%; T1 statistic centred at 0 (|mean| below 0.25 SD).
- **Planted I1:**
  - logit p = logit(0.75) + 0.10 · z(I1), with I1 computed from real feature counts.
  - The pipeline must recover a positive I1 coefficient within ±50% of the planted size, and T1 must be Confirmed.
  - The placebo must not be Confirmed.
- **Planted null-I1:** lab 1's planted formula (slot, K%, H/AB) with no I1 term. T1 must not be Confirmed.
- **Order:** the real hit column stays unread until this plan, the contact table, the scripts and the self-test logs
  are committed.

## 13. Limits and exposure

1. **Seen data.** 2022–2025 outcomes were already seen in lab 1 for M1 and B2.
   - Lab 2 tests one new feature group on one pre-registered metric.
   - But repeated looks at the same seasons raise multiplicity. With lab 1's two confirmatory tests and lab 2's one,
     the chance of at least one false confirmation across both labs is about 1 − 0.975³ = 0.073 if independent.
   - Only 2026, if built, is untouched.
2. **Whole slate.** The pool is the whole slate, which is optimistic, as in lab 1.
3. **Not modelled:**
   - pitch location and sequencing;
   - counts, and how the pitcher's mix changes by count;
   - relievers' arsenals.
4. **Not in the data:** superstition and clubhouse effects.
5. **Contact table.** It is fitted on 2019–21 and ignores park and weather. Expected hits are therefore slightly
   miscalibrated for later seasons and parks; this affects all families alike.
6. **Resolvable difference**, scaled from lab 1's T5 intervals. Those are paired, row-level differences between
   nested models, the same design as T1. The 95% half-widths of the log-loss difference:

   | group dropped | half-width |
   |---|---|
   | D | ±0.00002 |
   | G | ±0.00004 |
   | B | ±0.00007 |
   | E | ±0.00008 |
   | F | ±0.00028 |
   | C | ±0.00033 |
   | A | ±0.00039 |

   The width grows with how much the feature moves predictions.
   - A small matchup feature like I1 should land near the D–E end, about ±0.00003 to ±0.0001.
   - **Resolvable:** a true improvement of about 0.0001–0.0002 in log loss, which is roughly the gain group B (slot)
     gave.
   - **Not resolvable:** improvements below about 0.00005.
   - In top-1 success the test still cannot resolve differences under about 4–5 points (lab 1).

## Amendments before results

(none)

## Addendum

(none)
