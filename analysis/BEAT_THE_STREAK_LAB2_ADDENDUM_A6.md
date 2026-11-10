# Beat the Streak lab 2 — Addendum A6 (Step 1d definitions)

**Dated 2026-11-09. Written after the outcome-free placebo design check, and before the analysis script, any self-test,
model fit or batter-game outcome read.**
- It amends `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`), Sections 2, 7, 8,
  9, 11 and 12, as listed in item 16.
- It adds to addenda A1 (`074878e`, `74e182c7…08ba`), A2 (`478847a`, `f0300c33…55e9`), A3 (`097456c`,
  `21efe8e5…538f`), A4 (`c93c851`, `70cd9322…5af2`) and A5 (`1cb627a`, `3180e9f1…31d8`). None of those files is
  changed.
- **Basis:**
  - the lab 2 feature table `analysis/cache/btslab2_features.parquet` (content sha256
    `62445424bda0c468f7dab50225a5a1d7fb346206e113a5cb18ef7367b7beb8f7`), built and leak-checked in `c8957f2`;
  - the placebo design check `analysis/btslab2_placebo_check.py` (on-disk sha256
    `fd192e81bb874cae3a5b893db7a1e87fa94d08dfbf473ae10f12d6a9a8ab307a`) and its log
    `analysis/out/btslab2_placebo_check_log.txt` (on-disk sha256
    `d76663bbce20e720e8db594fa8ecb09943ac5ff7607e5dd07d188b489ffe9846`; committed blob sha256
    `8148478a8e65411c0c4dc8aa9dcb86fdc3e9670906f5e2196f92c9d1991f83a1`), both committed in `abff0a3`;
  - Claude Code's Step 1d review of 2026-11-09 (gaps G1 to G11) and its placebo-check report.
- The decisions are from Claude (chat) on 2026-11-09.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A6 sha256:`.
- Where this addendum and the plan or addenda A1–A5 differ, this addendum governs.

## Summary in plain language

Lab 2 is about to write its analysis script. Before that, this addendum settles every point the plan and earlier
addenda leave open for that script.

**The placebo changes.** The plan's placebo shuffles only each starter's pitch mix between games. A check that used no
outcomes showed that this barely disturbs the matchup score: the shuffled score tracks the real one closely
(correlation about 0.8, so about two-thirds of its ups and downs are shared). So if the real score worked, the
placebo would seem to work too, and the safety stop would fire for the wrong reason. A rule written before the check
was run picks the replacement: shuffle the starter's whole pitching profile against one side of batters (his mix, and
how much damage each pitch type allows) as one bundle. That shuffled score keeps almost nothing of the real one
(correlation about 0.1, so about 1% shared).

**A prediction is written down** for the made-up-outcome tests before they run.

**The open points of the review are closed:**
- the lab 2 table's own fingerprint is checked before anything is fitted;
- the checks that lab 1 and lab 3 numbers come out the same are skipped on made-up outcomes, as lab 3 did;
- how the exploratory models, the skill-spread diagnostic (T5) and the runtime guard work;
- how two loosely worded expectations are scored;
- one recalibration step that cannot affect any lab 2 result is recorded as unused;
- the missing-pitch report counts lost plate appearances, because lost pitches cannot be measured.

Two small quotation slips in A4 are corrected here. Nothing else changes.

## Items

### 1. The placebo design check and the rule it applied

**Refers to:**
- plan Section 8 T4 (line 240), "Repeat T1 with each starter-game's usage vector u(·|s) permuted across starter-games
  within the same season.";
- A3 item 7 (lines 363–365), "**Shrunk vectors are moved.** The vectors moved are the fold's shrunk u(·|s); the prior
  strengths m are not re-estimated on permuted data. I1 is then recomputed with each row's own r_b(f), r_p(f|s), R_b
  and R_p(s) and the donor's u(·|s)."

**The check.** `analysis/btslab2_placebo_check.py` reads only the lab 2 feature table (counts and league levels), the
plan and addenda A1–A5, and `PLAN_HASH.txt`. It read no outcome column, no box-score file and no 2026 data.
- Folds: the two walk-forward folds; in each, the fold's prior strengths are estimated and applied by the builder's
  own `fold_m()` and `apply_m()`, and the statistics use the test season's rows with I1_miss = 0 (42,759 rows for 2024,
  42,925 for 2025).
- Placebos, both computed from the same shuffle of units for each k = 0..4 (seeds 20261150 + k):
  - **U**, A3 item 7 as written: the donor's u(·|s) only;
  - **P**: the donor's u(·|s), r_p(·|s) and R_p(s) together.
- The real I1 recomputed by the check equals the builder's I1 exactly in both folds.

**The rule, written verbatim at the top of the log before any computation** (log line 4; timestamp line 6, "rule
written at 2026-11-09 17:14:36 -0600; computation starts now"):

"A placebo is usable only if the maximum Pearson correlation between real and placebo I1 over both folds and all five
permutations is at most 0.30. Reason: the planted effect (c = 0.10) gives a log-loss gain of about 0.0009; a placebo
sharing a fraction r^2 of the real feature's variance reproduces about r^2 of that gain, and r^2 of at most 0.09 keeps
it near 0.0001, the test's resolving power. Rule: if U is usable, T4 stays as written. Else if P is usable, P replaces U
as T4 and the placebo for M4, with the other mechanics of A3 item 7 unchanged. Else neither is a usable stop-rule
placebo: both are reported descriptively, and the Step 1d addendum drops the placebo stop rule and the planted test's
'placebo not Confirmed' condition, stating that plainly."

**Results** (log lines 42–49), correlation between real and placebo I1 over both folds and k = 0..4:

| placebo | mean | min | max | usable (max ≤ 0.30) |
|---|---|---|---|---|
| U | 0.7960 | 0.7751 | 0.8166 | no |
| P | 0.1081 | 0.0986 | 0.1175 | yes |

- SD(placebo I1) / SD(real I1): U 1.10 to 1.17; P 1.01 to 1.03.
- At least 99.96% of eligible units received another unit's profile in every shuffle.
- **The branch that applies** (log line 49): "U is not usable; P is usable: P replaces U as T4 and the placebo for M4,
  with the other mechanics of A3 item 7 unchanged."

**Disclosure.**
- The check's first run crashed after the per-fold table of correlations and SD ratios had been printed, and before
  the summary and branch lines. The cause was a column-access bug: `R.groupby("placebo").corr` reached pandas' `corr`
  method instead of the column named "corr".
- Only that access was fixed (now `R.groupby("placebo")["corr"].agg(["mean", "min", "max"])`) and the check was re-run
  in full. The rule, threshold, seeds and computations were not changed, and the re-run printed the same per-fold
  numbers. No separate log of the first run is kept: the re-run wrote the log file again.
- **The per-fold numbers were seen before the branch was printed.** The rule had been fixed in the code and written at
  the top of the log before any computation, so the branch follows from those numbers mechanically.

### 2. Placebo P: the T4 placebo and the M4 placebo for lab 2

**Refers to:**
- plan Section 8 T4 (lines 240–243), "Repeat T1 with each starter-game's usage vector u(·|s) permuted across
  starter-games within the same season.", "Permutation seed 20261150.", "The batter's and pitcher's family skills stay
  with their own rows; only the pitch mix is shuffled." and "I1 and M2 are refitted on the placebo feature.";
- plan Summary (line 33), "A **placebo** shuffles each pitcher's pitch mix between games.";
- A3 item 3 (line 183), "It is a placebo for M4 under the same rule as T4 (item 7).";
- A3 item 7 (lines 348–370):
  - "**Permutations.** The real run averages 50 permutations, seeds 20261150 + k for k = 0..49. Each null-study seed
    uses 10, k = 0..9.";
  - "**What is permuted.** As the plan says: usage vectors u(·|s) are permuted across starter-games within the same
    season, and the batter's and pitcher's family skills stay with their rows.";
  - "**Thin rows.** Rows with I1_miss = 1 keep I1 = 0 and I1_miss = 1, and are neither permuted nor used as donors.";
  - "**Refit.** The placebo refits M2, and M4 for the M4 placebo, at the fold's real penalty C with no inner CV, as lab
    3 did.";
  - "**Unit of permutation.** The (starter-game, side s) pair, permuted within (season, side s).";
  - "**One mapping per k.** One donor mapping per k is shared by both walk-forward folds and by the M2 and M4
    placebos.";
  - "The M2 placebo uses the plan's T4 seed, 20261101 + 4 = 20261105." and "The M4 placebo uses the seed in item 8."
- `analysis/btslab2_features.py` line 181, `F["I1_miss"] = (F.side_s.isna() | F.b_n.eq(0) | F.us_n.eq(0)).astype(int)`,
  where `us_n` is the starter's as-of pitches in the three families to side s.

**Placebo P, in code terms.**
- **Eligible units.** A unit is (`game_pk`, `opp_sp`, `side_s`). It is eligible if `side_s` is known and the starter
  has three-family pitches to side s in the window (`us_n` > 0), as in the check (line 79):
  `el = T.side_s.notna().to_numpy() & (T.us_n.to_numpy() > 0)`. Eligibility depends on the starter only. A unit stays
  eligible when some of its rows have I1_miss = 1 because the batter has no AB-ending pitches.
- **One donor mapping per k**, for k = 0..49 in the real run and k = 0..9 for each null-study seed, with
  `rng = np.random.default_rng(20261150 + k)`.
  - It is drawn once, over all eligible units of all four seasons 2022–2025, sorted by (`game_pk`, `opp_sp`,
    `side_s`).
  - Units are shuffled within (season, side s), the groups taken in sorted order (season ascending, then L before R),
    each by `rng.permutation`, as the check does (lines 91–93): `donor[idx] = idx[rng.permutation(len(idx))]`.
  - The same mapping is used for training and test rows alike, in both walk-forward folds, and by the M2 placebo and
    the M4 placebo.
- **What moves: the starter-side profile, as one bundle.** The recipient unit takes the donor unit's u(·|s),
  r_p(·|s) and R_p(s).
  - These are recomputed per fold: they are the values `apply_m()` gives on the donor unit's rows with that fold's
    prior strengths, from the donor unit's own as-of counts and league levels.
  - The prior strengths m are not re-estimated on shuffled data.
- **What stays.** Each recipient row recomputes I1 with its own r_b(·), R_b and L_f and the donor's bundle, in the
  builder's formula: `I1 += u * Lf * (rb * rp - Rb * Rp)` (`analysis/btslab2_features.py`, `apply_m()`). In the check
  this is line 103: `plP = np.where(keep, i1(du, Lf, rb, drp, Rb, dRp), 0.0)`.
- **Thin rows.** Rows with I1_miss = 1 keep I1 = 0 and I1_miss = 1, and receive no donor value. A3's "neither permuted
  nor used as donors" is read this way: donors are units, and a unit's profile is the starter's, so a row whose batter
  is thin never supplies anything.
- **Refit.** M2 (M2 placebo) and M4 (M4 placebo) are refitted on the placebo I1 at the fold's real penalty C, with no
  inner CV, as A3 item 7 says. Every other column is unchanged.
- **Unchanged from A3 item 7:** the 50 and 10 permutations, the seeds 20261150 + k, the averaging of per-row
  differences over the permutations, the date bootstraps (M2 placebo seed 20261105, M4 placebo seed 20261124) and the
  2.5–97.5 percentile range of the per-permutation statistics.
- **Registered planted runs (new rule; A3 left it unset).** The planted I1 run (seed 3300) and the planted null-I1 run
  (seed 3301) each use 10 permutations, k = 0..9 (seeds 20261150 + k), for the M2 and M4 placebos, like each
  null-study seed. The dry runs keep their 2 permutations (A3 item 4).
- **Accepted by Claude (chat) on 2026-11-09.** Three points above were set by Claude Code, not by the decision
  itself, and were then accepted:
  - the shuffle order: season ascending, then side L before R, over units sorted by (`game_pk`, `opp_sp`, `side_s`);
  - the donor's bundle is computed with the donor's own league levels; only L_f is taken from the recipient row;
  - the reading of A3's "neither permuted nor used as donors" as applying to units (see "Thin rows" above).
- **Correlation logged (descriptive).** The analysis script logs, for each k and each walk-forward fold, the Pearson
  correlation between the real I1 and the placebo-P I1 on the test-season rows with I1_miss = 0, with the mean,
  minimum and maximum over k. No decision is attached to it. If the maximum exceeds 0.30, that is stated as a
  limitation of the placebo, and nothing else changes.

**How this differs from the check.**
- The check drew a new generator per fold and shuffled only that fold's test-season units. The real procedure draws
  one mapping per k over all four seasons and applies it to training and test rows.
- So the real run's mappings are not the check's mappings. The real run's own correlation between real and placebo
  I1 is therefore measured and logged by the analysis script ("Correlation logged" above).
- Both shuffle only within (season, side s). The check's correlation is a property of within-season shuffles of this
  bundle, not of one particular mapping.

**Superseded for lab 2.**
- Plan Section 8 T4, "Repeat T1 with each starter-game's usage vector u(·|s) permuted across starter-games within the
  same season." and "The batter's and pitcher's family skills stay with their own rows; only the pitch mix is
  shuffled.": T4 is now placebo P. The pitcher's family vulnerabilities move with the mix; the batter's skills stay.
- Plan Summary, "A **placebo** shuffles each pitcher's pitch mix between games.": the placebo shuffles the starter's
  profile against one side.
- A3 item 7, the "What is permuted" and "Shrunk vectors are moved" points, as far as they move u(·|s) alone.

**Unchanged.**
- Stop rule 2 (plan line 335), "**Placebo confirmation** (T4 interval entirely below 0) stops before any other result
  is read.", applies to placebo P for M2 and for M4, as A3 item 3 already applies it to the M4 placebo.
- The null-study limits on false stops apply to P unchanged: A3 item 5, "False-confirmation limit ≤ 8% and false-stop
  limit ≤ 8%." and "**PROPOSED (Claude Code):** both limits apply to T1 and T1b, and to the M2 and M4 placebos alike."
- The planted tests' placebo conditions (A3 item 4) and expectation (d) now refer to placebo P.

### 3. Prediction recorded before any dry run

**Refers to:**
- A3 item 4 (lines 216–220), "**Planted I1:** M2's I1 coefficient (per SD, fit on 2022–24) within ±50% of 0.10 (0.05
  to 0.15), **and** T1 Confirmed, **and** the placebo not Confirmed.", "**PROPOSED (Claude Code):** the M4 placebo is
  also not Confirmed." and "**Planted null-I1:** T1 not Confirmed. Added: T1b not Confirmed.";
- A3 item 4 (lines 237–238), "A planted 0.10 per SD would then give a T1 gain of roughly ½ · 0.10² · 0.75 · 0.25 ≈
  0.0009 in log loss".

**Prediction** (scored met or not met; no decision rule):
- With placebo P, a planted effect (c = 0.10) survives the placebo only at about r² of its gain, about 1% (r ≈ 0.11).
  So the planted test's "placebo not Confirmed" condition should be passable for both the M2 and the M4 placebo.
- In the planted-null run, neither T1 nor T1b nor either placebo is Confirmed.

### 4. G1: stop rule 6 gains the lab 2 table

**Refers to:** A3 item 6 (lines 281–287), "Before anything is fitted, the analysis script recomputes and verifies the
content hashes (SHA-256 of `pandas.util.hash_pandas_object(df, index=False)`) of:" and "Any mismatch stops the run."

**The decision.** Stop rule 6 also checks lab 2's table `analysis/cache/btslab2_features.parquet`:
`62445424bda0c468f7dab50225a5a1d7fb346206e113a5cb18ef7367b7beb8f7`. Lab 1's and lab 3's hashes stay as A3 lists them.

### 5. G2: reproduction checks on fake outcomes

**Refers to:** A3 item 6 (lines 290–292), "M1 must reproduce lab 1's printed numbers, and M3 lab 3's, exactly." and
"Any mismatch stops before M2 or M4 is fitted."

**The decision.** Lab 1's and lab 3's printed numbers are real-outcome numbers. On fake outcomes the M1 and M3
reproduction checks are skipped and the skip is logged, as lab 3 did (`analysis/btslab3_analysis.py` line 329):
`say("skipped on fake outcomes (lab 1's numbers are real-outcome numbers); logged as skipped")`. Under `--real`, stop
rule 7 applies as written.

### 6. G3 and G7

- **G3** (penalty, grid and inner CV for M2, M3 and M4): no gap; nothing changes.
- **G7** (T4 donors in code terms): settled by item 2.

### 7. G4: the exploratory models (T6)

**Refers to:**
- plan Section 7 (lines 208–210), "**Exploratory** (labelled, no rules):", "M1 + I1..I4;" and "M1 + I1 (alternative
  grouping).";
- plan Section 6 (lines 189–190), "**Thin history:** if the batter has no AB-ending pitches, or the starter no pitches
  to side s, in the window, then I1 = 0 and the flag `I1_miss` = 1.";
- `analysis/btslab_analysis.py` `design()` (line 159), "Fill missing with the training mean, standardise with training
  mean/SD, prepend an intercept."

**The decision:**
- The exploratory models are fitted at M2's penalty C for the fold, with no inner CV of their own.
- I2, I3 and I4 are NaN on rows with I1_miss = 1, so `design()` fills them with the training mean.
- I1_miss is included in M1 + I1..I4 and in M1 + I1 (alternative grouping).
- The alternative-grouping I1 uses the same I1_miss as the primary I1. The two groupings cover the same pitch codes
  (A5 item 8), so the thin-history condition is the same.

### 8. G5: T5 in code terms

**Refers to:**
- plan Section 8 T5 (lines 249–253), "For batters (per family) and pitchers (per family and side), the method-of-moments
  estimate of the between-unit variance of true family-specific deviations beyond overall skill.", "Data: one row per
  unit-season, walk-forward training data 2022–24, as-of windows.", "Each is reported as variance and as a share of the
  overall-skill spread (SD ratio)." and "Point estimates; date-bootstrap intervals, seed 20261106.";
- A3 item 13e (lines 601–602), "**Decided (amendment 2):** T5 resamples units (players: batters, or pitchers), not
  dates. All of a unit's seasons are carried together, with 2,000 resamples and seed 20261106.";
- A5 item 8, "T5's method-of-moments estimate of the between-unit variance of true family-specific deviations uses the
  same pooled within-unit variance cells as the estimator in item 3, instead of the binomial bound: same x, same rows,
  same cells." and "The SD ratio is the SD of the family-specific between-unit deviations divided by that overall
  between-unit SD.";
- A5 item 3, "**Everything else in `estimate_m` is unchanged:** the nesting correction, the rows, the moment condition,
  and m = ∞ when the between-unit variance is not positive."; the moment line in `analysis/btslab_features.py` (line
  363) is `tau2 = ((n * (x / n - mu) ** 2).sum() - (w * c).sum()) / (n * c ** 2).sum()`.

**The decision:**
- **Data.** The walk-forward training rows of 2022–24 with their as-of windows, one row per unit-season selected as
  for the prior strengths (A3 item 13, amendment 8).
- **The estimator returns the between-unit variance** (τ² above) per cell, not only m, because T5 needs a separate
  variance per family (batters) and per family and side (pitchers), and the overall variance for R_b and per side for
  R_p(s).
- **Unclipped.** Each per-cell between-unit variance is reported as computed, including values at or below 0.
- **SD ratio.** Per cell, the SD of the family-specific between-unit deviations divided by the overall between-unit
  SD (A5 item 8). It is 0 when the family-specific variance is not positive. The number of resamples in which that
  happens is reported per cell.
- **Denominator not positive.** When a resample's overall-skill variance for a cell's denominator is not positive,
  that resample's ratio for that cell is undefined. It is left out of that cell's interval, and the number of such
  resamples is reported per cell, alongside the count of resamples in which the family-specific variance is not
  positive.
- **Bootstrap.** Unit resamples as A3 item 13e: 2,000, seed 20261106. The overall-skill SD in the denominator is
  computed on the same player resamples as the numerator, and the ratio is taken per resample.

### 9. G6: T6 also reports M1 + I1..I4 against M2

**Refers to:**
- plan Section 8 T6 (lines 257–258), "Log loss and top-1 success for the exploratory models in Section 7 against M1."
  and "Date bootstrap, seed 20261107.";
- plan Section 9 (line 292), "**(e)** I4 adds nothing beyond I1. In T6, M1 + I1..I4 vs M2: the log-loss difference
  interval includes 0."

**The decision.** T6 also reports M1 + I1..I4 against M2 (log loss, paired by row, date bootstrap, seed 20261107), for
scoring expectation (e) only. T6's comparisons against M1 stay as written.

### 10. G8: the runtime guard before the null study

**Refers to:** A3 item 9 (lines 426–428), "Before the null study starts, the script reports a timed single fit and a
runtime projection, as lab 3 did with `--time`." and "It refuses to start the 200-seed study without that report."

**The decision:**
- The timing run writes `analysis/out/btslab2_time_log.txt`, and that log records the on-disk SHA-256 of
  `analysis/btslab2_analysis.py`.
- The 200-seed null study refuses to start unless that log exists and the SHA-256 it records equals the current
  script's.

### 11. G9: scoring of the expectations

**Refers to:** plan Section 9 (lines 287–296):
- "**(a)** T1 improvement small: log loss(M1) − log loss(M2) between 0 and 0.0004, with about a 35% chance the interval
  excludes 0."
- "**(b)** The I1 coefficient is positive and small: under 0.03 per standard deviation."
- "**(c)** Top-1 success M2 − M1 within ±2 points, and no confirmation."
- "**(d)** The placebo (T4) is within noise of 0: its interval includes 0."
- "**(e)** I4 adds nothing beyond I1. In T6, M1 + I1..I4 vs M2: the log-loss difference interval includes 0."
- "**(f)** T5: family-specific spread is real but modest — positive, under half of the overall-skill spread (SD ratio
  < 0.5) — and larger for pitchers than for batters."
- "**(g)** Lab 2 is unlikely to turn the pooled edge over log5 into a confirmed one. M2 − B2 top-1 (T3) is not
  confirmed by the T1-style rule."

and A3 item 3 (lines 187–189):
- "**(h)** T1b improvement, loss(M3) − loss(M4), is between 0 and 0.0003, with about a 25% chance the interval excludes
  0."
- "**(i)** The T1b improvement is no larger than the T1 improvement: loss(M3) − loss(M4) ≤ loss(M1) − loss(M2)."

**The decision:**
- **(a) and (h):** the "% chance" parts are probabilities, not outcomes, and are not scored. The rest is scored
  literally.
- **(c):** met if the T3 M2 − M1 top-1 difference is within ±0.02 and its 95% interval includes 0.
- **(f):** met if every T5 cell has a point variance greater than 0, every T5 SD ratio is below 0.5, and the mean
  pitcher-cell SD ratio is above the mean batter-cell SD ratio.
- **(d):** scored literally on placebo P (item 2), as A3 item 5 says: "It is scored literally."
- **All others** are scored literally.

### 12. G10: the recalibration of daily top picks is unused

**Refers to:**
- plan Section 7 (line 215), "**Recalibration** of daily top picks: as in A2 change a (slope constrained to [0, 1]).";
- A3 item 3 (line 167), "T3b follows T3 exactly, including its recalibration."

**The decision.** The recalibration is a monotone map, so it cannot change a top-1 pick. Lab 2 has no day tiers and
no streak simulation, and its final test is withdrawn (A3 item 2). The recalibration therefore affects no lab 2
output, and lab 2 does not compute it. This supersedes the plan's recalibration for lab 2 and A3's "including its
recalibration" for T3b.

### 13. G11: the `I_gap` report

**Refers to:**
- plan Section 2 (lines 81–85), "**Games with dropped pitches** are flagged. These are games where box-score PA ≠
  pitch-row PA − truncated_pa − IBB: 276 of 9,704 in 2022–25, plus any found in 2019–21. Their remaining pitch rows are
  used.", "The flag `I_gap` is carried for the row's own game but is never a model input." and "The count of history
  pitches lost is reported.";
- A4 item 1 (lines 40–43), "A separate reporting-only script, named and written later with Step 1d:", "reads only the
  team PA and intentional-walk fields from the box-score JSON, never the batting hits field;", "produces the `I_gap`
  flag and the history-pitches-lost count exactly as plan Section 2 defines them;" and "writes its own separate file."

**The decision:**
- **Files.** `analysis/btslab2_igap_report.py` writes `analysis/out/btslab2_igap_log.txt` and
  `analysis/cache/btslab2_igap_flags.parquet` (git-ignored; its content hash is logged).
- **What it reports.** Plate appearances lost per game (box-score PA − (pitch-row PA − truncated_pa − IBB)), and
  plate appearances lost over history windows.
- **Wording replaced.** The plan asks for history pitches lost: "The count of history pitches lost is reported." That
  cannot be measured from box scores, which give plate appearances, not pitches. A6 replaces it with plate appearances
  lost. This also replaces A4 item 1's "history-pitches-lost count".
- **What it reads.** The parser loads each whole box-score file, but reads only the team plate-appearance and
  intentional-walk fields and discards the rest.
- **When.** Any time before `--real`. It is never a model input.

### 14. Carry-over corrections to A4 and A5 (neither file is edited)

**a. A4 line 65.** A4 shows lab 1's call as an assignment, `m["hab"] = bf.estimate_m(f(bat, "a_h"), f(bat, "a_ab"),
f(bat, "lg_h_ab"))`. The source, `analysis/btslab_analysis.py` line 89, is a dict entry:
`m = {"hab": bf.estimate_m(f(bat, "a_h"), f(bat, "a_ab"), f(bat, "lg_h_ab")),`. The call and its arguments are the
same, so A4's meaning is unchanged.

**b. A4 line 140.** A4 has "No pitches to side s" with a capital N. Plan line 189 has it lowercase: "or the starter no
pitches to side s".

**c. A5's header.** A5's header (lines 18–21) describes item 8 as of its 16:03 revision: "Item 8 adds T5 to the
replacement, and lists every other place the binomial variance meets an x-based measure." It does not mention the
later additions of T5's overall-skill spread and SD ratio and of the alternative grouping. Those are in A5 item 8 and
in the superseded-before-publication lines of `analysis/out/PLAN_HASH.txt` for A5.

### 15. Order of work and commit points

1. A6. Commit.
2. `analysis/btslab2_analysis.py` with its unit checks, the smoke run (seed 9109) and the timing log
   `analysis/out/btslab2_time_log.txt`. Commit.
3. Throwaway dry runs, seeds 9101–9108. Commit the logs. If they show the generator misbehaving, an addendum comes
   first (A3 item 4).
4. Registered planted I1 (seed 3300) and planted null-I1 (seed 3301). Commit.
5. Null study, seeds 3001–3200 (about 70 minutes). Commit.
6. The `I_gap` script and its log (item 13). Commit.
7. `--real`, once.

### 16. Scope: what changes

- **T4 placebo form:** placebo P replaces placebo U as T4 and as the M4 placebo (items 1–2). The plan's T4 usage
  permutation is superseded for lab 2.
- **Stop rule 6** gains the lab 2 table's content hash (item 4).
- **The reproduction skip** on fake outcomes (item 5).
- **The T5 and T6 definitions** (items 7–9).
- **The scoring of (c) and (f)**, and the unscored "% chance" parts of (a) and (h) (item 11).
- **The recalibration is unused** (item 12).
- **The `I_gap` wording:** plate appearances lost, not pitches (item 13).
- **The registered planted runs** use 10 placebo permutations each (item 2).
- **T5's undefined ratios:** a resample whose denominator variance is not positive is left out of that cell's
  interval and counted (item 8).
- The logged real-versus-placebo-P correlation (item 2) is descriptive and changes no rule.
- The runtime guard (item 10), the prediction (item 3), the corrections (item 14) and the order of work (item 15)
  add records and procedure; they change no rule.
- No other rule, test, threshold, seed or expectation changes.

## Unchanged

- Items 1–3 record the placebo design check, define placebo P (with its permutations for the registered planted runs
  and its logged correlation) and record a prediction.
- Items 4–13 close the Step 1d review's gaps G1–G11.
- Item 14 corrects A4's two quotation slips and notes A5's header.
- Items 15–16 set the order of work and list what changes.
- Everything else in the lab 2 plan and addenda A1–A5 stays as written.

## Addendum

(none)
