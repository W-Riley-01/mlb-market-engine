# Beat the Streak lab 2 — Addendum A5 (per-trial variance of the per-AB measure)

**Dated 2026-11-09. Written after the first Step 1c build stopped with I1 identically zero, and before any leak check,
batter-game outcome read or model fit.**
- It amends `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`), Section 6, for
  lab 2's x-based measures only.
- It adds to addenda A1 (`074878e`, `74e182c7…08ba`), A2 (`478847a`, `f0300c33…55e9`), A3 (`097456c`,
  `21efe8e5…538f`) and A4 (`c93c851`, `70cd9322…5af2`). None of those files is changed.
- **Basis:**
  - the first-run build log `analysis/out/btslab2_features_build_log_run1_STOPPED.txt`, renamed unchanged from
    `btslab2_features_build_log.txt` (on-disk sha256 `e8693a30dedb475b12f2ae97ddc225d422ce8e37ccecc59ab956b5a70a6bf2e2`);
  - the first-run table `analysis/cache/btslab2_features.parquet` (content sha256
    `d300f4e912c77050c9fac97b7f3596c1df08575f844a5f72e3a84beb95a36b50`);
  - the variance diagnosis `analysis/btslab2_variance_diagnosis.py` and its log
    `analysis/out/btslab2_variance_diagnosis_log.txt`;
  - Claude Code's Step 1c report of 2026-11-09.
- The decisions are from Claude (chat) on 2026-11-09.
- **Amended before publication on 2026-11-09.**
  - Item 8 adds T5 to the replacement, and lists every other place the binomial variance meets an x-based measure.
  - Item 6 and "Unchanged" are updated to match.
  - The superseded version had sha256 `c7cd5b64…57eb`, frozen 2026-11-09 15:58:07 -06:00.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A5 sha256:`.
- Where this addendum and the plan or addenda A1–A4 differ, this addendum governs.

## Summary in plain language

The first build of lab 2's features gave a matchup score that is exactly zero for every batter-game.

**Why.** The smoothing step decides how much a player's own record counts against the league average. To do that it
must know how noisy a single at-bat's value is. The plan used a safe upper limit for that noise. The value of an at-bat
here is an expected-hit probability, which is far less noisy than a plain hit-or-out, and the limit was about 2.4 times
too large. Measured that way, every difference between players looked like noise, so the smoothing pulled everyone all
the way to the average and the score vanished.

**The fix.** Use the noise actually measured inside each player's own at-bats.
- With this noise term given the old limit, the estimator must reproduce the old result exactly.
- The choice is made on statistical grounds, before the rebuilt feature is seen, and will not be revisited after.

## Items

### 1. Finding

**Refers to:**
- plan Section 6, "Family measures are subsets of the overall window, so they use lab 1's nested-variance estimator
  (`btslab_features.estimate_m` with N = all AB-ending pitches; A2 change c).";
- plan Section 6, "The binomial variance μ(1 − μ) is used as an upper bound for the per-trial variance, which can only
  add shrinkage."

**In the first-run build log:**
- the prior strengths of both pooled family estimators, r_b(f) and r_p(f|s), are m = ∞ in all six folds;
- R_p(s)'s is m = ∞ in 4 of 6 folds;
- R_b's is finite (857.6 to 1,034).

**Consequence.** With the family m = ∞, r_b(f) = R_b and r_p(f|s) = R_p(s) on every row. So I1 = Σ_f u(f|s) · L_f ·
(r_b(f) · r_p(f|s) − R_b · R_p(s)) = 0 on every row. The log shows "I1 mean … SD 0.000000" in every fold.

**Why m is infinite.** Shown for the walk-forward train-2022–24 fold, on rows with n ≥ 100:

| estimator | Σ n·(x/n − μ)² | Σ μ(1 − μ)·c | ratio | m |
|---|---|---|---|---|
| R_b | 382.0 | 272.0 | 1.41 | finite |
| R_p(s) | 245.2 | 247.9 | 0.99 | ∞ |
| pooled r_b(f) | 117.1 | 205.5 | 0.57 | ∞ |
| pooled r_p(f|s) | 105.2 | 113.6 | 0.93 | ∞ |

- When the left column is not larger than the noise term, the estimated between-unit variance is not positive, and
  `estimate_m` returns m = ∞.
- **Source.** This table was computed ad hoc in Claude Code's Step 1c report, from the first-run table's stored counts
  for the train-2022–24 fold. It is not in a log.

**The size of the bound.** From the variance diagnosis log, on 2022–2023 AB-ending pitches (327,250, the training
window of the first walk-forward fold):
- the per-trial variance of x divided by the binomial bound μ(1 − μ) is 0.37 to 0.44 in every cell;
- overall it is 0.0776 / 0.1849 = **0.42**.

| estimator | cell | units | within-unit variance | league variance | μ(1 − μ) | within / bound |
|---|---|---|---|---|---|---|
| R_b | all | 906 | 0.0775 | 0.0776 | 0.1849 | 0.419 |
| R_p(s) | side L | 424 | 0.0769 | 0.0772 | 0.1844 | 0.417 |
| R_p(s) | side R | 763 | 0.0771 | 0.0778 | 0.1853 | 0.416 |
| r_b(f) | fb | 719 | 0.0809 | 0.0803 | 0.1941 | 0.417 |
| r_b(f) | br | 473 | 0.0753 | 0.0737 | 0.1700 | 0.443 |
| r_b(f) | os | 42 | 0.0733 | 0.0723 | 0.1791 | 0.409 |
| r_p(f|s) | fb/L | 192 | 0.0804 | 0.0806 | 0.1932 | 0.416 |
| r_p(f|s) | fb/R | 364 | 0.0801 | 0.0801 | 0.1946 | 0.412 |
| r_p(f|s) | br/L | 37 | 0.0729 | 0.0729 | 0.1673 | 0.435 |
| r_p(f|s) | br/R | 160 | 0.0739 | 0.0741 | 0.1715 | 0.431 |
| r_p(f|s) | os/L | 29 | 0.0676 | 0.0712 | 0.1814 | 0.373 |
| r_p(f|s) | os/R | 43 | 0.0712 | 0.0733 | 0.1768 | 0.403 |

Units in the diagnosis are calendar unit-seasons with at least 100 AB-ending pitches in the cell. The diagnosis is
descriptive, with no fit and no m.

**Disclosure.**
- Claude Code's earlier ad hoc diagnosis, in its Step 1c report, used 2022–2024 AB-ending pitches. That includes 2024,
  the first walk-forward fold's test season.
- It was a league-level description of the feature's inputs, x from the vault's pitch history: no model was fitted
  and no batter-game label was read.
- The diagnostic log now uses 2022–2023 only.

### 2. Why the bound is valid but too loose here

**Refers to:**
- plan Section 5, "**Ball in play** (`type == "X"`, including home runs and reaching on error): the expected-hit value
  from Section 3.";
- plan Section 5, "**Strikeout:** 0 (strikeout, strikeout_double_play)." and "Any other AB-ending event without a ball
  in play is 0."

**The bound holds.** x lies in [0, 1], so x² ≤ x and E[x²] ≤ E[x]. Hence Var(x) = E[x²] − μ² ≤ μ − μ² = μ(1 − μ),
with equality only when x takes the values 0 and 1 alone.

**But it is loose.** For a plain hit indicator the bound is exact. For an expected-hit value, most balls in play take
fractional values, so the variance is well below the bound: 0.42 of it here.

**Effect.** In the moment estimator, an inflated noise term is subtracted from the observed spread between units.
Where the true between-unit spread is small relative to the noise, as for family-specific deviations, the remainder
falls to zero or below, and m becomes infinite. So the plan's "can only add shrinkage" holds, but here the extra
shrinkage is total.

### 3. Replacement rule

**Refers to:**
- plan Section 6, "The binomial variance μ(1 − μ) is used as an upper bound for the per-trial variance, which can only
  add shrinkage.";
- A3 item 13a, "**`min_n` = 100 for every lab 2 estimator (amendment 3),** in the estimator's own trial unit (item
  b).", "**If no row reaches 20 either,** `estimate_m` returns m = ∞, full shrinkage to the target, by its own code"
  and "**Logging (amendment 10).** The builder logs, per fold and per cell (estimator × family × side where
  applicable)";
- A4 item 2, "**Each of R_b and R_p(s) gets its own m,** estimated the way lab 1 estimates m for its overall rates";
- A4 item 3, "**min_n.** A3 item 13a's rules apply to these m's as well, with their logging: `min_n` = 100; if no row
  reaches it, `min_n` = 20; if none reaches that, m = ∞."

**The rule:**
- **What changes.** In the noise term of the method-of-moments estimator, the binomial variance μ(1 − μ) is replaced
  by the pooled within-unit sample variance of x.
- **Where it is estimated.** On the same qualifying training-fold rows the estimator already uses (A3 item 13a: min_n
  = 100, then 20, then m = ∞, with the same logging).
- **Per cell:**
  - R_b: one cell;
  - R_p(s): one cell per side;
  - r_b(f): one cell per family;
  - r_p(f|s): one cell per family and side.
- **The formula.** Pooled within-unit variance of a cell = the sum over units of (Σx² − n · (unit mean)²), divided by
  the sum over units of (n − 1).
- **Pooled estimators.** Each stacked row uses its own cell's variance in place of μ(1 − μ).
- **Everything else in `estimate_m` is unchanged:** the nesting correction, the rows, the moment condition, and m = ∞
  when the between-unit variance is not positive.
- **Scope.** The same replacement applies to R_b, R_p(s) and the two pooled family estimators, because all four use
  the same x. The usage shrinkage, u(f) and u(f|s), is on counts of pitches and is unchanged.

**Where the noise variance enters `estimate_m`.** In `analysis/btslab_features.py`:

```python
360    k = (n >= min_n) & np.isfinite(mu) & np.isfinite(c)
361    x, n, mu, c = x[k], n[k], mu[k], c[k]
362    w = mu * (1 - mu)
363    tau2 = ((n * (x / n - mu) ** 2).sum() - (w * c).sum()) / (n * c ** 2).sum()
364    return float(w.mean() / tau2) if tau2 > 0 else np.inf  # no between-unit spread: shrink fully
```

**The change.** `btslab_features.py` is not edited. A new function in `analysis/btslab2_features.py` takes one more
argument, v (the per-row variance from the rule above), subsets it with the other arrays on line 361, and changes
line 362:

```python
362    w = v          # was: w = mu * (1 - mu)
```

- Lines 363 and 364 are unchanged.
- Line 364 uses the same w, so the prior strength is m = mean(v) / τ²: the per-trial variance over the between-unit
  variance, as before with the bound.

**Equivalence assertion.**
- When the new function is given v = μ(1 − μ), it must reproduce `bf.estimate_m` exactly on every current input:
  both infinite, or a relative difference of at most 1e-12.
- The builder asserts this for every estimator call in every fold.

**Logging.** The build log shows, per fold and cell:
- the variance used;
- its ratio to the mean binomial bound μ(1 − μ) over that cell's qualifying rows.

### 4. Table change

**Refers to:** A3 item 13, "As in labs 1 and 3, the feature table stores as-of counts and league levels; shrinkage,
standardisation and fills are applied per training fold at fit time."
- The table stores the as-of sum of x² next to every stored sum of x, for both groupings, as counts.
- That covers the batter, the starter to side s and the league, overall and per family.
- Nothing else in the table changes.
- The table is rebuilt after this addendum is public. The first-run content hash above identifies the table it
  replaces.

### 5. Selection and stopping

- **Grounds.** The estimator is chosen on statistical grounds, not on the size of the resulting I1 spread.
- **No revision.** It will not be revised after the rebuilt table's I1 spread is seen.
- **Stop rule.** If any prior strength for R_b, R_p(s), r_b(f) or r_p(f|s) is still infinite in any fold after the
  rebuild, Step 1c stops and reports without choosing a fix.

### 6. Scope

- The plan sentence "The binomial variance μ(1 − μ) is used as an upper bound for the per-trial variance, which can
  only add shrinkage." is superseded for lab 2's x-based measures only: R_b, R_p(s), r_b(f) and r_p(f|s).
- It is also superseded for T5's method-of-moments estimate (item 8).
- Nothing else changes: no rule, test, threshold, seed or expectation.

### 7. Files

- `analysis/btslab2_features.py` and `analysis/btslab2_leakcheck.py` are kept as written in the first run and are
  revised under this addendum.
- `analysis/out/btslab2_features_build_log_run1_STOPPED.txt` is the first-run build log, renamed without change.
- The leak checks have not been run.

### 8. T5, and every other use of the binomial variance on an x-based measure

**Refers to:** plan Section 8 T5:
- "**T5 — DIAGNOSTIC: how much family-specific skill exists.**"
- "For batters (per family) and pitchers (per family and side), the method-of-moments estimate of the between-unit
  variance of true family-specific deviations beyond overall skill."
- "Data: one row per unit-season, walk-forward training data 2022–24, as-of windows."
- "Each is reported as variance and as a share of the overall-skill spread (SD ratio)."
- "Point estimates; date-bootstrap intervals, seed 20261106."
- "**Stated plainly:** if the shares are near zero, there is nothing for the fit score to find."

**The decision.**
- T5's method-of-moments estimate of the between-unit variance of true family-specific deviations uses the same
  pooled within-unit variance cells as the estimator in item 3, instead of the binomial bound: same x, same rows, same
  cells.
- That means one cell per family for batters, and one per family and side for pitchers.

**T5's overall-skill spread.**
- It is the between-unit spread of overall skill, estimated by the same method as item 3 with the within-unit
  variance:
  - one cell for batters (R_b);
  - one cell per side for pitchers (R_p(s)).
- The SD ratio is the SD of the family-specific between-unit deviations divided by that overall between-unit SD.

**Alternative grouping.**
- **Refers to:** plan Section 6, "**Alternative grouping:** FC with breaking, FS with fastball; I1 recomputed with
  it."
- The alternative grouping re-estimates every prior strength whose input depends on the grouping, with the alternative
  families as cells, using the within-unit variance per alternative-family cell.
  - That covers usage levels 1 and 2 (on counts, unchanged in kind), pooled r_b(f) and pooled r_p(f|s).
  - Item 4 stores the sums of x² for both groupings.
- **R_b and R_p(s) are reused, not re-estimated.** The plan's wording moves FC and FS between families and drops no
  code, so the alternative grouping covers the same pitches as the primary one (checked: the code sets are identical).
  The pitches that R_b, R_p(s) and L count therefore do not change.

**Search.** The lab 2 plan, addenda A1–A4 and `analysis/btslab2_features.py` were searched for "binomial", "μ(1 − μ)",
"mu * (1 - mu)", "per-trial variance", "method of moments", "method-of-moments", "estimate_m" and "variance". Every
place where the binomial variance is, or could be, applied to an x-based measure:

| # | location | text or code | does the A5 replacement apply? |
|---|---|---|---|
| 1 | plan Section 6, Shrinkage (lines 178–179) | "The per-AB measure is not 0/1 for balls in play. The binomial variance μ(1 − μ) is used as an upper bound for the per-trial variance, which can only add shrinkage." | Yes. This is the sentence superseded by item 3. |
| 2 | plan Section 6, Shrinkage (lines 176–177) | "Family measures are subsets of the overall window, so they use lab 1's nested-variance estimator (`btslab_features.estimate_m` with N = all AB-ending pitches; A2 change c)." | Yes: the nested estimator is kept, and its noise term is replaced (item 3). |
| 3 | plan Section 6, Shrinkage (line 180) | "For usage, one m per level: the n-weighted mean of the three families' binomial moment estimates." | No. Usage is counts of pitches, not x (item 3: "The usage shrinkage, u(f) and u(f\|s), is on counts of pitches and is unchanged."). |
| 4 | plan Section 8 T5 (lines 249–250) | "the method-of-moments estimate of the between-unit variance of true family-specific deviations beyond overall skill" | Yes, by this item. |
| 5 | plan Section 8 T5 (line 252) | "Each is reported as variance and as a share of the overall-skill spread (SD ratio)." | Yes: the overall-skill spread is estimated with the within-unit variance, one cell for R_b and one per side for R_p(s) (see "T5's overall-skill spread" above). |
| 6 | A3 item 13 (lines 514–515) | "It calls `btslab_features.estimate_m(x, n, mu, min_n=100, N=None)`." and "That function is the beta-binomial method of moments with per-trial variance μ(1 − μ)." | A description of lab 1's precedent, not a rule. The lab 2 x-based calls are replaced (item 3); lab 1's own calls (M1) are not touched. |
| 7 | A3 item 13b (lines 556–570) | "**b. Usage shrinkage, both levels.** The trial unit is pitches in the three families (codes per item c)." and "the n-weighted mean of the three families' binomial moment estimates" | No. Usage is counts, as row 3. |
| 8 | A4 item 2 (lines 64–71) | "**R_b:** `estimate_m(x = the batter's Σx, n = his AB-ending pitches, mu = L)`" and "**R_p(s):** `estimate_m(x = the starter's Σx allowed to side s, …, mu = L)`" | Yes (item 3). |
| 9 | A4 item 3 (lines 89–107) | "`estimate_m(x = Σx in family f, n = AB-ending pitches in f, mu = the batter's raw overall measure × L_f / L, N = all his AB-ending pitches)`" and the pitcher call | Yes (item 3). |
| 10 | `analysis/btslab2_features.py`, `est()` (lines 196–206) | `m = bf.estimate_m(x, n, mu, min_n=mn, N=Nn) if (k100 or k20) else np.inf`, called by `fold_m()` for usage levels 1–2, R_b, R_p(s), pooled r_b and pooled r_p | Yes for the R_b, R_p(s), pooled r_b and pooled r_p calls, which are revised under item 3. No for the two usage calls. |
| 11 | plan Section 6, exploratory I2, I3, I4 | "**I2** = Σ_f u(f\|s) · r_b(f)", "**I3** = Σ_f u(f\|s) · r_p(f\|s)", "**I4** … = max_f u(f\|s) · (r_p(f\|s) − 1) · r_b(f)" | Yes, through r_b(f) and r_p(f\|s). They have no variance term of their own. |
| 12 | plan Section 6, alternative grouping | "**Alternative grouping:** FC with breaking, FS with fastball; I1 recomputed with it." | Yes: grouping-dependent prior strengths are re-estimated with alternative-family cells; R_b and R_p(s) are reused (see "Alternative grouping" above). |
| 13 | A3 item 4, planted generator | "z(I1) is M2's own shrunk I1 column, computed with the train-2022–24 fold's prior strengths and standardised on that fold." | Yes, through I1. The generator has no variance term of its own. |
| 14 | A3 item 7, T4 placebo | "The vectors moved are the fold's shrunk u(·\|s); the prior strengths m are not re-estimated on permuted data." | Yes, through the fold's r_b(f), r_p(f\|s), R_b and R_p(s). The permutation adds no variance term. |
| 15 | plan Section 12, null study | "200 seeds (fake seeds 3001–3200), Bernoulli(0.75), 500 bootstrap resamples, parallel." | No variance term on x. The features it uses come from the estimators of item 3. |

A1 and A2 contain no hit.

## Unchanged

- Items 1–7 record the finding and replace the per-trial variance in the estimators of lab 2's x-based measures.
- Item 8 applies the same replacement to T5's method-of-moments estimate, and lists every other place the bound meets
  an x-based measure.
- Everything else in the lab 2 plan and addenda A1–A4 stays as written.

## Addendum

(none)
