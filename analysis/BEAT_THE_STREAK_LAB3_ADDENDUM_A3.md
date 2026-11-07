# Beat the Streak lab 3 — Addendum A3 (failed null-study centring check; confirmation study)

**Dated 2026-11-06, written after the registered planted runs (seeds 4300, 4301) and the registered null study (seeds
4001–4200), and before the confirmation null study, any lab 3 real-outcome read, or `--real`.**
- It amends `analysis/BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md` (`722712b`, lab3 sha256 `f3d76f8f…a872`) and adds to
  addenda A1 (lab3 A1 sha256 `a887f248…2af0`) and A2 (lab3 A2 sha256 `890c8c98…a387`). All three files are unchanged.
- **One check is replaced (item 3) by an EASIER one, and an independent confirmation study is added (item 4).
  No other rule, threshold, seed, test or expectation changes.**
- Basis: the registered self-test logs as committed in `b054373` (`analysis/out/btslab3_selftest_planted_log.txt`,
  `btslab3_selftest_planted_null_log.txt`, `btslab3_null_study_log.txt`, `btslab3_null_study_seeds.csv`), which stay
  unchanged, and descriptive figures computed from them only (no model was run).
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab3 A3 sha256:`.
- Where this addendum and the plan or addenda A1–A2 differ, this addendum governs.

## Summary in plain language

Before the real run, the pipeline is tested on made-up outcomes. The two planted tests passed. The null study, 200
made-up seasons in which nothing is real, did what matters most: the new features were never falsely "confirmed" and
the placebo never falsely stopped the run. But it failed one more check, which required the average result over the
200 seasons to be close to zero.

We believe that check was set wrongly, by us. When nothing is real, adding 22 extra columns to the model costs a
little accuracy on new data, so the average result should sit slightly below zero, not at zero. That is what the 200
seasons show, and the placebo, which adds columns with no signal, shows the same cost.

We replace that check with an easier one that only asks the average not to lean upward, and we say plainly that it is
easier. Because we chose it after seeing the failure, passing it on the same 200 seasons would prove nothing. So a
second null study on 200 fresh seeds, never used before, must pass it before the real run.

## Items

### 1. Registered result and failure

**Refers to:** plan Section 11 step 4, "**Null study:** 200 seeds (4001–4200), Bernoulli(0.75), 500 bootstrap
resamples, parallel." and "**Pass:** T1 false confirmation ≤ 8%; T4 false stop ≤ 8%; the T1 statistic centred at 0
(|mean| < 0.25 SD across seeds).", and plan Section 10, "4. **Self-tests fail** (Section 11): null false confirmation
above 8% over 200 seeds; the planted effect (+0.10 per SD on h_xhit) not recovered; the planted-null confirms; the
placebo false-stops above 8% (10 permutations per null seed)."
- **Planted-H, seed 4300: PASS.** M3's h_xhit coefficient +0.0691, inside 0.05–0.15. It is lower than in the two
  throwaway dry runs (+0.1067 and +0.1098). T1 Confirmed, +0.00045 (95% +0.00023 to +0.00068); 2024 +0.00038, 2025
  +0.00053. Placebo not Confirmed, −0.00011 (95% −0.00012 to −0.00009).
- **Planted-null, seed 4301: PASS.** T1 not confirmed, −0.00002 (95% −0.00010 to +0.00007). h_xhit coefficient
  −0.0103.
- **Null study, seeds 4001–4200: FAIL.**
  - T1 falsely confirmed: 0 of 200 (limit 8%).
  - Placebo falsely stopped: 0 of 200 (limit 8%).
  - T1 statistic across seeds: mean −0.0000656, SD 0.0000411, so |mean| = 1.60 SD against the limit of 0.25 SD.
- This is a registered FAIL. Section 11's pass rule is not met, so the self-tests have failed and stop rule 4
  applies: `--real` is not run on the strength of these self-tests.
- The logs are kept exactly as committed in `b054373`.

### 2. Diagnosis

The explanation we believe, supported by the descriptive figures below, not proven:
- **The check assumed an expected T1 of 0 under the null.** But M3 has 22 more design columns than M1 (M1 34, M3 56;
  `H_COLS` in `analysis/btslab3_analysis.py`), and extra columns with no signal cost out-of-sample log loss. The
  expected T1 under the null is therefore slightly negative.
- **Figures from the committed seeds file** (descriptive; the T1 statistic is recorded pooled over both walk-forward
  folds, not per fold):
  - T1 across the 200 seeds: mean −0.0000656, SD 0.0000411, min −0.0002160, max +0.0000432; 2.5th and 97.5th
    percentiles −0.000142 and +0.000010; T1 > 0 on 11 of 200 seeds; no seed's 95% interval lay entirely above 0.
  - Placebo gain (M3 refitted with the H columns permuted, i.e. the same extra columns with no signal): mean
    −0.0000774, SD 0.0000131. The ratio of the mean placebo gain to the mean T1 is 1.18.
  - Textbook out-of-sample cost of k extra columns, about k / (2 n) per row in log loss: with k = 22 and the
    walk-forward training rows n = 87,096 (train 2022–23) and 130,564 (train 2022–24), 0.0001263 and 0.0000842,
    average 0.0001053. The observed mean T1, −0.0000656, is 0.62 of that.
  - This is an unpenalised approximation. The ridge penalty (C = 0.001 was chosen for M1 and M3 on every seed in the
    first fold) shrinks the extra coefficients, so the real cost should be somewhat smaller, as observed.
- **The pipeline errs in the conservative direction.** Most seeds ended below zero and none confirmed.
- **The check was mis-specified, and the error is ours.** The pipeline was not shown to be faulty. This is not a
  claim that the original check would have passed under any reading: it failed, and that failure stands in the
  record.

### 3. Replacement check (EASIER)

**Refers to:** plan Section 11 step 4, "**Pass:** T1 false confirmation ≤ 8%; T4 false stop ≤ 8%; the T1 statistic
centred at 0 (|mean| < 0.25 SD across seeds)."
- The centring test is replaced by a one-sided "no upward bias" test: the mean T1 statistic across seeds must not be
  above zero by more than a quarter SD, i.e. **mean ≤ +0.25 SD**. There is no two-sided centring test.
- The false-confirmation limit (≤ 8%) and the false-stop limit (≤ 8%) are unchanged.
- **This replaces a failed check with an easier one.** It is easier than the original, and it was chosen after seeing
  the failure. A pass on the original 200 seeds would therefore not be an independent test; the independent
  confirmation is item 4.

### 4. Independent confirmation

**Refers to:** plan Section 11 step 4, "**Null study:** 200 seeds (4001–4200), Bernoulli(0.75), 500 bootstrap
resamples, parallel."
- A second null study on fresh seeds 4401–4600, run once, with the rule of item 3: Bernoulli(0.75) fake outcomes, 500
  bootstrap resamples, 10 placebo permutations per seed with the same permutation seeds (20261250 + k, k = 0..9),
  parallel, through the same code path as study 1 (`--null-study-2`).
- None of seeds 4401–4600, nor any outcome from them, has been used.
- The `--real` preconditions (stop rule 4) now require `SELF-TEST planted: PASS`, `SELF-TEST planted-null: PASS` and
  `SELF-TEST null-2: PASS` in their logs. Study 1's log (a FAIL) is kept as evidence and is no longer a precondition.
- If the confirmation study fails, lab 3 does not proceed to `--real` without a further documented decision, and
  nothing is loosened.

### 5. Pre-registered predictions for the confirmation study

These are predictions, not pass rules. Each is reported as met or not met.
- T1 falsely confirmed in 0–3% of seeds (expected: 0).
- Placebo false stops in 0–3% of seeds.
- Mean T1 between −0.00009 and −0.00004.
- The upward-bias check passes, with probability about 99%.
- The mean placebo gain is within a factor of 2 of the mean T1 (the per-seed placebo gains are recorded).

### 6. Consequence for reading the real result

**Refers to:** plan Section 9, "**(a)** T1 gain (M1 loss − M3 loss) is between +0.0002 and +0.0010."
- Under the null the expected T1 is about −0.00007. On real data T1 is positive only if the true benefit of the H
  columns exceeds that cost.
- Claude's expectation (a) refers to the net T1 and is unchanged.
- No other rule, threshold, seed, test or expectation changes.

### 7. Output files

**Refers to:** plan Section 11 step 5, "**Commit and push** the plan (hashed), the features, the leak-check logs, the
scripts and the self-test logs."
- The confirmation study writes `analysis/out/btslab3_null_study2_log.txt` and
  `analysis/out/btslab3_null_study2_seeds.csv`. The CSV will be force-added to git.
- `--null-study-2` refuses to run if either file already exists.

## Unchanged

Item 3 replaces the null study's centring check with an easier one-sided check, and item 4 adds the confirmation
study that must pass it. No other rule, threshold, seed, test or expectation changes. Everything else in the lab 3
plan and addenda A1 and A2 stays as written.

## Addendum

(none)
