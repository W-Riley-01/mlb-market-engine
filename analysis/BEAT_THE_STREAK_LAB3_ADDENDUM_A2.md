# Beat the Streak lab 3 — Addendum A2 (planted-test generator and clarifications)

**Dated 2026-11-06, written after the group H builder, the leak checks and the analysis script's smoke test, and
before any registered self-test seed (4001–4200, 4300, 4301), any lab 3 real-outcome read, or `--real`.**
- It amends `analysis/BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md` (`722712b`, lab3 sha256 `f3d76f8f…a872`) and adds to
  addendum A1 (lab3 A1 sha256 `a887f248…2af0`). Both files are unchanged.
- **ONE design item is changed (item 1: how the planted self-tests generate their fake outcomes). Everything else is a
  clarification. No threshold, seed, expectation, test or pass rule is changed.**
- Basis: `analysis/btslab3_analysis.py` and its smoke log `analysis/out/btslab3_smoke_log.txt` (fake outcomes,
  throwaway seeds only; two dry runs).
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab3 A2 sha256:`.
- Where this addendum and the plan or addendum A1 differ, this addendum governs.

## Summary in plain language

Before the real run, the analysis is tested on made-up outcomes. In one test we plant a known effect of the new
contact-quality feature (h_xhit) and check that the model finds it at about the planted size.

A trial on throwaway seeds went wrong in two ways: the model found +0.17 where +0.10 was planted, and in the matching
test with nothing planted the new features still "won". We believe the reason is not a bug. The made-up outcomes were
built from raw, unsmoothed batting rates, which the model sees only in smoothed form, so the old model was wrong in
that made-up world and the new columns partly repaired it.

The fix is to build the made-up outcomes from the models' own smoothed columns, with the same planted sizes. A second
trial on eight throwaway seeds then behaved as a working pipeline should. The pass rules are exactly the plan's. We
expect the registered planted test to pass about nine times in ten; it checks the pipeline in a world the model
describes correctly, while the pure null study keeps the job of catching false confirmations. The
remaining items write down how the script reads a few points the plan left open. None of them changes what is tested
or how it is judged.

## Items

### 1. Planted-test generator (CHANGED)

**Refers to:** plan Section 11 step 4, "**Planted H:** logit p = logit(0.75) + 0.30·[slot ≤ 4] − 0.40·z(K%) +
0.50·z(H/AB) + 0.10·z(raw h_xhit) (lab 1's planted formula plus the H term; Resolved choice 3). z is over all rows;
missing → 0. Raw h_xhit = unshrunk window mean.", "**Pass:** M3's h_xhit coefficient is within ±50% of 0.10 (0.05
to 0.15) **and** T1 is Confirmed; the placebo is not Confirmed." and "**Planted-null:** lab 1's planted formula with
no H term, seed 4301. **Pass:** T1 not Confirmed."

**(a) First dry run, old generator** (throwaway seeds, not registered; 20 bootstrap resamples, 2 placebo
permutations):
- planted-H seed 9001: M3's h_xhit coefficient +0.1698, outside 0.05–0.15;
- planted-null seed 9003: T1 Confirmed (+0.00078, 95% +0.00046 to +0.00095; 2024 +0.00056, 2025 +0.00100), with
  ablation gains for M1 + H4 of +0.00048 and M1 + H1 of +0.00032.

**(b) The explanation we believe (not proven).** The old generator draws on unshrunk rates (raw K%, raw H/AB, raw
h_xhit) that M1 sees only through shrunk columns, so M1 is misspecified in that world. M3's extra columns, including
speed's missing flag (a proxy for thin samples), partly repair that misspecification. This makes the planted-null
look like a gain and inflates the planted coefficient.

**(c) The change.** For lab 3's planted-H and planted-null fake outcomes only:
- logit p = logit(0.75) + 0.30·[slot ≤ 4] − 0.40·z(a_k) + 0.50·z(a_hab) + c·z(h_xhit), with c = 0.10 for planted-H
  and c = 0 for planted-null, then a Bernoulli draw with the seed;
- a_k and a_hab are M1's own shrunk columns (`btslab_analysis.derive()`), and h_xhit is M3's own shrunk column
  (`derive_h()` in `btslab3_analysis.py`), all computed with the prior strengths of the train-2022–24 fold and
  standardised with that fold's training mean and SD (`btslab_analysis.design()`), applied to all rows; missing → 0;
- the coefficients 0.30, −0.40, +0.50 and +0.10 are unchanged, and so are the seeds (4300, 4301);
- the pure null study (Bernoulli 0.75, seeds 4001–4200) is unchanged.

**(d) Pass criteria: exactly the plan's, unchanged.** Planted-H: M3's h_xhit coefficient (per SD of the shrunk
h_xhit, fit on 2022–24) within ±50% of 0.10 (0.05 to 0.15), **and** T1 Confirmed, **and** the placebo not Confirmed.
Planted-null: T1 not Confirmed. The coefficient difference between the two runs is printed as a descriptive line only;
it is not a pass condition.

**(e) Second dry run, new generator** (once; throwaway seeds, not registered; 20 bootstrap resamples, 2 placebo
permutations; code, generator and settings unchanged after the results). Freeze conditions fixed in advance: both
planted-H seeds pass the plan's rule; T1 Confirmed on at most 1 of the 6 planted-null seeds.

| run | seed | h_xhit coefficient | T1 (gain, 95% interval) | placebo (gain, 95% interval) | plan's rule |
|---|---|---|---|---|---|
| planted-H | 9001 | +0.1067 | Confirmed, +0.00026 (+0.00016 to +0.00043) | not Confirmed, −0.00010 (−0.00014 to −0.00001) | PASS |
| planted-H | 9002 | +0.1098 | Confirmed, +0.00034 (+0.00012 to +0.00052) | not Confirmed, −0.00005 (−0.00008 to +0.00000) | PASS |
| planted-null | 9003 | +0.0287 | not confirmed, −0.00008 (−0.00016 to −0.00003) | not Confirmed, −0.00009 (−0.00018 to −0.00002) | PASS |
| planted-null | 9004 | +0.0231 | not confirmed, −0.00008 (−0.00016 to −0.00000) | not Confirmed, −0.00004 (−0.00008 to +0.00001) | PASS |
| planted-null | 9005 | −0.0103 | not confirmed, −0.00005 (−0.00013 to +0.00004) | not Confirmed, −0.00011 (−0.00017 to −0.00006) | PASS |
| planted-null | 9006 | −0.0039 | not confirmed, −0.00012 (−0.00021 to +0.00000) | not Confirmed, −0.00016 (−0.00020 to −0.00007) | PASS |
| planted-null | 9007 | −0.0310 | not confirmed, −0.00010 (−0.00016 to −0.00001) | not Confirmed, −0.00008 (−0.00015 to −0.00003) | PASS |
| planted-null | 9008 | −0.0449 | not confirmed, −0.00009 (−0.00018 to −0.00003) | not Confirmed, −0.00013 (−0.00020 to −0.00003) | PASS |

- Freeze condition 1 (both planted-H seeds pass): holds.
- Freeze condition 2 (T1 Confirmed on at most 1 of 6 planted-null seeds): 0 of 6, holds.

**(f) Disclosure.**
- This is a design correction made after two dry runs on throwaway seeds and before any registered self-test seed was
  run. It is not a response to a failed registered check.
- The registered seeds are run once.
- A single registered null seed carries its own small chance (up to 8% in a working pipeline) of confirming T1. That
  would be reported as a failed check under the usual rules and not retried.

**(g) Expected power and scope, stated before any registered run.**
- **Expected power.** In the second dry run the planted-H T1 gains were +0.00026 and +0.00034, much smaller than a
  naive +0.0007 because h_xhit overlaps with columns M1 already has, so only its unique part counts. From the dry-run
  bootstrap intervals (estimated, not simulated) we expect the registered planted-H run (seed 4300, 2,000 resamples)
  to confirm T1 with probability roughly 85–95%. A chance failure is therefore possible, probably about 1 time in 10.
  If it happens it is reported as a failed registered check under the usual rules; the effect size, tolerance and
  seeds are not changed and the run is not repeated. The planted effect is deliberately not raised again after two
  dry-run-driven design corrections.
- **Scope.** The planted tests check that the pipeline (folds, shrinkage, inner CV, bootstrap, T1 rule, placebo)
  recovers a planted effect in a world the model describes correctly, and that M3 does not beat M1 when nothing is
  planted. They do not test robustness to a misspecified model. The false-confirmation role belongs to the pure null
  study (200 seeds, Bernoulli(0.75)), which is unchanged.

### 2. Placebo permutation

**Refers to:** plan Section 7 T4, "within each game date, the full vector of H columns (all sub-groups and flags
together) is permuted across that date's batter-games. All M1 columns stay with their rows."
- The permutation acts on the fold's derived H columns (after shrinkage), jointly, within each game date.
- One shuffle per k (seed 20261250 + k) is shared by both walk-forward folds.

### 3. T2 rule

**Refers to:** plan Section 7 T2, "T1's rule is applied and reported separately."
- "Positive in both test seasons" is read as positive in every held-out season (all four).

### 4. Stop rule 6 applied to M3

**Refers to:** plan Section 10 rule 6, "checked for M1 and M3 in each walk-forward fold: M1 beats B1 on batter-game
AUC in the training inner CV".
- For M3 the check reads "M3 beats B1 on batter-game AUC in the training inner CV", an added sanity check. The slot
  check (slots 1–4 minus slot 9 > 0) is applied to M1 and M3 alike.

### 5. `--real` preconditions (stop rules 1, 2 and 4)

**Refers to:** plan Section 10, "1. **Leak-check failure** stops everything", "2. **Base table mismatch:** the lab 1
table's content hash differs from `e1fb5d01…00f9a`." and "4. **Self-tests fail** (Section 11)".
- Before any model fit, `--real` stops unless all of these hold:
  - `analysis/out/btslab3_leakcheck_log.txt` contains `LEAK CHECK: PASS`;
  - lab 1's table content hash equals `e1fb5d01cf19f18eab46b561575a5975867348cca1f16f29656d203bf7600f9a`;
  - the lab 3 table content hash equals `d8dccfe335795fec26b7f5bceb138059c10209bbe9a557463072860f1889bfa1`;
  - `analysis/out/btslab3_selftest_planted_log.txt` contains `SELF-TEST planted: PASS`;
  - `analysis/out/btslab3_selftest_planted_null_log.txt` contains `SELF-TEST planted-null: PASS`;
  - `analysis/out/btslab3_null_study_log.txt` contains `SELF-TEST null: PASS`.

### 6. Sign conventions

**Refers to:** plan Section 7 T5, "Log-loss change and top-1 change for M1 + Hk against M1 and M3 − Hk against M3".
- Gain = log loss(reference) − log loss(model).
- Top-1 change = model − reference, paired by date.

### 7. T6 refits

**Refers to:** plan Section 7 T6, "T1's statistic retested with 2022 dropped from training (Section 6), seed
20261206."
- The 2022-dropped folds refit M1 and M3, each with its own inner CV.

### 8. Output files

**Refers to:** plan Section 11 step 3, "`--real` refuses to run if `analysis/out/lab3_results_log.txt` already exists
(run once)."
- Output files not named in the plan: `btslab3_smoke_log.txt`, `btslab3_selftest_planted_log.txt`,
  `btslab3_selftest_planted_null_log.txt`, `btslab3_null_study_log.txt`, `btslab3_null_study_seeds.csv` (all in
  `analysis/out/`).
- The real-run log uses the plan's exact name, `analysis/out/lab3_results_log.txt`.

## Unchanged

Item 1 changes only how the planted self-tests generate their fake outcomes, and states their expected power and
scope; items 2–8 are clarifications.
No threshold, seed, expectation, test or pass rule changes. Everything else in the lab 3 plan and addendum A1
stays as written.

## Addendum

(none)
