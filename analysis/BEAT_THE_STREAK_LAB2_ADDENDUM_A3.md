# Beat the Streak lab 2 — Addendum A3 (design fixes before anything is built)

**Dated 2026-11-09. Written before any lab 2 feature, leak check, model fit, self-test or batter-game outcome read, and
after lab 3's real run.**
- It amends `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`).
- It adds to addenda A1 (`074878e`, lab2 A1 sha256 `74e182c7…08ba`) and A2 (`478847a`, lab2 A2 sha256
  `f0300c33…55e9`). All three files are unchanged.
- Basis:
  - the lab 2 plan and addenda;
  - the Step 0 and Step 1b logs;
  - lab 1's code (`analysis/btslab_features.py`, `analysis/btslab_analysis.py`);
  - lab 3's code (`analysis/btslab3_features.py`, `analysis/btslab3_leakcheck.py`, `analysis/btslab3_analysis.py`);
  - lab 3's addenda A2 and A3;
  - lab 3's results log `analysis/out/lab3_results_log.txt` (committed in `aab2c84`; on-disk sha256 `fecec9e2…eca2`).
- No lab 2 code beyond the contact table and Step 0/1b exists. No 2026 file was opened to write this addendum.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A3 sha256:`.
- Where this addendum and the plan or addenda A1–A2 differ, this addendum governs.
- **Who decided what.** Items 1–12 are decisions from Claude chat. Item 13 and every point labelled
  **PROPOSED (Claude Code)** are implementation proposals by Claude Code (numbered P1–P17 in Claude Code's report of
  2026-11-09).
- **Reviewed by Claude (chat) on 2026-11-09: P1 to P18 accepted; amendments 1 to 7 added before publication;
  amendments 8 to 11 added after the unit-season finding.**
- **Claude Code note.** The unit-season finding was wrong and is retracted (item 13, amendment 8).
  - Amendment 8 is written to the corrected measurement, not to its first wording.
  - Item 13 j adds two new PROPOSED points (sort on load; shuffle seed 20261131).
  - All of this awaits review.
  - Amendment 1: item 3, "Confirmatory family".
  - Amendment 2: item 13 e.
  - Amendment 3: item 13 a.
  - Amendment 4: item 8.
  - Amendment 5: item 3, "M4 scope".
  - Amendment 6: item 3, "Resolving power".
  - Amendment 7: item 11.
  - Amendment 8: item 13, "Unit-season rows", and item 13 a.
  - Amendment 9: item 3, "M4 scope" (P18).
  - Amendment 10: item 13 a.
  - Amendment 11: item 13 j.

## Summary in plain language

Lab 2 has not built anything yet. Since its plan was frozen, lab 3 ran and taught us four things that bear on lab 2's
design, so we fix the design now, before any feature or result exists.

**No 2026 data in lab 2.**
- The 2026 test moves out of lab 2 into a separate combined plan, written later.
- Lab 2 works on 2022–2025 only and never opens 2026 data.

**A second comparison.**
- Lab 3 showed that its richer batter model (M3) beats lab 1's M1.
- Lab 2's matchup score uses the same contact table as one of M3's features, so beating M1 alone might show nothing
  new. Lab 2 therefore also tests the matchup score on top of M3.
- The score is carried forward only if it wins both comparisons.

**Lessons from lab 3's self-tests.**
- **Planted tests.** Made-up outcomes must be built from the models' own smoothed columns, or the test misleads.
- **Null study.** When nothing is real, a model with extra columns is expected to do very slightly worse, not exactly
  as well. The null-study check is restated to expect that, before the study runs this time rather than after it fails.

The remaining items:
- fix the placebo's mechanics, the seeds, the file names and two reproduction checks;
- write down how the builder handles several points the plan left open. Those are marked as proposals for review.

## Items

### 1. Scope of Step 1c: the 2022–2025 build only (decision)

**Refers to:**
- plan Section 10, step 4: "**Build and check.** Lab 1's feature builder, then the lab 2 features. Then the full leak
  checks of Section 11 on 2026 rows: structural; future deletion on 300 random rows plus 150 doubleheader rows; the
  plain-path spot check."
- A2 d: "**Walk-forward tests (2022–25)** use the existing vault only (pulled 2026-04-22)."

**The decision:**
- The lab 2 build (Step 1c) covers 2022–2025 only, for the walk-forward and leave-one-season-out designs.
- It reads `data/master_physics_vault.parquet` only, as A2 d requires. It never opens
  `data/statcast_current_2025_2026.parquet`.
- 2026 schedule files are filtered out of the schedule loader exactly as lab 3's `load_schedule()` does in
  `analysis/btslab3_features.py`: `bf.load_schedule()` with files named `2026…` skipped.
- **Section 10 step 4 is withdrawn from lab 2.**

### 2. The 2026 final test moves to a separate combined plan (decision; design change)

**Refers to:**
- plan Section 8, "**FINAL TEST on 2026** (only if Section 10's data step passes)";
- plan Section 8, "**F1:** M1 − B2 top-1 success on 2026 … seed 20261111" and "**F2:** M2 − M1 log loss on 2026 …
  Seed 20261112.";
- plan Section 10, step 4 (quoted in item 1);
- plan Summary, "If 2026 pitch data can be pulled and checked, the frozen models get one final test on that untouched
  season.";
- plan Section 13 item 1, "Only 2026, if built, is untouched."

**The decision:**
- The FINAL TEST (F1, F2) and Section 10 step 4 are withdrawn from lab 2. They move to a separate combined 2026 plan,
  to be written and frozen later, before 2026 is opened.
- Lab 2 does not open 2026 at all.
- **This is a design change, made before any lab 2 result exists.**
- Seeds 20261111 and 20261112 are not used.
- 2026 stays untouched by lab 2. Section 13 item 1's statement now applies to the combined plan.

**Open items for the combined plan** (recorded here, not solved):
- **2026 source for lab 1's other groups.** Lab 1's non-arsenal groups read `data/pitch_matrix.parquet`, which ends
  2025-11-01, so their 2026 source is undecided.
- **Label vintage.** A2 d's label-vintage rule, "pitch-type features must never mix label vintages inside one test",
  has to be applied to the combined test.

### 3. Comparators: T1 unchanged; T1b, T3b and the M4 placebo added (decision)

**Refers to:**
- plan Section 7, "**M2 (primary)** = M1 + I1 + I1_miss.";
- plan Section 8 common rules, "**Confirmatory family:** T1 alone. Its rule fires only if the 95% interval lies
  entirely on the side of M2 improving, about 0.025 under the null.";
- plan Section 8 T1, "**Rule:** **"Confirmed"** only if the 95% interval lies entirely below 0 (M2 better). Otherwise
  **"not confirmed"**."

**Unchanged.** T1 and everything else about M1 and M2 stay exactly as frozen:
- the primary T1 compares M2 with M1;
- the sign convention is loss(new) − loss(old);
- "Confirmed" requires the 95% interval to lie entirely below 0.

**Added models:**
- **M3** = lab 3's model as it ran:
  - the same group H columns (`M3_COLS` in `analysis/btslab3_analysis.py`);
  - built from `analysis/cache/btslab3_features.parquet` through the lab 3 code, imported and not copied.
- **M4** = M3 + I1 + I1_miss.
- M3 and M4 each tune their own penalty by inner CV on the plan's grid: 5 contiguous date blocks, 12 C from 10⁻³ to
  10².

**T1b, added.**
- The statistic is the mean over walk-forward test rows (2024 and 2025 pooled) of log loss(M4) − log loss(M3), paired
  by row.
- It uses the same rule, the same paired test and the same date-clustered bootstrap design as T1. Like the plan's T1,
  and unlike lab 3's, it has no per-season condition.
- **Why.** Lab 3 confirmed that M3 beats M1, and I1 may overlap with group H: lab 3's h_xhit uses the same contact
  table. A win over M1 alone could therefore be redundant.

**Inclusion rule.**
- I1 enters the combined 2026 model only if T1 **and** T1b are both Confirmed.
- If either is not, I1 stays out.

**Confirmatory family (amendment 1).**
- **Refers to:**
  - plan Section 8, "**Confirmatory family:** T1 alone.";
  - plan Section 13 item 1, "With lab 1's two confirmatory tests and lab 2's one, the chance of at least one false
    confirmation across both labs is about 1 − 0.975³ = 0.073 if independent."
- **T1b is confirmatory.** With F1 and F2 withdrawn (item 2), lab 2's confirmatory rules are T1 and T1b.
- **What the old figure counted.** It counted three rules: lab 1's two confirmatory tests and lab 2's T1.
  - Section 13 names lab 1's two only as "lab 1's two confirmatory tests" and does not identify them. This addendum does
    not guess which they are.
  - None of the three is gone. Lab 1's two are part of lab 1's finished record, and lab 2's T1 is unchanged.
  - F1 and F2, now withdrawn, were not among the three: Section 13 counts "lab 2's one".
- **The new family, in plain words.** Each rule fires falsely about 2.5% of the time under a null.
  - **Either rule firing.** The chance that at least one of T1 and T1b fires falsely is at most about 5%: 1 − 0.975² =
    0.049 if the two are independent, and never more than 0.05 whatever their dependence.
  - **I1 wrongly included.** Inclusion needs both to fire, so that chance is at most about 2.5%. Both firing can be no
    likelier than either one firing.
- **Across labs (descriptive).** Counting lab 1's two tests, lab 2's two and lab 3's one, the chance of at least one
  false confirmation is about 1 − 0.975⁵ = 0.119 if independent.

**M4 scope (amendment 5).**
- M4 gets:
  - T1b;
  - T3b;
  - the M4 placebo;
  - M4's standardised I1 coefficient with its interval, reported from the walk-forward fit (2022–24) and its date
    bootstrap. This is descriptive, with no rule.
- M4 gets no T2 and no T6.
- T3b follows T3 exactly, including its recalibration. Plan Section 7: "**Recalibration** of daily top picks: as in A2
  change a (slope constrained to [0, 1])."
- **P18, accepted by Claude (chat) on 2026-11-09 (amendment 9):** the M4 coefficient's bootstrap seed is 20261120, the same as M2's coefficient. The two
  are computed on the same resamples, as the plan's coefficient bootstrap is one design. No new seed is introduced.

**Resolving power (amendment 6).**
- **Refers to:** plan Section 13 item 6, "A small matchup feature like I1 should land near the D–E end, about ±0.00003
  to ±0.0001."
- No resolving-power estimate is given for T1b.
- The planted test shows only that a gain of the planted size is detectable.
- The plan's extrapolated figure for T1 is not claimed for T1b.

**T3b, added (descriptive, no rule).**
- Top-1 daily success, M4 − M3, paired by date as in T3.

**M4 placebo, added.**
- It is a placebo for M4 under the same rule as T4 (item 7).
- Stop rule 2 applies to it as written: if its interval lies entirely below 0, stop before any other result is read.

**Claude's pre-registered expectations, added** (each scored met or not met; no decision rule):
- **(h)** T1b improvement, loss(M3) − loss(M4), is between 0 and 0.0003, with about a 25% chance the interval excludes
  0.
- **(i)** The T1b improvement is no larger than the T1 improvement: loss(M3) − loss(M4) ≤ loss(M1) − loss(M2).

### 4. Planted-test generator (CHANGED; decision)

**Refers to:** plan Section 12:
- "**Planted I1:** logit p = logit(0.75) + 0.10 · z(I1), with I1 computed from real feature counts. The pipeline must
  recover a positive I1 coefficient within ±50% of the planted size, and T1 must be Confirmed. The placebo must not be
  Confirmed."
- "**Planted null-I1:** lab 1's planted formula (slot, K%, H/AB) with no I1 term. T1 must not be Confirmed."

**Why.** Lab 3 addendum A2 item 1(b), the explanation it believed (not proven): "The old generator draws on unshrunk
rates (raw K%, raw H/AB, raw h_xhit) that M1 sees only through shrunk columns, so M1 is misspecified in that world.
M3's extra columns … partly repair that misspecification. This makes the planted-null look like a gain and inflates
the planted coefficient." Lab 2's planted null-I1 uses the same raw-rate formula.

**The change.** For planted I1 and planted null-I1 only:
- logit p = logit(0.75) + 0.30·[slot ≤ 4] − 0.40·z(a_k) + 0.50·z(a_hab) + c·z(I1), with c = 0.10 for planted I1 and
  c = 0 for planted null-I1, then a Bernoulli draw with the seed.
- As in lab 3 A2 item 1(c), "a_k and a_hab are M1's own shrunk columns (`btslab_analysis.derive()`) … all computed
  with the prior strengths of the train-2022–24 fold and standardised with that fold's training mean and SD
  (`btslab_analysis.design()`), applied to all rows; missing → 0".
- z(I1) is M2's own shrunk I1 column, computed with the train-2022–24 fold's prior strengths and standardised on that
  fold.
- **PROPOSED (Claude Code):** rows with I1_miss = 1 get z(I1) = 0, the analogue of lab 3's "missing → 0".
- The pure null study is unchanged: Bernoulli(0.75).

**Pass rules: the plan's, unchanged, plus the same for T1b.**
- **Planted I1:** M2's I1 coefficient (per SD, fit on 2022–24) within ±50% of 0.10 (0.05 to 0.15), **and** T1
  Confirmed, **and** the placebo not Confirmed.
  - Added: T1b Confirmed.
  - **PROPOSED (Claude Code):** the M4 placebo is also not Confirmed.
- **Planted null-I1:** T1 not Confirmed. Added: T1b not Confirmed.

**Dry runs first.**
- Throwaway dry runs on seeds 9101–9109 come before the registered runs.
- If they show the generator misbehaving, an addendum is written before any registered seed runs.
- **PROPOSED (Claude Code), allocation and conditions fixed now, as lab 3 A2 item 1(e) did:**
  - 9101–9102 planted I1;
  - 9103–9108 planted null-I1;
  - 9109 smoke;
  - 20 bootstrap resamples and 2 placebo permutations;
  - code, generator and settings not changed after the results.
- **Conditions for proceeding to the registered runs:**
  - both planted seeds pass the rule above;
  - T1, and separately T1b, Confirmed on at most 1 of the 6 planted-null seeds.

**Expected size (Claude Code estimate, not simulated).**
- I1 is built as a deviation from overall skill, so it should overlap little with M1's columns.
- A planted 0.10 per SD would then give a T1 gain of roughly ½ · 0.10² · 0.75 · 0.25 ≈ 0.0009 in log loss, well above
  the plan's I1 half-width estimate (Section 13 item 6, "about ±0.00003 to ±0.0001").
- If I1 turns out to overlap more, the gain shrinks, as h_xhit's did in lab 3.

### 5. Null-study centring check (REPLACED with a different check; decision)

**Refers to:** plan Section 12, "**Pass:** T1 false-confirmation ≤ 8%; T4 false stop ≤ 8%; T1 statistic centred at 0
(|mean| below 0.25 SD)."

**The change.**
- The two-sided centring check is replaced by lab 3 A3 item 3's one-sided form.
- With D = loss(new) − loss(old), the mean of D over the 200 seeds must be **at least −0.25 SD(D)**: no favourable
  bias.
- It is applied to T1 (D = loss(M2) − loss(M1)) and to T1b (D = loss(M4) − loss(M3)).
- **This is a different check from the one the plan froze.**

**Why.**
- Extra columns cost out-of-sample accuracy under the null, so a model with more columns is expected to sit slightly on
  the unfavourable side.
- Lab 3's two-sided check failed for exactly that reason: mean T1 −0.0000656, |mean| = 1.60 SD, in lab 3's sign
  convention where positive favours the new model.
- Unlike lab 3 A3, this check is set **before** the null study runs, not after a failure.

**Unchanged.**
- False-confirmation limit ≤ 8% and false-stop limit ≤ 8%.
- **PROPOSED (Claude Code):** both limits apply to T1 and T1b, and to the M2 and M4 placebos alike.

**Claude's pre-registered predictions for the null study** (scored met or not met; no decision rule):
- 0–3% false confirmations, and 0–3% false stops, for each of T1 and T1b.
- Mean D positive (the new model slightly worse), between 0 and +0.00002, for each of T1 and T1b.
- The one-sided check passes.

**Note on expectation (d).**
- Expectation (d), "The placebo (T4) is within noise of 0: its interval includes 0", may be missed because of the same
  column cost: in lab 3 the placebo interval lay entirely on the unfavourable side.
- It is scored literally.

### 6. Reproduction and base-table rules (new stop rules 6 and 7; rule 5 extended; decision)

**Refers to:** plan Section 11, "5. **Inherited from lab 1 (Section 10, rules 2–3)**, checked for M1 and M2 in each
walk-forward fold: M1 must beat B1 on batter-game AUC in the training inner CV; the training slot 1–4 minus slot 9
predicted P(hit) must be > 0."

**Stop rule 6, added: table identity.**
- Before anything is fitted, the analysis script recomputes and verifies the content hashes (SHA-256 of
  `pandas.util.hash_pandas_object(df, index=False)`) of:
  - lab 1's table `analysis/cache/btslab_features.parquet`:
    `e1fb5d01cf19f18eab46b561575a5975867348cca1f16f29656d203bf7600f9a`;
  - lab 3's table `analysis/cache/btslab3_features.parquet`:
    `d8dccfe335795fec26b7f5bceb138059c10209bbe9a557463072860f1889bfa1`.
- Any mismatch stops the run.

**Stop rule 7, added: reproduction.**
- M1 must reproduce lab 1's printed numbers, and M3 lab 3's, exactly.
- Content is compared, not file bytes, because Python logs are CRLF on disk.
- Any mismatch stops before M2 or M4 is fitted.

**Lab 1 strings compared,** from `analysis/out/btslab_results_log.txt`. These are the same fields and precision as
lab 3's `REPRO_FOLDS` and `REPRO_WF`.
- Fold rows: C, inner-CV log loss, AUC M1 (CV), AUC B1 (CV), wiring.
  - `| WF | 2022-2023 | 2024 | 0.001 | 0.65839 | 0.5731 | 0.5492 | +0.1185 |`
  - `| WF | 2022-2023-2024 | 2025 | 0.001 | 0.65864 | 0.5726 | 0.5480 | +0.1220 |`
  - `| LOSO | 2023-2024-2025 | 2022 | 0.001 | 0.65843 | 0.5695 | 0.5429 | +0.1249 |`
  - `| LOSO | 2022-2024-2025 | 2023 | 0.001 | 0.65973 | 0.5711 | 0.5471 | +0.1226 |`
  - `| LOSO | 2022-2023-2025 | 2024 | 0.001 | 0.65879 | 0.5715 | 0.5462 | +0.1211 |`
  - `| LOSO | 2022-2023-2024 | 2025 | 0.001 | 0.65864 | 0.5726 | 0.5480 | +0.1220 |`
- Walk-forward test log loss and AUC: `| M1 (reference) | 0.783 (0.738–0.822) | +0.0220 | 0.65947 | 0.5740 |`
  (`0.65947`, `0.5740`).
- Top-1:
  - `| WF | M1 | pooled | 365 | 1 | 0.783 (0.738–0.822) |`
  - `| WF | M1 | 2024 | 183 | 0 | 0.765 (0.699–0.821) |`
  - `| WF | M1 | 2025 | 182 | 1 | 0.801 (0.737–0.853) |`

  The rates are compared: `0.783`, `0.765`, `0.801`.
- M1 − B2: `WF: M1 − B2 = +0.0220 (95% -0.0220 to +0.0687), 364 paired dates`, with seed 20261027 as in lab 1.

**Lab 3 strings compared,** from `analysis/out/lab3_results_log.txt`.
- **PROPOSED (Claude Code):** this selection of strings.
- M3 fold rows (C, inner-CV log loss, AUC (CV), AUC B1 (CV), wiring):
  - `| WF | 2022-2023 | 2024 | M3 | 0.001 (grid edge) | 0.65755 | 0.5763 | 0.5492 | +0.1188 |`
  - `| WF | 2022-2023-2024 | 2025 | M3 | 0.001 (grid edge) | 0.65782 | 0.5756 | 0.5480 | +0.1223 |`
  - `| LOSO | 2023-2024-2025 | 2022 | M3 | 0.001 (grid edge) | 0.65778 | 0.5721 | 0.5429 | +0.1252 |`
  - `| LOSO | 2022-2024-2025 | 2023 | M3 | 0.001 (grid edge) | 0.65891 | 0.5742 | 0.5471 | +0.1229 |`
  - `| LOSO | 2022-2023-2025 | 2024 | M3 | 0.001 (grid edge) | 0.65803 | 0.5745 | 0.5462 | +0.1213 |`
  - `| LOSO | 2022-2023-2024 | 2025 | M3 | 0.001 (grid edge) | 0.65782 | 0.5756 | 0.5480 | +0.1223 |`
- Walk-forward test log loss and AUC: `batter-game log loss M1 0.65947, M3 0.65878; AUC M1 0.5740, M3 0.5771`.
- Lab 3's T1, pooled: `pooled 2024–25: +0.00069 (95% +0.00040 to +0.00099), rows 87021`, with lab 3's seed 20261201.
- Top-1: `M3 0.810 (0.767–0.847)` (the rate `0.810`).
- Lab 3's results log is committed in `aab2c84`. **PROPOSED (Claude Code):** the analysis script records its on-disk
  SHA-256.

**Stop rule 5, extended.** It is checked for M1, M2, M3 and M4 in each walk-forward fold:
- each must beat B1 on batter-game AUC in the training inner CV;
- each must have a training slot 1–4 minus slot 9 predicted P(hit) > 0.

**Order in `--real`.**
- Rule 6 first.
- Then the self-test preconditions (item 13 i).
- Then rule 7, M1 and then M3.
- Then rule 5.
- Then the placebos (stop rule 2), before any other result is read.

### 7. T4 mechanics (decision)

**Refers to:** plan Section 8 T4:
- "Repeat T1 with each starter-game's usage vector u(·|s) permuted across starter-games within the same season.
  Permutation seed 20261150. The batter's and pitcher's family skills stay with their own rows; only the pitch mix is
  shuffled."
- "I1 and M2 are refitted on the placebo feature."

**The decision:**
- **Permutations.** The real run averages 50 permutations, seeds 20261150 + k for k = 0..49. Each null-study seed
  uses 10, k = 0..9.
- **What is permuted.** As the plan says: usage vectors u(·|s) are permuted across starter-games within the same
  season, and the batter's and pitcher's family skills stay with their rows.
- **Thin rows.** Rows with I1_miss = 1 keep I1 = 0 and I1_miss = 1, and are neither permuted nor used as donors.
  I1_miss is otherwise unchanged.
- **Refit.** The placebo refits M2, and M4 for the M4 placebo, at the fold's real penalty C with no inner CV, as lab 3
  did.
- Stop rule 2 applies as written, and to the M4 placebo.

**PROPOSED (Claude Code):**
- **Unit of permutation.** The (starter-game, side s) pair, permuted within (season, side s). A vector for batters on
  one side goes only to another starter-game's vector for the same side.
- **One mapping per k.** One donor mapping per k is shared by both walk-forward folds and by the M2 and M4 placebos.
  This follows lab 3 A2 item 2: "One shuffle per k (seed 20261250 + k) is shared by both walk-forward folds."
- **Shrunk vectors are moved.** The vectors moved are the fold's shrunk u(·|s); the prior strengths m are not
  re-estimated on permuted data. I1 is then recomputed with each row's own r_b(f), r_p(f|s), R_b and R_p(s) and the
  donor's u(·|s).
- **Placebo gain.** As in lab 3, per-row differences are averaged over the permutations and the mean is
  date-bootstrapped (2,000).
  - The M2 placebo uses the plan's T4 seed, 20261101 + 4 = 20261105.
  - The M4 placebo uses the seed in item 8.
  - The 2.5–97.5 percentile range of the per-permutation statistics is also reported.

### 8. Seeds (decision)

**Refers to:** plan Section 8, "**Bootstrap:** clustered by date, 2,000 resamples; seed = 20261101 + test number.",
and plan Section 12, "200 seeds (fake seeds 3001–3200)".

**Seed table:**

| use | seeds |
|---|---|
| null study 1 (registered) | 3001–3200 (plan) |
| confirmation null study (reserved; used only if a registered check fails) | 3401–3600 |
| planted I1 | 3300 |
| planted null-I1 | 3301 |
| throwaway and smoke | 9101–9109 |
| placebo permutations | 20261150 + k (item 7) |
| real-run bootstrap base | 20261120; the plan's own seed is used wherever it states one for a test |
| leak check, future-deletion random rows | 20261130 |
| leak check, doubleheader rows | all 150, or the number available, drawn as lab 3 did (item 13 g) |
| leak check, plain-path rows | 20261132 |

**Seeds of the plan's own tests (amendment 4).**
- **Quoted.** The plan's rule is "seed = 20261101 + test number" (Section 8). It states three values explicitly: T5
  "seed 20261106", T6 "seed 20261107" and the coefficients "seed 20261120".
- **A reading, not a quoted rule.** The values for T1 to T4 are this addendum's arithmetic from that rule; the plan
  states them nowhere:
  - T1 20261102;
  - T2 20261103;
  - T3 20261104;
  - T4 20261105.
- They agree with T5's and T6's stated values.

**Seeds of the new tests (amendment 4, confirmed).**
- The base value 20261120 stays the plan's coefficient-bootstrap seed.
- New tests are offset from it:
  - T1b 20261121;
  - T3b 20261123;
  - M4 placebo 20261124.
- No two tests share a seed.

**The confirmation null study** is used only if a registered check fails, under a rule stated in an addendum written
before it runs.

**Collision check** (grep of all plans, addenda, scripts and logs, `analysis/.venv`, `analysis/cache`, `data` and
`.git` excluded, on 2026-11-09):
- 3001–3600, 9101–9109, 20261130, 20261132 and 20261151–20261199: no use anywhere.
  - The one match near the 3xxx range is the number of seconds in an hour in `btslab3_analysis.py`, not a seed.
- 20261150 appears only in the lab 2 plan's own T4.
- 20261120 appears only in the lab 2 plan's own coefficient bootstrap, which is why the new tests take offsets from 1
  up.
- 20261121–20261124: no use anywhere.

### 9. Runtime projection (decision)

**Refers to:** plan Section 12, "**Null study:** 200 seeds … parallel."
- Before the null study starts, the script reports a timed single fit and a runtime projection, as lab 3 did with
  `--time`.
- It refuses to start the 200-seed study without that report.

### 10. File names (decision)

The plan names no files. Lab 3's scheme is followed:
- builder `analysis/btslab2_features.py`;
- leak checks `analysis/btslab2_leakcheck.py`;
- analysis `analysis/btslab2_analysis.py`;
- logs `analysis/out/btslab2_*_log.txt`;
- feature table `analysis/cache/btslab2_features.parquet` (git-ignored);
- the real run's results log `analysis/out/lab2_results_log.txt`.

The analysis script refuses `--real` if `lab2_results_log.txt` exists.

### 11. Status of earlier statements (decision)

Some statements were true when written and are not changed:
- `analysis/out/lab2_step0_log.txt`: "pybaseball is not installed" and "No 2026 pitch data".
- A2's header: "No 2026 Statcast data has been pulled".

Since then (Step 1b log):
- pybaseball 2.2.7 is installed in `analysis/.venv`;
- the current-vintage pull `data/statcast_current_2025_2026.parquet` exists (sha256 `23c6e74d…a23e`).

Lab 2 does not open it (item 1).

**Section 10 is history for lab 2 (amendment 7).**
- Steps 1 to 3 were done in Step 1b: the validation, the 2026 schedules and box scores, and no re-pull of 2019–2025.
- Step 4 is withdrawn by item 2.

**Section 12's "Order" (amendment 7).**
- The rule: "the real hit column stays unread until this plan, the contact table, the scripts and the self-test logs
  are committed."
- A3 does not amend that rule. But A3 is committed before any lab 2 feature is built.

### 12. Line endings and identities (decision)

- Any hash of a Python-written log recorded during lab 2 is labelled "on-disk" and computed on disk. Logs redirected
  from Python on Windows are CRLF on disk and LF in git, so the two hashes differ.
- Tables are identified by content hash: the SHA-256 of `pandas.util.hash_pandas_object(df, index=False)`.

### 13. Implementation clarifications (PROPOSED (Claude Code) unless marked as following lab 1)

**Lab 1's precedent, used throughout:**
- `btslab_analysis.derive()` estimates every prior strength on the training rows only (`T = X[tr]`), from one row
  per unit-season:
  - batters: `T.drop_duplicates(["id", "season"], keep="last")`;
  - starters: `T.drop_duplicates(["opp_sp", "season"], keep="last")`;
  - platoon splits: `T.drop_duplicates(["id", "season", "c_hand"], keep="last")`.
**Unit-season rows (amendment 8; written to the corrected measurement, see "Retraction" below).**
- **The rule, by value.** Lab 2 keeps lab 1's rule. Within each unit-season, the row used for estimating prior
  strengths is the training row with the latest game date. If several training rows share that date, the one with the
  largest `game_pk` is used, then the largest `id`. Where lab 1 splits by `c_hand`, the rule applies within each split;
  for lab 2's side-specific quantities, within each side s.
- **How lab 1 arrives at it.**
  - `derive()` (`analysis/btslab_analysis.py` lines 84–86, and line 107 for the `c_hand` split) applies
    `drop_duplicates(..., keep="last")` to the training rows of the table returned by `load()`.
  - `load()` (line 59) sorts that table explicitly: `X.sort_values(["date", "game_pk", "id"])`.
  - Lab 3's `load()` (`analysis/btslab3_analysis.py` lines 63–71) calls lab 1's `load()` and then a left merge, which
    keeps that order.
  - Lab 3's `derive_h()` selects the same way: line 121 for batter-seasons and line 138 for venue-season-side rows.
- **Measured** on 2026-11-09 on key columns only (`game_pk`, `id`, `date`, `season`, `opp_sp`, `venue_id`, `c_hand`;
  no outcome column), in `load()`'s order. The selected row has the unit's latest training date for:
  - 1,484 of 1,484 starter-seasons;
  - 2,627 of 2,627 batter-seasons;
  - 120 of 120 venue-seasons;
  - 5,974 of 5,974 batter-season-hand splits.
- **Ties within a date change no value.** Every training row of one unit on one date has the same as-of window
  [D − 365, D), and so the same counts. This is shown by reasoning, not measured.
- **Properties.**
  - The rule is deterministic.
  - It is not a leak: each selected row carries its own as-of counts.
  - Lab 1's and lab 3's results were produced this way and reproduce. M1 and M3 are unchanged.
  - Lab 2 introduces no different rule, so that all estimators in one model are formed alike.
  - The effect of this choice on any result has not been measured.
- **Retraction.**
  - The previous frozen version of this addendum (sha256 `416251e2…ed24`) said lab 1 selects the row with the largest
    `game_pk` on a table in `game_pk` order, with matching dates for only 335 of 1,484 starter-seasons and 121 of
    2,627 batter-seasons. **That was wrong.**
  - Those figures came from the parquet file's own row order, which is `game_pk` order (`pool_rows()` and
    `full_build()` in `analysis/btslab_features.py`). But `derive()` never sees that order: `load()` re-sorts by date
    first.
  - Lab 3's plan Section 4 wording, "the last training row of each batter-season, as `derive()` does", and lab 1's
    `analysis/out/btslab_analysis_notes.txt` item 5 are correct.
  - Amendment 8, as first decided, restated the wrong finding. It is written here to the corrected measurement, and
    the difference is flagged for review.
- It calls `btslab_features.estimate_m(x, n, mu, min_n=100, N=None)`.
  - That function is the beta-binomial method of moments with per-trial variance μ(1 − μ).
  - With `N`, it uses c = 1 − n/N for a subset of a window of N trials whose raw rate is μ.
  - It returns ∞ when no between-unit spread is found.
- Shrinkage is `btslab_features.shrink(x, n, mu, m)` = (x + m·μ) / (n + m).
- Lab 1's platoon term estimates m against the batter's **raw** window H/AB with `N` = the window's AB, then shrinks
  toward the batter's **shrunk** H/AB.
- `min_n` is 100 everywhere except recent form, where lab 1 used 20 because 100 AB is unreachable in 14–30 days.
- Lab 1's pitcher counts (`raw_tables()` "pit") include every regular-season appearance of the pitcher, starts and
  relief alike.
- As in labs 1 and 3, the feature table stores as-of counts and league levels; shrinkage, standardisation and fills are
  applied per training fold at fit time.

**a. Pitcher unit-season rows and minimum n.**
- **Lab 2 does the same as lab 1,** with the row selected by value as in amendment 8 above:
  - Starter quantities that do not depend on side (u(f)) use one row per (`opp_sp`, season).
  - Side-specific quantities (u(f|s), R_p(s), r_p(f|s)) use one row per (`opp_sp`, season, s). This is the pattern of
    lab 1's platoon rows.
  - Batter quantities (R_b, r_b(f)) use one row per (`id`, season).
  - Each row's window counts are the as-of counts of that row.
  - A pitcher's counts include all his regular-season pitches, any role, as lab 1's "pit" table.
- **`min_n` = 100 for every lab 2 estimator (amendment 3),** in the estimator's own trial unit (item b). This is lab 1's
  rule: lab 1 plan Section 4, "estimated once per training fold by method of moments on training-season players with
  n ≥ 100", and `estimate_m(..., min_n: float = 100, ...)` in `analysis/btslab_features.py`.
- **Where 100 cannot be reached in a cell, lab 1's own rule for that case applies.**
  - **Lab 1's rule.** `derive()` in `analysis/btslab_analysis.py` lowers `min_n` to 20:
    `m["g14"] = bf.estimate_m(f(bat, "g_h14"), f(bat, "g_ab14"), raw(bat), min_n=20, N=f(bat, "a_ab"))  # 100 AB
    unreachable`, and the same for `g30`.
  - **When it applies (accepted, amendment 10).** "Cannot be reached" means no qualifying unit-season row (n ≥ 100)
    for that estimator in that fold. Then lab 1's rule applies.
  - **If no row reaches 20 either,** `estimate_m` returns m = ∞, full shrinkage to the target, by its own code:
    `return float(w.mean() / tau2) if tau2 > 0 else np.inf`.
  - No other threshold is introduced.
- **Why it can matter here.** In Step 0, a starter's offspeed PA-ending pitches to one side over 365 days had a median
  of 17–22 and a 90th percentile of 100–108, so few rows reach 100 in the thinnest cells.
- **Logging (amendment 10).** The builder logs, per fold and per cell (estimator × family × side where applicable):
  - the number of qualifying unit-seasons at n ≥ 100;
  - which estimators used `min_n=20`, and which ended with m = ∞.

  These are computed from stored counts only. The builder reads no outcome column.
- If those counts look unsafe, Claude (chat) may ask for an addendum before any outcome is read.

**b. Usage shrinkage, both levels.** The trial unit is pitches in the three families (codes per item c).
- **Level 1: u(f) toward league usage for the starter's hand.** PROPOSED.
  - For each family f: `estimate_m(x = the starter's pitches in f, n = his pitches in all three families, mu = the
    league share of f among pitches by pitchers of the starter's hand in the same window)`, with c = 1 (not nested: the
    target is the league).
  - The hand is lab 1's `c_hand`.
- **Level 2: u(f|s) toward u(f).** Lab 2 does the same as lab 1's platoon term.
  - For each family f: `estimate_m(x = pitches in f to side s, n = pitches to side s, mu = the starter's raw u(f)
    over all sides, N = his pitches to all sides)`, the nested form.
  - The shrink target at fit time is the shrunk u(f).
- **One m per level.** The plan says "the n-weighted mean of the three families' binomial moment estimates".
  - PROPOSED: weights = each family's total pitches over the qualifying rows. The trials n are the same for all three
    families, so weighting by n alone would give equal weights.
  - Infinite family m's are left out of the mean; if all three are infinite, m = ∞.
- One common m per level makes each level a convex combination of distributions, so the shares sum to 1.

**c. Dropped codes (KN, EP, PO, UN).** PROPOSED: excluded everywhere a family or measure is formed:
- usage;
- the family measures;
- the overall measures R_b and R_p(s);
- the league levels L and L_f.

Reasons:
- The plan defines L "over all three families".
- I1's two sums are only comparable when the overall and family measures cover the same pitches.

The usual-side rule (item f) counts all pitches, because it is a count of sides, not a family measure.

**d. History `events` and `type` are permitted inputs.**
- The builder reads each batter's and pitcher's history `events` and `type` from the vault:
  - to find AB-ending pitches;
  - to form the value x;
  - for balls in play with no launch data, which "use their actual result" (Section 3), i.e. 1 if `events` is a single,
    double, triple or home run, else 0.
- This is before-D history, never the scored game's own pitches (plan Section 2: "A game's own pitches are same-day
  data and never enter its features"), and the leak checks cover it (item g).
- Lab 3's builder made the same kind of read through `raw_tables()`.
- The target hit column in `analysis/cache/btslab_features.parquet` is never read. The builder reads only `game_pk`,
  `id`, `date`, `season` and `opp_sp` from that table, after checking its file SHA-256 against
  `14cf9d8b75516ce1349558a7e378e130a517fd145296b35ab68b7692e8f63497`, as lab 3's builder does. The leak check also
  reads `home_id` and `away_id`, only to draw the doubleheader sample, as lab 3's does.
- PROPOSED: this reading of "actual result".

**e. T5's bootstrap.**
- **Refers to:** plan Section 8 T5, "Point estimates; date-bootstrap intervals, seed 20261106."
- **Decided (amendment 2):** T5 resamples units (players: batters, or pitchers), not dates. All of a unit's seasons
  are carried together, with 2,000 resamples and seed 20261106.
- **This departs from the plan's wording "date-bootstrap".**
- **Why.** T5's data are unit-season rows, one per player-season, not dated batter-games.
  - Each row carries one date, the date of its selected row (amendment 8). Those dates cluster at season ends.
    Measured on key columns only, 2022–24:
    - 63.0% of 1,966 batter-seasons fall in their season's last 7 days, and the 5 busiest dates hold 49.5%.
    - For 1,117 starter-seasons the figures are 43.3% and 14.2%.
  - So a date bootstrap over them would barely vary, and it would not resample the main source of dependence: the same
    player across seasons.
  - The previous version of this item said the opposite, relying on the retracted finding (amendment 8). That
    sentence is withdrawn.
- T5 is a diagnostic with no rule.

**f. Side s.** The plan is followed as written; no change.
- "the batter's usual side (the side of most pitches seen, 2019 to D)".
- Switch hitter: "at least 20% of pitches seen from each side", in which case s is the side opposite the starter's
  hand.
- Starter hand as lab 1's `c_hand`.
- Counting covers regular-season vault pitches from 2019-01-01 to before D.
- This differs from lab 3, which used the bio `bat_side`. The two can disagree for some batters.
- A batter with no pitches before D has no s, and has I1_miss = 1.

**g. Leak-check design.**
- **Structural.**
  - Every row's lab 2 max-source date (`I_max_date`: the latest pitch date used by any lab 2 count, including the
    usual-side count) is before D.
  - The contact table's training max date is before 2022-01-01, read from the CSV header and asserted, as in lab 3.
- **Future deletion.**
  - 300 random rows: seed 20261130, 15 dates × 5 rows per season, as lab 3.
  - Doubleheader rows: both games of 25 doubleheader pairs, 3 rows each, drawn from the same generator after the random
    rows, as lab 3. That is 150 rows, or the number available.
  - Every source row dated on or after D is deleted, every lab 2 column is rebuilt, and the result is compared exactly.
- **Plain-path recompute.**
  - 20 rows, seed 20261132.
  - The raw counts are recomputed by plain filtering of the vault rows.
  - Then u(f|s), r_b(f), r_p(f|s) and I1 are recomputed by a separate plain-Python loop. Shrinkage and the I1 formula
    are written out directly, with the train-2022–24 fold's prior strengths passed in.
  - Tolerance is 1e-9 relative, as lab 3's `close()`.
- **Usage sums (stop rule 3).**
  - The shrunk shares sum to 1 within 1e-9 on every row with history.
  - Asserted by the analysis script in each fold, because shrinkage is applied at fit time.
  - The builder asserts that the stored family counts sum to the stored totals.

**h. Expected runtimes** (estimates, not timings).
- Builder: 1–3 minutes. Lab 3's builder took 17 s; lab 2 forms about 5–10 times as many windowed tables.
- Leak checks: 15–30 minutes. Lab 3's took 781 s, dominated by the future-deletion rebuilds.
- Null study: about twice lab 3's 2,006 s on 16 processes. Each seed now fits M1, M2, M3 and M4 with inner CV and two
  sets of 10 placebo refits.

**i. `--real` preconditions.**
- PROPOSED, as lab 3 A2 item 5: `--real` stops before any fit unless:
  - the leak-check log contains `LEAK CHECK: PASS`;
  - the planted, planted-null and null-study logs contain their PASS lines;
  - stop rule 6 passes.

**j. Row order (amendment 11).**
- **No lab 2 computation may depend on table row order.**
- Wherever an ordering matters (most recent, last, first, rolling or cumulative), the code sorts explicitly by game
  date and then by within-day order.
- **Tie-breakers** (lab 1's own orders):
  - batter-game rows: (`date`, `game_pk`, `id`), as lab 1's `load()`;
  - pitch rows: (game date, `game_pk`, `at_bat_number`, `pitch_number`), as lab 1's `raw_tables()` sort.
- **Within-day order does not affect as-of values.** `game_pk` is not chronological between the games of a
  doubleheader. But every as-of quantity uses only dates strictly before D, so within-day order changes no stored
  value; it only has to be deterministic.
  - One exception follows lab 1 as is: the starter's hand is "the most recent `p_throws` before D". On the latest
    earlier date it takes the last pitch in `game_pk` order (`raw_tables()` "hand").
- **PROPOSED (Claude Code; new in this revision):** the builder sorts every input by its key immediately after
  loading:
  - the vault by (`game_pk`, `at_bat_number`, `pitch_number`);
  - lab 1's key columns by (`date`, `game_pk`, `id`).

  The output is then a function of the input's contents only.
- **Shuffle check (leak checks).**
  - The builder is rerun on a copy of every input with its rows shuffled.
  - The output, sorted by (`game_pk`, `id`), must be identical to the original: exact equality of every column.
  - **PROPOSED (Claude Code; new in this revision):** shuffle seed 20261131, unused anywhere in the repo (grep of
    2026-11-09). Exact equality is expected because of the sort-on-load rule; float sums are then formed in the same
    order.

## Unchanged

- Items 1–2 withdraw the 2026 final test and the 2026 build from lab 2.
- Item 3 adds:
  - M3 and M4;
  - T1b (confirmatory), T3b and the M4 placebo;
  - M4's I1 coefficient (descriptive);
  - an inclusion rule;
  - expectations (h) and (i).
- Item 4 changes the planted-test generator.
- Item 5 replaces the null-study centring check with a different, one-sided check.
- Items 6–12 add stop rules, mechanics, seeds and conventions.
- Item 13 records implementation clarifications.
- T1, the primary test, is unchanged. No threshold of the plan or of addenda A1–A2 changes.
- Everything else in the lab 2 plan and addenda A1 and A2 stays as written.

## Addendum

(none)
