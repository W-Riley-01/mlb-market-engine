# Beat the Streak lab 3 — Addendum A1 (implementation clarifications)

**Dated 2026-11-06, written after the group H feature builder and leak checks passed, and before any lab 3 model fit
or outcome read.**
- It clarifies `analysis/BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md` (`722712b`, lab3 sha256 `f3d76f8f…a872`). The plan file
  is unchanged.
- **No rule, test, threshold, seed or expectation changes.**
- Basis: the build log `analysis/out/btslab3_features_build_log.txt` and the leak-check log
  `analysis/out/btslab3_leakcheck_log.txt` (structural 0 violations; future deletion 0 of 25,650 cells mismatched;
  plain-path recompute 0 of 265 cells mismatched).
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab3 A1 sha256:`.
- Where this addendum and the plan differ, this addendum governs.

## Summary in plain language

The group H features are built and pass every leak check. While building them, five points came up where the plan
names a file differently, asks for something the safety rules forbid at that step, or did not say exactly what to do.
This addendum records how each was handled. None of them changes what is tested, how it is judged, or what was
expected.

In short: the files follow lab 1's naming; the base table is checked by its file bytes instead of a hash that would
need the outcome columns; the analysis script and self-tests come in part 2; the speed feature's date stamp is the last
game of the previous season; and 210 pitches from 2021 with no recorded zone are handled by the plan's literal
definitions.

## Clarifications

### 1. File names

**Refers to:** plan Section 11, "1. **Feature builder** `analysis/lab3_features.py` →
`analysis/cache/lab3_features.parquet`.", "2. **Leak checks** `analysis/lab3_leakcheck.py`:" and "3. **Analysis
script** `analysis/lab3_analysis.py`".
- The implementation uses lab 1's naming:
  - builder `analysis/btslab3_features.py`;
  - leak checks `analysis/btslab3_leakcheck.py`;
  - feature table `analysis/cache/btslab3_features.parquet` (git-ignored);
  - logs `analysis/out/btslab3_features_build_log.txt` and `analysis/out/btslab3_leakcheck_log.txt`;
  - the analysis script will be `analysis/btslab3_analysis.py`.
- Roles are unchanged.

### 2. Base-table identity check

**Refers to:** plan Section 2, "The builder recomputes it before anything else and stops on a mismatch (Section 10,
rule 2)."
- The content hash `e1fb5d01…00f9a` covers the outcome columns, which the builder must not load.
- The builder instead verified the base table's file SHA-256 against the recorded value
  `14cf9d8b75516ce1349558a7e378e130a517fd145296b35ab68b7692e8f63497` (recorded in `analysis/out/lab2_step1b_log.txt`
  as the file whose content hash is `e1fb5d01…`; the build log reports the match). Identical bytes imply identical
  content.
- The content-hash recompute is done by the analysis script in `--real` mode, where outcomes are loaded anyway. It runs
  before any model fit, and the run stops on a mismatch. In fake-outcome mode it is skipped and the skip is logged.
- The lab 3 feature table's own content hash is
  `d8dccfe335795fec26b7f5bceb138059c10209bbe9a557463072860f1889bfa1` (SHA-256 of
  `pandas.util.hash_pandas_object(df, index=False)`). Its file hash is in the build log.

### 3. Scope of part 1

**Refers to:** plan Section 11, "In order. Each step's log is committed before the next step that depends on it."
- The analysis script (Section 11 step 3) and the self-tests (step 4) are built in part 2.
- Part 1 delivered only the builder and the leak checks.

### 4. H4_max_date

**Refers to:** plan Section 3, "H4's is the end of season s − 1."
- Set to the last regular-season game date of season s − 1, only when a speed exists.
- The structural check (max source date before the game date) passes equally for any date up to Dec 31 of s − 1,
  because all of them precede the season's opening day.

### 5. Null zones

**Refers to:** plan Section 4, "There is no zone 10 and no null zone." and ""In zone" = 1–9; "out of zone" = 11–14."
- The plan's statement that there is no null zone was verified on 2022–25 only.
- The vault has 210 regular-season pitches with a null zone, all in 2021, which fall inside 2022 trailing windows.
- By the plan's literal definitions they count as pitches, and as swings in h_whiff when the description is a swing.
- They are neither in-zone (1–9) nor out-of-zone (11–14), so they enter neither the numerator nor the denominator of
  h_chase or h_zcon.

## Unchanged

Everything else in the lab 3 plan stays as written.

## Addendum

(none)
