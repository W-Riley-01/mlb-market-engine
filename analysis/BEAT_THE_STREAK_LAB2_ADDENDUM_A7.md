# Beat the Streak lab 2 — Addendum A7 (analysis-script procedure and readings)

**Dated 2026-11-09. Written after the analysis script's unit checks, smoke run and timing run, and before the dry runs,
any registered self-test, the null study or any batter-game outcome read.**
- It adds to `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`) and addenda A1
  (`074878e`, `74e182c7…08ba`), A2 (`478847a`, `f0300c33…55e9`), A3 (`097456c`, `21efe8e5…538f`), A4 (`c93c851`,
  `70cd9322…5af2`), A5 (`1cb627a`, `3180e9f1…31d8`) and A6 (`1da2e32`, `99902c15…6fab`). None of those files is
  changed.
- **Basis:** the analysis script `analysis/btslab2_analysis.py` (on-disk sha256
  `604115b8dbc4acb11efd33c026c225caf521bd19e9512206838ca9e08bab82d2`), its smoke log and time log (item 7), and
  Claude Code's reports of 2026-11-09 on writing and running it. Items 1 to 6 were decided while earlier versions of
  the script were on disk (the last was `d2f2e0cb…d219`); the T5 values quoted in item 6 were the same in the console
  capture and in both smoke logs, and the version committed with this addendum is the one named in item 7.
- The decisions are from Claude (chat) on 2026-11-09. Items 1 to 4 accept Claude Code's judgment calls or set new
  procedure; item 5 records readings; item 6 adds one unscored column and a disclosure.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the last line starting `lab2 A7 sha256:`.
- Where this addendum and the plan or addenda A1–A6 differ, this addendum governs.

## Summary in plain language

The analysis script is written and has passed its own checks on made-up outcomes. Writing it raised a few practical
points the earlier documents did not settle. This addendum records how they were settled:
- how the table fingerprints are checked when the real hit column may not be loaded;
- the order of the safety checks before the real run;
- what the missing-pitch report must contain before the real run may start;
- how the real run is made truly run-once, even if it crashes;
- three small readings of the rules;
- what the skill-spread diagnostic (T5) actually measures, an extra column that is not scored, and the fact that its
  numbers were seen early. T5 reads no outcomes, so seeing them early reveals nothing about hits.

No test, threshold, seed or expectation changes.

## Items

### 1. Stop rule 6 in fake modes

**Refers to:**
- A3 item 6 (`analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A3.md` lines 281–287), "Before anything is fitted, the analysis
  script recomputes and verifies the content hashes (SHA-256 of `pandas.util.hash_pandas_object(df, index=False)`)
  of:" and "Any mismatch stops the run.";
- A6 item 4, which adds lab 2's table `62445424bda0c468f7dab50225a5a1d7fb346206e113a5cb18ef7367b7beb8f7` to that list;
- plan Section 12 (`analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` line 344), "Lab 1's process is reused with fake
  outcomes on AB ≥ 1 rows; the real hit column is never loaded.";
- lab 3's addendum A1 item 2 (`analysis/BEAT_THE_STREAK_LAB3_ADDENDUM_A1.md` lines 45 and 50–51), "The content hash
  `e1fb5d01…00f9a` covers the outcome columns, which the builder must not load." and "The content-hash recompute is
  done by the analysis script in `--real` mode, where outcomes are loaded anyway. It runs before any model fit, and the
  run stops on a mismatch. In fake-outcome mode it is skipped and the skip is logged."

**The decision:**
- Lab 1's table content hash covers its outcome column, so fake modes (smoke, dry runs, the registered planted and
  planted-null runs, the null study) do not recompute it. They check lab 1's recorded file SHA-256 instead
  (`14cf9d8b75516ce1349558a7e378e130a517fd145296b35ab68b7692e8f63497`) and log that the recompute was skipped.
- Lab 3's and lab 2's content hashes are recomputed in full in every mode.
- `--real` recomputes all three content hashes.
- Any mismatch stops the run, as A3 says.

### 2. Order of checks under `--real`

**Refers to:** A3 item 6 (lines 332–337), "**Order in `--real`.**", "Rule 6 first.", "Then the self-test
preconditions (item 13 i).", "Then rule 7, M1 and then M3.", "Then rule 5." and "Then the placebos (stop rule 2),
before any other result is read."

**The decision.** Under `--real` the order is:
1. the run-once checks (item 4);
2. the PASS logs of A3 item 13i and the `I_gap` completion line (item 3). No data file is read before these pass;
3. stop rule 6 (item 1);
4. the started marker (item 4);
5. the fits, then rule 7 (M1, then M3), rule 5 and the placebos, as A3 says.

The only change is that the log checks come before stop rule 6 rather than after. No check is removed or loosened.

### 3. The `I_gap` precondition

**Refers to:** A6 item 13 (`analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A6.md` line 360), "**When.** Any time before
`--real`. It is never a model input."

**The decision.** `--real` requires `analysis/out/btslab2_igap_log.txt` to exist and its last non-blank line to be
exactly `I_GAP REPORT: COMPLETE`. There is no pass/fail threshold: the report is descriptive. The log is read before
any data file; it is not data.

### 4. Run-once

**Refers to:** A3 item 10 (line 440), "The analysis script refuses `--real` if `lab2_results_log.txt` exists."

**The decision:**
- `--real` refuses if `analysis/out/lab2_real_started.txt` or `analysis/out/lab2_results_log.txt` exists, saying that
  an addendum must record what happened before another run.
- The started marker holds the UTC and local start times and the script's on-disk SHA-256. It is written after every
  gate in items 1 to 3 has passed, immediately before the first fit that uses the real outcome, and is flushed to disk
  at once.
- A gate failure (a missing or failed precondition log, or stop rule 6) writes neither file.
- A run that crashes after the marker cannot be run again without an addendum recording what happened.
- The registered seeds 3300 and 3301 refuse if their log exists (`btslab2_selftest_planted_log.txt`,
  `btslab2_selftest_planted_null_log.txt`); the null study refuses if `btslab2_null_study_log.txt` or
  `btslab2_null_study_seeds.csv` exists. These files are written at the end of the run (or, for the self-tests, when a
  stop rule fires). A crash leaves none of them, and a rerun is then allowed: no real outcome is involved.

### 5. Readings

**a. T5's minimum n, per cell.**
- **Refers to:** A3 item 13a (lines 535 and 544), "**`min_n` = 100 for every lab 2 estimator (amendment 3),** in the
  estimator's own trial unit (item b)." and "**If no row reaches 20 either,** `estimate_m` returns m = ∞, full
  shrinkage to the target, by its own code"; A5 item 8 (lines 225–227), "same x, same rows, same cells."
- **Reading.** The fallback (100; if no row in the cell reaches it, 20; if none reaches 20, no estimate) is applied
  separately in each T5 cell. In the smoke run every T5 cell used 100.

**b. Sign of the placebo statistic.**
- **Refers to:** A3 item 3 (line 117), "the sign convention is loss(new) − loss(old);"; plan Section 8 T4 (lines
  245–246), "**Stop rule:** if the placebo's interval lies entirely below 0 (M2 better), stop and investigate before
  reading any other result (Section 11, rule 2)."
- **Reading.** The placebo statistic is loss(placebo model) − loss(reference model): M2 placebo against M1, M4 placebo
  against M3. Stop rule 2 fires when the upper end of its 95% interval is below 0.

**c. An undefined point ratio in expectation (f).**
- **Refers to:** A6 item 11 (lines 323–324), "**(f):** met if every T5 cell has a point variance greater than 0, every
  T5 SD ratio is below 0.5, and the mean pitcher-cell SD ratio is above the mean batter-cell SD ratio."
- **Reading.** If a cell's point SD ratio is undefined (its overall-skill variance is not positive), (f) scores "not
  met".

### 6. T5: what the registered ratio measures, an unscored companion, and an early view

**Refers to:**
- plan Section 8 T5 (lines 248–250), "For batters (per family) and pitchers (per family and side), the
  method-of-moments estimate of the between-unit variance of true family-specific deviations beyond overall skill.";
- A5 item 8 (`analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A5.md` lines 225–227 and 235), "T5's method-of-moments estimate
  of the between-unit variance of true family-specific deviations uses the same pooled within-unit variance cells as
  the estimator in item 3, instead of the binomial bound: same x, same rows, same cells." and "The SD ratio is the SD
  of the family-specific between-unit deviations divided by that overall between-unit SD.";
- A6 item 8 (lines 258–259 and 264–266), "the moment line in `analysis/btslab_features.py` (line 363) is
  `tau2 = ((n * (x / n - mu) ** 2).sum() - (w * c).sum()) / (n * c ** 2).sum()`." and "**The estimator returns the
  between-unit variance** (τ² above) per cell, not only m, because T5 needs a separate variance per family (batters)
  and per family and side (pitchers), and the overall variance for R_b and per side for R_p(s)."

**What the registered ratio measures.**
- In the nested estimator, a unit's family rate minus its overall rate equals c times (family rate minus the rate on
  the rest of its arsenal), with c = 1 − n/N, up to the league-level scaling that A4 puts into the overall-rate target
  mu. With the denominator Σ n·c², the moment line's τ² is therefore the between-unit variance of the true
  **family-minus-rest-of-arsenal** differences.
- So the registered SD ratio is: SD of (family minus rest-of-arsenal) differences divided by SD of overall skill.
- The plan's phrase "beyond overall skill" leans toward family minus the unit's overall skill. A6 item 8 pins the
  computation to `estimate_m`'s τ². **The registered T5 stays exactly as A6 defines it,** and expectation (f) is scored
  on it alone.

**Unscored companion column.**
- For each family cell, κ² = Σ(n·c²) / Σ n over the cell's qualifying rows, with the estimator's own c.
- The column "deviation-from-overall ratio" = registered point ratio × κ. It is a point estimate only, has no interval,
  is not scored and feeds nothing.

**Early view, disclosed.**
- The T5 point estimates were first seen on 2026-11-09 in the console capture
  `analysis/out/btslab2_smoke_console_run2_CRASHED.txt` (on-disk sha256
  `6d4acb5edb0746a3cbb693aad89a82900f7875481cf1edbdeade1b9bd92c1acf`; local modified time 2026-11-09 21:28:35 -06:00,
  the time the capture was copied into `analysis/out`; the run itself ended at 21:27:28 -06:00), with 20 resamples.
- T5 reads no outcome. Its values are fixed by the Step 1c feature table and its prior strengths (`c8957f2`); any mode
  gives the same point values within rounding.
- The question of what the ratio measures was raised after these values were seen. No scored definition was changed
  because of it: the registered ratio is A6's, and the new column is unscored.

**The values seen** (smoke log `analysis/out/btslab2_smoke_log.txt`, 20 resamples):

| cell | τ² | κ² | registered ratio | deviation-from-overall ratio (unscored) |
|---|---|---|---|---|
| batter fb | +3.137e-04 | 0.2021 | 0.773 | 0.347 |
| batter br | +2.298e-04 | 0.4607 | 0.661 | 0.449 |
| batter os | +2.596e-04 | 0.6206 | 0.703 | 0.554 |
| pitcher fb/L | +1.544e-03 | 0.1854 | 1.874 | 0.807 |
| pitcher br/L | +1.166e-03 | 0.3312 | 1.629 | 0.937 |
| pitcher os/L | +1.111e-03 | 0.3238 | 1.590 | 0.905 |
| pitcher fb/R | +1.117e-03 | 0.1796 | 1.605 | 0.680 |
| pitcher br/R | +9.263e-04 | 0.3267 | 1.462 | 0.836 |
| pitcher os/R | +1.980e-03 | 0.4290 | 2.138 | 1.400 |

- The registered point ratios are above 0.5 in all nine cells. Point estimates do not depend on the number of
  resamples and T5 reads no outcome, so the final point values are expected to equal these. Expectation (f) therefore
  will score "not met".
- The unscored column would lead to the same score: it is above 0.5 in seven of nine cells.

### 7. Procedure notes and evidence

**Refers to:** A3 item 4 (lines 223 and 226–228), "Throwaway dry runs on seeds 9101–9109 come before the registered
runs.", "9101–9102 planted I1;", "9103–9108 planted null-I1;" and "9109 smoke;".

**Procedure:**
- The unit checks (35) run at the start of `--smoke`, before the smoke run; a failure stops it.
- There is no seed flag. The smoke run uses seed 9109: the planted generator, then the null-study code path on 9109
  with one process. `--time` also uses seed 9109.
- The dry runs 9101–9108 run through `--dry-criterion` and are appended to the smoke log.
- Command lines, from the repository root:
  - dry runs: `analysis\.venv\Scripts\python.exe analysis/btslab2_analysis.py --dry-criterion`
  - planted I1, seed 3300: `analysis\.venv\Scripts\python.exe analysis/btslab2_analysis.py --fake planted`
  - planted null-I1, seed 3301: `analysis\.venv\Scripts\python.exe analysis/btslab2_analysis.py --fake planted-null`
  - null study: `analysis\.venv\Scripts\python.exe analysis/btslab2_analysis.py --null-study --procs 16`
  - real run: `analysis\.venv\Scripts\python.exe analysis/btslab2_analysis.py --real`

**Evidence files** (kept unchanged; on-disk hashes; Python logs are CRLF on disk):
- `analysis/out/btslab2_smoke_log_run1_FAILED.txt`, sha256
  `fa95a2a82678668cd42d56089805d4f8bbfb1f7d2602bf77481ed2aace718795`: the first smoke attempt stopped at the unit
  checks. The fault was in the test code of the estimator equivalence check (it did not accept "both infinite"); the
  test was fixed to A5 item 3's form, and the estimator was not changed.
- `analysis/out/btslab2_smoke_console_run2_CRASHED.txt`, sha256
  `6d4acb5edb0746a3cbb693aad89a82900f7875481cf1edbdeade1b9bd92c1acf`: the second attempt crashed in the
  information-only pass-rule printout at the end of the smoke run, after every test section had printed. Fixed by
  parsing the line `selftest_data()` returns.

**The version that produced the quoted numbers:**
- script `analysis/btslab2_analysis.py`, sha256 `604115b8dbc4acb11efd33c026c225caf521bd19e9512206838ca9e08bab82d2`;
- smoke log `analysis/out/btslab2_smoke_log.txt`, sha256
  `7d4eb85518a3fb6ab030fb2a0dcc84c0520e6d1e39f50b639b63981616a50da2`;
- time log `analysis/out/btslab2_time_log.txt`, sha256
  `02829b884afd506b3468096f060f6d47b5a2b36d5e61146446db0d25891f5b9c`, which records the same script hash.
- Any later edit to the script means redoing the unit checks, the smoke run and `--time` before any further step.
  The dry runs append to the smoke log, so later versions of that file differ from the hash above; the hash given is
  that of the file as first committed. CRLF files are stored as LF in git (autocrlf), so a committed blob's hash
  differs from the on-disk hashes given in this addendum.

**Paths that first run under `--real` itself.** No unit check or smoke run can reach these, because each needs the
real outcome column:
- the real branch of the lab 1 content-hash recompute (stop rule 6 under `--real`);
- the real-outcome load;
- the passing case of stop rule 7 (M1 and M3 reproduction).

The new tests cover the rest of the real-only code on constructed or fake inputs.

### 8. Not implemented here

- **The confirmation null study (seeds 3401–3600).** A3 item 8 (lines 411–412): "**The confirmation null study** is
  used only if a registered check fails, under a rule stated in an addendum written before it runs." It stays
  reserved for such an addendum.
- **The `I_gap` script.** It is still to be written, as A6 item 13 defines it.

### 9. Scope of changes

- Items 1 to 4 change procedure only: how stop rule 6 is checked in fake modes, the order of checks under `--real`,
  the `I_gap` completion line, and the run-once marker.
- Item 5 records readings.
- Item 6 adds one unscored column and a disclosure; the registered T5 and the scoring of (f) are unchanged.
- Items 7 and 8 record procedure and status.
- Nothing else changes: no test, threshold, seed, expectation or pass rule.

## Unchanged

- Everything in the lab 2 plan and addenda A1–A6 stays as written, except as items 1 to 4 set procedure.

## Addendum

(none)
