# Beat the Streak lab 2 — Addendum A4 (implementation definitions for Step 1c)

**Dated 2026-11-09. Written before any lab 2 feature table exists and before any lab 2 outcome was read.**
- It adds to `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`) and addenda A1
  (`074878e`, lab2 A1 sha256 `74e182c7…08ba`), A2 (`478847a`, lab2 A2 sha256 `f0300c33…55e9`) and A3 (`097456c`,
  lab2 A3 sha256 `21efe8e5…538f`). None of those files is changed.
- **What it is.** Definitions that the plan and A3 leave out, found when Claude Code read them before writing the
  Step 1c builder. Claude Code stopped before writing any code. The decisions below are from Claude (chat) on
  2026-11-09.
- **What it is not.** Nothing frozen is changed: no rule, test, threshold, seed, expectation or pass rule.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A4 sha256:`.

## Summary in plain language

Before building lab 2's feature table, Claude Code found four points the plan and A3 do not settle. They are:
- a flag that would need box scores, which also hold the outcome;
- what two overall skill measures are shrunk toward;
- how the family-level measures are smoothed;
- a stale line in A3's header.

This addendum records the decisions on those points, five readings of the plan's wording, and the exact list of
folds the build logs. The builder can then be written without deciding anything new.

## Items

### 1. `I_gap` and the "history pitches lost" count: deferred out of Step 1c

**Refers to:** plan Section 2:
- "**Games with dropped pitches** are flagged. These are games where box-score PA ≠ pitch-row PA − truncated_pa
  − IBB: 276 of 9,704 in 2022–25, plus any found in 2019–21. Their remaining pitch rows are used."
- "The flag `I_gap` is carried for the row's own game but is never a model input. A game's own pitches are same-day
  data and never enter its features."
- "The count of history pitches lost is reported."

**The problem.** Building `I_gap` needs box-score PA counts. Each cached box-score file also holds the batter-game
hits (`btslab_features.load_box()` reads `stats.batting.hits`).

**The decision:**
- Step 1c's builder and leak checks do not open any box-score file.
- A separate reporting-only script, named and written later with Step 1d:
  - reads only the team PA and intentional-walk fields from the box-score JSON, never the batting hits field;
  - produces the `I_gap` flag and the history-pitches-lost count exactly as plan Section 2 defines them;
  - writes its own separate file.
- The feature table has no `I_gap` column.
- `I_gap` is never a model input, so nothing the builder, the leak checks or any fit uses depends on it.

### 2. Shrinkage target of R_b and R_p(s): L

**Refers to:**
- plan Section 6, "The batter's per-AB measure against family f is shrunk toward R_b · L_f, where R_b = the batter's
  shrunk overall per-AB measure / L.";
- plan Section 6, "Likewise for the starter, on AB-ending pitches to side s, shrunk toward R_p(s) · L_f." and
  "R_p(s) = the starter's shrunk overall per-AB measure allowed to side s / L.";
- plan Section 5, "L = the league per-AB measure over all three families.";
- lab 1 plan Section 4 (`analysis/BEAT_THE_STREAK_LAB_PLAN.md`), "μ = the league rate in the same window."

Neither the plan nor A3 says what the two overall measures are shrunk toward.

**The decision:**
- R_b's and R_p(s)'s overall measures are shrunk toward L, the league overall per-AB measure in the same window, so
  R_b and R_p(s) shrink toward 1. This follows lab 1's common rule quoted above.
- R_p(s) does not use a side-specific league level. The plan defines only one league level, L, and no side-specific
  one.
- **Each of R_b and R_p(s) gets its own m,** estimated the way lab 1 estimates m for its overall rates (`derive()`,
  e.g. `m["hab"] = bf.estimate_m(f(bat, "a_h"), f(bat, "a_ab"), f(bat, "lg_h_ab"))`): not nested, with μ = the
  league level.
  - **R_b:** `estimate_m(x = the batter's Σx, n = his AB-ending pitches, mu = L)`, on one row per (`id`, season).
  - **R_p(s):** `estimate_m(x = the starter's Σx allowed to side s, n = his AB-ending pitches to side s, mu = L)`, on
    one row per (`opp_sp`, season, s), both sides in one call. This gives one m for R_p(s), as A3 item 13a
    specifies for "Side-specific quantities (u(f|s), R_p(s), r_p(f|s))".
  - Units, row selection and min_n follow A3 item 13a: x and n cover the three families only (A3 item 13c).

### 3. Family measures r_b(f) and r_p(f|s): one pooled m each

**Refers to:**
- plan Section 6, "Empirical Bayes by method of moments, per training fold, on one row per unit-season (lab 1 notes
  item 5).";
- plan Section 6, "Family measures are subsets of the overall window, so they use lab 1's nested-variance estimator
  (`btslab_features.estimate_m` with N = all AB-ending pitches; A2 change c).";
- plan Section 6, "The binomial variance μ(1 − μ) is used as an upper bound for the per-trial variance, which can only
  add shrinkage."

**The precedent.** Lab 1's platoon term pools both hand splits into one `estimate_m` call, giving one m:
- `pl_ = T.drop_duplicates(["id", "season", "c_hand"], keep="last")`
- `m["plat"] = bf.estimate_m(f(pl_, "a_vs_h"), np.nan_to_num(f(pl_, "a_vs_ab")), raw(pl_), N=f(pl_, "a_ab"))`

Both are in `btslab_analysis.derive()`.

**The decision.** Per fold:
- **Batters: one pooled m for r_b(f), pooled across the three families.**
  - The rows are the batter unit-season rows (A3 item 13a), stacked once per family.
  - `estimate_m(x = Σx in family f, n = AB-ending pitches in f, mu = the batter's raw overall measure × L_f / L,
    N = all his AB-ending pitches)`.
- **Pitchers: one pooled m for r_p(f|s), pooled across the three families and the two sides.**
  - The rows are the starter unit-season-side rows (A3 item 13a), stacked once per family.
  - `estimate_m(x = Σx allowed in family f to side s, n = AB-ending pitches in f to side s, mu = his raw overall
    measure allowed to side s × L_f / L, N = all his AB-ending pitches to side s)`.
- **The estimation μ.**
  - A unit's estimation μ is its raw (unshrunk) overall measure times L_f / L. That is the same baseline the shrinkage
    target (R_b · L_f, or R_p(s) · L_f) is built on.
  - **Reason:** using the unscaled overall measure as μ would count the league-wide difference between families as
    between-unit variation. That would inflate the estimated spread, give a smaller m and shrink too little.
  - Raw overall measure = Σx / AB-ending pitches over the three families. A unit with none has no finite μ and is left
    out of the estimate, as `estimate_m` does for non-finite μ or c.
- **Applying it.**
  - Shrunk family measure = (x + m · R_b · L_f) / (n + m), or with R_p(s) for pitchers.
  - r_b(f) = that value / L_f, and likewise r_p(f|s).
- **min_n.** A3 item 13a's rules apply to these m's as well, with their logging: `min_n` = 100; if no row reaches it,
  `min_n` = 20; if none reaches that, m = ∞.

### 4. A3's header line "All of this awaits review" is stale

**Refers to:** A3 header:
- "**Claude Code note.** The unit-season finding was wrong and is retracted (item 13, amendment 8)."
- "Amendment 8 is written to the corrected measurement, not to its first wording."
- "Item 13 j adds two new PROPOSED points (sort on load; shuffle seed 20261131)."
- "All of this awaits review."

**Status.** These were reviewed in the Claude (chat) session on 2026-11-09 and accepted before A3 was committed
(`097456c`):
- the rewritten amendment 8;
- item 13 j's two proposals: sort every input by its key on load, and shuffle-check seed 20261131.

A3 is not edited.

### 5. Readings accepted for Step 1c

**a. `truncated_pa` rows.**
- **Refers to:** plan Section 2, "**`truncated_pa` rows** (an inning ended by a baserunning out mid-PA) are not PAs.
  They are excluded from every lab 2 measure."
- Pitch rows whose `events` is `truncated_pa` are dropped from every lab 2 count: usage, the measures, the
  usual-side count and the league levels.
- The other pitches in that plate appearance stay.

**b. Thin history.**
- **Refers to:**
  - plan Section 6, "**Thin history:** if the batter has no AB-ending pitches, or the starter no pitches to side s,
    in the window, then I1 = 0 and the flag `I1_miss` = 1.";
  - A3 item 13c, "excluded everywhere a family or measure is formed".
- "No pitches to side s" and "no AB-ending pitches" count only the three families' pitches.
- A starter who threw only other-coded pitches (KN, EP, PO, UN) to side s gets `I1_miss` = 1.

**c. Starter hand.**
- **Refers to:** plan Section 6, "Starter hand as in lab 1 (most recent `p_throws` before D)."
- `c_hand` is recomputed from vault PA-ending rows exactly as lab 1's `raw_tables()` "hand" does:
  `E.groupby(["pitcher", "date"]).p_throws.last()` on the PA-ending rows, sorted by (`game_pk`, `at_bat_number`,
  `pitch_number`), then the latest date strictly before D.

**d. Where the shrinkage and I1 code lives.**
- **Refers to:** A3 item 13, "As in labs 1 and 3, the feature table stores as-of counts and league levels; shrinkage,
  standardisation and fills are applied per training fold at fit time."
- The shrinkage and I1 code is a function in `analysis/btslab2_features.py` and is part of Step 1c.
- The builder uses it to log the per-fold m counts (A3 item 13a), and the leak check uses it for I1 with the
  train-2022–24 fold's m. The analysis script will import it.
- The table stores counts and league levels only.

**e. Folds logged.**
- **Refers to:** A3 item 13a, "**Logging (amendment 10).** The builder logs, per fold and per cell (estimator ×
  family × side where applicable)".
- The folds are exactly lab 1's six, `ba.WF + ba.LOSO` in `analysis/btslab_analysis.py`, labelled as in lab 1's
  results log `analysis/out/btslab_results_log.txt`:

| design | train | test |
|---|---|---|
| WF | 2022-2023 | 2024 |
| WF | 2022-2023-2024 | 2025 |
| LOSO | 2023-2024-2025 | 2022 |
| LOSO | 2022-2024-2025 | 2023 |
| LOSO | 2022-2023-2025 | 2024 |
| LOSO | 2022-2023-2024 | 2025 |

- **One shared training set.** The leave-out-2025 fold (LOSO, train 2022-2023-2024) has the same training set as
  walk-forward 2022-2023-2024, and therefore the same m values. The logs say so.
- **Lab 3's T6 folds are excluded.** Those are `T6 = [((2023,), 2024), ((2023, 2024), 2025)]` in
  `analysis/btslab3_analysis.py`. Lab 2 has no 2022-dropped design, and A3 item 3 says "M4 gets no T2 and no T6."

## Unchanged

- Items 1–5 define points the plan and A3 leave out, and record the status of A3's header line.
- No rule, test, threshold, seed, expectation or pass rule of the plan or of addenda A1–A3 changes.

## Addendum

(none)
