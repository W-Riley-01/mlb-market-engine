# Beat the Streak lab 2 — Addendum A1 (amendments before results)

**Dated 2026-10-26, written before any lab 2 feature exists.**
- It amends `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`). The plan file
  is unchanged.
- Basis: two contact-table runs, logs `analysis/out/lab2_contact_table_log_run1_FAILED.txt` and
  `analysis/out/lab2_contact_table_log_run2_FAILED.txt`. Those runs built the table and applied its checks. No lab 2
  feature, model or batter-game outcome analysis exists.
- This version was revised before its first commit. An earlier, unpublished version had sha256 `fe3e202d…`; it is
  superseded in full by this one.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A1 sha256:`.
- Where this addendum and the plan differ, this addendum governs.

## Summary in plain language

Lab 2 scores each ball hit into play by how often balls hit at that speed and angle become hits.

The plan's second check assumed that, at a given angle, harder-hit balls are never much less likely to be hits. That
was wrong: soft fly balls drop in, medium ones are caught, hard ones get through. A replacement check, which built
the table from 2019–2020 and predicted 2021, also failed: 2021 had fewer hits on hard-hit balls than 2019–2020.

The cause is that 2019 was different. Its ball carried further, and it recorded badly tracked balls with stand-in
values far more often than later seasons. So the table now leaves 2019 out and is built from 2020–2021. It is then
tested on 2022: it must predict hits clearly better than a single league-wide rate. How far its predicted rates drift
from 2022's actual rates is reported, without a pass mark, because the league's hit rate moves from year to year. The
thresholds that failed are not loosened. The fastest balls, which are rare, now borrow their rate from the next
slower speed at the same angle rather than from all speeds.

## Version history and changes

### a. First failure: sanity check 2 (monotonicity)

**Amends:** plan Section 3, sanity checks, and Section 11, stop rule 4.
- **Check 2**, at most a 0.03 fall in P(hit) between adjacent speed bins within each angle bin from −10° to 40°, among
  cells with ≥ 500 balls: 11 of 51 pairs failed, with a largest drop of 0.386 (table fitted on 2019–2021).
- The plan's assumption was wrong at 20–40°. Hit probability is non-monotone in speed there: bloopers at 65–75 mph,
  routine fly-outs at 80–90 mph, hard hits at 100 mph and up. For example, at 30–40°: 0.808 at 65–70 mph, 0.029 at
  80–85 mph, 0.873 at 105–110 mph.
- Check 2 is withdrawn.

### b. Second failure: check 2b (fit 2019–2020, predict 2021)

The earlier version of this addendum replaced check 2 with an out-of-time check. It failed:
- **Overall:** predicted 0.3368 against an actual 0.3248, a difference of +0.012 against a limit of ±0.01.
- **By speed band** (limit ±0.03 in each of the 12 bands with ≥ 1,000 2021 balls): 90–95 mph +0.033 and 100–105 mph
  +0.034 failed; 60–65 mph was −0.025.
- **Log loss** (no threshold): 0.445 for the table against 0.631 for a constant rate.
- **Balls at 115 mph or more** (179 balls, not checked): predicted 0.534 against an actual 0.827.

Check 2b is withdrawn.

### c. Diagnosis

- **2019 is a different regime.**
  - It was the juiced-ball season.
  - Two repeated, placeholder-like recorded values, 82.9 mph at −21° and 80 mph at 69°, fill two cells: 80–85 mph at
    −30° to −20°, and 80–85 mph at ≥ 60°. In 2019 these cells hold about 3.8% of balls in play; from 2020 on, about
    0.5% and 1.0%.
- **2021 hit rates fell.** Balls in play became hits less often in 2021 (0.325) than in 2019–2020 (0.334), mostly at
  90–105 mph.
- **The thresholds are NOT loosened.** The fix is to leave 2019 out of the table and to replace a calibration test
  that a season-to-season level shift fails with a discrimination test plus a reported drift.

### d. Changes

1. **Training seasons.** The table is fitted on 2020–2021 regular-season balls in play only (schedule gameType R,
   type X, with launch speed and angle). No row dated before 2020 or after 2021 is used in the fit. This replaces the
   plan's 2019–2021.
2. **Check 2b is replaced by a discrimination rule and a descriptive drift report.** Applied to 2022 regular-season
   balls in play with launch speed and angle:
   - **Discrimination (pass/fail):** the table's mean log loss (natural log) must beat a constant-rate baseline by at
     least 0.10. The constant is the 2020–2021 hit rate on the fitting balls.
   - **Drift (no pass/fail):** mean predicted against actual hit rate, overall and for each of the 13 speed bins
     (pooled over angles), with counts.
   - Only league-level ball-in-play outcomes for 2022 are read (the `events` of 2022 regular-season balls in play). No
     batter-game outcome is read, and nothing is grouped by batter or game.
   - **If the discrimination rule fails:** stop (stop rule 4) and report. The threshold is not changed.
3. **Sparse cells at high speed.** A cell with fewer than 100 balls at speed 105 mph or more (the 105–110, 110–115 and
   ≥ 115 mph bins) takes the rate of the next-lower-speed cell in the same angle bin when that cell has at least 100
   balls. The rule is not chained: if that cell also has fewer than 100 balls, the sparse cell keeps the angle-bin
   pooled rate. All other sparse cells keep the angle-bin pooled rule.
4. **Check 1** (barrels, ≥ 98 mph at 20–35°, against weak contact, < 80 mph; mean P(hit) difference at least +0.25) is
   unchanged, and is computed on the 2020–2021 table.
5. **Descriptive, no rule:** the per-season count of balls in play in the two oversized cells, and its share of the
   season's balls in play, continues for 2019–2025, from launch data only. Their hit rates are reported for the
   fitting seasons 2020–2021 only.

### e. Disclosure

Every lab 2 feature is a ratio to the league level in the same window. That cancels a uniform shift in the league's
hit rate on balls in play, such as the 2021 fall, but not a shift that varies by pitch family.

### f. Unchanged

Everything else in the plan stays as written, including:
- the binning (13 left-closed speed bins, 11 angle bins) and the 100-ball sparse threshold;
- the angle-bin pooled rule for sparse cells other than those in d3;
- all features, models, tests, stop rules and self-tests.

## Addendum

(none)
