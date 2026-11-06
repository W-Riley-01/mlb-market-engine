# Beat the Streak lab 2 — Addendum A2 (2026 data validation and label vintage)

**Dated 2026-11-06, written before any 2026 pitch pull and before the second validation week is pulled.**
- It amends `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`), Section 10. The
  plan file and addendum A1 (lab2 A1 sha256 `74e182c7…08ba`) are unchanged.
- Basis: the Step 1b log, `analysis/out/lab2_step1b_log.txt`. No 2026 Statcast data has been pulled. The 2026
  schedules and box scores are cached. No lab 2 feature, model or batter-game outcome analysis exists.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `lab2 A2 sha256:`.
- Where this addendum and the plan or addendum A1 differ, this addendum governs.

## Summary in plain language

Before pulling 2026 pitch data, the plan checks the download tool by re-downloading one week of 2025 and comparing it
with the copy we already have. The rows and the speeds matched exactly. But the count of offspeed pitches was 0.66%
higher, over the plan's 0.5% limit.

The reason is not the tool. Baseball Savant has since renamed some pitches: for one pitcher, 24 pitches once called
sinkers are now called splitters. Every tool reads the same Savant data, so every tool fails the same way.

We change the check to ask, pitch by pitch, whether each pitch is in the same family (fastball, breaking, offspeed) in
both copies. That passes for this week, but only because we chose it after seeing the failure. So a second, untouched
week must also pass before any 2026 data is pulled.

Because Savant keeps renaming pitches, the 2026 test also uses one fresh download covering 2025 and 2026 together,
never mixed with the older copy.

## Changes

### a. What happened

**Plan Section 10.1 validation**, re-pull of 2025-06-02 to 2025-06-08 against the vault (27,134 rows; the same rows and
keys as the vault):

| plan check | result |
|---|---|
| row count within 0.5% | 27,134 vs 27,134, **passes** |
| release speed and launch speed equal on ≥ 99.5% of matched pitches | 100%, **passes** |
| pitch-family counts within 0.5% | fastball −0.23%, breaking +0.12%, offspeed **+0.66%, FAILS** |

- **Cause.** Savant relabelled pitch types for 9 of 421 pitchers after the vault was pulled on 2026-04-22. The
  offspeed miss comes from one pitcher whose 24 sinkers (SI) became splitters (FS).
- **No tool can pass.** The results are identical for pybaseball 2.2.7, pybaseballstats 0.6.4 and a direct Savant CSV
  pull, which all read the same endpoint.
- **A stricter rule was also run.** The Step 1b instruction used an exact-label rule that is not in the plan:
  pitch_type, events and launch_speed each agreeing on at least 99.5% of matched pitches. Pitch_type exact agreement
  was 99.488% (139 of 27,134 differ); events 100%; launch_speed 100%.

### b. The 139 relabels

- **Within a family, 105:** SL to ST 70, CU to KC 23, FS to CH 9, KC to ST 2, SI to FF 1.
- **Across families, 34**, i.e. 0.125% of matched pitches: SI to FS 24, FC to SL 10.

### c. Change 1: per-pitch family agreement

**Amends:** plan Section 10.1, the pitch-family count check.
- **The rule.** The plan's family-count tolerance (each family's count within 0.5%) is replaced by per-pitch family
  agreement: among matched pitches (game_pk, at_bat_number, pitch_number), at least 99.5% must be in the same family
  in both pulls. Families are those of plan Section 4. Codes outside the three families count as one "other" group.
  Observed in week 1: 99.875%.
- **Rationale.** A count rule can fail on one pitcher's relabel, or pass when relabels cancel out. Per-pitch agreement
  measures what lab 2 needs: that each pitch lands in the same family.
- **Disclosure.**
  - This change was made after seeing the plan's own rule fail.
  - It replaces a failed check with one that passes.
  - The week-1 pass is NOT an independent test.
- **Independent confirmation.** Validate a second week, 2025-08-04 to 2025-08-10 (not yet pulled), with pybaseball,
  under the amended rule and the unchanged checks below. It must pass before any 2026 pull. If it fails, stop and
  report.
- **Unchanged.** All other Section 10.1 checks: row count within 0.5%; release speed and launch speed equal on at least
  99.5% of matched pitches.
- **Reported descriptively only:** exact-label agreement and events agreement.

### d. Change 2: label vintage

**Amends:** plan Section 10.3, "2019–2025 is not re-pulled".
- Savant keeps revising labels, so pitch-type features must never mix label vintages inside one test.
- **Walk-forward tests (2022–25)** use the existing vault only (pulled 2026-04-22).
- **The 2026 final test** uses ONE current-vintage pull covering 2025-03-01 through the end of 2026, both years from the
  same pull. The vault's 2025 rows are not used for the 2026 arsenal windows.
- The pulled file is hashed, and the hash is recorded in the Step 1b log.
- **Disclosure.** Models trained on vault-vintage labels are applied to current-vintage 2026 features. About 0.125% of
  pitches changed family between vintages in the validation week.

### e. Expectation fix

The 2026 regular season has 2,430 listed games: 2,429 Final and 1 cancelled, played 2026-03-25 to 2026-09-27.

### f. Unchanged

Everything else in the lab 2 plan and addendum A1 stays as written.

## Addendum

(none)
