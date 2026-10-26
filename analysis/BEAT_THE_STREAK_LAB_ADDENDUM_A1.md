# Beat the Streak lab — Addendum A1 (amendments before results)

**Dated 2026-10-26, written before any feature build or model fit.**
- It amends `analysis/BEAT_THE_STREAK_LAB_PLAN.md`, committed in `f10661f` (btslab sha256 `6fba909a…ee68`). The plan
  file itself is unchanged.
- Basis: the Step 0 data inventory, log `analysis/out/btslab_step0_log.txt`. Step 0 counted rows and keys only. It
  looked at no hit outcome against any feature.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the line starting `btslab A1 sha256:`.
- Where this addendum and the plan differ, this addendum governs.

## Summary in plain language

Before building anything, we checked what data we actually have. Seven things need changing.

- **No 2026 test.** The 2026 data is incomplete, so the lab will draw all its conclusions from 2022–2025 only.
- **Exhibition and neutral-site games out.** Spring-training and postseason games are kept out. So are 18 games
  played at neutral sites, such as London and Tokyo.
- **Actual starter, not the announced one.** Our record of the announced ("probable") starting pitcher turned out to
  be overwritten after the games. We therefore use the pitcher who actually started. This gives the models a small
  unfair advantage. The earlier backtests had the same advantage.
- **Betting odds out of the main model.** We cannot prove when the betting odds were recorded, so they leave the
  main model and are tried only as a side experiment. As far as we can check, they behave like normal pre-game odds.
- **Park effects need older schedules.** They are measured at the ballpark where each game was actually played.
  Building them for 2022 needs the 2019–2021 schedules, which will be downloaded.
- **Weather from the box score.** Temperature and roof status come from the box score's game-time record.
- **Data sources clarified.** The note on where each data field lives is corrected.

The tests, expectations and stop rules do not change.

## Changes

### 1. 2026 is dropped as a final test

**Amends:** Section 0 (Step 0, 2026 rule) and Section 6 (Design, "2026").

**Evidence (Step 0, item 3)**
- `pitch_matrix` has 0 rows for 2026.
- There are no cached box scores or schedules for 2026.
- The Postgres database has 2026 games only from 2026-04-28, and its lineups (`player_predictions.batting_order`)
  are the app's own and cannot be checked against the true starting nine.
- The odds file ends on 2026-04-11, before the database's first 2026 game.

**Change**
- 2026 is not used.
- All conclusions come from 2022–2025: walk-forward is primary and LOSO secondary, as in the plan.
- No result in this lab is out-of-sample in the sense of untouched data (plan Section 11, item 5).

### 2. Regular season from the schedule; neutral sites excluded

**Amends:** Section 2 (Data and unit), Section 4 (common rules and F) and Section 3 (as-of windows).

**Evidence (Step 0, items 1 and 4)**
- `pitch_matrix` has no game-type column and contains spring-training and postseason games: 2022 has 2,572 games
  there against 2,429 eligible regular-season games, and dates from 2022-03-17 to 2022-11-05.
- 18 games in 2022–2025 were played at neutral sites:

| venue (id) | games |
|---|---|
| London Stadium (5381) | 4 |
| Estadio Alfredo Harp Helu, Mexico City (5340) | 4 |
| Rickwood Field (3949) | 1 |
| Field of Dreams (5445) | 1 |
| Williamsport: Muncy Bank Ballpark / Journey Bank Ballpark (2735) | 4 |
| Tokyo Dome (2397) | 2 |
| Gocheok Sky Dome, Seoul (5150) | 2 |

**Change**
- **Regular season** is defined by the schedule's `gameType == "R"`, for the pool and for every feature window.
- Features built from `pitch_matrix` use only pitch rows whose `game_pk` is a regular-season game in the schedule.
  This is why 2019–2021 schedules are fetched (change 5).
- The 18 neutral-site games are excluded from the pick pool and from park factors.
- They still count in batter and pitcher rate windows, since those games were played.

### 3. Opposing starter = the starter who actually pitched

**Amends:** Section 2 ("Games used"), Section 3 (as-of rule exceptions) and Section 4C (opposing starter).

**Evidence (Step 0, item 1e).** The cached schedule's `probablePitcher` equals the box score's actual starter in
99.7–100% of team-games:

| season | team-games equal |
|---|---|
| 2022 | 4,844 of 4,858 |
| 2023 | 4,841 of 4,846 |
| 2024 | 4,850 of 4,852 |
| 2025 | 4,845 of 4,852 |

The schedules were fetched after the games. Late changes of starter normally make this match noticeably less than
complete, so the field was evidently overwritten after the games. It cannot be shown to be pre-game.

**Change**
- The opposing starter is the pitcher who threw the team's first pitch (box score `pitchers[0]`).
- The starter's identity is one more same-day exception to the as-of rule in Section 3. All of his features still use only
  games before D.
- The rule "games where a side has no probable starter are excluded" no longer applies.

**Stated limitation**
- This is a small optimism. Real picks are made without knowing a late scratch.
- An opener's game is described by the opener's history, as it would be pre-game only if the opener was announced.
- R2 and the earlier v4 backtest (`*_starter_pred`, taken from the same cached field) inherit the same optimism.

### 4. Implied runs dropped from M1; kept as exploratory "M1 + market"

**Amends:** Section 4B (implied runs) and Section 5 (models, exploratory list).

**Evidence (Step 0, item 5)**
- **Provenance:** `data/MLB_Basic.csv` has no timestamp or source column and no README. The only note is a code
  comment, "from OddsWarehouse". The snapshot time is undocumented.
- **Market AUC against actual home wins, 2022–2025 (de-vigged moneyline):**
  - close: 0.605 pooled (0.620, 0.603, 0.607, 0.586 by season);
  - open: 0.601 pooled.
  - This is consistent with pre-game lines, far from the ≥ 0.9 a post-game record would give, but it is not proof
    of the timing.
- **Coverage:** 11 of the 9,704 games have no closing odds: 8 unmatched and 2 ambiguous doubleheaders in 2023,
  spread over June–September. Miami (5) and Tampa Bay (3) appear most often.

**Change**
- Per the plan's own rule (4B), implied runs is **dropped from M1**. M1 is groups A and C–G, plus lineup slot from B.
- The plan's exploratory "M1 without B-implied-runs" is therefore M1 itself and is not run separately.
- New exploratory group, labelled **"M1 + market"**: M1 plus team implied runs from the closing total and moneyline,
  exactly as defined in 4B, with the league-average fill and missing flag.
- "M1 + market" carries no rule and enters no confirmatory test.

### 5. Park factors by schedule venue id, with 2019–2021 schedules fetched

**Amends:** Section 4F (park factor).

**Evidence (Step 0, item 4)**
- The schedule's `venue.id` and the box score's info "Venue" agree in 9,704 of 9,704 games.
- The box score's `teams.{side}.team.venue` is the team's home park, not the game's site (the 2025 Tokyo game lists
  Wrigley Field), so it is not used.
- There is no local venue record for 2019–2021. The schedule cache covers 2022-03-01..2025-10-01 only.
- All 81 of the Athletics' 2025 home games are recorded at Sutter Health Park (venue 2529).

**Change**
- **Venue** = the schedule's `venue.id`.
- **2019–2021 schedules** are fetched with the existing per-date schedule fetch in `analysis/backtest_v4.py`:
  `get_json`, with the same URL (`/schedule?sportId=1&date={D}&hydrate=probablePitcher,linescore`).
  - They are cached to the existing `analysis/cache/api/schedule/{D}.json`.
  - Schedules only, no box scores.
  - They supply venue id and game type for the prior seasons of the 2022 park factors.
  - `analysis/fetch_schedule.py` is not used: it is fixed to the 2026 date range and writes a CSV.
- **Excluded from park factors:**
  - 2020 games played at a relocated or neutral site, i.e. any venue that is not the home team's usual park that
    season;
  - all neutral-site games (change 2).
  - The fetched 2019–2021 venue ids identify both.
- **Sacramento (venue 2529)** starts at the neutral factor 1 and moves only with games played there before D, as the
  plan says.

### 6. Temperature and roof from the box score's game-time record

**Amends:** Section 4F (temperature, roof handling) and Section 11, item 3 (weather limitation).

**Evidence (Step 0, items 1 and 4)**
- **Box score weather text:** present for every eligible game, e.g. "68 degrees, Roof Closed". It records "Roof
  Closed" or "Dome" per game:
  - Tropicana Field: 242 dome.
  - Retractable roofs: Chase Field, Rogers Centre, American Family Field, T-Mobile Park, Minute Maid/Daikin Park,
    loanDepot park and Globe Life Field each show both closed and open games.
- **`data/game_weather_log.parquet`:** daily per home team, with no hour and no venue (10,220 rows a year, including
  days without games).

**Change**
- **Temperature** = the box score's game-time degrees.
- **Roof:**
  - "Roof Closed" or "Dome" → league-mean temperature, plus the indoor flag;
  - every other game (open-air parks and retractable roofs recorded open) → the observed temperature, flag off.
- This replaces the plan's blanket treatment of retractable-roof parks as unknown.
- `game_weather_log` is **not used**.
- Wind remains exploratory only, from the same box text.

**Stated limitation.** These are post-game recordings of conditions that exist at first pitch. They stand in for a
pre-game forecast and are slightly optimistic: a forecast is noisier, and roof decisions are sometimes announced
late.

### 7. Fields and where they live (corrects the Step 0 report)

**Amends:** Section 4 (sources) and Section 0 (Step 0's wording about "the database").

**Correction.** The Step 0 report listed all four fields as "present" and then said "none is in the database". The
precise statement is below.

**Postgres (`mlb_engine`)**
- It holds only the live app's 2026 logs: `game_predictions`, `game_outcomes`, `player_predictions`,
  `player_outcomes`, 2026-04-28..2026-10-16.
- It contains **none** of the four fields below and **no** 2022–2025 data.

**The four fields**

| field | present | where |
|---|---|---|
| Batter handedness | yes | `data/pitch_matrix.parquet`, column `stand` (per pitch) |
| Pitcher handedness | yes | `data/pitch_matrix.parquet`, column `p_throws` |
| Batters faced per start | yes | cached box scores (`analysis/cache/api/box/{gamePk}.json`), `stats.pitching.battersFaced` |
| Pitches per reliever per game | yes | cached box scores, `stats.pitching.numberOfPitches` |

- **Batter handedness:** a switch hitter's `stand` depends on the pitcher's hand. The platoon split (4A) uses the
  batter's PAs against each pitcher hand, not a single handedness.
- **Batters faced per start:** could also be derived by counting PA rows of the first pitcher in `pitch_matrix`.
- **Pitches per reliever:** the game date comes from the schedule.
- **Change:** the box scores are the source for batters faced and relief pitch counts. They are only cached for the
  2022–2025 eligible games, so as-of windows reaching before 2022-03-01 use the `pitch_matrix` derivation for
  2019–2021, restricted to regular-season games (change 2).

## Unchanged

- All tests (T1–T5), including T1 and T2's decision rules, seeds and bootstrap settings.
- Claude's expectations (a)–(f) and William's hypotheses W1–W7.
- The stop rules (Section 10).
- The T4 design: N-best-days policy family, primary N = 100, Double Down sensitivity, Streak Saver ignored.
- The as-of rule, except for the added same-day exception in change 3.

## Addendum

(none)
