# Beat the Streak lab 2 — Addendum A8 (website replay of 2026 World Series Game 5)

**Dated 2026-11-18. Written before any 2026 file is opened for the replay, before `analysis/replay_ws5.py` exists,
and before anything about the game is simulated.**
- It adds to `analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` (`3bd02aa`, lab2 sha256 `0c226655…27ee`) and addenda A1
  (`074878e`, `74e182c7…08ba`), A2 (`478847a`, `f0300c33…55e9`), A3 (`097456c`, `21efe8e5…538f`), A4 (`c93c851`,
  `70cd9322…5af2`), A5 (`1cb627a`, `3180e9f1…31d8`), A6 (`1da2e32`, `99902c15…6fab`) and A7 (`92e3843`,
  `71134cd6…2b0c`). None of those files is changed.
- **Basis:** Claude Code's read-only scoping report for the website replay. That work opened no 2026 data file, no
  2026 schedule or box score and no database, and ran no engine, backtest or data pull. It ran `tests/test_wiring.py`,
  which reads no data (5 of 5 passed).
- The decisions are from Claude (chat) on 2026-11-18.
- The SHA-256 of this file is in `analysis/out/PLAN_HASH.txt` on the last line starting `lab2 A8 sha256:`.
- Where this addendum and the plan or addenda A1–A7 differ, this addendum governs.

## Summary in plain language

The public site, app.diamondmetrics.dev, will show one finished game in the off-season: what the engine predicts for
it, next to what happened. The game is 2026 World Series Game 5. The engine never predicted it live, and the stored
2026 live predictions have home and away crossed, so the numbers come from one replay with the fixed (v4) engine.

A3 keeps 2026 closed until the combined 2026 plan is frozen. This addendum makes one narrow exception for the replay:
- it reads three 2026 files, all about this one game: the cached schedule for 2026-10-28, the game's box score and the
  game's line score;
- it reads no 2026 pitch data, and no field the script uses is a 2026 regular-season outcome;
- the engine uses only data from 2025 and earlier, the same data window the live engine had.

This is a display, not a test. No lab result, rule or expectation depends on it.

## Items

### 1. Purpose

- The public site (app.diamondmetrics.dev) shows, in the off-season, one finished game as predicted versus actual.
- **This is a display, not a test.** No confirmatory claim, no expectation and no stop rule of any lab applies to its
  numbers.
- Only the game simulator runs: `advanced_simulator.run_advanced_monte_carlo` (`advanced_simulator.py` line 317), called
  from a new script `analysis/replay_ws5.py`. That script imports helpers from `analysis/backtest_v4.py` and changes no
  engine file.
- No lab model (M1 to M4) is fitted or run, and no lab feature table is read.

### 2. Scope

- Exactly one game: 2026 World Series Game 5, 2026-10-28, Milwaukee Brewers at Tampa Bay Rays, Tropicana Field.
- Its gamePk is taken from the cached 2026-10-28 schedule (item 4).
- `analysis/backtest_v4.py` skips every game that is not regular season (line 155, `if g.get("gameType") != "R":`).
  The replay script accepts this one game, of type W, and no other. `analysis/backtest_v4.py` is not edited.

### 3. Inputs ("as live")

**Refers to:**
- A3 item 2 (`analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A3.md` lines 93–98), "They move to a separate combined 2026 plan,
  to be written and frozen later, before 2026 is opened.", "Lab 2 does not open 2026 at all." and "2026 stays
  untouched by lab 2. Section 13 item 1's statement now applies to the combined plan.";
- plan Section 13 item 1 (`analysis/BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md` line 362), "Only 2026, if built, is
  untouched.";
- A3 item 2's open items (lines 101–102), "Lab 1's non-arsenal groups read `data/pitch_matrix.parquet`, which ends
  2025-11-01";
- A2 d (`analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A2.md` lines 76–79), "Savant keeps revising labels, so pitch-type
  features must never mix label vintages inside one test." and "**The 2026 final test** uses ONE current-vintage pull
  covering 2025-03-01 through the end of 2026, both years from the same pull. The vault's 2025 rows are not used for the
  2026 arsenal windows."

**The decision.** The replay runs the v4 engine as the leak-free backtest ran it, with the same data window the live
engine had: nothing after the 2025 season.
- **Matrices:** `data/pitch_matrix.parquet` and `data/contact_matrix_env_fixed.parquet`. Each is cut to rows dated
  before 2026-10-28, as `analysis/backtest_v4.py` does (lines 144–145,
  `resolver.pqm = master_pqm[master_pqm["game_date"] < Dts]` and
  `resolver.cqm = master_cqm[master_cqm["game_date"] < Dts]`).
- **Contact matrix: the corrected one, not the live one.** The live engine read the uncorrected file:
  - `resolver.py` line 13 (the default), `cqm_path='./data/contact_matrix_env.parquet',`;
  - the live call path, `auto_log_predictions.py` line 114, `cqm_path="./data/contact_matrix_env.parquet",`.

  The replay uses `data/contact_matrix_env_fixed.parquet` instead, the corrected contact matrix used by the leak-free
  v4 backtest.
- **League baseline:** the hard-coded values in `resolver.py` (lines 34–41). The backtest's expanding-window baseline is
  not used.
- **Bullpens:** empty for both teams. The resolver then uses its league-average bullpen card (`resolver.py` line 642,
  "No bullpen data — use league avg with a slight K boost (relievers K more)").
- "The card the live engine used before first pitch" is inferred from the live code path, not checked against a
  stored prediction. The live engine took its bullpen from the box score (`daily_scraper.py` line 114,
  `bullpen_ids = pitchers[1:] if len(pitchers) > 1 else []`), and before first pitch that list would hold no reliever.
- **Recent-form and arsenal-profile loaders:** off (`load_recent_form=False, load_arsenal_profile=False`).
- **Weather:** `weather.get_game_weather` (`weather.py` line 327). Tropicana Field is marked as a dome (line 96), so it
  returns a neutral entry without calling any service (line 357, "Domes: synthesize a neutral entry, no API call").
  As in the live engine, a dome gets no weather adjustment.
- **Iterations:** 5,000, the live count (`auto_log_predictions.py` line 120, `default=5000`).
- **Seed:** the gamePk, set as `analysis/backtest_v4.py` does (line 198, `np.random.seed(pk % (2 ** 32))`).
- `density_ratio=1.0`, as in the live engine and the backtest.

**Stop rule for the replay script.** Before anything else, the script:
- reads the date column of both matrices;
- logs both files' latest dates and content hashes (SHA-256 of `pandas.util.hash_pandas_object(df, index=False)`);
- stops without simulating if either matrix holds any row dated 2026-01-01 or later.

**Identity checks before simulating.** The script stops without simulating unless all of these hold:
- the schedule holds exactly one game on 2026-10-28;
- that game is gameType W, with the Milwaukee Brewers away and the Tampa Bay Rays at home, at Tropicana Field;
- the box score's teams match the schedule's;
- each side has nine starters;
- `weather.get_game_weather` returns the dome entry.

The first check turns "2026-10-28 was a World Series-only date" from an assumption into a check.

**Option B rejected.** Building the matrices from `data/statcast_current_2025_2026.parquet` was rejected for two
reasons:
- it would read the whole 2026 season, which the plan and A3 item 2 reserve for the combined 2026 plan;
- it would mix label vintages, which A2 d forbids: the vault's rows for 2019–2025 with the current pull's 2026 rows.

### 4. The 2026 files read, and the order

**Only three 2026 files are read:**
- (a) `analysis/cache/api/schedule/2026-10-28.json`, already cached (5,689 bytes, saved 2026-11-06);
- (b) the Game 5 box score, fetched once from statsapi.mlb.com/api/v1/game/{gamePk}/boxscore and saved under
  `analysis/cache/api/box/`. It is not in the cache now;
- (c) the Game 5 line score, fetched once from statsapi.mlb.com/api/v1/game/{gamePk}/linescore, only after the
  prediction file is written and hashed, and saved under `analysis/cache/api/linescore/`.

**Why a line-score file.** The /boxscore endpoint has no line score. `record_outcomes.py` fetches the live feed for
that reason (lines 120–122, "Pull the live feed for a game. The /live endpoint is more verbose than /boxscore but gives
us first-inning play-by-play for NRFI resolution in one request, plus the full boxscore."). A cached 2025 box score
(`analysis/cache/api/box/777505.json`, gamePk taken from `analysis/cache/api/schedule/2025-06-15.json`) has these
top-level keys only: copyright, teams, officials, info, pitchingNotes, topPerformers. The first-5 and run-in-the-1st
results come from the line-score file.

**Order:**
1. **Before simulating**, the script uses only these fields:
   - from the schedule: gamePk, teams, gameDate, gameType, venue and probable pitchers;
   - from the box score: team IDs and names, each team's batting order and pitcher lists, and each listed player's
     ID, name and battingOrder code. The starting nine are the players whose own battingOrder code equals slot times
     100, as `starting_nine` in `analysis/backtest_v4.py` reads it (lines 98–99, "box battingOrder (the team list)
     holds each slot's END-of-game occupant;" and "a player's own battingOrder code is slot*100 only for the
     starter"). The starting pitcher is the first entry of the team's pitcher list.
   - Nothing from any stats or seasonStats block is used before the prediction file is hashed.
2. It runs the identity checks (item 3), simulates, writes the prediction file and logs that file's SHA-256.
3. **Only then** does it fetch the line score and use the result fields:
   - from the line score: the innings, for the first-5 and run-in-the-1st results;
   - from the box score: the final score, each starter's strikeouts, and each batter's at-bats, hits, doubles,
     triples, home runs, RBI and stolen bases.

   It then writes the page file with a results block (item 6).

The files are parsed whole, as JSON is. "Reads" here means which fields the script uses, and at which step. The
cached schedule also holds this game's line score (it was fetched with `hydrate=probablePitcher,linescore`); the
script does not use it.

**Starters.**
- The probable pitchers come from the cached schedule. If one is missing, the script falls back to the box score's
  starting pitcher, as `analysis/backtest_v4.py` does (line 188, `sp[s] = prob or actual`).
- **Disclosure:** the schedule was cached after the game, on 2026-11-06. Its probable pitcher may have been
  overwritten with the actual starter after the game, as in every backtest.

**Not read:**
- `data/statcast_current_2025_2026.parquet`;
- any other 2026 schedule, box score or line score;
- the Postgres database;
- any lab feature table;
- `data/MLB_Basic.csv`.

**Stated plainly:**
- No field the script uses is a 2026 regular-season outcome.
- Blocks that may hold 2026 regular-season figures do exist in these files: each player's seasonStats in the box
  score, and each team's leagueRecord in the schedule.
- The script never uses, prints or logs any field outside this item's lists. seasonStats is never used.
- Nobody opens the 2026 files by hand.
- The parsing is written and tested on 2025 files or made-up data, never on the 2026 files.
- The 2026 holdout (2026 regular-season batter-games) stays closed.
- A3 item 2 stays in force for the combined 2026 plan, unchanged.

### 5. Run once

**The replay runs once.**

**The log records:**
- the script's SHA-256;
- both matrices' content hashes and latest dates;
- the SHA-256 of each 2026 input file: the schedule, the box score and the line score;
- the gamePk, iterations and seed;
- the timings;
- the SHA-256 of the prediction file and of the page file.

**Reruns:**
- A crash before the prediction file is written may be fixed, and the script rerun.
- After the prediction file exists, a rerun is allowed only if it reproduces the prediction file byte for byte (same
  inputs, same seed). If the new prediction differs, the script stops and writes nothing.
- Any change that alters the numbers needs a new addendum first.
- So that a rerun can be compared byte for byte, the prediction file holds no time stamp or timing. Those go in the log
  and the page file.

### 6. Outputs

**Where they go:**
- The prediction file, the page file and the log go to `analysis/out/site/`, named
  `analysis/out/site/ws5_prediction.json`, `analysis/out/site/ws5_page.json` and
  `analysis/out/site/replay_ws5_log.txt`.
- They are committed in this repository as the record.
- The page file is later copied to a separate site repository, diamondmetrics-site.

**The page file's shape:**
- It uses the JSON shape below, from the scoping report.
- Home win = 1 − away win, and first-5 home = 1 − first-5 away − first-5 tie, as `prediction_logger.py` derives them
  (lines 375 and 377).
- The prediction file has the same shape without the results block, the scores and the status.

**Beat the Streak:** the top 5 is ranked by 1+ hit chance from this game's 18 batters only.

```json
{
  "date": "2026-10-28", "updated_at": "…Z", "model_version": "v4", "note": "replay",
  "games": [{
    "game_id": 0, "away_team": "Milwaukee Brewers", "home_team": "Tampa Bay Rays",
    "start_time_utc": "…", "park": "Tropicana Field", "roof": "dome",
    "status": "final", "away_score": null, "home_score": null, "inning": null, "inning_half": null, "outs": null,
    "markets": {"away_win": 0, "home_win": 0, "f5_away": 0, "f5_home": 0, "f5_tie": 0,
                "no_run_first": 0, "median_total": 0, "total_over": {"7.5": 0, "8.5": 0}},
    "weather": {"temp_f": 72, "humidity_pct": 50, "pressure_inhg": 29.92, "wind_mph": 0,
                "wind_from": "—", "wind_out_mph": 0, "wind_label": "Indoors",
                "carry_ft": 0, "hr_score": 50, "hr_label": "Neutral (Dome)"},
    "starters": {"away": {"id": 0, "name": "", "median_k": 0, "k_over": {"3": 0, "4": 0, "5": 0, "6": 0, "7": 0},
                          "first_inning": {"era": 0, "scoreless": 0, "not_scoreless": 0, "opp_obp": 0, "starts": 0}},
                 "home": {}},
    "lineups": {"away": [{"slot": 1, "id": 0, "name": "", "hit": 0, "tb2": 0, "hr": 0, "rbi": 0, "sb": 0}],
                "home": []},
    "results": {"away_score": 0, "home_score": 0, "led_after_5": "away|home|tied",
                "run_in_first": true, "total_runs": 0,
                "starter_k": {"away": 0, "home": 0},
                "batters": {"away": [{"id": 0, "h": 0, "tb": 0, "hr": 0, "rbi": 0, "sb": 0, "line": "2-4, 2B, RBI"}],
                            "home": []}}
  }],
  "beat_the_streak": [{"name": "", "team": "", "opponent": "", "home": true, "hit": 0}]
}
```

### 7. Labelling on the page

The page shows this text, verbatim:

> Replay, run November 2026 with the v4 engine and data through the end of the 2025 season, using the real starting
> lineups. Not a live pre-game prediction.

### 8. Known limits

- **Data through 2025 only.** The engine sees nothing from the 2026 season for any player. The resolver uses every
  matrix row before the cut, with no recency window:
  - a batter with fewer than 10 batted balls in the contact matrix gets the league-average card (`resolver.py` lines
    438–440, `if overall_n < 10:` and "Truly no data — rookie or missing — return league card");
  - a starter with fewer than 150 plate-appearance results gets league strikeout and walk rates (`resolver.py` line
    46, `self.PITCHER_THRESHOLD  = 150  # terminal events, not pitches`, and lines 341–344,
    `if terminal_pa < self.PITCHER_THRESHOLD:`, `k_rate  = self.league_avg['k_rate']` and
    `bb_rate = self.league_avg['bb_rate']`).
- **No extra innings.** The simulator stops after nine (`advanced_simulator.py` line 272, `while self.inning <= 9:`).
  A game tied after nine counts as not an away win, so "home win" includes ties, as in the live engine.
- **League-average bullpens** (item 3).
- **The probable-pitcher caveat** (item 4).
- **Recent-form and arsenal profiles are off.** That matches what the live engine effectively had on game days:
  - their lookups need a snapshot for the exact date (`recent_form.py` line 262,
    `row = self._h_idx.loc[(as_of_date, batter_id)]`);
  - snapshots exist only for 2026-04-22/23 (`analysis/BACKTEST_V4_PLAN.md` line 16, "the local snapshots only exist
    for 2026-04-22/23").

### 9. Scope of changes

- This addendum adds one display exception to A3 item 2, limited to the three files in item 4.
- No frozen test, threshold, seed, expectation or stop rule of any lab changes.
- Lab 2 stays paused at the dry runs.

## Unchanged

- Everything in the lab 2 plan and addenda A1–A7 stays as written, except that item 4 allows the three reads it lists for
  this display.

## Addendum

(none)
