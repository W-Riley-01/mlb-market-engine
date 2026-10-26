# As-of audit: can a backtest game on date D see data from D or later?

Audited code: current `backtester.py`, `resolver.py`, `daily_scraper.py`, `advanced_simulator.py`, `matrix_builder.py`,
`enviroment_merger.py`, `weather_builder.py`, `recent_form.py`, `batter_arsenal_profile.py`.

**Caveat:** `data/backtest_ledger.csv` was written on 2026-04-22, six days before the first commit (`ac85dc5`, 04-28).
The exact code that produced it is not in git. The findings below describe the current code; findings marked (*)
depend on code that may have changed since 04-22.

| # | input | can game D use data from D or later? | evidence |
|---|---|---|---|
| 1 | **Relief pitchers ("bullpen")** | **YES: post-game selection leak** | `daily_scraper.py:112–114`: `pitchers = team_data['pitchers']; starter_id = pitchers[0]; bullpen_ids = pitchers[1:]` from the game-D **box score** (`BOXSCORE_URL`, line 8). For a completed game this is the set of relievers who *actually pitched*, including closers used only in close/winning games and position players pitching in blowouts. Their rates come from blinded pre-D data, but *which* arms are pooled depends on how game D unfolded. `backtester.py:220` calls it for every game. |
| 2 | Starting pitcher | Yes, minor | Same lines: `pitchers[0]` is the pitcher who actually started, not the announced probable. This leaks late scratches and openers. |
| 3 | Lineups / batting order | Yes, minor | `daily_scraper.py:115`: `battingOrder` from the game-D box score is the actual starting lineup. It is normally posted pre-game, so it mainly leaks late scratches. |
| 4 | Pitch matrix (PQM): pitcher K/BB, batter K/BB, fastball %, 1st-inning stats, bullpen rates | No | `backtester.py:202–205`: `resolver.pqm = master_pqm[master_pqm['game_date'] < cutoff]`, re-sliced each date. All resolver paths read `self.pqm` (`resolver.py:142, 202–205, 338, 613`). No cached aggregates are built at `__init__` (`resolver.py:17–18` only loads). |
| 5 | Contact matrix (CQM): batter contact/platoon rates | No | `backtester.py:206`: `resolver.cqm = master_cqm[... < cutoff]`. Read at `resolver.py:429–432, 652`. The weather columns merged in by `enviroment_merger.py:47–51` are per row, for that row's own past game. |
| 6 | League-average anchors | **Yes, season aggregates** | `resolver.py:24–37`: `k_rate 0.235, bb_rate 0.084` documented as "2019–2025" rates, plus contact rates; fixed constants that include D and later seasons. Every rate shrinks toward these. (*) |
| 7 | Simulator constants | Partly (*) | `advanced_simulator.py:116–117` (SB attempt 0.09 / success 0.79, "post pitch-clock"), `:145` (pitches/PA from the pitch matrix), `resolver.py:342–347` (fb% 0.55, "real 2025"). These were tuned on data that includes 2023–2025. They were introduced in v3 (07-20), *after* the ledger was written, so they probably did not affect it. |
| 8 | Recent form (hitters/pitchers) | No (not used) | Only read when `as_of_date` is given (`resolver.py:331–333`; bullpen `:606–607`). `backtester.py:228–238` passes no `as_of_date`. |
| 9 | Batter arsenal profiles, pitch-type baselines | No (not used) | `resolver.py:488–490`: only with `as_of_date`, so `batter_profile` is None and the arsenal blend falls back (`:499–500`). The pitcher arsenal mix comes from blinded `p_data`. |
| 10 | xwOBA lookup | No (not used) | Not referenced in `resolver.py`. Only `batter_arsenal_profile.py` / `matrix_builder.py` (CQM column) use it, and that path is inactive (row 9). |
| 11 | Park and weather factors | No (not applied) | `backtester.py:236`: `density_ratio=1.0`. The backtester has no post-sim weather shift. `game_weather_log` (`weather_builder.py:43–55`, temperature randomized by month) only feeds the CQM `density_ratio` column, which the resolver does not read. There are no explicit park factors. |
| 12 | Early-inning tracker parquet | No (not used) | First-inning stats are computed from the blinded PQM (`resolver.py:186–219`), not from `early_inning_tracker.parquet`. |
| 13 | Closing odds (comparator) | n/a, but mismatches possible | `backtester.py:254`: `oddsbook.get(date_str, away_abbr, home_abbr)` keys on date + teams, so the 104 doubleheader pairs in the ledger cannot be told apart. |
| 14 | Outcomes | Data issue | 76 ledger games have tied final scores (likely suspended/incomplete). `ledger_market.py` scored them as home losses. |

## Conclusion
**The audit is not clean.** Row 1 is a direct post-game information leak: the bullpen pool is chosen from the
pitchers who actually appeared in game D. That could produce exactly the kind of small, positive moneyline signal
`ledger_market.py` found (model coefficient 0.57, log-loss gain 0.0024). Rows 2–3 are minor pre-game-adjacent
leaks, and row 6 is mild season-level leakage.

**Under the LEDGER_PLAN rule, the sides result stays provisional.** Clearing it needs a re-run of the backtest with:
- the fixed (v4) simulator;
- probable starters instead of `pitchers[0]`;
- an announced or depth-chart bullpen instead of `pitchers[1:]`;
- as-of league anchors.

The totals null result (slope 0.02, −0.06 to 0.11) stands as is.
