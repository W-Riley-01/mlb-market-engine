# Leak-free walk-forward backtest (v4 simulator): plan and pilot scope

Written 2026-10-22, before the pilot was run. Script: `analysis/backtest_v4.py`. Output ledger:
`analysis/out/ledger_v4_pilot.csv` (new file; `data/backtest_ledger.csv` is never written).

## Inputs for a game on ET date D (all pre-game)
| input | source | as-of rule |
|---|---|---|
| Simulator | `advanced_simulator.run_advanced_monte_carlo` with the v4 wiring fix | — |
| Pitch / contact matrices | `data/pitch_matrix.parquet`, `data/contact_matrix_env.parquet` | rows with game_date < D (re-sliced each date, as in `backtester.py`) |
| League baselines (`resolver.league_avg`) | k_rate, bb_rate per terminal PA from the PQM; single/double/triple/HR per batted ball from the CQM | expanding window, all rows with game_date < D (vault starts 2019-03-20). Replaces the hardcoded 2019–2025 constants for this run only |
| Starters | probable pitchers from the schedule endpoint (`hydrate=probablePitcher`); **fallback**: actual starter (box score `pitchers[0]`) | `starter_source` column = `probable` / `actual_fallback` |
| Lineups | box-score starting nine (`battingOrder`) | **This is the posted lineup, known to the market before first pitch.** It does leak late scratches; accepted. |
| Bullpen | that team's **relief** pitchers in the 30 days before D (from the PQM, rows with D−30 ≤ game_date < D; a relief pitch = any pitch by a team's non-first pitcher of the game), weighted by relief pitches; today's starter removed | passed as `{pitcher_id: pitches}`; `resolver.generate_bullpen_probabilities` averages per-pitcher K/BB by those weights. Empty or thin pool (≤ 50 pitches) → existing league-average branch |
| Weather / park | none (density_ratio = 1.0, no post-sim shift), same as the old ledger | — |
| Recent form, arsenal profiles | not used (no `as_of_date`), same as the old ledger; the local snapshots only exist for 2026-04-22/23 | — |
| Iterations | 1,000; `np.random.seed(gamePk)` per game | — |

## Game selection and outcomes
- Regular season (`gameType R`) games listed as Final on date D.
- **Excluded:**
  - suspended or resumed games (`resumedFrom`/`resumeDate` present, or "Suspended" in the state);
  - postponed games;
  - **tied final scores**;
  - games without a full starting nine on both sides.
- Outcomes from the box score and linescore: runs, F5, NRFI, starters' Ks, batter hits/TB/HR/RBI/SB by lineup slot.

## Odds (comparator, not a model input)
- `data/MLB_Basic.csv` via `odds_ingestion.OddsBook` normalization, matched on date + away + home.
- When more than one row matches (doubleheaders), disambiguate by the final score in the odds file.
- `odds_match` = `unique` / `score_disambiguated` / `ambiguous` / `none`. Ambiguous and none carry no odds.

## Pilot (stop after this)
- The first **300** eligible games from **2025-04-01** onward, in date order.
- **Report:**
  - seconds per game (simulation only, and wall-clock including API calls and data slicing);
  - projected hours for all 2024–2025 regular-season games at 1,000 iterations.
- **Metrics on the pilot,** and on the old leaky ledger for the same gamePks (old: as stored and swapped):
  - home-win AUC;
  - starter-K correlation (predicted median K vs that starter's actual K, and vs the other starter's), restricted to games where the predicted starter actually started;
  - per-batter prop AUC (1+ hit, 1+ HR) vs own outcome and vs the opposite-side same-slot outcome.
- No market-information test on the pilot: n = 300 is too small, and that test waits for the full run.
