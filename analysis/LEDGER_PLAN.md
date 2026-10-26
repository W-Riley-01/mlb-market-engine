# Ledger market-information test: frozen plan (written 2026-10-22, before computing anything)

Source: `data/backtest_ledger.csv` (git-ignored, read-only). Script: `analysis/ledger_market.py`.
Anything not listed here is post-hoc and labeled as such.

## Data
- Seasons, date range, game count; counts with closing ML (both sides), closing total, and both closing total juices.
- Games without a final score are dropped. Rows missing a field are dropped only from the analysis that needs that field.
- Season = calendar year of `date`.

## Known properties of the ledger (from code reading, before computing)
- `backtester.py` blinds the pitch/contact matrices to games strictly before each date. Recent-form and arsenal
  profiles are not used (no `as_of_date`). League-average constants in `resolver.py` are 2019–2025 rates (mild leakage).
  Weather is not applied (density_ratio = 1.0, no post-sim shift).
- The ledger stores `model_median_total`, not `total_mean`. The totals test uses the median (deviation from the request).
- The ledger was produced on 2026-04-22 by an unversioned simulator that predates the first commit.

## 1. Orientation
AUC of the model home win prob (1 − model_away_win) vs home won, all games with a final score.
If AUC < 0.5, the ledger is treated as crossed and the swapped value is used: **model P(home win) = model_away_win**
(the same logic as `swapped.py`). The choice is stated in the output.

## 2. Moneyline
- De-vig: multiplicative (proportional) normalization of the closing American odds.
  imp = 100/(o+100) if o > 0 else −o/(−o+100); p_home = imp_home / (imp_home + imp_away).
- Model: home_won ~ a + b_mkt·logit(p_mkt) + b_mod·logit(p_model), logistic, probabilities clipped to [1e-6, 1−1e-6].
- **Cross-fit by season:** for each season, fit on the other seasons and predict the held-out season.
- Report:
  - b_mod with a 95% CI (full-sample fit, game bootstrap B = 2,000) and the per-fold b_mod.
  - Log loss and Brier on the cross-fitted predictions for:
    - (i) market (de-vigged, raw)
    - (ii) model (raw, oriented)
    - (iii) model recalibrated alone (cross-fitted a + b·logit(model))
    - (iv) blend
  - The paired difference blend − market in log loss and Brier, with a game-bootstrap 95% CI.
  - All of the above per season and pooled.

## 3. Totals
- OLS: (actual total − closing total) ~ c + s·(model median total − closing total). Report the slope s with a
  game-bootstrap 95% CI (B = 2,000), pooled and per season. s = 0 means the model adds nothing beyond the line.

## 4. Edge reliability (no threshold search)
- edge = p_model (oriented, raw) − p_market (de-vigged), home side.
- **10 equal-count bins** of edge (fixed; no other bin counts).
- Per bin: n, mean edge, mean market-implied P(home), realized home win rate with a Wilson 95% CI, and
  realized − market.

## Addendum (2026-10-22, written AFTER the computation in `ledger_market.py` had already been run and reported)
- **Orientation:** the ledger is confirmed crossed (AUC 0.414 / 0.412, n=8,354). Swapped probabilities are used throughout;
  the step-1 check is still reported.
- **Decision rule:** a null result (model coefficient near 0) is accepted as is. A positive result is **provisional** until
  (1) the as-of audit (`LEDGER_AUDIT.md`) is clean and (2) it is confirmed by a re-run of the backtest with the fixed (v4) code.
  This rule was added after the results were seen. It is applied to them unchanged, and no analysis was re-run or added because of it.

## 5. Verdict
A plain statement: does the model add measurable information beyond the closing price on sides (b_mod CI
excludes 0 AND cross-fitted blend log loss < market log loss with CI excluding 0) and on totals (slope CI
excludes 0)? n is reported for every number.
