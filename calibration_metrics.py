"""
calibration_metrics.py
----------------------
Task 2: metrics, reliability binning, and per-market reports built on top
of the long-format comparison view in calibration.py.

What this module computes
-------------------------
- Brier score, log loss, baseline Brier, skill score (one value per market)
- Reliability curves: predicted-vs-actual binned, with Wilson CIs and
  low-n flags
- Expected Calibration Error (ECE) and bias per market
- A single structured `calibration_report` dict bundling all the above,
  ready to render to terminal, dump to JSON, or feed into a PDF later

Design principles
-----------------
1. Single-version discipline.
   calibration_report() takes one model_version. Never pool v1 and v2 in
   one report. To compare versions, generate two reports and diff with
   compare_versions().

2. Default to iter >= 5000.
   v1 has 15 iter=2000 rows from early April that the handoff flagged as
   MC-noise-heavy. Default min_iterations=5000 excludes them; pass
   min_iterations=0 to include everything.

3. Skill score for readability.
   skill_score = 1 - brier / baseline_brier
     >0    beats "always predict the base rate"
     =0    tied with the naive baseline (no useful signal)
     <0    actively worse than naive — model is mispredicting
   Lets us rank markets without flipping mental sign between "lower
   Brier is better" and "higher accuracy is better."

4. Low-n bins surface explicitly.
   Any reliability bin with n < LOW_N_THRESHOLD gets low_confidence=True
   in the output. Per-market ECE still includes those bins, since dropping
   them would bias the ECE downward — but the caller can see which bin
   readings are statistically thin.

5. Composable.
   brier(), log_loss(), baseline_brier(), skill_score() each take a tiny
   DataFrame with `predicted_prob` and `actual_outcome`. reliability_curve
   and per_market_metrics wrap them with binning and groupby. The caller
   can use either layer depending on whether they want a single number,
   a per-market table, or the full report.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from calibration import build_comparison_view


# Clip used for log loss to avoid log(0) on confidently-wrong predictions.
# 1e-15 matches sklearn's default; in practice it caps the per-row penalty
# around 34 nats.
LOG_LOSS_EPS = 1e-15

# z-score for the 95% Wilson confidence interval on bin actual rates.
WILSON_Z_95 = 1.96

# Bins with fewer than this many rows are flagged low-confidence. At n=20
# the Wilson CI on a 50% rate is still ±22 percentage points, so calling
# anything below that thin is generous.
LOW_N_THRESHOLD = 20

# Default minimum iterations for the report. Excludes the v1 iter=2000
# early-April rows. Caller can pass min_iterations=0 to disable.
DEFAULT_MIN_ITERATIONS = 5000


# ---------------------------------------------------------------------------
#  Market family taxonomy
# ---------------------------------------------------------------------------
def market_family(market_type: str) -> str:
    """Map a specific market_type to its broader category.

    Useful for grouping in summary tables when there are many sub-markets
    of the same logical category (e.g. seven over_total_X.X markets all
    belong to family 'total'; ten pitcher_K_side markets all belong to
    family 'pitcher_k').
    """
    if market_type in ("away_ml", "home_ml"):
        return "moneyline"
    if market_type.startswith("f5_"):
        return "first_five"
    if market_type == "nrfi":
        return "nrfi"
    if market_type.startswith("over_total_"):
        return "total"
    if market_type.startswith("run_line_"):
        return "run_line"
    if market_type.startswith("pitcher_") and market_type.endswith(("_away", "_home")):
        return "pitcher_k"
    if market_type.startswith("player_"):
        return "player_prop"
    return "other"


# ---------------------------------------------------------------------------
#  Core scalar metrics — operate on (predicted_prob, actual_outcome) only
# ---------------------------------------------------------------------------
def brier(df: pd.DataFrame) -> float:
    """Mean squared error between predicted_prob and actual_outcome.
    NaN on empty input. Lower is better. Range [0, 1]."""
    if df.empty:
        return float("nan")
    diff = df["predicted_prob"].to_numpy() - df["actual_outcome"].to_numpy()
    return float(np.mean(diff ** 2))


def log_loss(df: pd.DataFrame) -> float:
    """Binary cross-entropy. Probabilities clipped to LOG_LOSS_EPS to
    avoid -inf on confidently-wrong predictions.

    Penalizes confident wrong predictions much more harshly than Brier,
    so it surfaces a different kind of miscalibration: the high-K pitcher
    rows that consistently predict 0.9 and resolve 0.78 will look much
    worse here than they do in Brier.

    NaN on empty input. Lower is better. Unbounded above."""
    if df.empty:
        return float("nan")
    p = np.clip(df["predicted_prob"].to_numpy(), LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    y = df["actual_outcome"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def baseline_brier(df: pd.DataFrame) -> float:
    """Brier score for the trivial 'always predict the base rate' model.
    Equals base_rate * (1 - base_rate).

    This is the floor any real model needs to beat to demonstrate skill.
    NaN on empty input. Zero when all outcomes are identical (no variance
    to predict)."""
    if df.empty:
        return float("nan")
    base_rate = float(df["actual_outcome"].mean())
    return base_rate * (1.0 - base_rate)


def skill_score(df: pd.DataFrame) -> float:
    """1 - brier / baseline_brier. Positive beats baseline; negative is
    actively worse than predicting the base rate.

    NaN when baseline_brier is 0 (all outcomes the same — skill is
    undefined since there's nothing to predict)."""
    bb = baseline_brier(df)
    if not bb or np.isnan(bb):
        return float("nan")
    return 1.0 - brier(df) / bb


# ---------------------------------------------------------------------------
#  Reliability curve
# ---------------------------------------------------------------------------
def _wilson_interval(n_pos: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """95% Wilson confidence interval for a binomial proportion. Handles
    small n and rates near 0/1 better than the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = n_pos / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def reliability_curve(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-vs-actual reliability binning for a single market.

    Uniform bins on [0, 1]. For each non-empty bin:
      bin_lower, bin_upper  -- inclusive lower, exclusive upper
                               (last bin is inclusive on both ends)
      n                     -- row count in the bin
      mean_predicted        -- average predicted_prob in the bin
      mean_actual           -- empirical rate of actual outcomes
      ci_lower, ci_upper    -- 95% Wilson CI on the actual rate
      abs_error             -- |mean_predicted - mean_actual|
      low_confidence        -- True if n < LOW_N_THRESHOLD

    A perfectly calibrated model would show mean_predicted ≈ mean_actual
    in every bin, with the diagonal y=x line passing through each
    bin's (mean_predicted, mean_actual) point. Systematic bias shows as
    a parallel shift; non-linear miscalibration (e.g. only the
    high-probability bins are off) shows as bend in the curve.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "bin_lower", "bin_upper", "n", "mean_predicted", "mean_actual",
            "ci_lower", "ci_upper", "abs_error", "low_confidence",
        ])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # side='right' on searchsorted means a value equal to an edge gets
    # assigned to the *upper* bin. Then -1 to get a 0-indexed bin. Clip
    # so that the value 1.0 (which would map to n_bins) lands in the
    # last bin.
    bin_idx = np.clip(
        np.searchsorted(edges, df["predicted_prob"].to_numpy(), side="right") - 1,
        0, n_bins - 1,
    )

    p = df["predicted_prob"].to_numpy()
    y = df["actual_outcome"].to_numpy()

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        bin_p = p[mask]
        bin_y = y[mask]
        n_pos = int(bin_y.sum())
        mean_pred = float(bin_p.mean())
        mean_act = float(bin_y.mean())
        ci_lo, ci_hi = _wilson_interval(n_pos, n)
        rows.append({
            "bin_lower":      float(edges[i]),
            "bin_upper":      float(edges[i + 1]),
            "n":              n,
            "mean_predicted": mean_pred,
            "mean_actual":    mean_act,
            "ci_lower":       ci_lo,
            "ci_upper":       ci_hi,
            "abs_error":      abs(mean_pred - mean_act),
            "low_confidence": n < LOW_N_THRESHOLD,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Per-market aggregation
# ---------------------------------------------------------------------------
def _ece(rel_df: pd.DataFrame, total_n: int) -> float:
    """Expected Calibration Error: bin-n-weighted mean of |pred - actual|.
    Operates on the output of reliability_curve."""
    if rel_df.empty or total_n == 0:
        return float("nan")
    return float((rel_df["abs_error"] * rel_df["n"]).sum() / total_n)


def per_market_metrics(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """One row per market_type with every scalar metric we compute.

    Columns:
      market_type, family, n,
      mean_predicted, mean_actual,
      brier, log_loss, baseline_brier, skill_score,
      ece, bias

    bias = mean_predicted - mean_actual. Positive = over-predicting.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for market_type, sub in df.groupby("market_type"):
        rel = reliability_curve(sub, n_bins=n_bins)
        n = len(sub)
        mean_pred = float(sub["predicted_prob"].mean())
        mean_act = float(sub["actual_outcome"].mean())
        rows.append({
            "market_type":    market_type,
            "family":         market_family(market_type),
            "n":              n,
            "mean_predicted": mean_pred,
            "mean_actual":    mean_act,
            "brier":          brier(sub),
            "log_loss":       log_loss(sub),
            "baseline_brier": baseline_brier(sub),
            "skill_score":    skill_score(sub),
            "ece":            _ece(rel, n),
            "bias":           mean_pred - mean_act,
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["family", "market_type"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
#  Top-level report
# ---------------------------------------------------------------------------
def calibration_report(
    model_version: str = "v2",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    min_iterations: int = DEFAULT_MIN_ITERATIONS,
    market_filter: Iterable[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Build a complete calibration report for one model version.

    Pulls fresh long-format data via calibration.build_comparison_view,
    applies iteration/date/market filters, then computes per-market
    metrics and reliability curves.

    Returns a dict:
      metadata    -- model_version, date_range, min_iterations, counts
      markets     -- {market_type: {scalar metrics + reliability DataFrame}}
      summary     -- DataFrame from per_market_metrics
    """
    df = build_comparison_view(
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
    )

    metadata_base = {
        "model_version":  model_version,
        "min_iterations": min_iterations,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }

    if df.empty:
        return {
            "metadata": {
                **metadata_base,
                "date_range":   (None, None),
                "n_total_rows": 0,
                "n_markets":    0,
            },
            "markets":  {},
            "summary":  pd.DataFrame(),
        }

    if min_iterations > 0:
        df = df[df["iterations"] >= min_iterations].copy()

    if market_filter is not None:
        df = df[df["market_type"].isin(set(market_filter))].copy()

    summary = per_market_metrics(df, n_bins=n_bins)

    markets: dict[str, dict] = {}
    for market_type, sub in df.groupby("market_type"):
        rel = reliability_curve(sub, n_bins=n_bins)
        n = len(sub)
        mean_pred = float(sub["predicted_prob"].mean())
        mean_act = float(sub["actual_outcome"].mean())
        markets[market_type] = {
            "family":         market_family(market_type),
            "n":              n,
            "mean_predicted": mean_pred,
            "mean_actual":    mean_act,
            "brier":          brier(sub),
            "log_loss":       log_loss(sub),
            "baseline_brier": baseline_brier(sub),
            "skill_score":    skill_score(sub),
            "ece":            _ece(rel, n),
            "bias":           mean_pred - mean_act,
            "reliability":    rel,
        }

    return {
        "metadata": {
            **metadata_base,
            "date_range":   (str(df["game_date"].min()), str(df["game_date"].max())),
            "n_total_rows": len(df),
            "n_markets":    len(markets),
        },
        "markets":  markets,
        "summary":  summary,
    }


# ---------------------------------------------------------------------------
#  Sort / pretty-print helpers
# ---------------------------------------------------------------------------
def summary_table(report: dict, sort_by: str = "skill_score") -> pd.DataFrame:
    """Get the report's summary DataFrame, optionally re-sorted.

    sort_by options:
      'skill_score'  -- highest skill first (default)
      'brier'        -- lowest Brier first
      'ece'          -- lowest ECE first
      'abs_bias'     -- smallest absolute bias first
      'n'            -- largest sample first
      'market_type'  -- alphabetical
    """
    summary = report["summary"].copy()
    if summary.empty:
        return summary

    if sort_by == "skill_score":
        return summary.sort_values("skill_score", ascending=False).reset_index(drop=True)
    if sort_by == "brier":
        return summary.sort_values("brier", ascending=True).reset_index(drop=True)
    if sort_by == "ece":
        return summary.sort_values("ece", ascending=True).reset_index(drop=True)
    if sort_by == "abs_bias":
        summary["abs_bias"] = summary["bias"].abs()
        return summary.sort_values("abs_bias", ascending=False).reset_index(drop=True)
    if sort_by == "n":
        return summary.sort_values("n", ascending=False).reset_index(drop=True)
    if sort_by == "market_type":
        return summary.sort_values("market_type").reset_index(drop=True)
    return summary


def compare_versions(
    versions: Iterable[str] = ("v1", "v2"),
    metrics: Iterable[str] = ("n", "brier", "skill_score", "ece", "bias"),
    **report_kwargs,
) -> pd.DataFrame:
    """Side-by-side per-market metrics across model versions.

    Returns a wide DataFrame indexed by market_type, with columns
    multi-indexed by (version, metric). Useful for the v1↔v2 comparison
    once v1 has enough resolved data to compare against.

    All kwargs except `versions` and `metrics` are forwarded to
    calibration_report (e.g. min_iterations, n_bins, date filters)."""
    pieces = {}
    for v in versions:
        rep = calibration_report(model_version=v, **report_kwargs)
        if rep["summary"].empty:
            continue
        s = rep["summary"].set_index("market_type")[list(metrics)]
        pieces[v] = s
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1, names=["version", "metric"])


# ---------------------------------------------------------------------------
#  CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model_version = sys.argv[1] if len(sys.argv) > 1 else "v2"

    print(f"=== Calibration report: {model_version} "
          f"(iter >= {DEFAULT_MIN_ITERATIONS}) ===")
    report = calibration_report(model_version=model_version)

    meta = report["metadata"]
    print(f"Date range:  {meta['date_range'][0]} -> {meta['date_range'][1]}")
    print(f"Total rows:  {meta['n_total_rows']:,}")
    print(f"Markets:     {meta['n_markets']}")
    print()

    if report["summary"].empty:
        print("(no data to report)")
        sys.exit(0)

    # --- Main summary table, sorted by skill score ---
    print("=== Summary (ranked by skill score) ===")
    print("skill_score > 0 means the model beats 'always predict base rate'")
    print()
    table = summary_table(report, sort_by="skill_score")
    display_cols = [
        "market_type", "family", "n",
        "mean_predicted", "mean_actual",
        "brier", "baseline_brier", "skill_score",
        "ece", "bias",
    ]
    print(table[display_cols].round(4).to_string(index=False))
    print()

    # --- Markets with the largest absolute bias (most miscalibrated) ---
    print("=== Top 5 most-biased markets (|pred_mean - actual_rate|) ===")
    biased = summary_table(report, sort_by="abs_bias").head(5)
    print(biased[["market_type", "n", "mean_predicted", "mean_actual",
                  "bias", "skill_score"]].round(4).to_string(index=False))
    print()

    # --- Reliability detail for the three most-biased markets ---
    print("=== Reliability curves: top 3 most-biased markets ===")
    print("Bins with low_confidence=True have n < {} (treat as thin).".format(
        LOW_N_THRESHOLD))
    print()
    for mt in biased.head(3)["market_type"]:
        m = report["markets"][mt]
        print(f"--- {mt}  (n={m['n']}, "
              f"pred_mean={m['mean_predicted']:.3f}, "
              f"actual_rate={m['mean_actual']:.3f}, "
              f"bias={m['bias']:+.3f}, "
              f"skill={m['skill_score']:+.3f}) ---")
        print(m["reliability"].round(4).to_string(index=False))
        print()

    print("Next steps:")
    print("  - From Python, use calibration_report() and reliability_curve()")
    print("    to drill into any specific market.")
    print("  - Use compare_versions() once v1 backfill is fully done to get")
    print("    a side-by-side v1/v2 metrics table.")