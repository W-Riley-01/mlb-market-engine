"""Market-information test on data/backtest_ledger.csv (plan: LEDGER_PLAN.md). Read-only."""
from pathlib import Path

import numpy as np
import pandas as pd

import stats

LEDGER = Path(__file__).parents[1] / "data" / "backtest_ledger.csv"
B = 2000
EPS = 1e-6


def implied(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logit(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (mu * (1 - mu))[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_mean(x, seed):
    r = stats.rng(seed)
    bt = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return np.quantile(bt, .025), np.quantile(bt, .975)


def main():
    pd.set_option("display.width", 220)
    L = pd.read_csv(LEDGER)
    L["season"] = pd.to_datetime(L.date).dt.year
    n_all = len(L)
    L = L[L.actual_away_score.notna() & L.actual_home_score.notna()].copy()
    has_ml = L.odds_away_ml_close.notna() & L.odds_home_ml_close.notna()
    has_tot = L.odds_total_close.notna()
    has_juice = L.odds_over_close_juice.notna() & L.odds_under_close_juice.notna()
    print("=== 0. Data ===")
    print(f"rows {n_all}; with final score {len(L)}; dates {L.date.min()}..{L.date.max()}")
    print(pd.DataFrame({"games": L.groupby("season").size(), "closing_ML": has_ml.groupby(L.season).sum(),
                        "closing_total": has_tot.groupby(L.season).sum(), "total_juice": has_juice.groupby(L.season).sum(),
                        "first": L.groupby("season").date.min(), "last": L.groupby("season").date.max()}).to_string())
    print(f"totals: closing ML {int(has_ml.sum())}, closing total {int(has_tot.sum())}, juice {int(has_juice.sum())}")

    # --- 1. orientation ---
    y = (L.actual_home_score > L.actual_away_score).astype(int).to_numpy()
    auc_raw = stats.auc(1 - L.model_away_win, y)
    crossed = auc_raw < 0.5
    L["p_model"] = L.model_away_win if crossed else 1 - L.model_away_win
    L["y"] = y
    print(f"\n=== 1. Orientation === AUC(1 - model_away_win -> home won) = {auc_raw:.3f}, n={len(L)}. "
          f"{'CROSSED: using swapped model P(home) = model_away_win' if crossed else 'Not crossed: using 1 - model_away_win'}")
    print(f"AUC after orientation = {stats.auc(L.p_model, y):.3f}")

    # --- 2. moneyline ---
    M = L[has_ml].copy()
    ih, ia = implied(M.odds_home_ml_close), implied(M.odds_away_ml_close)
    M["p_mkt"] = ih / (ih + ia)
    M["overround"] = ih + ia - 1
    X = np.column_stack([np.ones(len(M)), logit(M.p_mkt), logit(M.p_model)])
    yv = M.y.to_numpy().astype(float)
    beta = fit_logit(X, yv)
    r = stats.rng(101)
    bt = np.array([fit_logit(X[i], yv[i]) for i in (r.integers(0, len(M), len(M)) for _ in range(B))])
    print(f"\n=== 2. Moneyline (n={len(M)}; mean overround {M.overround.mean():.3f}) ===")
    print(f"full-sample: a={beta[0]:.3f}  b_mkt={beta[1]:.3f} ({np.quantile(bt[:, 1], .025):.3f}..{np.quantile(bt[:, 1], .975):.3f})"
          f"  b_mod={beta[2]:.3f} ({np.quantile(bt[:, 2], .025):.3f}..{np.quantile(bt[:, 2], .975):.3f})")
    M["p_blend"] = np.nan
    M["p_modcal"] = np.nan
    folds = []
    for s in sorted(M.season.unique()):
        tr, te = M.season.ne(s).to_numpy(), M.season.eq(s).to_numpy()
        b = fit_logit(X[tr], yv[tr])
        M.loc[te, "p_blend"] = 1 / (1 + np.exp(-X[te] @ b))
        bm = fit_logit(X[tr][:, [0, 2]], yv[tr])
        M.loc[te, "p_modcal"] = 1 / (1 + np.exp(-X[te][:, [0, 2]] @ bm))
        folds.append({"held_out": s, "n_test": int(te.sum()), "b_mkt": b[1], "b_mod": b[2]})
    print(pd.DataFrame(folds).round(3).to_string(index=False))
    rows = []
    for s, d in [("pooled", M)] + [(str(k), g) for k, g in M.groupby("season")]:
        yy = d.y.to_numpy()
        row = {"set": s, "n": len(d)}
        for name, col in (("market", "p_mkt"), ("model_raw", "p_model"), ("model_recal", "p_modcal"), ("blend", "p_blend")):
            row[f"LL_{name}"] = ll(d[col].to_numpy(), yy).mean()
            row[f"Brier_{name}"] = ((d[col] - yy) ** 2).mean()
        dll = ll(d.p_blend.to_numpy(), yy) - ll(d.p_mkt.to_numpy(), yy)
        dbr = (d.p_blend - yy) ** 2 - (d.p_mkt - yy) ** 2
        row["dLL(blend-mkt)"], (row["dLL_lo"], row["dLL_hi"]) = dll.mean(), boot_mean(dll, 102)
        row["dBrier(blend-mkt)"], (row["dBr_lo"], row["dBr_hi"]) = dbr.mean(), boot_mean(dbr.to_numpy(), 103)
        rows.append(row)
    print(pd.DataFrame(rows).round(5).to_string(index=False))

    # --- 3. totals ---
    T = L[has_tot & L.model_median_total.notna()].copy()
    T["resid"] = T.actual_away_score + T.actual_home_score - T.odds_total_close
    T["x"] = T.model_median_total - T.odds_total_close
    rows = []
    for s, d in [("pooled", T)] + [(str(k), g) for k, g in T.groupby("season")]:
        Xt = np.column_stack([np.ones(len(d)), d.x])
        yt = d.resid.to_numpy()
        c = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r = stats.rng(104)
        sl = [np.linalg.lstsq(Xt[i], yt[i], rcond=None)[0][1] for i in (r.integers(0, len(d), len(d)) for _ in range(B))]
        rows.append({"set": s, "n": len(d), "mean_actual-line": yt.mean(), "mean_model-line": d.x.mean(),
                     "sd_model-line": d.x.std(), "intercept": c[0], "slope": c[1],
                     "slope_lo": np.quantile(sl, .025), "slope_hi": np.quantile(sl, .975)})
    print("\n=== 3. Totals: (actual - close) ~ (model median - close) ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("model_median_total distinct values:", sorted(T.model_median_total.unique())[:15], "...")

    # --- 4. edge reliability ---
    M["edge"] = M.p_model - M.p_mkt
    M["bin"] = pd.qcut(M.edge.rank(method="first"), 10, labels=False)
    rows = []
    for b_, d in M.groupby("bin"):
        k, n = int(d.y.sum()), len(d)
        lo, hi = stats.wilson(k, n)
        rows.append({"bin": b_ + 1, "n": n, "edge_min": d.edge.min(), "edge_max": d.edge.max(), "mean_edge": d.edge.mean(),
                     "mean_model": d.p_model.mean(), "mean_market": d.p_mkt.mean(), "realized": k / n,
                     "lo": lo, "hi": hi, "realized-market": k / n - d.p_mkt.mean()})
    E = pd.DataFrame(rows)
    print("\n=== 4. Edge reliability (10 equal-count bins of model - market, home side) ===")
    print(E.round(3).to_string(index=False))
    out = Path(__file__).parent / "out"
    M[["date", "season", "game_id", "y", "p_mkt", "p_model", "p_modcal", "p_blend", "edge"]].to_csv(out / "ledger_ml.csv", index=False)
    E.to_csv(out / "ledger_edge_bins.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "ledger_edge_bins.csv", index=False)


if __name__ == "__main__":
    main()
