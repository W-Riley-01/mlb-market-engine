"""POST-HOC (B4, B5). Swapped-back probabilities: calibration intercept/slope, and Brier skill after
recalibration (intercept only; intercept + slope), cross-fitted over 5 game-grouped folds so the
recalibration is never scored on the games it was fitted on. Props: skill vs baseline (c) and AUC of (c)."""
import numpy as np
import pandas as pd

import lib
import stats
from swapped import add_swapped
from headline import skill_vs

FOLDS = 5


def fit2(p, y, iters=50):
    x = stats.logit(np.asarray(p, float))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (mu * (1 - mu))[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def crossfit(d: pd.DataFrame, mode: str) -> np.ndarray:
    out = np.empty(len(d))
    fold = d["gp_id"].to_numpy() % FOLDS
    x = stats.logit(d["p"].to_numpy())
    y = d["y"].to_numpy().astype(float)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        if mode == "int":
            a, _ = stats.calib_fit(d["p"].to_numpy()[tr], y[tr])
            z = a + x[te]
        else:
            a, b = fit2(d["p"].to_numpy()[tr], y[tr])
            z = a + b * x[te]
        out[te] = 1 / (1 + np.exp(-z))
    return out


def main():
    g = lib.analysis_games()
    p = lib.load_players()
    T = add_swapped(lib.target_rows(g, p), g, p)
    targets = ["home_win"] + [f"k{t}_{s}" for t in lib.K_THRESHOLDS for s in ("away", "home")] + list(lib.PROPS)
    rows = []
    for tgt in targets:
        for ver in ("v2", "v3"):
            d = T[T.target.eq(tgt) & T.version.eq(ver) & T.p_sw.notna()].copy()
            d["p"] = d["p_sw"]
            prop = tgt in lib.PROPS
            o = stats.bias_skill(d, seed=91)
            rm = stats.row_metrics(d, B=300 if prop else stats.B_ROW, seed=92)
            row = {"target": tgt, "version": ver, "n": o["n"], "events": o["events"], "mean_p": o["mean_p"],
                   "rate": o["rate"], "bias": o["bias"], "bias_lo": o["bias_lo"], "bias_hi": o["bias_hi"],
                   **{k: rm[k] for k in ("cal_int", "cal_int_lo", "cal_int_hi", "cal_slope", "cal_slope_lo",
                                         "cal_slope_hi", "auc", "auc_lo", "auc_hi")},
                   "bss_raw": o["bss"], "bss_raw_lo": o["bss_lo"], "bss_raw_hi": o["bss_hi"]}
            for mode in ("int", "int_slope"):
                dr = d.assign(p=crossfit(d, mode))
                r = stats.bias_skill(dr, seed=93)
                row.update({f"bss_{mode}": r["bss"], f"bss_{mode}_lo": r["bss_lo"], f"bss_{mode}_hi": r["bss_hi"]})
                if prop and tgt in ("p_1h", "p_1hr"):
                    e, lo, hi = skill_vs(dr, "base_own")
                    row.update({f"bss_vs_own_{mode}": e, f"bss_vs_own_{mode}_lo": lo, f"bss_vs_own_{mode}_hi": hi})
            if prop and tgt in ("p_1h", "p_1hr"):
                e, lo, hi = skill_vs(d, "base_own")
                row.update({"bss_vs_own_raw": e, "bss_vs_own_raw_lo": lo, "bss_vs_own_raw_hi": hi})
                rb = stats.row_metrics(d.assign(p=d["base_own"]), B=300, seed=94)
                row.update({"auc_base_own": rb["auc"], "auc_base_own_lo": rb["auc_lo"], "auc_base_own_hi": rb["auc_hi"]})
            rows.append(row)
            print(tgt, ver, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(lib.OUT / "posthoc_recal.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    c1 = ["target", "version", "n", "events", "mean_p", "rate", "bias", "bias_lo", "bias_hi", "cal_int", "cal_int_lo",
          "cal_int_hi", "cal_slope", "cal_slope_lo", "cal_slope_hi"]
    c2 = ["target", "version", "auc", "auc_lo", "auc_hi", "bss_raw", "bss_raw_lo", "bss_raw_hi", "bss_int", "bss_int_lo",
          "bss_int_hi", "bss_int_slope", "bss_int_slope_lo", "bss_int_slope_hi"]
    print(R[c1].round(3).to_string())
    print(R[c2].round(4).to_string())
    c3 = ["target", "version", "auc", "auc_base_own", "auc_base_own_lo", "auc_base_own_hi", "bss_vs_own_raw",
          "bss_vs_own_raw_lo", "bss_vs_own_raw_hi", "bss_vs_own_int", "bss_vs_own_int_lo", "bss_vs_own_int_hi",
          "bss_vs_own_int_slope", "bss_vs_own_int_slope_lo", "bss_vs_own_int_slope_hi"]
    print(R[R.target.isin(["p_1h", "p_1hr"])][c3].round(4).to_string())


if __name__ == "__main__":
    main()
