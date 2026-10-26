"""Section 1: totals ladder, weather shift (pre-specified tests b and c), totals dispersion."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib
import stats

FIG = lib.OUT / "figures"


def boot_ci(x: np.ndarray, fn, B=stats.B_DEFAULT, seed=11):
    r = stats.rng(seed)
    bt = np.array([fn(x[r.integers(0, len(x), len(x))]) for _ in range(B)])
    return np.quantile(bt, .025), np.quantile(bt, .975)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    g = lib.analysis_games()
    g["resid"] = g["total_mean"] - g["total_runs"]  # predicted - actual (same sign as bias)
    g["z"] = (g["total_runs"] - g["total_mean"]) / g["total_std"]
    g["unshifted_mean"] = g["total_mean"] - g["shift"]
    out = {}

    # --- continuous totals accuracy by version / regime -------------------------
    rows = []
    for key, sub in [("v2", g[g.version.eq("v2")]), ("v3", g[g.version.eq("v3")]),
                     ("v3a", g[g.regime.eq("v3a")]), ("v3b", g[g.regime.eq("v3b")])]:
        r = sub["resid"].to_numpy()
        z = sub["z"].to_numpy()
        lo, hi = boot_ci(r, np.mean)
        slo, shi = boot_ci(z, lambda a: a.std(ddof=1))
        rows.append({"set": key, "n_games": len(sub), "mean_pred_total": sub["total_mean"].mean(),
                     "mean_actual": sub["total_runs"].mean(), "bias(pred-actual)": r.mean(),
                     "bias_lo": lo, "bias_hi": hi, "MAE": np.abs(r).mean(),
                     "mean_total_std": sub["total_std"].mean(), "SD_z": z.std(ddof=1), "SD_z_lo": slo, "SD_z_hi": shi})
    cont = pd.DataFrame(rows)
    out["totals_continuous"] = cont

    # --- (c) whole vs half rung --------------------------------------------------
    rows = []
    for ver in ("v2", "v3"):
        sub = g[g.version.eq(ver)]
        for x in range(6, 11):
            half, whole = f"{x}.5", f"{x + 1}.0"
            d = pd.DataFrame({"gp_id": sub["gp_id"],
                              "y": (sub["total_runs"] >= x + 1).astype(int),
                              "p_half": sub["total_thresholds"].map(lambda t: t[half]),
                              "p_whole": sub["total_thresholds"].map(lambda t: t[whole])})
            d["se_half"] = (d.p_half - d.y) ** 2
            d["se_whole"] = (d.p_whole - d.y) ** 2
            pdiff = stats.paired_diff(d, "se_whole", "se_half", seed=x)
            rows.append({"version": ver, "event": f"total >= {x + 1}", "half": half, "whole": whole,
                         "n": pdiff["n"], "events": int(d.y.sum()),
                         "brier_half": d.se_half.mean(), "brier_whole": d.se_whole.mean(),
                         "diff(whole-half)": pdiff["diff"], "lo": pdiff["lo"], "hi": pdiff["hi"],
                         "bias_half": d.p_half.mean() - d.y.mean(), "bias_whole": d.p_whole.mean() - d.y.mean(),
                         "pct_pairs_differ": (abs(d.p_half - d.p_whole) > 1e-9).mean()})
    test_c = pd.DataFrame(rows)
    out["test_c_whole_vs_half"] = test_c

    # --- (b) weather shift vs totals accuracy ------------------------------------
    w = g[~g["weather_missing"]].copy()
    w["se_shift"] = (w["total_runs"] - w["total_mean"]) ** 2
    w["se_unshift"] = (w["total_runs"] - w["unshifted_mean"]) ** 2
    w["ae_shift"] = (w["total_runs"] - w["total_mean"]).abs()
    w["ae_unshift"] = (w["total_runs"] - w["unshifted_mean"]).abs()
    rows = []
    for key, sub in [("v2+v3 (primary)", w), ("v2", w[w.version.eq("v2")]), ("v3", w[w.version.eq("v3")])]:
        nz = sub[sub["shift"].ne(0)]
        dm = stats.paired_diff(nz, "se_shift", "se_unshift", seed=21)
        da = stats.paired_diff(nz, "ae_shift", "ae_unshift", seed=22)
        X = np.column_stack([np.ones(len(sub)), sub["shift"].to_numpy()])
        yv = (sub["total_runs"] - sub["unshifted_mean"]).to_numpy()
        beta = np.linalg.lstsq(X, yv, rcond=None)[0][1]
        r = stats.rng(23)
        bt = []
        for _ in range(stats.B_DEFAULT):
            i = r.integers(0, len(sub), len(sub))
            bt.append(np.linalg.lstsq(X[i], yv[i], rcond=None)[0][1])
        rows.append({"set": key, "n_games_with_weather": len(sub), "n_shift_nonzero": dm["n"],
                     "dMSE(shift-unshift)": dm["diff"], "dMSE_lo": dm["lo"], "dMSE_hi": dm["hi"], "dMSE_p": dm["p"],
                     "dMAE": da["diff"], "dMAE_lo": da["lo"], "dMAE_hi": da["hi"],
                     "beta(actual-unshifted ~ shift)": beta, "beta_lo": np.quantile(bt, .025),
                     "beta_hi": np.quantile(bt, .975)})
    test_b = pd.DataFrame(rows)
    out["test_b_shift"] = test_b

    # --- by |shift| bucket ---------------------------------------------------------
    rows = []
    g["bucket"] = g["abs_shift_b"].fillna("weather missing")
    for ver in ("v2", "v3"):
        for b in ["<=0.5", "0.5-1.5", "1.5-2.5", ">2.5", "weather missing"]:
            sub = g[g.version.eq(ver) & g.bucket.eq(b)]
            if sub.empty:
                continue
            r = sub["resid"].to_numpy()
            lo, hi = boot_ci(r, np.mean) if len(r) > 1 else (np.nan, np.nan)
            row = {"version": ver, "abs_shift": b, "n_games": len(sub), "mean_shift": sub["shift"].mean(),
                   "bias(pred-actual)": r.mean(), "bias_lo": lo, "bias_hi": hi,
                   "bias_unshifted": (sub["unshifted_mean"] - sub["total_runs"]).mean(),
                   "MAE": np.abs(r).mean(), "MAE_unshifted": (sub["unshifted_mean"] - sub["total_runs"]).abs().mean(),
                   "SD_z": sub["z"].std(ddof=1),
                   "ladder_pairs_inconsistent": sub["total_thresholds"].map(
                       lambda t: np.mean([abs(t[f"{x}.5"] - t[f"{x + 1}.0"]) > 1e-9 for x in range(6, 11)])).mean()}
            for rung in ("7.5", "8.5", "9.5"):
                p = sub["total_thresholds"].map(lambda t: t[rung])
                y = (sub["total_runs"] >= float(rung) + 0.5).astype(int)
                row[f"brier_{rung}"] = ((p - y) ** 2).mean()
                row[f"bias_{rung}"] = p.mean() - y.mean()
                row[f"events_{rung}"] = int(y.sum())
            rows.append(row)
    buckets = pd.DataFrame(rows)
    out["shift_buckets"] = buckets

    # carry decomposition (why the shift is so large)
    w["wind_ft"] = lib.WIND_FT_PER_MPH * w["weather_wind_out_mph"]
    w["density_ft"] = w["weather_carry_delta_ft"] - w["wind_ft"]
    dec = pd.DataFrame({
        "carry_ft": w["weather_carry_delta_ft"].describe(percentiles=[.05, .5, .95]),
        "wind_ft": w["wind_ft"].describe(percentiles=[.05, .5, .95]),
        "density_ft": w["density_ft"].describe(percentiles=[.05, .5, .95]),
        "shift_runs": w["shift"].describe(percentiles=[.05, .5, .95])}).T
    out["carry_decomposition"] = dec
    top = w.loc[w["shift"].abs().nlargest(6).index,
                ["et_date", "park", "shift", "weather_carry_delta_ft", "wind_ft", "density_ft",
                 "weather_wind_out_mph", "weather_temp_f", "total_mean", "unshifted_mean", "total_runs"]]
    out["largest_shifts"] = top
    coors = w[w.park.eq("Colorado Rockies")]
    out["coors_density"] = pd.DataFrame([{"n": len(coors), "mean_density_ft": coors["density_ft"].mean(),
                                          "mean_shift_runs": coors["shift"].mean(),
                                          "bias_shifted": (coors["total_mean"] - coors["total_runs"]).mean(),
                                          "bias_unshifted": (coors["unshifted_mean"] - coors["total_runs"]).mean()}])

    # --- figures -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ver, c in (("v2", "#3b6ea8"), ("v3", "#c4572e")):
        s = w[w.version.eq(ver)]
        ax.scatter(s["shift"], s["total_runs"] - s["unshifted_mean"], s=6, alpha=.25, color=c, label=f"{ver} games")
    bins = pd.cut(w["shift"], [-4, -1.5, -0.5, -0.01, 0.01, 0.5, 1.5, 2.5, 7])
    m = (w["total_runs"] - w["unshifted_mean"]).groupby(bins, observed=True).agg(["mean", "sem"])
    mid = w["shift"].groupby(bins, observed=True).mean()
    ax.errorbar(mid, m["mean"], yerr=1.96 * m["sem"], fmt="o-", color="black", label="binned mean ±95%")
    xs = np.linspace(-4, 7, 10)
    ax.plot(xs, xs, "--", color="grey", label="slope 1 (shift right-sized)")
    ax.axhline(0, color="grey", lw=.5)
    ax.set_xlabel("weather shift applied (runs)")
    ax.set_ylabel("actual − unshifted predicted mean (runs)")
    ax.set_title("Does the weather shift predict the residual? (games with weather)")
    ax.set_ylim(-12, 15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "totals_shift_vs_residual.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, ver in enumerate(("v2", "v3")):
        s = test_c[test_c.version.eq(ver)]
        xpos = np.arange(len(s)) + (i - .5) * .3
        ax.errorbar(xpos, s["diff(whole-half)"], yerr=[s["diff(whole-half)"] - s["lo"], s["hi"] - s["diff(whole-half)"]],
                    fmt="o", capsize=3, label=ver)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks(range(5), test_c[test_c.version.eq("v2")]["event"])
    ax.set_ylabel("Brier(whole) − Brier(half)")
    ax.set_title("Test (c): whole-number vs half-run rung, same event (<0 = whole rung better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "totals_whole_vs_half.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for k, v in out.items():
        v.to_csv(lib.OUT / f"s1_{k}.csv", index=False)
        print(f"\n=== {k} ===")
        print(v.round(4).to_string())


if __name__ == "__main__":
    main()
