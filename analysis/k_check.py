"""Section 4: starter K markets. Pre-specified test (a) pitcher_4k_away v2 vs v3,
all thresholds descriptively, and full-distribution scoring (RPS, randomized PIT)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib
import stats

FIG = lib.OUT / "figures"
KMAX = 16  # outcomes 0..15, 16+ lumped


def boot_mean(x, B=stats.B_DEFAULT, seed=31):
    r = stats.rng(seed)
    return np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    g = lib.analysis_games()
    T = lib.target_rows(g)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # --- (a) pre-specified: k4_away v2 vs v3 --------------------------------------
    rows, boots = [], {}
    for ver in ("v2", "v3"):
        d = T[T.target.eq("k4_away") & T.version.eq(ver)]
        o = stats.bias_skill(d, seed=41)
        boots[ver] = boot_mean((d["p"] - d["y"]).to_numpy(), seed=42 if ver == "v2" else 43)
        rows.append({"version": ver, **{k: o[k] for k in ("n", "events", "mean_p", "rate", "bias", "bias_lo",
                                                          "bias_hi", "brier", "bss", "bss_lo", "bss_hi")}})
    chg = boots["v3"] - boots["v2"]
    test_a = pd.DataFrame(rows)
    test_a.loc[2] = {"version": "v3 - v2", "bias": rows[1]["bias"] - rows[0]["bias"],
                     "bias_lo": np.quantile(chg, .025), "bias_hi": np.quantile(chg, .975)}

    # --- all thresholds, descriptive ------------------------------------------------
    rows = []
    for t in lib.K_THRESHOLDS:
        for side in ("away", "home"):
            for ver in ("v2", "v3"):
                d = T[T.target.eq(f"k{t}_{side}") & T.version.eq(ver)]
                o = stats.bias_skill(d, seed=t)
                rows.append({"target": f"k{t}_{side}", "version": ver,
                             **{k: o[k] for k in ("n", "events", "nonevents", "mean_p", "rate", "bias",
                                                  "bias_lo", "bias_hi", "bss", "bss_lo", "bss_hi")}})
    allk = pd.DataFrame(rows)

    # --- full distribution: RPS, PIT, expected K ------------------------------------
    rows = []
    r = stats.rng(51)
    for _, x in g.iterrows():
        for side in ("away", "home"):
            if x[f"{side}_starter_pitched"] is not True or pd.isna(x[f"{side}_starter_actual_k"]):
                continue
            dist = np.minimum(np.asarray(x[f"{side}_k_dist"], int), KMAX - 1)
            pmf = np.bincount(dist, minlength=KMAX) / len(dist)
            cdf = np.cumsum(pmf)
            k = min(int(x[f"{side}_starter_actual_k"]), KMAX - 1)
            obs = (np.arange(KMAX) >= k).astype(float)
            F_lo = cdf[k - 1] if k > 0 else 0.0
            rows.append({"gp_id": x["gp_id"], "version": x["version"], "regime": x["regime"], "side": side,
                         "actual_k": k, "exp_k": float((np.arange(KMAX) * pmf).sum()),
                         "median_k": x[f"{side}_pitcher_median_k"],
                         "rps": float(((cdf - obs) ** 2)[:-1].sum() / (KMAX - 1)),
                         "pit": F_lo + r.uniform() * (cdf[k] - F_lo), "cdf": cdf})
    kd = pd.DataFrame(rows)
    # climatology baseline: in-sample empirical K distribution per version
    clim = {}
    for ver in ("v2", "v3"):
        a = kd.loc[kd.version.eq(ver), "actual_k"].to_numpy()
        clim[ver] = np.cumsum(np.bincount(a, minlength=KMAX) / len(a))
    kd["rps_clim"] = [float(((clim[v] - (np.arange(KMAX) >= k)) ** 2)[:-1].sum() / (KMAX - 1))
                      for v, k in zip(kd.version, kd.actual_k)]
    rows = []
    for ver in ("v2", "v3", "v3a", "v3b"):
        s = kd[kd.version.eq(ver)] if ver in ("v2", "v3") else kd[kd.regime.eq(ver)]
        dd = stats.paired_diff(s.assign(a=s.rps, b=s.rps_clim), "a", "b", seed=52)
        bt = boot_mean((s.exp_k - s.actual_k).to_numpy(), seed=53)
        rows.append({"set": ver, "starts": len(s), "mean_exp_k": s.exp_k.mean(), "mean_actual_k": s.actual_k.mean(),
                     "bias_k(exp-actual)": (s.exp_k - s.actual_k).mean(),
                     "bias_lo": np.quantile(bt, .025), "bias_hi": np.quantile(bt, .975),
                     "mean_median_k": s.median_k.mean(),
                     "RPS": s.rps.mean(), "RPS_clim": s.rps_clim.mean(), "RPSS": 1 - s.rps.mean() / s.rps_clim.mean(),
                     "RPS-clim": dd["diff"], "lo": dd["lo"], "hi": dd["hi"],
                     "PIT_mean": s.pit.mean(), "PIT_sd": s.pit.std(), "PIT_sd_ideal": 1 / np.sqrt(12)})
    kfull = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ver in zip(axes, ("v2", "v3")):
        s = kd[kd.version.eq(ver)]
        ax.hist(s.pit, bins=10, range=(0, 1), color="#3b6ea8" if ver == "v2" else "#c4572e", alpha=.8)
        ax.axhline(len(s) / 10, color="black", ls="--", lw=1)
        ax.set_title(f"{ver}: randomized PIT, starter Ks (n={len(s)})")
        ax.set_xlabel("PIT (uniform = calibrated; mass at right = under-predicts Ks)")
    fig.tight_layout()
    fig.savefig(FIG / "k_pit.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, ver in enumerate(("v2", "v3")):
        s = allk[allk.version.eq(ver)]
        xpos = np.arange(len(s)) + (i - .5) * .3
        ax.errorbar(xpos, s["bias"], yerr=[s["bias"] - s["bias_lo"], s["bias_hi"] - s["bias"]], fmt="o", capsize=2, label=ver)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks(range(10), allk[allk.version.eq("v2")]["target"], rotation=45)
    ax.set_ylabel("bias = mean predicted − rate")
    ax.set_title("Starter K thresholds: bias with 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "k_threshold_bias.png", dpi=130)
    plt.close(fig)

    for name, df in (("test_a_k4_away", test_a), ("k_thresholds", allk), ("k_fulldist", kfull)):
        df.to_csv(lib.OUT / f"s4_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(df.round(4).to_string())


if __name__ == "__main__":
    main()
