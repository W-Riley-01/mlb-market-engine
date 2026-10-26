"""POST-HOC (B8). K PIT histogram (10 bins) per version, dispersion after removing mean bias,
and slope of K>=5 bias on month (cluster bootstrap by game). Distributions are used as stored and
also swapped back (P1): the swapped view pairs each starter with the K distribution the sim produced
for him (stored under the other side)."""
import numpy as np
import pandas as pd

import lib
import stats

KMAX = 16
MONTH = {"May": 0, "Jun": 1, "Jul": 2, "Aug": 3, "Sep": 4}


def starts(g, swapped):
    r = stats.rng(51)
    rows = []
    for _, x in g.iterrows():
        for side in ("away", "home"):
            if x[f"{side}_starter_pitched"] is not True or pd.isna(x[f"{side}_starter_actual_k"]):
                continue
            src = {"away": "home", "home": "away"}[side] if swapped else side
            dist = np.minimum(np.asarray(x[f"{src}_k_dist"], int), KMAX - 1)
            pmf = np.bincount(dist, minlength=KMAX) / len(dist)
            cdf = np.cumsum(pmf)
            k = min(int(x[f"{side}_starter_actual_k"]), KMAX - 1)
            F_lo = cdf[k - 1] if k > 0 else 0.0
            mean = float((np.arange(KMAX) * pmf).sum())
            rows.append({"gp_id": x.gp_id, "version": x.version, "month": x.month, "actual": k, "mean": mean,
                         "sd": float(np.sqrt((((np.arange(KMAX) - mean) ** 2) * pmf).sum())),
                         "pit": F_lo + r.uniform() * (cdf[k] - F_lo), "p5": float(1 - cdf[4]),
                         "y5": int(k >= 5)})
    return pd.DataFrame(rows)


def boot_games(d, fn, B=stats.B_DEFAULT, seed=5):
    games = d.gp_id.unique()
    groups = {gid: idx for gid, idx in d.groupby("gp_id").indices.items()}
    r = stats.rng(seed)
    out = []
    for _ in range(B):
        pick = r.choice(games, len(games))
        out.append(fn(d.iloc[np.concatenate([groups[gid] for gid in pick])]))
    return np.quantile(out, .025), np.quantile(out, .975)


def main():
    g = lib.analysis_games()
    hist_rows, disp_rows, slope_rows = [], [], []
    for view in ("as stored", "swapped"):
        K = starts(g, view == "swapped")
        for ver in ("v2", "v3"):
            d = K[K.version.eq(ver)].reset_index(drop=True)
            h = np.histogram(d.pit, bins=10, range=(0, 1))[0]
            hist_rows.append({"view": view, "version": ver, "starts": len(d),
                              **{f"{i / 10:.1f}-{(i + 1) / 10:.1f}": int(c) for i, c in enumerate(h)},
                              "expected_per_bin": len(d) / 10})
            # dispersion after removing mean bias: SD of (actual - mean - mean bias) vs RMS of predicted SD
            fn = lambda s: ((s.actual - s["mean"]) - (s.actual - s["mean"]).mean()).std() / np.sqrt((s.sd ** 2).mean())
            lo, hi = boot_games(d, fn, seed=6)
            disp_rows.append({"view": view, "version": ver, "starts": len(d),
                              "mean_bias(pred-actual)": (d["mean"] - d.actual).mean(),
                              "SD_resid_after_bias": ((d.actual - d["mean"]) - (d.actual - d["mean"]).mean()).std(),
                              "RMS_pred_SD": np.sqrt((d.sd ** 2).mean()), "ratio": fn(d), "ratio_lo": lo, "ratio_hi": hi})
            if view == "as stored":  # K>=5 bias pooled over both sides is identical under the swap
                d["e"] = d.p5 - d.y5
                d["m"] = d.month.map(MONTH)

                def slope(s):
                    X = np.column_stack([np.ones(len(s)), s.m])
                    return np.linalg.lstsq(X, s.e.to_numpy(), rcond=None)[0][1]
                lo, hi = boot_games(d, slope, seed=7)
                slope_rows.append({"version": ver, "starts": len(d), "months": sorted(d.month.unique(), key=MONTH.get),
                                   "slope_bias_per_month": slope(d), "lo": lo, "hi": hi})
    H, Dp, S = pd.DataFrame(hist_rows), pd.DataFrame(disp_rows), pd.DataFrame(slope_rows)
    for n, df in (("k_pit_hist", H), ("k_dispersion", Dp), ("k5_month_slope", S)):
        df.to_csv(lib.OUT / f"{n}.csv", index=False)
        print(f"\n=== {n} ===")
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
