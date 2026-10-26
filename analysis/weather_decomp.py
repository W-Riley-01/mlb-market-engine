"""POST-HOC (B6). Split the totals weather shift into density and wind terms.
shift_runs = 0.05 * carry_ft;  carry_ft = density_ft + 3.8 * wind_out_mph  (weather.score_hr_conditions).
density_ft is reconstructed as carry_ft - 3.8 * wind_out_mph from the stored (rounded) values:
carry is stored to 0.1 ft and wind_out to 0.1 mph, so density_ft carries up to ~±0.24 ft (±0.012 runs) rounding error."""
import numpy as np
import pandas as pd

import lib
import stats

B = stats.B_DEFAULT


def ols_boot(y, X, seed):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = stats.rng(seed)
    bt = []
    for _ in range(B):
        i = r.integers(0, len(y), len(y))  # one row per game, so this is the cluster bootstrap
        bt.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bt = np.array(bt)
    return beta, np.quantile(bt, .025, axis=0), np.quantile(bt, .975, axis=0)


def main():
    g = lib.analysis_games()
    w = g[~g.weather_missing].copy()
    w["wind_runs"] = lib.WIND_FT_PER_MPH * w.weather_wind_out_mph * lib.RUNS_PER_FT_CARRY
    w["density_runs"] = w["shift"] - w["wind_runs"]
    w["unshifted"] = w.total_mean - w["shift"]
    w["y"] = w.total_runs - w.unshifted
    dome = w.weather_is_dome.astype(bool)
    counts = pd.Series({"games with weather fields": len(w), "  domes (shift 0 by design)": int(dome.sum()),
                        "  outdoor": int((~dome).sum()),
                        "  outdoor with carry exactly 0": int((~dome & w["shift"].eq(0)).sum()),
                        "shift != 0 (test b primary n)": int(w["shift"].ne(0).sum()),
                        "weather missing (excluded from both)": int(g.weather_missing.sum())})
    print("=== why n=759 (test b) vs n=1,153 (regression) ===")
    print(counts.to_string())

    dec = pd.DataFrame({c: w.loc[~dome, c].describe(percentiles=[.05, .5, .95])
                        for c in ("weather_carry_delta_ft", "density_runs", "wind_runs", "shift")}).T
    print("\n=== carry decomposition, outdoor games (runs unless _ft) ===")
    print(dec.round(3).to_string())

    rows = []
    for name, sub in [("all with weather", w), ("outdoor non-Coors", w[~dome & w.park.ne("Colorado Rockies")]),
                      ("Coors", w[w.park.eq("Colorado Rockies")]), ("v2", w[w.version.eq("v2")]),
                      ("v3", w[w.version.eq("v3")])]:
        X = np.column_stack([np.ones(len(sub)), sub.density_runs, sub.wind_runs])
        beta, lo, hi = ols_boot(sub.y.to_numpy(), X, seed=len(sub))
        rows.append({"set": name, "n_games": len(sub), "intercept": beta[0], "b_density": beta[1],
                     "b_density_lo": lo[1], "b_density_hi": hi[1], "b_wind": beta[2], "b_wind_lo": lo[2],
                     "b_wind_hi": hi[2], "sd_density_runs": sub.density_runs.std(), "sd_wind_runs": sub.wind_runs.std()})
    reg = pd.DataFrame(rows)
    print("\n=== (actual - unshifted mean) ~ density_runs + wind_runs; coefficient 1 = right-sized ===")
    print(reg.round(3).to_string())

    rows = []
    for name, sub in [("all with weather", w), ("outdoor", w[~dome]), ("outdoor non-Coors", w[~dome & w.park.ne("Colorado Rockies")])]:
        res = {}
        for s in (0, 0.33, 1):
            pred = sub.unshifted + sub.density_runs + s * sub.wind_runs
            res[s] = (sub.total_runs - pred) ** 2
        res["none"] = (sub.total_runs - sub.unshifted) ** 2
        r = stats.rng(7)
        idx = [r.integers(0, len(sub), len(sub)) for _ in range(B)]
        for s in (0, 0.33, "none"):
            dlt = (res[s] - res[1]).to_numpy()
            bt = np.array([dlt[i].mean() for i in idx])
            rows.append({"set": name, "n_games": len(sub), "wind_scale": s, "MSE": res[s].mean(), "MSE_at_scale_1": res[1].mean(),
                         "MSE - MSE(scale 1)": dlt.mean(), "lo": np.quantile(bt, .025), "hi": np.quantile(bt, .975)})
    mse = pd.DataFrame(rows)
    print("\n=== totals MSE by wind-term scale (density term kept at 1; 'none' = no weather shift at all) ===")
    print(mse.round(3).to_string())
    for n, d in (("weather_decomp_counts", counts.to_frame("n")), ("weather_decomp_carry", dec),
                 ("weather_decomp_regression", reg), ("weather_decomp_mse", mse)):
        d.to_csv(lib.OUT / f"{n}.csv")


if __name__ == "__main__":
    main()
