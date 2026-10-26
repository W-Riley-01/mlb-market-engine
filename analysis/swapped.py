"""POST-HOC (not in SLICES.md). Re-score with home/away swapped back.

advanced_simulator.run_advanced_monte_carlo builds `away_starter_card` = away batters vs home starter,
but AdvancedSimulatedGame uses its `away_starter_card` argument as the AWAY PITCHER's card (bottom half,
indexed by home_batter_index). Net effect: the sim's "away" side is the real home offense vs the away
starter, and vice versa. So, approximately:
  real home win   ~ stored away_win_prob      real away starter K ~ stored home_p_Tk
  real home F5    ~ stored f5_away_prob       real away batter slot i ~ stored home batter slot i props
Approximate because the sim bats that offense in the top half (walk-off / bottom-9th rules flip).
"""
import numpy as np
import pandas as pd

import lib
import stats


def add_swapped(T: pd.DataFrame, g: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Return T with column p_sw = the swapped-back probability (NaN where no swap applies)."""
    T = T.copy()
    T["p_sw"] = np.nan
    gi = g.set_index("gp_id")
    sw = {"home_win": "away_win_prob", "f5_home": "f5_away_prob", "f5_away": "f5_home_prob",
          "rl_home": "run_line_away_minus_1_5_prob", "rl_away": "run_line_home_minus_1_5_prob"}
    sw.update({f"k{t}_away": f"home_p_{t}k" for t in lib.K_THRESHOLDS})
    sw.update({f"k{t}_home": f"away_p_{t}k" for t in lib.K_THRESHOLDS})
    for tgt, col in sw.items():
        m = T.target.eq(tgt)
        T.loc[m, "p_sw"] = T.loc[m, "gp_id"].map(gi[col]).to_numpy()
    pp = p[["gp_id", "side", "batting_order"] + list(lib.PROPS)].copy()
    pp["side"] = pp["side"].map({"away": "home", "home": "away"})
    pp = pp.drop_duplicates(["gp_id", "side", "batting_order"])
    pf = lib.player_frame(g, p)[["gp_id", "side", "batting_order"]].merge(
        pp, on=["gp_id", "side", "batting_order"], how="left")
    for tgt in lib.PROPS:
        m = T.target.eq(tgt)
        assert (T.loc[m, "gp_id"].to_numpy() == pf["gp_id"].to_numpy()).all()
        T.loc[m, "p_sw"] = pf[tgt].to_numpy()
    return T


def main():
    g = lib.analysis_games()
    p = lib.load_players()
    T = lib.target_rows(g, p)

    # swapped game-level probabilities
    sw = {"home_win": "away_win_prob", "f5_home": "f5_away_prob", "f5_away": "f5_home_prob",
          "rl_home": "run_line_away_minus_1_5_prob", "rl_away": "run_line_home_minus_1_5_prob"}
    pmap = {}
    for tgt, col in sw.items():
        pmap[tgt] = g.set_index("gp_id")[col]
    for t in lib.K_THRESHOLDS:
        pmap[f"k{t}_away"] = g.set_index("gp_id")[f"home_p_{t}k"]
        pmap[f"k{t}_home"] = g.set_index("gp_id")[f"away_p_{t}k"]

    # swapped player props: same slot, other side
    pp = p[["gp_id", "side", "batting_order"] + list(lib.PROPS)].copy()
    pp["side"] = pp["side"].map({"away": "home", "home": "away"})
    pp = pp.drop_duplicates(["gp_id", "side", "batting_order"])
    pf_all = lib.player_frame(g, p)  # same row order as T's player rows

    rows = []
    targets = ["home_win", "f5_home", "f5_away", "rl_home", "rl_away", "nrfi", "over_8.5"] + \
              [f"k{t}_{s}" for t in lib.K_THRESHOLDS for s in ("away", "home")] + list(lib.PROPS)
    for tgt in targets:
        for ver in ("v2", "v3"):
            d = T[T.target.eq(tgt) & T.version.eq(ver)].copy()
            variants = {"as stored": d}
            if tgt in pmap:
                variants["swapped"] = d.assign(p=d.gp_id.map(pmap[tgt]).values)
            elif tgt in lib.PROPS:
                pf = pf_all[pf_all.version.eq(ver)][["gp_id", "side", "batting_order"]].merge(
                    pp, on=["gp_id", "side", "batting_order"], how="left")
                assert (pf["gp_id"].to_numpy() == d["gp_id"].to_numpy()).all()
                ds = d.copy()
                ds["p"] = pf[tgt].to_numpy()
                variants["swapped"] = ds[ds.p.notna()]
            for name, dv in variants.items():
                o = stats.bias_skill(dv, seed=81)
                rm = stats.row_metrics(dv, B=300 if tgt in lib.PROPS else stats.B_ROW, seed=82)
                rows.append({"target": tgt, "version": ver, "scoring": name,
                             **{k: o[k] for k in ("n", "events", "mean_p", "rate", "bias", "bias_lo", "bias_hi",
                                                  "brier", "bss", "bss_lo", "bss_hi")},
                             **{k: rm[k] for k in ("auc", "auc_lo", "auc_hi", "cal_slope", "cal_slope_lo", "cal_slope_hi")}})
            print(tgt, ver, flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(lib.OUT / "posthoc_swapped.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
