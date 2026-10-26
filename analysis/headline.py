"""Headline metrics per target x version/regime, baselines, post-stratification, monthly series,
reliability figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zlib

import numpy as np
import pandas as pd

import lib
import stats

FIG = lib.OUT / "figures"
GAME_T = ["home_win", "nrfi", "f5_home", "f5_away", "f5_tie", "rl_home", "rl_away"] + \
         [f"over_{r}" for r in lib.HALF_RUNGS] + ["k5_away", "k5_home"]
PLAYER_T = list(lib.PROPS)
SERIES_T = ["home_win", "nrfi", "over_8.5", "k5", "p_1h", "p_2tb", "p_1hr", "p_1rbi", "p_1sb"]
SETS = [("v2", "version"), ("v3", "version"), ("v3a", "regime"), ("v3b", "regime")]


def skill_vs(d: pd.DataFrame, base_col: str, B=stats.B_DEFAULT, seed=61):
    """Brier skill of the sim vs a baseline probability column; cluster bootstrap CI."""
    x = d.assign(sm=(d.p - d.y) ** 2, sb=(d[base_col] - d.y) ** 2).groupby("gp_id")[["sm", "sb"]].sum()
    W = stats._boot_counts(len(x), B, stats.rng(seed))
    bt = 1 - (W @ x.sm.to_numpy()) / (W @ x.sb.to_numpy())
    est = 1 - x.sm.sum() / x.sb.sum()
    return est, np.quantile(bt, .025), np.quantile(bt, .975)


def poststrat_weights(g: pd.DataFrame) -> pd.Series:
    """Per-game weight = completed regular-season schedule count / sample count, by version x month x start."""
    s = pd.read_csv(lib.CACHE / "schedule_2026.csv")
    s["listed_date"] = pd.to_datetime(s["listed_date"])
    s = s[s.game_type.eq("R") & s.detailed_state.isin(["Final", "Completed Early"])].drop_duplicates("game_pk", keep="last")
    s["version"] = np.select([s.listed_date.between("2026-05-01", "2026-07-19"),
                              s.listed_date.between("2026-07-20", "2026-09-27")], ["v2", "v3"], None)
    s = s[s.version.notna()]
    s["month"] = s.listed_date.dt.strftime("%b")
    s["start"] = lib._bucket(pd.to_datetime(s.scheduled_utc).dt.tz_convert("America/New_York").dt.hour,
                             [0, 16, 19, 21, 24], ["<4pm", "4-7pm", "7-9pm", "9pm+"])
    pop = s.groupby(["version", "month", "start"]).size()
    samp = g.groupby(["version", "month", "start"]).size()
    w = (pop / samp).rename("w")
    missing = pop.index.difference(samp.index)
    assert len(missing) == 0, f"population cells with no sample: {list(missing)}"
    return g.join(w, on=["version", "month", "start"]).set_index("gp_id")["w"]


def weighted(d: pd.DataFrame, B=stats.B_DEFAULT, seed=71):
    x = d.assign(wp=d.w * d.p, wy=d.w * d.y, wse=d.w * (d.p - d.y) ** 2).groupby("gp_id")[["w", "wp", "wy", "wse"]].sum()
    W = stats._boot_counts(len(x), B, stats.rng(seed))
    S = {c: W @ x[c].to_numpy() for c in x}
    bias_b = (S["wp"] - S["wy"]) / S["w"]
    brier_b = S["wse"] / S["w"]
    tot = x.sum()
    return {"w_bias": (tot.wp - tot.wy) / tot.w, "w_bias_lo": np.quantile(bias_b, .025), "w_bias_hi": np.quantile(bias_b, .975),
            "w_brier": tot.wse / tot.w, "w_brier_lo": np.quantile(brier_b, .025), "w_brier_hi": np.quantile(brier_b, .975),
            "w_rate": tot.wy / tot.w}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    g = lib.analysis_games()
    T = lib.target_rows(g, lib.load_players())
    w = poststrat_weights(g)
    T["w"] = T.gp_id.map(w)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    rows, rel_store = [], {}
    for tgt in GAME_T + PLAYER_T:
        for name, col in SETS:
            d = T[T.target.eq(tgt) & T[col].eq(name)]
            o = stats.bias_skill(d, seed=zlib.crc32(f"{tgt}|{name}".encode()))
            rm = stats.row_metrics(d, seed=zlib.crc32(f"{name}|{tgt}".encode()))
            rel = stats.reliability(d)
            rel_store[(tgt, name)] = rel
            row = {"target": tgt, "set": name, **o, **rm, "ece_eqcount": stats.ece(rel), "rel_bins": len(rel),
                   "qualifies": o["events"] >= stats.MIN_EV and o["nonevents"] >= stats.MIN_EV}
            if name in ("v2", "v3"):
                row.update(weighted(d))
            if tgt in PLAYER_T:
                for b in ("base_league", "base_slot", "base_own"):
                    e, lo, hi = skill_vs(d, b)
                    row.update({f"bss_{b}": e, f"bss_{b}_lo": lo, f"bss_{b}_hi": hi})
            rows.append(row)
            print(tgt, name, "done", flush=True)
    H = pd.DataFrame(rows)
    H.to_csv(lib.OUT / "headline.csv", index=False)

    # --- monthly series -----------------------------------------------------------
    rows = []
    for tgt in SERIES_T:
        for ver in ("v2", "v3"):
            for mon, d in T[T.target.eq(tgt) & T.version.eq(ver)].groupby("month"):
                o = stats.bias_skill(d, seed=7)
                rows.append({"target": tgt, "version": ver, "month": mon,
                             **{k: o[k] for k in ("n", "events", "nonevents", "mean_p", "rate", "bias", "bias_lo", "bias_hi",
                                                  "brier", "bss", "bss_lo", "bss_hi")}})
    M = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(["May", "Jun", "Jul", "Aug", "Sep"])}
    M["m_i"] = M.month.map(order) + np.where(M.version.eq("v3") & M.month.eq("Jul"), 0.3, 0) - \
        np.where(M.version.eq("v2") & M.month.eq("Jul"), 0.3, 0)
    M = M.sort_values(["target", "m_i"])
    M.to_csv(lib.OUT / "monthly.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, tgt in zip(axes.flat, SERIES_T):
        for ver, c in (("v2", "#3b6ea8"), ("v3", "#c4572e")):
            s = M[M.target.eq(tgt) & M.version.eq(ver)]
            ax.errorbar(s.m_i, s.bias, yerr=[s.bias - s.bias_lo, s.bias_hi - s.bias], fmt="o-", capsize=2, color=c, label=ver)
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xticks(range(5), list(order))
        ax.set_title(tgt)
    axes.flat[0].legend()
    fig.suptitle("Monthly bias (mean predicted − rate), 95% cluster-bootstrap CI; Sep flagged: call-ups / resting starters")
    fig.tight_layout()
    fig.savefig(FIG / "monthly_bias.png", dpi=130)
    plt.close(fig)

    # --- reliability figures --------------------------------------------------------
    rel_t = ["home_win", "nrfi", "over_8.5", "k5_away", "k5_home", "p_1h", "p_2tb", "p_1hr", "p_1rbi"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, tgt in zip(axes.flat, rel_t):
        for ver, c in (("v2", "#3b6ea8"), ("v3", "#c4572e")):
            r = rel_store[(tgt, ver)]
            if r.empty:
                continue
            ax.errorbar(r.mean_p, r.rate, yerr=[r.rate - r.lo, r.hi - r.rate], fmt="o-", capsize=2, color=c,
                        label=f"{ver} ({len(r)} bins)")
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="grey")
        ax.set_title(tgt)
        ax.set_xlabel("mean predicted")
        ax.set_ylabel("observed rate")
        ax.legend(fontsize=8)
    fig.suptitle("Reliability, equal-count bins (≥30 events and non-events per bin), Wilson 95% CI")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=130)
    plt.close(fig)

    cols = ["target", "set", "n", "events", "mean_p", "rate", "bias", "bias_lo", "bias_hi", "brier", "bss", "bss_lo",
            "bss_hi", "auc", "auc_lo", "auc_hi", "cal_int", "cal_int_lo", "cal_int_hi", "cal_slope", "cal_slope_lo",
            "cal_slope_hi", "log_loss", "ece_eqcount", "w_bias", "w_bias_lo", "w_bias_hi", "w_brier"]
    print(H[cols].round(3).to_string())
    pc = ["target", "set", "bss_base_league", "bss_base_league_lo", "bss_base_league_hi", "bss_base_slot",
          "bss_base_slot_lo", "bss_base_slot_hi", "bss_base_own", "bss_base_own_lo", "bss_base_own_hi"]
    print(H[H.target.isin(PLAYER_T)][pc].round(4).to_string())


if __name__ == "__main__":
    main()
