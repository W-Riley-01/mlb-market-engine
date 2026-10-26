"""POST-HOC re-analysis (B7). Re-test each tested cell's bias as
   (cell bias - bias of all other rows of the same target in the same version),
joint cluster bootstrap over the version's games, BH over the same family of cells.
Replication rule as in SLICES.md, applied to the difference."""
import zlib

import numpy as np
import pandas as pd
from scipy import stats as sps

import lib
import stats
from slices import cells, INDEP


def main():
    g = lib.analysis_games()
    T = lib.target_rows(g, lib.load_players())
    S = pd.read_csv(lib.OUT / "slices_family.csv")
    rows = []
    for seg, lvl, ver, tgt, d in cells(T):
        base = {"segment": seg, "level": str(lvl), "version": ver, "target": tgt, "n": len(d)}
        ev = int(d.y.sum())
        if ev < stats.MIN_EV or len(d) - ev < stats.MIN_EV:
            continue
        allv = T[T.target.eq(tgt) & T.version.eq(ver)]
        comp = allv.drop(index=d.index)
        if comp.empty:
            rows.append({**base, "status": "no complement"})
            continue
        games = pd.Index(allv.gp_id.unique())
        c1 = d.assign(r=d.p - d.y).groupby("gp_id").agg(r=("r", "sum"), n=("y", "size")).reindex(games, fill_value=0)
        c0 = comp.assign(r=comp.p - comp.y).groupby("gp_id").agg(r=("r", "sum"), n=("y", "size")).reindex(games, fill_value=0)
        W = stats._boot_counts(len(games), stats.B_DEFAULT, stats.rng(zlib.crc32(f"d{seg}{lvl}{ver}{tgt}".encode())))
        with np.errstate(invalid="ignore", divide="ignore"):
            bt = (W @ c1.r.to_numpy()) / (W @ c1.n.to_numpy()) - (W @ c0.r.to_numpy()) / (W @ c0.n.to_numpy())
        bt = bt[np.isfinite(bt)]
        est = c1.r.sum() / c1.n.sum() - c0.r.sum() / c0.n.sum()
        se = bt.std(ddof=1)

        def half_diff(h):
            a, b = d[d.half.eq(h)], comp[comp.half.eq(h)]
            if a.empty or b.empty:
                return np.nan
            return (a.p - a.y).mean() - (b.p - b.y).mean()

        rows.append({**base, "status": "tested", "cell_bias": (d.p - d.y).mean(), "rest_bias": (comp.p - comp.y).mean(),
                     "diff": est, "lo": np.quantile(bt, .025), "hi": np.quantile(bt, .975),
                     "p": 2 * sps.norm.sf(abs(est) / se), "diff_H1": half_diff("H1"), "diff_H2": half_diff("H2")})
    D = pd.DataFrame(rows)
    t = D.status.eq("tested")
    D.loc[t, "q"], D.loc[t, "sig"] = stats.bh(D.loc[t, "p"])
    D["sig"] = D["sig"].fillna(False).astype(bool)

    def other(r):
        if r.segment not in INDEP or r.segment == "wlabel":
            return np.nan
        if r.segment == "weather_missing":
            lvl = r.level.replace("(v2)", "(X)").replace("(v3a)", "(v2)").replace("(X)", "(v3a)")
            o = D[D.segment.eq(r.segment) & D.level.eq(lvl) & D.target.eq(r.target)]
        else:
            o = D[D.segment.eq(r.segment) & D.level.eq(r.level) & D.target.eq(r.target) & D.version.ne(r.version)]
        o = o[o.status.eq("tested")]
        return o["diff"].iloc[0] if len(o) else np.nan

    D["diff_other_version"] = [other(r) if s else np.nan for r, s in zip(D.itertuples(), D.sig)]
    same = (np.sign(D.diff_H1) == np.sign(D.diff_H2)) & (np.sign(D.diff_H1) == np.sign(D["diff"]))
    cross = D.diff_other_version.isna() | (np.sign(D.diff_other_version) == np.sign(D["diff"]))
    D["class"] = np.where(~D.sig, "", np.where(same & cross, "finding", "lead"))
    old = S.set_index(["segment", "level", "version", "target"])["bias_class"]
    D["old_bias_class"] = D.set_index(["segment", "level", "version", "target"]).index.map(old)
    D.to_csv(lib.OUT / "slices_diff.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(f"cells re-tested: {int(t.sum())}  (no complement: {int(D.status.eq('no complement').sum())})")
    print("old bias findings:", int(D.old_bias_class.eq("finding").sum()),
          "| surviving as relative findings:", int((D.old_bias_class.eq("finding") & D["class"].eq("finding")).sum()),
          "| relative leads:", int(D["class"].eq("lead").sum()), "| new relative findings:",
          int((D["class"].eq("finding") & D.old_bias_class.ne("finding")).sum()))
    cols = ["segment", "level", "version", "target", "n", "cell_bias", "rest_bias", "diff", "lo", "hi", "q",
            "diff_H1", "diff_H2", "diff_other_version", "class", "old_bias_class"]
    print(D[D["class"].ne("")][cols].sort_values(["class", "target", "segment"]).round(4).to_string())


if __name__ == "__main__":
    main()
