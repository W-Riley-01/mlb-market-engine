"""Frozen slice family (SLICES.md): bias + skill tests per cell, BH across the family,
replication by date halves and across versions, EB shrinkage + permutation for park/team."""
import zlib

import numpy as np
import pandas as pd

import lib
import stats

SLICE_T = ["home_win", "nrfi", "over_8.5", "k5", "p_1h", "p_2tb", "p_1hr", "p_1rbi", "p_1sb"]
PLAYER_T = list(lib.PROPS)
INDEP = {"start", "lead", "park", "wlabel", "weather_missing", "abs_shift_b"}  # model-independent segments


def cells(T):
    """Yield (segment, level, version, target, frame) for the frozen family."""
    for tgt in SLICE_T:
        t = T[T.target.eq(tgt)]
        is_p = tgt in PLAYER_T
        for reg in ("v2", "v3a", "v3b"):
            yield "regime", reg, reg[:2], tgt, t[t.regime.eq(reg)]
        for ver in ("v2", "v3"):
            tv = t[t.version.eq(ver)]
            for seg in ("month", "start", "lead", "abs_shift_b", "park"):
                for lvl, d in tv.groupby(seg):
                    yield seg, lvl, ver, tgt, d
            if is_p or tgt == "k5":
                for lvl, d in tv.groupby("side"):
                    yield "side", lvl, ver, tgt, d
            if is_p:
                for seg in ("slot", "bat_team"):
                    for lvl, d in tv.groupby(seg):
                        yield seg, lvl, ver, tgt, d
        for lvl, d in t[t.regime.eq("v3b")].groupby("wlabel"):
            yield "wlabel", lvl, "v3", tgt, d
        for reg in ("v2", "v3a"):
            for lvl, d in t[t.regime.eq(reg)].groupby("weather_missing"):
                yield "weather_missing", f"{'missing' if lvl else 'present'} ({reg})", reg[:2], tgt, d


def sign_in_halves(d, stat):
    out = []
    for h in ("H1", "H2"):
        s = d[d.half.eq(h)]
        if s.empty or s.y.nunique() < 2:
            out.append(np.nan)
            continue
        if stat == "bias":
            out.append(s.p.mean() - s.y.mean())
        else:
            r = s.y.mean()
            out.append(1 - ((s.p - s.y) ** 2).mean() / (r * (1 - r)))
    return out


def entity_tests(T, ent, targets, n_perm=1000):
    """EB-shrunk bias per entity + permutation test of max |z| (entity label shuffled across units)."""
    rows, perm = [], []
    for tgt in targets:
        for ver in ("v2", "v3"):
            d = T[T.target.eq(tgt) & T.version.eq(ver)].copy()
            unit = ["gp_id"] if ent == "park" else ["gp_id", "side"]
            u = d.assign(r=d.p - d.y).groupby(unit).agg(r=("r", "sum"), n=("y", "size"), ev=("y", "sum"),
                                                        lab=(ent, "first")).reset_index()
            agg = u.groupby("lab").agg(n=("n", "sum"), ev=("ev", "sum"))
            ok = agg.index[(agg.ev >= stats.MIN_EV) & (agg.n - agg.ev >= stats.MIN_EV)]
            u = u[u.lab.isin(ok)].reset_index(drop=True)
            if len(ok) < 3:
                perm.append({"entity": ent, "target": tgt, "version": ver, "qualifying": len(ok), "perm_p": np.nan})
                continue
            codes, labs = pd.factorize(u.lab)
            r_, n_ = u.r.to_numpy(), u.n.to_numpy()

            def zstats(c):
                R = np.bincount(c, r_)
                N = np.bincount(c, n_)
                b = R / N
                v = np.bincount(c, (r_ - b[c] * n_) ** 2)
                return b, np.sqrt(v) / N

            b, se = zstats(codes)
            zmax = np.max(np.abs(b / se))
            rg = stats.rng(zlib.crc32(f"{ent}{tgt}{ver}".encode()))
            null = np.array([np.max(np.abs(np.divide(*zstats(rg.permutation(codes))))) for _ in range(n_perm)])
            shr, tau2, mu = stats.eb_shrink(b, se)
            perm.append({"entity": ent, "target": tgt, "version": ver, "qualifying": len(ok), "max_abs_z": zmax,
                         "perm_p": (1 + (null >= zmax).sum()) / (1 + n_perm), "tau": np.sqrt(tau2), "pooled_bias": mu})
            for lab, bi, si, sh in zip(labs, b, se, shr):
                rows.append({"entity": ent, "level": lab, "target": tgt, "version": ver, "bias_raw": bi, "se": si,
                             "bias_shrunk": sh, "pooled_bias": mu, "tau": np.sqrt(tau2)})
    return pd.DataFrame(rows), pd.DataFrame(perm)


def main():
    g = lib.analysis_games()
    T = lib.target_rows(g, lib.load_players())
    rows = []
    for seg, lvl, ver, tgt, d in cells(T):
        ev = int(d.y.sum())
        base = {"segment": seg, "level": str(lvl), "version": ver, "target": tgt, "n": len(d), "events": ev,
                "nonevents": len(d) - ev}
        if ev < stats.MIN_EV or len(d) - ev < stats.MIN_EV:
            rows.append({**base, "status": "insufficient"})
            continue
        o = stats.bias_skill(d, seed=zlib.crc32(f"{seg}{lvl}{ver}{tgt}".encode()))
        hb, hs = sign_in_halves(d, "bias"), sign_in_halves(d, "bss")
        rows.append({**base, "status": "tested", **{k: o[k] for k in (
            "mean_p", "rate", "bias", "bias_lo", "bias_hi", "p_bias", "brier", "bss", "bss_lo", "bss_hi", "p_bss")},
            "bias_H1": hb[0], "bias_H2": hb[1], "bss_H1": hs[0], "bss_H2": hs[1]})
    S = pd.DataFrame(rows)

    # BH across the whole family: both tests of every tested cell
    P = pd.concat([S["p_bias"].rename("p").to_frame().assign(test="bias"),
                   S["p_bss"].rename("p").to_frame().assign(test="bss")])
    P = P[P.p.notna()].reset_index()
    P["q"], P["sig"] = stats.bh(P["p"].reset_index(drop=True))
    for t in ("bias", "bss"):
        sub = P[P.test.eq(t)].set_index("index")
        S[f"q_{t}"] = sub["q"]
        S[f"sig_{t}"] = sub["sig"].astype("boolean")
    m_tests = len(P)

    # replication: same sign in both halves; for model-independent segments, same sign in other version
    def other_version(row, stat):
        if row.segment not in INDEP or row.segment == "wlabel":
            return np.nan
        if row.segment == "weather_missing":
            other_lvl = row.level.replace("(v2)", "(X)").replace("(v3a)", "(v2)").replace("(X)", "(v3a)")
            o = S[(S.segment.eq(row.segment)) & S.level.eq(other_lvl) & S.target.eq(row.target)]
        else:
            o = S[(S.segment.eq(row.segment)) & S.level.eq(row.level) & S.target.eq(row.target)
                  & S.version.ne(row.version)]
        o = o[o.status.eq("tested")]
        return o[stat].iloc[0] if len(o) else np.nan

    for stat in ("bias", "bss"):
        sig = S[f"sig_{stat}"].fillna(False).astype(bool)
        same = np.sign(S[f"{stat}_H1"]) == np.sign(S[f"{stat}_H2"])
        same &= np.sign(S[f"{stat}_H1"]) == np.sign(S[stat])
        S[f"{stat}_other_version"] = [other_version(r, stat) if s else np.nan for r, s in zip(S.itertuples(), sig)]
        ov = S[f"{stat}_other_version"]
        cross_ok = ov.isna() | (np.sign(ov) == np.sign(S[stat]))
        S[f"{stat}_class"] = np.where(~sig, "", np.where(same & cross_ok, "finding", "lead"))

    S.to_csv(lib.OUT / "slices_family.csv", index=False)
    ent_rows, perm = [], []
    for ent, tg in (("park", SLICE_T), ("bat_team", PLAYER_T)):
        e, pm = entity_tests(T, ent, tg)
        ent_rows.append(e)
        perm.append(pm)
    E, PM = pd.concat(ent_rows), pd.concat(perm)
    E.to_csv(lib.OUT / "slices_entity_shrunk.csv", index=False)
    PM.to_csv(lib.OUT / "slices_entity_permutation.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    print(f"cells: {len(S)}  tested: {int(S.status.eq('tested').sum())}  insufficient: {int(S.status.eq('insufficient').sum())}"
          f"  BH family size (tests): {m_tests}")
    ins = S[S.status.eq("insufficient")].groupby(["segment", "target"]).size().unstack(fill_value=0)
    tot = S.groupby(["segment", "target"]).size().unstack(fill_value=0)
    print("\nInsufficient / total cells by segment x target:\n", (ins.astype(str) + "/" + tot.astype(str)).to_string())
    cols = ["segment", "level", "version", "target", "n", "events", "bias", "bias_lo", "bias_hi", "q_bias",
            "bias_H1", "bias_H2", "bias_other_version", "bias_class"]
    print("\nBH-significant BIAS cells:\n", S[S.bias_class.ne("")][cols].round(4).sort_values(["target", "segment"]).to_string())
    cols2 = ["segment", "level", "version", "target", "n", "events", "bss", "bss_lo", "bss_hi", "q_bss",
             "bss_H1", "bss_H2", "bss_other_version", "bss_class"]
    print("\nBH-significant SKILL cells:\n", S[S.bss_class.ne("")][cols2].round(4).sort_values(["target", "segment"]).to_string())
    print("\nPermutation tests (max |z| across qualifying entities):\n", PM.round(4).to_string())


if __name__ == "__main__":
    main()
