"""Leak checks for analysis/cache/btslab3_features.parquet (lab 3 plan Section 11, step 2). Exits 1 on any mismatch.
No outcome is read: lab 1's table is read for key, date and id columns only.

  python analysis/btslab3_leakcheck.py > analysis/out/btslab3_leakcheck_log.txt
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab3_features as b3  # noqa: E402
import btslab_features as bf  # noqa: E402

SEED = 20261201  # plan Section 11 step 2: future deletion; the plain-path recompute uses SEED + 1
GROUPS = {"H1": ["h1_" + c for c in b3.H1C] + ["lg_hard", "lg_gb", "lg_ld", "lg_pu", "lg_xhit", "H1_miss", "H1_max_date"],
          "H2": ["h2_" + c for c in b3.H2C] + ["lg_whiff", "lg_chase", "lg_zcon", "H2_miss", "H2_max_date"],
          "H3": ["bat_side", "sp_hand", "c_hand", "hp_side", "h_same", "h_same_miss", "hp_venue_h", "hp_venue_pa", "hp_road_h",
                 "hp_road_pa", "H3_max_date"],
          "H4": ["h_speed", "h_speed_season", "h_speed_miss", "H4_max_date"],
          "H5": ["h_age", "h_age2", "sp_age", "sp_age2", "t_h_old", "t_ab_old", "lg_h_ab_old", "t_bip_old", "t_S_old",
                 "lg_xhit_old", "H5_trend_miss", "H5_max_date"]}


def same(a, b) -> np.ndarray:
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    return ((a == b) | (a.isna() & b.isna())).to_numpy()


def close(built, plain) -> bool:
    if pd.isna(built) or pd.isna(plain):
        return pd.isna(built) and pd.isna(plain)
    if isinstance(built, (str, pd.Timestamp)) or isinstance(plain, (str, pd.Timestamp)):
        return built == plain
    return abs(float(built) - float(plain)) <= 1e-9 * max(1.0, abs(float(plain)))  # float sums differ in order only


def main():
    t0 = time.time()
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 30)
    print(f"Lab 3 leak checks ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z})")
    X = b3.load_base()
    F = pd.read_parquet(b3.OUT_PARQUET)
    print(f"table {b3.OUT_PARQUET.name}: rows {len(F):,}; content sha256 {b3.content_hash(F)}")
    key_ok = len(F) == len(X) and (F.game_pk.to_numpy() == X.game_pk.to_numpy()).all() and (F.id.to_numpy() == X.id.to_numpy()).all() \
        and (F.date.to_numpy() == X.date.to_numpy()).all()
    print(f"same rows and order as lab 1's table (game_pk, id, date): {key_ok}")
    fail = not key_ok

    # a. structural
    print("\n## a. Structural: H<k>_max_date < game date, every row; speed season; contact table fit dates")
    rows = []
    for g in GROUPS:
        bad = int((F[f"{g}_max_date"].notna() & (F[f"{g}_max_date"] >= F.date)).sum())
        rows.append({"group": f"{g}_max_date", "rows": len(F), "with history": int(F[f"{g}_max_date"].notna().sum()), "violations": bad})
        fail |= bad > 0
    print(pd.DataFrame(rows).to_string(index=False))
    k = F.h_speed.notna()
    bad_sp = int((F.h_speed_season[k] != F.season[k] - 1).sum())
    print(f"speed season = game season - 1: rows with a speed {int(k.sum()):,}, violations {bad_sp}")
    head = "".join(l for l in b3.CONTACT.read_text().splitlines(True) if l.startswith("#"))
    fit_max = pd.Timestamp(re.search(r"max game_date (\d{4}-\d{2}-\d{2})", head).group(1))
    fit_yrs = re.search(r"training: ([\d, ]+) regular-season", head).group(1)
    print(f"contact table header: training seasons {fit_yrs.strip()}, max game_date {fit_max.date()} < 2022-01-01: {fit_max < pd.Timestamp('2022-01-01')}")
    fail |= bad_sp > 0 or not fit_max < pd.Timestamp("2022-01-01")
    print("Static attributes, by design (plan Section 3): bat_side, sp_hand (bio), birth dates; ages are computed on D.")

    # b. future deletion: 300 random rows + 150 doubleheader rows (lab 1's sampling, seed 20261201)
    print("\n## b. Future deletion: rebuild with every source row dated on or after D deleted, compare exactly")
    XF = pd.concat([X, F.drop(columns=["game_pk", "id", "date", "season"])], axis=1)
    rng = np.random.default_rng(SEED)
    pick = []
    for s in bf.POOL_SEASONS:
        days = np.sort(XF.loc[XF.season.eq(s), "date"].unique())
        for d in rng.choice(days, 15, replace=False):
            pick += list(rng.choice(XF.index[XF.date.eq(d)], 5, replace=False))
    pairs = XF.groupby(["date", "home_id", "away_id"]).game_pk.unique().loc[lambda s: s.str.len().eq(2)]
    dh = []
    for s, kk in zip(bf.POOL_SEASONS, (6, 6, 6, 7)):
        ps = pairs[pairs.index.get_level_values("date").year == s]
        for g in ps.iloc[np.sort(rng.choice(len(ps), kk, replace=False))]:
            for pk in g:
                dh += list(rng.choice(XF.index[XF.game_pk.eq(pk)], 3, replace=False))
    S = XF.loc[sorted(set(pick) | set(dh))]
    S = S.assign(sample=np.where(S.index.isin(dh), "doubleheader", "random"))
    inp = b3.load_inputs()
    pmd_all = pd.to_datetime(inp["pm"].game_date)
    mism = []
    for d, rr in S.groupby("date"):
        tr = {"sched": inp["sched"][inp["sched"].date < d], "pm": inp["pm"][pmd_all < d],
              "vault": inp["vault"][inp["vault"].game_date < d], "bio": inp["bio"],
              "sprint": inp["sprint"][inp["sprint"].season < rr.season.iloc[0]], "contact": inp["contact"]}
        Fd = b3.build(tr, rr[b3.BASE_COLS])
        for c in F.columns:
            ok = same(rr[c], Fd[c])
            mism += [{"date": d.date(), "row": int(i), "sample": rr["sample"].iloc[j], "col": c, "full": rr[c].iloc[j],
                      "truncated": Fd[c].iloc[j]} for j, i in enumerate(rr.index) if not ok[j]]
        print(f"  {d.date()}: {len(rr)} rows, {F.shape[1]} columns compared", flush=True)
    by = S.groupby("sample").agg(rows=("game_pk", "size"), dates=("date", "nunique"))
    by["mismatched cells"] = [sum(m["sample"] == kk for m in mism) for kk in by.index]
    by["columns"], by["cells"] = F.shape[1], by.rows * F.shape[1]
    print(f"seasons {S.season.value_counts().sort_index().to_dict()}; doubleheader pairs 25 (both games)")
    print(by.to_string())
    print(f"total cells {len(S) * F.shape[1]}, mismatches {len(mism)}")
    if mism:
        print(pd.DataFrame(mism).head(50).to_string(index=False))
        fail = True

    # c. plain-path recompute: 5 rows per sub-group (seed 20261202), by plain filtering of the raw sources
    print("\n## c. Plain-path recompute: 5 rows per sub-group, plain filtering of vault, pitch_matrix, bio and speed rows")
    sched = inp["sched"].set_index("game_pk")
    v = inp["vault"][inp["vault"].game_pk.isin(sched.index)].copy()
    v["d"] = v.game_pk.map(sched.date)
    bip = v[v.type.eq("X") & v.launch_speed.notna() & v.launch_angle.notna()].copy()
    T = pd.read_csv(b3.CONTACT, comment="#")
    s_lo, a_lo = np.sort(T.speed_lo.unique()), np.sort(T.angle_lo.unique())
    cell = {(r.speed_lo, r.angle_lo): r.p_hit for r in T.itertuples()}
    # league: the cell whose lower edge is the largest lower edge <= the value (left-closed), via searchsorted
    bip["p"] = [cell[(s_lo[i], a_lo[j])] for i, j in zip(np.searchsorted(s_lo, bip.launch_speed, "right") - 1,
                                                          np.searchsorted(a_lo, bip.launch_angle, "right") - 1)]
    pm = inp["pm"][inp["pm"].game_pk.isin(sched.index)].copy()
    pm["d"] = pm.game_pk.map(sched.date)
    pa = pm[pm.events.notna()]
    bio = inp["bio"].set_index("mlbam_id")
    usual = {yy: sched[sched.season.eq(yy)].groupby("home_id").venue_id.agg(lambda q: q.mode().iloc[0]) for yy in (2019, 2021)}
    win = lambda df, lo, hi: df[(df.d >= lo) & (df.d < hi)]
    last = lambda df, first: df.d.max() if len(df) and first(df) > 0 else pd.NaT  # lab 1's window: no date if its first count is 0
    hits = lambda e: int(e.isin(bf.H_EV).sum())
    abs_ = lambda e: int((~e.isin(bf.NON_AB)).sum())

    def h1_plain(x, D):
        b = win(bip[bip.batter.eq(x.id)], D - pd.Timedelta(days=365), D)
        ps = []
        for ls, la in zip(b.launch_speed, b.launch_angle):  # plain loop over the table's cells
            ps += [r.p_hit for r in T.itertuples() if r.speed_lo <= ls < r.speed_hi and r.angle_lo <= la < r.angle_hi]
        assert len(ps) == len(b)
        lg = win(bip, D - pd.Timedelta(days=365), D)
        n = len(lg)
        return {"h1_bip": len(b), "h1_hard": int((b.launch_speed >= 95).sum()), "h1_gb": int((b.launch_angle < 10).sum()),
                "h1_ld": int(b.launch_angle.between(10, 25, inclusive="left").sum()), "h1_pu": int((b.launch_angle > 50).sum()),
                "h1_S": sum(ps), "h1_Q": sum(p * p for p in ps), "lg_hard": (lg.launch_speed >= 95).sum() / n,
                "lg_gb": (lg.launch_angle < 10).sum() / n, "lg_ld": lg.launch_angle.between(10, 25, inclusive="left").sum() / n,
                "lg_pu": (lg.launch_angle > 50).sum() / n, "lg_xhit": lg.p.sum() / n, "H1_miss": int(len(b) == 0),
                "H1_max_date": max([q for q in (last(b, len), last(lg, len)) if pd.notna(q)], default=pd.NaT)}

    def h2_counts(p):
        sw, ms = p.description.isin(b3.SWING), p.description.isin(b3.MISS)
        out, inz = p.zone.isin([11, 12, 13, 14]), p.zone.isin(range(1, 10))
        return {"pitch": len(p), "swing": int(sw.sum()), "miss": int(ms.sum()), "out": int(out.sum()), "out_swing": int((out & sw).sum()),
                "in_swing": int((inz & sw).sum()), "in_miss": int((inz & ms).sum())}

    def h2_plain(x, D):
        b, lg = win(v[v.batter.eq(x.id)], D - pd.Timedelta(days=365), D), win(v, D - pd.Timedelta(days=365), D)
        c, L = h2_counts(b), h2_counts(lg)
        return {**{f"h2_{k}": c[k] for k in c}, "lg_whiff": L["miss"] / L["swing"], "lg_chase": L["out_swing"] / L["out"],
                "lg_zcon": (L["in_swing"] - L["in_miss"]) / L["in_swing"], "H2_miss": int(c["pitch"] == 0),
                "H2_max_date": max([q for q in (last(b, len), last(lg, len)) if pd.notna(q)], default=pd.NaT)}

    def h3_plain(x, D):
        bs, sh = bio.bat_side.get(x.id), bio.pitch_hand.get(x.opp_sp)
        ph = pa[pa.pitcher.eq(x.opp_sp) & (pa.d < D)].sort_values(["d", "game_pk", "at_bat_number", "pitch_number"])
        ch = ph.p_throws.iloc[-1] if len(ph) else np.nan
        if bs in ("R", "L"):
            side = bs
        else:
            ref = sh if sh in ("R", "L") else ch
            side = {"R": "L", "L": "R"}.get(ref, np.nan)
        out = {"bat_side": bs, "sp_hand": sh, "c_hand": ch, "hp_side": side, "h_same": int(sh in ("R", "L") and side == sh),
               "h_same_miss": int(sh not in ("R", "L"))}
        pg = sched[(sched.date >= pd.Timestamp(f"{x.season - 3}-01-01")) & (sched.date < D) & ~sched.venue_id.isin(bf.NEUTRAL_PARK)]
        pg = pg[pg.season.ne(2020) | pg.venue_id.eq(pg.home_id.map(usual[2019])) | pg.venue_id.eq(pg.home_id.map(usual[2021]))]
        e = pa[pa.stand.eq(side)]
        vg, rg = e[e.game_pk.isin(pg.index[pg.venue_id.eq(x.venue_id)])], e[e.game_pk.isin(pg.index[pg.away_id.eq(x.home_id)])]
        hv, hr = hits(vg.events), hits(rg.events)
        out.update({"hp_venue_h": hv, "hp_venue_pa": len(vg), "hp_road_h": hr, "hp_road_pa": len(rg),
                    "H3_max_date": max([q for q in (vg.d.max() if hv else pd.NaT, rg.d.max() if hr else pd.NaT) if pd.notna(q)], default=pd.NaT)})
        return out

    def h4_plain(x, D):
        sp = [r for r in inp["sprint"].itertuples() if r.mlbam_id == x.id and r.season == x.season - 1]
        return {"h_speed": sp[0].sprint_speed if sp else np.nan, "h_speed_season": float(x.season - 1) if sp else np.nan,
                "h_speed_miss": int(not sp), "H4_max_date": sched[sched.season.eq(x.season - 1)].date.max() if sp else pd.NaT}

    def h5_plain(x, D):
        y365, y730 = D - pd.Timedelta(days=365), D - pd.Timedelta(days=730)
        age = lambda pid: (D - bio.birth_date.get(pid)).days / 365.2425 - 28
        bo, lo_ = win(pa[pa.batter.eq(x.id)], y730, y365), win(pa, y730, y365)
        xo, xlo = win(bip[bip.batter.eq(x.id)], y730, y365), win(bip, y730, y365)
        bn, ln = win(pa[pa.batter.eq(x.id)], y365, D), win(pa, y365, D)
        h1 = h1_plain(x, D)["H1_max_date"]
        srcs = [last(bo, lambda q: hits(q.events)), last(lo_, lambda q: hits(q.events)), last(xo, len), last(xlo, len),
                last(bn, lambda q: hits(q.events)), last(ln, lambda q: hits(q.events)), h1]
        return {"h_age": age(x.id), "h_age2": age(x.id) ** 2, "sp_age": age(x.opp_sp), "sp_age2": age(x.opp_sp) ** 2,
                "t_h_old": hits(bo.events), "t_ab_old": abs_(bo.events), "lg_h_ab_old": hits(lo_.events) / abs_(lo_.events),
                "t_bip_old": len(xo), "t_S_old": xo.p.sum(), "lg_xhit_old": xlo.p.sum() / len(xlo), "H5_trend_miss": int(abs_(bo.events) == 0),
                "H5_max_date": max([q for q in srcs if pd.notna(q)], default=pd.NaT)}

    plain = {"H1": h1_plain, "H2": h2_plain, "H3": h3_plain, "H4": h4_plain, "H5": h5_plain}
    rs = np.random.default_rng(SEED + 1)
    out = []
    for g, fn in plain.items():
        for i in rs.choice(XF.index, 5, replace=False):
            x = XF.loc[i]
            P = fn(x, x.date)
            assert set(P) == set(GROUPS[g])
            out += [{"group": g, "row": int(i), "date": x.date.date(), "batter": int(x.id), "column": c, "built": x[c],
                     "plain": P[c], "match": close(x[c], P[c])} for c in GROUPS[g]]
    O = pd.DataFrame(out)
    print(O.to_string(index=False))
    print(f"plain-path cells {len(O)} ({O.row.nunique()} rows), mismatches {int((~O.match).sum())}")
    fail |= not O.match.all()

    print(f"\nruntime {time.time() - t0:.0f} s")
    print("LEAK CHECK:", "FAIL - STOP" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
