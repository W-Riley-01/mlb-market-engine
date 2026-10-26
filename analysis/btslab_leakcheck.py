"""Leak checks for analysis/cache/btslab_features.parquet (plan Section 3; Step 1 checks a-c). Exits 1 on any mismatch.

  python analysis/btslab_leakcheck.py > analysis/out/btslab_leakcheck_log.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab_features as bf  # noqa: E402
import v4_analysis as va  # noqa: E402

SEED = 20261026  # plan Section 3
ROWCOLS = ["date", "id", "slot", "side", "opp_sp", "opp_team", "bat_team", "home_id", "venue_id", "season"]


def same(a: pd.Series, b: pd.Series) -> np.ndarray:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return ((a == b) | (a.isna() & b.isna())).to_numpy()


def main():
    pd.set_option("display.width", 220, "display.max_rows", 400)
    X = pd.read_parquet(bf.OUT_PARQUET)
    groups = [c for c in X if c.endswith("_max_date")]
    fail = False

    # a. structural: every group's latest source date is strictly before the game date
    print("## a. Structural: <group>_max_date < game date, every row")
    rows = []
    for g in groups:
        bad = int((X[g].notna() & (X[g] >= X.date)).sum())
        rows.append({"group": g, "rows": len(X), "with history": int(X[g].notna().sum()), "violations": bad})
        fail |= bad > 0
    print(pd.DataFrame(rows).to_string(index=False))
    print("Same-day attributes, by design (plan Section 3 and addendum A1 changes 3 and 6): slot, opp_sp, venue_id, "
          "f_temp_f, f_indoor, x_wind_mph, x_mkt_team_runs. They carry no max date.")

    # b. future-deletion test
    print("\n## b. Future deletion: rebuild with only data dated before D, compare exactly")
    rng = np.random.default_rng(SEED)
    pick = []
    for s in bf.POOL_SEASONS:
        days = np.sort(X.loc[X.season.eq(s), "date"].unique())
        for d in rng.choice(days, 15, replace=False):
            idx = X.index[X.date.eq(d)]
            pick += list(rng.choice(idx, 5, replace=False))
    # plus 150 rows from doubleheaders: 25 pairs (6/6/6/7 by season), 3 rows from each of the two games
    pairs = X.groupby(["date", "home_id", "away_id"]).game_pk.unique().loc[lambda s: s.str.len().eq(2)]
    dh = []
    for s, k in zip(bf.POOL_SEASONS, (6, 6, 6, 7)):
        ps = pairs[pairs.index.get_level_values("date").year == s]
        for g in ps.iloc[np.sort(rng.choice(len(ps), k, replace=False))]:
            for pk in g:
                dh += list(rng.choice(X.index[X.game_pk.eq(pk)], 3, replace=False))
    S = X.loc[sorted(set(pick) | set(dh))]
    S = S.assign(sample=np.where(S.index.isin(dh), "doubleheader", "random"))
    sched, pm = bf.load_schedule(), bf.load_pitch_matrix()
    G = sched[sched.pool_eligible & sched.season.isin(bf.POOL_SEASONS)]
    _, boxp, info = bf.load_box(G)
    read = pd.read_parquet
    mism = []
    for d, rr in S.groupby("date"):
        sch = sched[sched.date < d]
        pmd = pm[pd.to_datetime(pm.game_date) < d]
        r = rr[ROWCOLS].reset_index(drop=True)
        Fd = bf.build(bf.raw_tables(pmd, sch, boxp[boxp.game_pk.isin(sch.game_pk)]), r)
        va.pd.read_parquet = lambda path, columns=None, **k: pmd[columns]  # baseline_tables reads the truncated file
        try:
            Bd = bf.baselines(r, pmd, va.baseline_tables())
        finally:
            va.pd.read_parquet = read
        # same-day attributes come from the row's own game, rebuilt the same way as the full build
        own = rr[["game_pk", "side"]].reset_index(drop=True)
        day = own.merge(bf.weather(info[info.game_pk.isin(own.game_pk)]), on="game_pk", how="left").merge(
            bf.market(G[G.game_pk.isin(own.game_pk)], info), on="game_pk", how="left")
        day["x_mkt_team_runs"] = np.where(day.side.eq("home"), day.x_mkt_home_runs, day.x_mkt_away_runs)
        sameday = ["f_indoor", "f_temp_f", "x_wind_mph", "x_mkt_team_runs"]
        # identity and outcome columns (game_pk, id, slot, ab, h, ...) are carried over from the row itself
        keep = rr.drop(columns=[c for c in rr if c in Fd or c in Bd or c in sameday or c == "sample"]).reset_index(drop=True)
        Fd = pd.concat([keep, Fd, Bd, day[sameday]], axis=1)
        for c in X.columns:
            ok = same(rr[c], Fd[c])
            mism += [{"date": d.date(), "row": int(i), "sample": rr["sample"].iloc[j], "col": c, "full": rr[c].iloc[j],
                      "truncated": Fd[c].iloc[j]} for j, i in enumerate(rr.index) if not ok[j]]
        print(f"  {d.date()}: {len(rr)} rows, {X.shape[1]} columns compared", flush=True)
    by = S.groupby("sample").agg(rows=("game_pk", "size"), dates=("date", "nunique"))
    by["mismatched cells"] = [sum(m["sample"] == k for m in mism) for k in by.index]
    by["columns"], by["cells"] = X.shape[1], by.rows * X.shape[1]
    print(f"seasons {S.season.value_counts().sort_index().to_dict()}; doubleheader pairs 25 (both games)")
    print(by.to_string())
    print(f"total cells {len(S) * X.shape[1]}, mismatches {len(mism)}")
    if mism:
        print(pd.DataFrame(mism).head(50).to_string(index=False))
        fail = True

    # c. plan's 20-row spot check, by plain filtering of the raw data
    print("\n## c. Spot check: 20 rows, recomputed by plain filtering")
    rs = np.random.default_rng(SEED).choice(X.index, 20, replace=False)
    sched = sched.set_index("game_pk")
    pr = pm[pm.game_pk.isin(sched.index) & pm.events.notna()].copy()
    pr["d"] = pr.game_pk.map(sched.date)
    pmr = pm[pm.game_pk.isin(sched.index)].copy()  # all regular-season pitch rows, dated like the build
    pmr["d"] = pmr.game_pk.map(sched.date)
    pmall = pm.assign(gd=pd.to_datetime(pm.game_date))
    usual19, usual21 = (sched[sched.season.eq(yy)].groupby("home_id").venue_id.agg(lambda v: v.mode().iloc[0]) for yy in (2019, 2021))
    out = []
    for i in rs:
        x = X.loc[i]
        D, y = x.date, x.date - pd.Timedelta(days=365)
        b = pr[(pr.batter == x.id) & (pr.d >= y) & (pr.d < D)]
        p = pr[(pr.pitcher == x.opp_sp) & (pr.d >= y) & (pr.d < D)]
        b14 = pr[(pr.batter == x.id) & (pr.d >= D - pd.Timedelta(days=14)) & (pr.d < D)]
        h = lambda e: int(e.isin(bf.H_EV).sum())
        ab = lambda e: int((~e.isin(bf.NON_AB)).sum())
        # relief pitches by the opposing team in [D-3, D): box score when cached, else pitch rows
        g3 = sched[(sched.date >= D - pd.Timedelta(days=3)) & (sched.date < D)
                   & ((sched.home_id == x.opp_team) | (sched.away_id == x.opp_team))]
        fat = 0
        for pk, g in g3.iterrows():
            side = "home" if g.home_id == x.opp_team else "away"
            f = bf.API / "box" / f"{pk}.json"
            if f.exists():
                t = json.loads(f.read_bytes())["teams"][side]
                fat += sum(t["players"][f"ID{q}"]["stats"]["pitching"].get("numberOfPitches", 0) for q in t["pitchers"][1:])
            else:
                rows_ = pm[(pm.game_pk == pk) & pm.inning_topbot.eq("Top" if side == "home" else "Bot")]
                rows_ = rows_.sort_values(["at_bat_number", "pitch_number"])
                fat += int(rows_.pitcher.ne(rows_.pitcher.iloc[0]).sum())
        # starter's last 10 starts before D: games where he threw his team's first pitch; BF from box, else PAs
        gs = pmr[pmr.pitcher.eq(x.opp_sp) & (pmr.d < D)][["game_pk", "inning_topbot", "d"]].drop_duplicates()
        st = []
        for g in gs.itertuples():
            half = pmr[(pmr.game_pk == g.game_pk) & (pmr.inning_topbot == g.inning_topbot)].sort_values(["at_bat_number", "pitch_number"])
            if half.pitcher.iloc[0] != x.opp_sp:
                continue
            side = "home" if g.inning_topbot == "Top" else "away"
            f = bf.API / "box" / f"{g.game_pk}.json"
            bfc = (json.loads(f.read_bytes())["teams"][side]["players"][f"ID{x.opp_sp}"]["stats"]["pitching"]["battersFaced"]
                   if f.exists() else int((half.pitcher.eq(x.opp_sp) & half.events.notna()).sum()))
            st.append((g.d, bfc))
        last10 = sorted(st)[-10:]
        # park counts: eligible games at the venue / the home team's road games, Jan 1 (season-3) .. D
        pg = sched[(sched.date >= pd.Timestamp(f"{x.season - 3}-01-01")) & (sched.date < D) & ~sched.venue_id.isin(bf.NEUTRAL_PARK)]
        pg = pg[pg.season.ne(2020) | pg.venue_id.eq(pg.home_id.map(usual19)) | pg.venue_id.eq(pg.home_id.map(usual21))]
        vg, rg = pr[pr.game_pk.isin(pg.index[pg.venue_id.eq(x.venue_id)])], pr[pr.game_pk.isin(pg.index[pg.away_id.eq(x.home_id)])]
        rate = lambda hh, nn: hh / nn if nn else np.nan
        pf_built = rate(x.f_venue_h, x.f_venue_pa) / rate(x.f_road_h, x.f_road_pa)
        pf_plain = rate(h(vg.events), len(vg)) / rate(h(rg.events), len(rg))
        # B2 as R2 defines it: all pitch_matrix rows with game_date < D in [D-365, D)
        w = pmall[(pmall.gd >= y) & (pmall.gd < D)]
        e = w[w.events.notna()].sort_values(["game_pk", "at_bat_number"])
        Lr = h(e.events) / len(e)
        eb, ep = e[e.batter.eq(x.id)], e[e.pitcher.eq(x.opp_sp)]
        bb_ = (h(eb.events) + 200 * Lr) / (len(eb) + 200)
        qq = (h(ep.events) + 400 * Lr) / (len(ep) + 400)
        xx = (bb_ * qq / Lr) / (bb_ * qq / Lr + (1 - bb_) * (1 - qq) / (1 - Lr))
        e = e.assign(side=np.where(e.inning_topbot.eq("Bot"), "home", "away"))
        e["sp"] = e.groupby(["game_pk", "side"]).pitcher.transform("first")
        fst = e.drop_duplicates(["game_pk", "side", "batter"]).copy()
        fst["slot"] = fst.groupby(["game_pk", "side"]).cumcount() + 1
        fst = fst[fst.slot.eq(x.slot) & fst.side.eq(x.side)][["game_pk", "side", "batter"]]
        sl = e.merge(fst, on=["game_pk", "side", "batter"])
        E_, s_ = len(sl) / sl.game_pk.nunique(), sl.pitcher.eq(sl.sp).mean()
        b2_plain = 1 - (1 - xx) ** (s_ * E_) * (1 - bb_) ** ((1 - s_) * E_)
        for name, built, plain in (("starter BF, last 10 starts", x.c_bf_last10, sum(v for _, v in last10)),
                                   ("starter starts counted (<= 10)", x.c_starts_last10, len(last10)),
                                   ("park: venue PA", x.f_venue_pa, len(vg)), ("park: venue H", x.f_venue_h, h(vg.events)),
                                   ("park: road PA", x.f_road_pa, len(rg)), ("park: road H", x.f_road_h, h(rg.events)),
                                   ("park factor, raw (venue/road H per PA)", pf_built, pf_plain),
                                   ("B2 P(1+ hit)", x.base_b2, b2_plain)):
            tol = 1e-12 if name.startswith(("park factor", "B2")) else 0  # float paths differ in summation order
            out.append({"row": int(i), "date": D.date(), "batter": int(x.id), "feature": name, "built": float(built),
                        "plain": float(plain), "match": bool(abs(float(built) - float(plain)) <= tol
                                                             or (np.isnan(built) and np.isnan(plain)))})
        for name, built, plain in (("batter H, 365d", x.a_h, h(b.events)), ("batter AB, 365d", x.a_ab, ab(b.events)),
                                   ("batter PA, 365d", x.a_pa, len(b)),
                                   ("starter BF, 365d", x.c_pa, len(p)), ("starter K, 365d", x.c_k, int(p.events.isin(bf.K_EV).sum())),
                                   ("batter H, 14d", x.g_h14, h(b14.events)), ("batter AB, 14d", x.g_ab14, ab(b14.events)),
                                   ("opp relief pitches, 3d", x.e_fat_pitches_3d, fat)):
            out.append({"row": int(i), "date": D.date(), "batter": int(x.id), "feature": name, "built": float(built),
                        "plain": float(plain), "match": float(built) == float(plain)})
    O = pd.DataFrame(out)
    print(O.to_string(index=False))
    print(f"spot-check cells {len(O)}, mismatches {int((~O.match).sum())}")
    fail |= not O.match.all()

    print("\nLEAK CHECK:", "FAIL - STOP" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
