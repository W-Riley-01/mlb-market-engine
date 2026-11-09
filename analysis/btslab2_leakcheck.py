"""Leak checks for analysis/cache/btslab2_features.parquet (lab 2 plan Section 11 rule 1; addendum A3 items 13g, 13j;
A4). Exits 1 on any mismatch. No outcome column is read: lab 1's table is read for game_pk, id, date, season, opp_sp
and, for the doubleheader sample only, home_id and away_id (A3 item 13d). No box-score file and no 2026 data.

  python analysis/btslab2_leakcheck.py > analysis/out/btslab2_leakcheck_log.txt
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab2_features as b2  # noqa: E402

SEED_DEL, SEED_SHUF, SEED_PLAIN = 20261130, 20261131, 20261132  # A3 item 8, item 13j
SEASONS = (2022, 2023, 2024, 2025)
# plain-path definitions, written out here rather than imported from the builder
FAMILY = {"fb": {"FF", "SI", "FA", "FC"}, "br": {"SL", "ST", "SV", "CU", "KC", "CS"}, "os": {"CH", "FS", "FO", "SC"}}
FAMILY_ALT = {"fb": {"FF", "SI", "FA", "FS"}, "br": {"SL", "ST", "SV", "CU", "KC", "CS", "FC"}, "os": {"CH", "FO", "SC"}}
NOT_AB = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf",
          "truncated_pa"}
HITS = {"single", "double", "triple", "home_run"}


def same(a, b) -> np.ndarray:
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    return ((a == b) | (a.isna() & b.isna())).to_numpy()


def close(built, plain) -> bool:
    if pd.isna(built) or pd.isna(plain):
        return pd.isna(built) and pd.isna(plain)
    if isinstance(built, (str, pd.Timestamp)) or isinstance(plain, (str, pd.Timestamp)):
        return built == plain
    if np.isinf(float(built)) or np.isinf(float(plain)):
        return float(built) == float(plain)
    return abs(float(built) - float(plain)) <= 1e-9 * max(1.0, abs(float(plain)))  # 1e-9 relative, as lab 3


def main():
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 1000, "display.max_columns", 30)
    print(f"Lab 2 leak checks ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z})")
    raw = b2.load_raw()
    keys = b2.load_base(b2.BASE_COLS + ["home_id", "away_id"]).sort_values(["date", "game_pk", "id"], kind="mergesort").reset_index(drop=True)
    F = pd.read_parquet(b2.OUT_PARQUET)
    print(f"table {b2.OUT_PARQUET.name}: rows {len(F):,}, cols {F.shape[1]}; content sha256 {b2.b3.content_hash(F)}")
    key_ok = len(F) == len(keys) and all((F[c].to_numpy() == keys[c].to_numpy()).all() for c in ("game_pk", "id", "date"))
    print(f"same rows and order as lab 1's table sorted by (date, game_pk, id): {key_ok}")
    fail = not key_ok

    # a. structural
    print("\n## a. Structural: I_max_date (every lab 2 count, including the usual-side count and the starter-hand date) < D")
    bad = int((F.I_max_date.notna() & (F.I_max_date >= F.date)).sum())
    print(f"rows {len(F):,}; with a source date {int(F.I_max_date.notna().sum()):,}; violations {bad}")
    head = "".join(l for l in b2.b3.CONTACT.read_text().splitlines(True) if l.startswith("#"))
    fit_max = pd.Timestamp(re.search(r"max game_date (\d{4}-\d{2}-\d{2})", head).group(1))
    print(f"contact table header: max game_date {fit_max.date()} < 2022-01-01: {fit_max < pd.Timestamp('2022-01-01')}")
    fail |= bad > 0 or not fit_max < pd.Timestamp("2022-01-01")

    # b. future deletion: 300 random rows (15 dates x 5 rows per season) + 150 doubleheader rows, seed 20261130 (as lab 3)
    print("\n## b. Future deletion: rebuild with every source row dated on or after D deleted, all lab 2 columns, exact")
    XF = pd.concat([keys, F.drop(columns=b2.BASE_COLS)], axis=1)
    rng = np.random.default_rng(SEED_DEL)
    pick = []
    for s in SEASONS:
        days = np.sort(XF.loc[XF.season.eq(s), "date"].unique())
        for d in rng.choice(days, 15, replace=False):
            pick += list(rng.choice(XF.index[XF.date.eq(d)], 5, replace=False))
    pairs = XF.groupby(["date", "home_id", "away_id"]).game_pk.unique().loc[lambda s: s.str.len().eq(2)]
    dh = []
    for s, kk in zip(SEASONS, (6, 6, 6, 7)):
        ps = pairs[pairs.index.get_level_values("date").year == s]
        for g in ps.iloc[np.sort(rng.choice(len(ps), kk, replace=False))]:
            for pk in g:
                dh += list(rng.choice(XF.index[XF.game_pk.eq(pk)], 3, replace=False))
    S = XF.loc[sorted(set(pick) | set(dh))]
    S = S.assign(sample=np.where(S.index.isin(dh), "doubleheader", "random"))
    cols = list(F.columns)
    mism = []
    for d, rr in S.groupby("date"):
        tr = {"base": rr[b2.BASE_COLS], "sched": raw["sched"][raw["sched"].date < d],
              "vault": raw["vault"][raw["vault"].game_date < d], "contact": raw["contact"]}
        Fd = b2.build(b2.prepare(tr), rr[b2.BASE_COLS])
        for c in cols:
            ok = same(rr[c], Fd[c])
            mism += [{"date": d.date(), "row": int(i), "sample": rr["sample"].iloc[j], "col": c, "full": rr[c].iloc[j],
                      "truncated": Fd[c].iloc[j]} for j, i in enumerate(rr.index) if not ok[j]]
        print(f"  {d.date()}: {len(rr)} rows, {len(cols)} columns compared", flush=True)
    by = S.groupby("sample").agg(rows=("game_pk", "size"), dates=("date", "nunique"))
    by["mismatched cells"] = [sum(m["sample"] == k for m in mism) for k in by.index]
    by["columns"], by["cells"] = len(cols), by.rows * len(cols)
    print(f"seasons {S.season.value_counts().sort_index().to_dict()}; doubleheader pairs {len(dh) // 6} (both games, 3 rows each)")
    print(by.to_string())
    print(f"total cells {len(S) * len(cols)}, mismatches {len(mism)}")
    if mism:
        print(pd.DataFrame(mism).head(50).to_string(index=False))
        fail = True

    # c. shuffled inputs (A3 item 13j): every input's rows shuffled, rebuild, sort by key, exact equality
    print("\n## c. Shuffled-input rebuild: rows of every input shuffled (seed 20261131); output sorted by (game_pk, id) must be identical")
    rs = np.random.default_rng(SEED_SHUF)
    shuf = {k: v.iloc[rs.permutation(len(v))].reset_index(drop=True) for k, v in raw.items()}
    Fs = b2.build(b2.prepare(shuf), b2.prepare(shuf)["base"])
    A = F.sort_values(["game_pk", "id"]).reset_index(drop=True)
    B = Fs.sort_values(["game_pk", "id"]).reset_index(drop=True)
    nbad = {c: int((~same(A[c], B[c])).sum()) for c in cols} if A.shape == B.shape and list(A.columns) == list(B.columns) else None
    shuf_ok = nbad is not None and sum(nbad.values()) == 0
    print(f"inputs shuffled: base {len(shuf['base']):,}, schedule {len(shuf['sched']):,}, vault {len(shuf['vault']):,}, contact {len(shuf['contact'])} rows")
    print(f"shape {B.shape} vs {A.shape}; columns {len(cols)}; cells {A.size:,}; mismatched cells {sum(nbad.values()) if nbad else 'n/a'}; "
          f"dtypes equal {A.dtypes.equals(B.dtypes)}")
    if nbad and sum(nbad.values()):
        print({c: k for c, k in nbad.items() if k})
    fail |= not shuf_ok or not A.dtypes.equals(B.dtypes)

    # d. plain-path recompute: 20 rows (seed 20261132), plain filtering of the vault, plain Python shrinkage and I1,
    #    with the train-2022-24 fold's prior strengths as given
    print("\n## d. Plain-path recompute: 20 rows; counts by plain filtering of vault rows; u(f|s), r_b(f), r_p(f|s), I1 by plain "
          "Python with the train-2022-24 fold's m (as given)")
    tr = F.season.isin([2022, 2023, 2024]).to_numpy()
    m, _ = b2.fold_m(F, tr)
    print("m (train 2022-2023-2024): " + ", ".join(f"{k} {v:.6g}" for k, v in m.items()))
    built = pd.concat([F, b2.apply_m(F, m).drop(columns="I1_miss")], axis=1)
    sched = raw["sched"]
    v = raw["vault"][raw["vault"].game_pk.isin(sched.game_pk)].copy()
    v["d"] = v.game_pk.map(dict(zip(sched.game_pk, sched.date)))
    for c in ("batter", "pitcher"):
        v[c] = v[c].astype("int64")
    vt = v[v.events.ne("truncated_pa") | v.events.isna()]
    T = raw["contact"]
    cells = [(r.speed_lo, r.speed_hi, r.angle_lo, r.angle_hi, r.p_hit) for r in T.itertuples()]
    s_lo, a_lo = np.sort(T.speed_lo.unique()), np.sort(T.angle_lo.unique())
    cell = {(r.speed_lo, r.angle_lo): r.p_hit for r in T.itertuples()}

    def x_loop(df):  # one value per AB-ending pitch, plain loop over the table's cells
        out = []
        for typ, ev, ls, la in zip(df.type, df.events, df.launch_speed, df.launch_angle):
            if typ != "X":
                out.append(0.0)
            elif pd.isna(ls) or pd.isna(la):
                out.append(1.0 if ev in HITS else 0.0)
            else:
                hit = [p for slo, shi, alo, ahi, p in cells if slo <= ls < shi and alo <= la < ahi]
                assert len(hit) == 1
                out.append(hit[0])
        return np.array(out)

    def x_search(df):  # league: the cell whose lower edges are the largest <= the value (left-closed)
        out = np.zeros(len(df))
        bip = (df.type == "X").to_numpy()
        lnc = bip & df.launch_speed.notna().to_numpy() & df.launch_angle.notna().to_numpy()
        i = np.searchsorted(s_lo, df.launch_speed.to_numpy(float)[lnc], "right") - 1
        j = np.searchsorted(a_lo, df.launch_angle.to_numpy(float)[lnc], "right") - 1
        out[lnc] = [cell[(s_lo[a], a_lo[b])] for a, b in zip(i, j)]
        out[bip & ~lnc] = [1.0 if e in HITS else 0.0 for e in df.events.to_numpy()[bip & ~lnc]]
        return out

    def shr(x, n, mu, mm):
        return mu if np.isinf(mm) else (x + mm * mu) / (n + mm)

    def plain(x):
        D, pid, sp = x.date, int(x.id), int(x.opp_sp)
        lo = D - pd.Timedelta(days=365)
        P = {}
        seen = vt[(vt.batter == pid) & (vt.d >= pd.Timestamp("2019-01-01")) & (vt.d < D)]
        nL, nR = int((seen.stand == "L").sum()), int((seen.stand == "R").sum())
        P.update({"sd_n": nL + nR, "sd_L": nL, "sd_R": nR})
        pa = v[v.events.notna() & (v.pitcher == sp) & (v.d < D)].sort_values(["d", "game_pk", "at_bat_number", "pitch_number"])
        ch = pa.p_throws.iloc[-1] if len(pa) else np.nan
        P["c_hand"] = ch
        if nL + nR == 0:
            s = np.nan
        elif min(nL, nR) / (nL + nR) >= 0.20:
            s = {"R": "L", "L": "R"}.get(ch, np.nan)
        else:
            s = "L" if nL > nR else "R"
        P["side_s"] = s
        w = vt[(vt.d >= lo) & (vt.d < D)]
        dates = [seen.d.max() if len(seen) else pd.NaT, pa.d.max() if len(pa) else pd.NaT]
        for g, fams in (("", FAMILY), ("_alt", FAMILY_ALT)):
            codes = set().union(*fams.values())
            w3 = w[w.pitch_type.isin(codes)]
            ab = w3[w3.events.notna() & ~w3.events.isin(NOT_AB)]
            ua, us, lh = w3[w3.pitcher == sp], w3[(w3.pitcher == sp) & (w3.stand == s)], w3[w3.p_throws == ch]
            bb, pp = ab[ab.batter == pid], ab[(ab.pitcher == sp) & (ab.stand == s)]
            xb, xp, xl = x_loop(bb), x_loop(pp), x_search(ab)
            for pre, df, xs in (("ua_", ua, None), ("us_", us, None), ("lh_", lh, None), ("b_", bb, xb), ("p_", pp, xp), ("lg_", ab, xl)):
                if g == "":
                    P[f"{pre}n"] = len(df)
                    if xs is not None:
                        P[f"{pre}x"] = float(np.sum(xs))
                    dates.append(df.d.max() if len(df) else pd.NaT)
                for f, cs in fams.items():
                    k = df.pitch_type.isin(cs).to_numpy()
                    P[f"{pre}n_{f}{g}"] = int(k.sum())
                    if xs is not None:
                        P[f"{pre}x_{f}{g}"] = float(np.sum(xs[k]))
            for f in fams:
                P[f"L_{f}{g}"] = P[f"lg_x_{f}{g}"] / P[f"lg_n_{f}{g}"]
        P["L"] = P["lg_x"] / P["lg_n"]
        P["I1_miss"] = int(pd.isna(s) or P["b_n"] == 0 or P["us_n"] == 0)
        P["I_max_date"] = max([q for q in dates if pd.notna(q)], default=pd.NaT)
        # derived, written out directly (plan Section 6, A3 item 13b, A4 items 2-3)
        L = P["L"]
        Rb = shr(P["b_x"], P["b_n"], L, m["Rb"]) / L
        Rp = shr(P["p_x"], P["p_n"], L, m["Rp"]) / L
        P["R_b"], P["R_p"] = Rb, Rp
        I1 = 0.0
        for f in FAMILY:
            Lf = P[f"L_{f}"]
            u1 = shr(P[f"ua_n_{f}"], P["ua_n"], P[f"lh_n_{f}"] / P["lh_n"] if P["lh_n"] else np.nan, m["u1"])
            u = shr(P[f"us_n_{f}"], P["us_n"], u1, m["u2"])
            rb = shr(P[f"b_x_{f}"], P[f"b_n_{f}"], Rb * Lf, m["rb"]) / Lf
            rp = shr(P[f"p_x_{f}"], P[f"p_n_{f}"], Rp * Lf, m["rp"]) / Lf
            P[f"u_{f}"], P[f"r_b_{f}"], P[f"r_p_{f}"] = u, rb, rp
            I1 += u * Lf * (rb * rp - Rb * Rp)
        P["I1"] = 0.0 if P["I1_miss"] else I1
        return P

    rp_ = np.random.default_rng(SEED_PLAIN)
    out = []
    for i in rp_.choice(len(built), 20, replace=False):
        x = built.iloc[i]
        P = plain(x)
        missing = [c for c in built.columns if c not in P and c not in b2.BASE_COLS]
        assert not missing, missing
        out += [{"row": int(i), "date": x.date.date(), "batter": int(x.id), "I1_miss": int(x.I1_miss), "column": c,
                 "built": x[c], "plain": P[c], "match": close(x[c], P[c])} for c in P]
    O = pd.DataFrame(out)
    key = O[O.column.isin(["u_fb", "u_br", "u_os", "r_b_fb", "r_b_br", "r_b_os", "r_p_fb", "r_p_br", "r_p_os", "I1", "side_s", "c_hand"])]
    print(key.to_string(index=False))
    print(f"\nplain-path rows {O.row.nunique()} (I1_miss = 1 on {int(O.drop_duplicates('row').I1_miss.sum())}); cells {len(O)} "
          f"(every table column plus R_b, R_p, u(f|s), r_b(f), r_p(f|s), I1); mismatches {int((~O.match).sum())}")
    if not O.match.all():
        print(O[~O.match].to_string(index=False))
    fail |= not O.match.all()

    print(f"\nruntime {time.time() - t0:.0f} s")
    print("LEAK CHECK:", "FAIL - STOP" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
