"""Beat the Streak lab 2, Step 1c: pitch-type matchup (I1) as-of feature table (plan BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md
Sections 2-6, addenda A1-A4).

One row per row of lab 1's table, keyed (game_pk, id). Only lab 1's game_pk, id, date, season and opp_sp are read; the
base file is checked by its recorded file sha256, so no outcome column is ever loaded. No box-score file is opened
(A4 item 1) and no 2026 row is read (A3 item 1). Every history count uses games with effective date strictly before D.
The table stores as-of counts and league levels only; shrinkage and I1 are fit-time quantities (A3 item 13), computed
per training fold by fold_m() and apply_m() below (A4 item 5d).

  python analysis/btslab2_features.py > analysis/out/btslab2_features_build_log.txt
writes analysis/cache/btslab2_features.parquet
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
import btslab_analysis as ba  # noqa: E402
import btslab_features as bf  # noqa: E402
import btslab3_features as b3  # noqa: E402
from lab2_contact_table import ANGLE_EDGES, SPEED_EDGES, bins  # noqa: E402

BASE_COLS = ["game_pk", "id", "date", "season", "opp_sp"]  # A3 item 13d: nothing else is read from lab 1's table
OUT_PARQUET = ROOT / "analysis" / "cache" / "btslab2_features.parquet"
VAULT = ROOT / "data" / "master_physics_vault.parquet"
VAULT_COLS = ["game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher", "pitch_type", "events", "type",
              "stand", "p_throws", "launch_speed", "launch_angle"]
VKEY = ["game_pk", "at_bat_number", "pitch_number"]
SIDE_LO = pd.Timestamp("2019-01-01")  # usual side: pitches seen 2019 to D (plan Section 6, A3 item 13f)
F3 = ["fb", "br", "os"]
FAM = {"FF": "fb", "SI": "fb", "FA": "fb", "FC": "fb", "SL": "br", "ST": "br", "SV": "br", "CU": "br", "KC": "br", "CS": "br",
       "CH": "os", "FS": "os", "FO": "os", "SC": "os"}  # plan Section 4; KN, EP, PO, UN dropped (A3 item 13c)
GROUPINGS = {"": FAM, "_alt": {**FAM, "FC": "br", "FS": "fb"}}  # exploratory alternative grouping (plan Section 6)
MIN_N, MIN_N_FALLBACK = 100, 20  # A3 item 13a


# ---------------------------------------------------------------- inputs (sorted by key on load, A3 item 13j)
def contact_array(T: pd.DataFrame) -> np.ndarray:
    """13 x 11 array of p_hit by (speed bin, angle bin), independent of the CSV's row order."""
    P = np.full((len(SPEED_EDGES) - 1, len(ANGLE_EDGES) - 1), np.nan)
    for r in T.itertuples():
        i, j = SPEED_EDGES.index(r.speed_lo), ANGLE_EDGES.index(r.angle_lo)
        assert (SPEED_EDGES[i + 1], ANGLE_EDGES[j + 1]) == (r.speed_hi, r.angle_hi)
        P[i, j] = r.p_hit
    assert len(T) == 143 and not np.isnan(P).any()
    return P


def load_base(cols=BASE_COLS) -> pd.DataFrame:
    """Lab 1's table, checked by its recorded file sha256 as lab 3's builder does; only `cols` are read."""
    assert b3.sha(b3.BASE) == b3.BASE_FILE_SHA, "stop: lab 1 table differs from the recorded file"
    return pd.read_parquet(b3.BASE, columns=cols)


def load_raw() -> dict:
    return {"base": load_base(), "sched": b3.load_schedule(), "vault": pd.read_parquet(VAULT, columns=VAULT_COLS),
            "contact": pd.read_csv(b3.CONTACT, comment="#")}


def prepare(raw: dict) -> dict:
    """Sort every input by its key; keep regular-season vault rows dated by their game's effective date."""
    sched = raw["sched"].sort_values("game_pk").reset_index(drop=True)
    assert sched.game_pk.is_unique
    gd = sched.set_index("game_pk").date
    v = raw["vault"]
    v = v[v.game_pk.isin(gd.index)].copy()
    v["date"] = v.game_pk.map(gd).astype("datetime64[ns]")
    for c in ("game_pk", "batter", "pitcher", "at_bat_number", "pitch_number"):
        assert v[c].notna().all(), c
        v[c] = v[c].to_numpy("int64")  # nullable Int64 in the vault; merge keys must match the base's int64
    for c in ("stand", "p_throws"):
        assert v[c].isin(["L", "R"]).all(), c
        v[c] = v[c].to_numpy(object)
    v = v.sort_values(["date"] + VKEY, kind="mergesort").reset_index(drop=True)
    assert not v.duplicated(VKEY).any()
    base = raw["base"].sort_values(["date", "game_pk", "id"], kind="mergesort").reset_index(drop=True)
    return {"base": base, "sched": sched, "vault": v, "contact": contact_array(raw["contact"])}


# ---------------------------------------------------------------- pitch-level daily tables
def _daily(df: pd.DataFrame, keys: list, fam: pd.Series, x: pd.Series | None, g: str) -> pd.DataFrame:
    """Per keys+date: family counts (and x sums) for one grouping, columns named n{f}{g} / x{f}{g}."""
    k = keys + ["date"]
    n = df.groupby(k + [fam.rename("_f")]).size().unstack("_f", fill_value=0).reindex(columns=F3, fill_value=0)
    out = n.rename(columns={f: f"n_{f}{g}" for f in F3})
    if x is not None:
        s = df.assign(_x=x).groupby(k + [fam.rename("_f")])._x.sum().unstack("_f", fill_value=0.0).reindex(columns=F3, fill_value=0.0)
        out = out.join(s.rename(columns={f: f"x_{f}{g}" for f in F3}))
    return out


def daily_tables(v: pd.DataFrame, P: np.ndarray) -> dict:
    hand = v[v.events.notna()].groupby(["pitcher", "date"]).p_throws.last().reset_index()  # lab 1's raw_tables() "hand" (A4 5c)
    vt = v[v.events.ne("truncated_pa") | v.events.isna()]  # truncated_pa rows dropped from every lab 2 count (A4 5a)
    side = vt.groupby(["batter", "date"]).stand.value_counts().unstack("stand", fill_value=0).reindex(columns=["L", "R"], fill_value=0)
    side = side.assign(n=side.L + side.R)[["n", "L", "R"]].reset_index()
    fam0 = vt.pitch_type.map(FAM)
    v3 = vt[fam0.notna()]  # the three families only
    ab = v3[v3.events.notna() & ~v3.events.isin(bf.NON_AB)].copy()  # AB-ending pitches (plan Section 5)
    launch = ab.launch_speed.notna() & ab.launch_angle.notna()
    bip = ab.type.eq("X")
    x = np.zeros(len(ab))
    li = (bip & launch).to_numpy()
    x[li] = P[bins(ab.launch_speed[li], SPEED_EDGES).to_numpy(int), bins(ab.launch_angle[li], ANGLE_EDGES).to_numpy(int)]
    x[(bip & ~launch).to_numpy()] = ab.events[bip & ~launch].isin(bf.H_EV).to_numpy(float)  # "use their actual result"
    ab["x"] = x
    T = {"hand": hand, "side": side}
    parts = {"ua": [], "us": [], "lh": [], "bab": [], "pab": [], "lab": []}
    for g, fm in GROUPINGS.items():
        f3, fa = v3.pitch_type.map(fm), ab.pitch_type.map(fm)
        parts["ua"].append(_daily(v3, ["pitcher"], f3, None, g))
        parts["us"].append(_daily(v3, ["pitcher", "stand"], f3, None, g))
        parts["lh"].append(_daily(v3, ["p_throws"], f3, None, g))
        parts["bab"].append(_daily(ab, ["batter"], fa, ab.x, g))
        parts["pab"].append(_daily(ab, ["pitcher", "stand"], fa, ab.x, g))
        parts["lab"].append(_daily(ab.assign(one=1), ["one"], fa, ab.x, g))
    for k, ps in parts.items():
        d = pd.concat(ps, axis=1).fillna(0)
        d.insert(0, "n", d[[f"n_{f}" for f in F3]].sum(axis=1))
        if any(c.startswith("x_") for c in d.columns):
            d.insert(1, "x", d[[f"x_{f}" for f in F3]].sum(axis=1))
        T[k] = d.reset_index()
    return T


# ---------------------------------------------------------------- build
def _win(T, keys, rows, qkeys, lo, prefix):
    cols = [c for c in T.columns if c not in keys + ["date"]]
    w = bf.window(T, keys, rows, qkeys, cols, lo)
    return w.rename(columns={c: f"{prefix}{c}" for c in cols}), w.src


def build(prep: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Lab 2 counts and league levels for `rows` (game_pk, id, date, season, opp_sp)."""
    rows = rows.reset_index(drop=True)
    D = rows.date
    y365 = D - pd.Timedelta(days=365)
    T = daily_tables(prep["vault"], prep["contact"])
    F = rows[BASE_COLS].copy()
    srcs = []

    # side s: usual side (most pitches seen, 2019 to D); switch hitter (>= 20% each side) bats opposite the starter's hand
    sd, src = _win(T["side"], ["batter"], rows, ["id"], pd.Series(SIDE_LO, index=rows.index), "sd_")
    srcs.append(src)
    hand = pd.merge_asof(rows[["date", "opp_sp"]].assign(_i=np.arange(len(rows))).sort_values("date"),
                         T["hand"].rename(columns={"pitcher": "opp_sp"}).assign(hdate=lambda x: x.date).sort_values("date"),
                         on="date", by="opp_sp", allow_exact_matches=False).sort_values("_i")
    srcs.append(hand.hdate.reset_index(drop=True))
    F["c_hand"] = hand.p_throws.to_numpy()
    for c in ("sd_n", "sd_L", "sd_R"):
        F[c] = sd[c].to_numpy()
    n = F.sd_n.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        switch = (n > 0) & (np.minimum(F.sd_L, F.sd_R) / n >= 0.20)
    opp = F.c_hand.map({"R": "L", "L": "R"})
    F["side_s"] = np.where(n == 0, None, np.where(switch, opp, np.where(F.sd_L > F.sd_R, "L", "R")))
    F["side_s"] = F.side_s.where(F.side_s.isin(["L", "R"]))  # None for no history, or a switch hitter with unknown hand
    qs = rows.assign(stand=F.side_s.fillna("N").to_numpy(object), p_throws=F.c_hand.fillna("N").to_numpy(object), one=1)

    for key, tkeys, qkeys, pre in (("ua", ["pitcher"], ["opp_sp"], "ua_"), ("us", ["pitcher", "stand"], ["opp_sp", "stand"], "us_"),
                                   ("lh", ["p_throws"], ["p_throws"], "lh_"), ("bab", ["batter"], ["id"], "b_"),
                                   ("pab", ["pitcher", "stand"], ["opp_sp", "stand"], "p_"), ("lab", ["one"], ["one"], "lg_")):
        w, src = _win(T[key], tkeys, qs, qkeys, y365, pre)
        srcs.append(src)
        for c in w.columns:
            if c != "src":
                F[c] = w[c].to_numpy()
    for g in GROUPINGS:
        with np.errstate(invalid="ignore", divide="ignore"):
            if g == "":
                F["L"] = F.lg_x / F.lg_n
            for f in F3:
                F[f"L_{f}{g}"] = F[f"lg_x_{f}{g}"] / F[f"lg_n_{f}{g}"]
    F["I1_miss"] = (F.side_s.isna() | F.b_n.eq(0) | F.us_n.eq(0)).astype(int)  # plan Section 6, A4 item 5b
    F["I_max_date"] = pd.concat([s.reset_index(drop=True) for s in srcs], axis=1).max(axis=1).to_numpy()
    return F


def check(F: pd.DataFrame) -> None:
    assert not (F.I_max_date >= F.date).any(), "I_max_date not before D"
    lv = ["L"] + [f"L_{f}{g}" for g in GROUPINGS for f in F3]
    assert np.isfinite(F[lv].to_numpy(float)).all(), "stop: a league level L or L_f is not finite"
    for g in GROUPINGS:  # stored family counts sum to the stored totals (A3 item 13g)
        for p in ("ua_", "us_", "lh_", "b_", "p_", "lg_"):
            assert (F[[f"{p}n_{f}{g}" for f in F3]].sum(axis=1) == F[f"{p}n"]).all(), f"{p}n{g}"
        for p in ("b_", "p_", "lg_"):
            assert np.allclose(F[[f"{p}x_{f}{g}" for f in F3]].sum(axis=1), F[f"{p}x"], rtol=1e-12, atol=1e-9), f"{p}x{g}"


# ---------------------------------------------------------------- fit time: prior strengths and I1 (A3 item 13, A4)
def est(x, n, mu, N=None):
    """estimate_m with A3 item 13a's fallback: min_n 100; if no row qualifies, 20; if none, m = inf (estimate_m's own rule)."""
    x, n, mu = (np.asarray(a, float) for a in (x, n, mu))
    Nn = None if N is None else np.asarray(N, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.ones(n.shape) if Nn is None else 1 - n / np.where(Nn > 0, Nn, np.nan)
        ok = np.isfinite(mu) & np.isfinite(c)
        k100, k20 = int((ok & (n >= MIN_N)).sum()), int((ok & (n >= MIN_N_FALLBACK)).sum())
        mn = MIN_N if k100 else MIN_N_FALLBACK
        m = bf.estimate_m(x, n, mu, min_n=mn, N=Nn) if (k100 or k20) else np.inf
    return {"m": m, "min_n": mn, "k100": k100, "k20": k20, "used": ok & (n >= mn)}


def units(T: pd.DataFrame) -> tuple:
    """Unit-season rows (A3 amendment 8): latest training date, ties to the largest game_pk, then the largest id."""
    T = T.sort_values(["date", "game_pk", "id"], kind="mergesort")
    return (T.drop_duplicates(["id", "season"], keep="last"), T.drop_duplicates(["opp_sp", "season"], keep="last"),
            T[T.side_s.notna()].drop_duplicates(["opp_sp", "season", "side_s"], keep="last"))


def _comb(parts, wts):
    """Usage: one m per level, the family-total-weighted mean of the finite family m's (A3 item 13b)."""
    fin = [(p["m"], w) for p, w in zip(parts, wts) if np.isfinite(p["m"]) and w > 0]
    return float(sum(m * w for m, w in fin) / sum(w for _, w in fin)) if fin else np.inf


def fold_m(X: pd.DataFrame, tr: np.ndarray, g: str = "") -> tuple[dict, list]:
    """Prior strengths on the training rows only; returns m and a log of every estimator cell."""
    bat, sp, sps = units(X[tr])
    f = lambda d, c: d[c].to_numpy(float)
    m, log = {}, []
    with np.errstate(invalid="ignore", divide="ignore"):
        for lvl, d, mu_of, N in (("usage L1 u(f)", sp, lambda ff: f(sp, f"lh_n_{ff}{g}") / f(sp, "lh_n"), None),
                                 ("usage L2 u(f|s)", sps, lambda ff: f(sps, f"ua_n_{ff}{g}") / f(sps, "ua_n"), f(sps, "ua_n"))):
            pre = "ua_" if lvl.startswith("usage L1") else "us_"
            parts, wts = [], []
            for ff in F3:
                r = est(f(d, f"{pre}n_{ff}{g}"), f(d, f"{pre}n"), mu_of(ff), N)
                parts.append(r)
                wts.append(float(f(d, f"{pre}n_{ff}{g}")[r["used"]].sum()))
                log.append({"estimator": lvl, "cell": ff, "k100": r["k100"], "k20": r["k20"], "min_n": r["min_n"], "m": r["m"], "weight": wts[-1]})
            m["u1" if pre == "ua_" else "u2"] = _comb(parts, wts)
            log.append({"estimator": lvl, "cell": "combined", "m": m["u1" if pre == "ua_" else "u2"]})
        r = est(f(bat, "b_x"), f(bat, "b_n"), f(bat, "L"))
        m["Rb"] = r["m"]
        log.append({"estimator": "R_b", "cell": "all", **{k: r[k] for k in ("k100", "k20", "min_n", "m")}})
        r = est(f(sps, "p_x"), f(sps, "p_n"), f(sps, "L"))
        m["Rp"] = r["m"]
        log.append({"estimator": "R_p(s)", "cell": "both sides", **{k: r[k] for k in ("k100", "k20", "min_n", "m")},
                    "k100 by side": {s: int((f(sps, "p_n")[sps.side_s.eq(s).to_numpy()] >= MIN_N).sum()) for s in ("L", "R")}})
        # pooled family measures (A4 item 3): rows stacked once per family, mu = raw overall x L_f / L, N = all AB-ending
        for name, d, pre, keyc in (("pooled r_b(f)", bat, "b_", None), ("pooled r_p(f|s)", sps, "p_", "side_s")):
            raw = f(d, f"{pre}x") / f(d, f"{pre}n")
            xs = np.concatenate([f(d, f"{pre}x_{ff}{g}") for ff in F3])
            ns = np.concatenate([f(d, f"{pre}n_{ff}{g}") for ff in F3])
            mus = np.concatenate([raw * f(d, f"L_{ff}{g}") / f(d, "L") for ff in F3])
            Ns = np.tile(f(d, f"{pre}n"), 3)
            r = est(xs, ns, mus, Ns)
            m["rb" if pre == "b_" else "rp"] = r["m"]
            cells = {}
            for i, ff in enumerate(F3):
                nn = f(d, f"{pre}n_{ff}{g}")
                if keyc is None:
                    cells[ff] = int((nn >= MIN_N).sum())
                else:
                    for s in ("L", "R"):
                        cells[f"{ff}/{s}"] = int((nn[d[keyc].eq(s).to_numpy()] >= MIN_N).sum())
            log.append({"estimator": name, "cell": "pooled", **{k: r[k] for k in ("k100", "k20", "min_n", "m")}, "k100 by cell": cells})
    return m, log


def apply_m(X: pd.DataFrame, m: dict, g: str = "") -> pd.DataFrame:
    """u(f|s), R_b, R_p(s), r_b(f), r_p(f|s) and I1 for every row of X, given one fold's m (plan Section 6, A4 items 2-3)."""
    f = lambda c: X[c].to_numpy(float)
    out = pd.DataFrame(index=X.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = f("L")
        Rb = bf.shrink(f("b_x"), f("b_n"), L, m["Rb"]) / L
        Rp = bf.shrink(f("p_x"), f("p_n"), L, m["Rp"]) / L
        out["R_b"], out["R_p"] = Rb, Rp
        I1 = np.zeros(len(X))
        usum = np.zeros(len(X))
        for ff in F3:
            u1 = bf.shrink(f(f"ua_n_{ff}{g}"), f("ua_n"), f(f"lh_n_{ff}{g}") / f("lh_n"), m["u1"])
            u = bf.shrink(f(f"us_n_{ff}{g}"), f("us_n"), u1, m["u2"])
            Lf = f(f"L_{ff}{g}")
            rb = bf.shrink(f(f"b_x_{ff}{g}"), f(f"b_n_{ff}{g}"), Rb * Lf, m["rb"]) / Lf
            rp = bf.shrink(f(f"p_x_{ff}{g}"), f(f"p_n_{ff}{g}"), Rp * Lf, m["rp"]) / Lf
            out[f"u_{ff}"], out[f"r_b_{ff}"], out[f"r_p_{ff}"] = u, rb, rp
            I1 += u * Lf * (rb * rp - Rb * Rp)
            usum += u
    miss = X.I1_miss.to_numpy() == 1
    assert (np.abs(usum[~miss] - 1) <= 1e-9).all(), "stop rule 3: usage shares do not sum to 1"
    out["I1"] = np.where(miss, 0.0, I1)
    out["I1_miss"] = miss.astype(int)
    return out


FOLDS = [("WF", tr, te) for tr, te in ba.WF] + [("LOSO", tr, te) for tr, te in ba.LOSO]  # lab 1's six folds (A4 item 5e)


def fold_label(d, tr, te):
    return f"{d} | {'-'.join(map(str, tr))} | {te}"


# ---------------------------------------------------------------- main
def describe_m(log: list) -> str:
    rows = []
    for r in log:
        extra = {k: v for k, v in r.items() if k in ("k100 by side", "k100 by cell")}
        rows.append({"estimator": r["estimator"], "cell": r["cell"], "k100": r.get("k100", ""), "k20": r.get("k20", ""),
                     "min_n": r.get("min_n", ""), "m": f"{r['m']:.4g}", "weight": r.get("weight", ""),
                     "detail": next(iter(extra.values()), "")})
    return pd.DataFrame(rows).to_string(index=False)


def main():
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 200)
    print(f"Lab 2 Step 1c feature build ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}); python {sys.version.split()[0]}, "
          f"pandas {pd.__version__}")
    for n in ("ARSENAL_PLAN", "ADDENDUM_A1", "ADDENDUM_A2", "ADDENDUM_A3", "ADDENDUM_A4"):
        p = ROOT / "analysis" / f"BEAT_THE_STREAK_LAB2_{n}.md"
        print(f"{p.name} sha256 {b3.sha(p)}")
    raw = load_raw()
    print(f"base {b3.BASE.relative_to(ROOT)}: file sha256 matches the recorded {b3.BASE_FILE_SHA[:8]}... (content e1fb5d01..., "
          f"not recomputed here: that needs the outcome column); columns read {BASE_COLS}; rows {len(raw['base']):,}")
    prep = prepare(raw)
    v, s = prep["vault"], prep["sched"]
    print(f"schedule: regular-season games {len(s):,}, seasons {sorted(s.season.unique().tolist())} (2026 files skipped as lab 3's "
          f"load_schedule); vault {VAULT.name} rows read {len(raw['vault']):,}, regular-season rows kept {len(v):,} "
          f"({v.date.min():%Y-%m-%d} to {v.date.max():%Y-%m-%d}); duplicate vault keys 0")
    print(f"contact table {b3.CONTACT.name} sha256 {b3.sha(b3.CONTACT)}; vault sha256 not hashed (136 MB, unchanged file per Step 1b log)")
    tp = v.events.eq("truncated_pa")
    other = v.pitch_type.map(FAM).isna()
    print(f"truncated_pa rows dropped {int(tp.sum()):,}; other-code pitches (KN, EP, PO, UN) {int(other.sum()):,} "
          f"by code {v.pitch_type[other].value_counts().to_dict()}")
    F = build(prep, prep["base"])
    check(F)
    assert len(F) == len(raw["base"]) and not F.duplicated(["game_pk", "id"]).any()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET.relative_to(ROOT)}: rows {len(F):,}, cols {F.shape[1]}; order (date, game_pk, id); duplicate keys 0")
    print(f"content sha256 {b3.content_hash(F)} (sha256 of pandas.util.hash_pandas_object(df, index=False)); file sha256 {b3.sha(OUT_PARQUET)}")
    print("checks: I_max_date < D on every row; L and every L_f finite on every row (both groupings); family counts sum to totals")

    print("\n## I1_miss by season")
    g = F.groupby("season")
    print(pd.DataFrame({"rows": g.size(), "I1_miss": g.I1_miss.sum(), "rate": g.I1_miss.mean().round(4),
                        "no side s": g.side_s.apply(lambda x: int(x.isna().sum())), "batter no AB-ending": g.b_n.apply(lambda x: int(x.eq(0).sum())),
                        "starter no pitches to s": g.us_n.apply(lambda x: int(x.eq(0).sum()))}).to_string())
    print(f"all seasons: {int(F.I1_miss.sum()):,} of {len(F):,} ({F.I1_miss.mean():.4f})")
    sw = (F.sd_n > 0) & (np.minimum(F.sd_L, F.sd_R) / F.sd_n.where(F.sd_n > 0) >= 0.20)
    print(f"\nside s: L {int(F.side_s.eq('L').sum()):,}, R {int(F.side_s.eq('R').sum()):,}, missing {int(F.side_s.isna().sum()):,}; "
          f"rows with a switch hitter (>= 20% each side) {int(sw.sum()):,}; c_hand missing {int(F.c_hand.isna().sum()):,}")
    print("league levels (per-AB measure), mean over rows: L {:.5f}; ".format(F.L.mean())
          + "; ".join(f"L_{f}{g} {F[f'L_{f}{g}'].mean():.5f}" for g in GROUPINGS for f in F3))

    print("\n## Prior strengths per fold (A3 item 13a, A4 items 2-3, 5e); n >= 100 qualifying unit-season rows")
    print("The LOSO fold training on 2022-2023-2024 has the same training set as WF 2022-2023-2024, hence the same m values.")
    X = F
    for d, tr, te in FOLDS:
        trm = X.season.isin(tr).to_numpy()
        m, log = fold_m(X, trm)
        bat, sp, sps = units(X[trm])
        print(f"\n### {fold_label(d, tr, te)}  (unit-seasons: batter {len(bat):,}, starter {len(sp):,}, starter-side {len(sps):,})")
        print(describe_m(log))
        fb = [r for r in log if r.get("min_n") == MIN_N_FALLBACK]
        fi = [r for r in log if "k100" in r and not np.isfinite(r["m"])]
        print(f"fallbacks to min_n = 20: {[(r['estimator'], r['cell']) for r in fb] or 'none'}; "
              f"m = inf: {[(r['estimator'], r['cell']) for r in fi] or 'none'}")
        a = apply_m(X, m)
        print(f"I1 on all rows with this fold's m: rows with I1_miss = 0 {int((a.I1_miss == 0).sum()):,}; "
              f"I1 mean {a.I1[a.I1_miss == 0].mean():+.6f}, SD {a.I1[a.I1_miss == 0].std():.6f}; usage sums to 1 within 1e-9: True")
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
