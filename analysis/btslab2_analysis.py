"""Beat the Streak lab 2, Step 1d: models and tests (plan BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md + addenda A1-A6).

M1 is lab 1's model and M3 lab 3's, both imported (btslab_analysis, btslab3_analysis), not copied. M2 = M1 + I1 + I1_miss and
M4 = M3 + I1 + I1_miss, with I1 shrunk per training fold by btslab2_features.fold_m()/apply_m(). The placebo is placebo P
(A6 item 2). Outcomes are FAKE unless --real is given; without --real the hit column is never loaded (lab 1's load() guard).
  python analysis/btslab2_analysis.py --smoke                 # unit checks, then smoke seed 9109 -> btslab2_smoke_log.txt
  python analysis/btslab2_analysis.py --time                  # times one null seed and the real-run pieces -> btslab2_time_log.txt
  python analysis/btslab2_analysis.py --dry-criterion         # throwaway dry runs 9101-9108 (A3 item 4), appended to the smoke log
  python analysis/btslab2_analysis.py --fake planted          # registered, seed 3300 -> btslab2_selftest_planted_log.txt
  python analysis/btslab2_analysis.py --fake planted-null     # registered, seed 3301 -> btslab2_selftest_planted_null_log.txt
  python analysis/btslab2_analysis.py --null-study            # seeds 3001-3200, parallel -> btslab2_null_study_log.txt
  python analysis/btslab2_analysis.py --real                  # the real run, once -> analysis/out/lab2_results_log.txt
"""
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab2_features as b2  # noqa: E402
import btslab2_placebo_check as pc  # noqa: E402
import btslab3_analysis as l3  # noqa: E402
import btslab3_features as b3  # noqa: E402
import btslab_analysis as ba  # noqa: E402
import btslab_features as bf  # noqa: E402
from btslab_analysis import design, fit_l2, pred, say  # noqa: E402
from v4_analysis import ci, ll, md_table  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = ba.OUT
RESULTS = OUT / "lab2_results_log.txt"  # A3 item 10: --real refuses to run if it exists
STARTED = OUT / "lab2_real_started.txt"  # written before the first real-outcome fit; --real refuses if it exists
LEAK_LOG = OUT / "btslab2_leakcheck_log.txt"
SELFTEST = {"planted": OUT / "btslab2_selftest_planted_log.txt", "planted-null": OUT / "btslab2_selftest_planted_null_log.txt",
            "null": OUT / "btslab2_null_study_log.txt"}
NULL_CSV = OUT / "btslab2_null_study_seeds.csv"
IGAP_LOG = OUT / "btslab2_igap_log.txt"  # A6 item 13
IGAP_DONE = "I_GAP REPORT: COMPLETE"  # its last non-blank line (Claude (chat), 2026-11-09)
TIME_LOG = OUT / "btslab2_time_log.txt"  # A6 item 10
SMOKE_LOG = OUT / "btslab2_smoke_log.txt"
LAB2_CONTENT = "62445424bda0c468f7dab50225a5a1d7fb346206e113a5cb18ef7367b7beb8f7"  # A6 item 4 (stop rule 6)
# seeds: plan Section 8 "seed = 20261101 + test number"; A3 item 8 for T1b, T3b, the M4 placebo; A3 item 7 / A6 item 2 permutations
SEED = {"T1": 20261102, "T2": 20261103, "T3": 20261104, "T4": 20261105, "T5": 20261106, "T6": 20261107, "coef": 20261120,
        "T1b": 20261121, "T3b": 20261123, "M4 placebo": 20261124}
PERM_SEED = 20261150  # + k
REGISTERED = {"planted": 3300, "planted-null": 3301}  # A3 item 8
NULL_SEEDS = range(3001, 3201)
DRY = [("planted", 9101), ("planted", 9102)] + [("planted-null", s) for s in range(9103, 9109)]  # A3 item 4
SMOKE_SEED = 9109
WF, LOSO = ba.WF, ba.LOSO
F3 = b2.F3
UNIT = pc.UNIT  # (game_pk, opp_sp, side_s): placebo P unit (A6 item 2)
M2_COLS = ba.M1_COLS + ["I1", "I1_miss"]  # plan Section 7
M4_COLS = l3.M3_COLS + ["I1", "I1_miss"]  # A3 item 3
X1_COLS = ba.M1_COLS + ["I1", "I2", "I3", "I4", "I1_miss"]  # T6, A6 item 7
XA_COLS = ba.M1_COLS + ["I1_alt", "I1_miss"]
PROF = [f"u_{f}" for f in F3] + [f"r_p_{f}" for f in F3] + ["R_p"]  # the starter-side bundle (A6 item 2)
OPT = {"nperm": 50, "ignore_stops": False}
# lab 3's printed M3 numbers (analysis/out/lab3_results_log.txt) compared under --real (A3 item 6, stop rule 7)
REPRO_M3_FOLDS = {("WF", (2022, 2023), 2024): ("0.001 (grid edge)", "0.65755", "0.5763", "0.5492", "+0.1188"),
                  ("WF", (2022, 2023, 2024), 2025): ("0.001 (grid edge)", "0.65782", "0.5756", "0.5480", "+0.1223"),
                  ("LOSO", (2023, 2024, 2025), 2022): ("0.001 (grid edge)", "0.65778", "0.5721", "0.5429", "+0.1252"),
                  ("LOSO", (2022, 2024, 2025), 2023): ("0.001 (grid edge)", "0.65891", "0.5742", "0.5471", "+0.1229"),
                  ("LOSO", (2022, 2023, 2025), 2024): ("0.001 (grid edge)", "0.65803", "0.5745", "0.5462", "+0.1213"),
                  ("LOSO", (2022, 2023, 2024), 2025): ("0.001 (grid edge)", "0.65782", "0.5756", "0.5480", "+0.1223")}
REPRO_M3_WF = {"test": "batter-game log loss M1 0.65947, M3 0.65878; AUC M1 0.5740, M3 0.5771",
               "lab 3 T1": "pooled 2024–25: +0.00069 (95% +0.00040 to +0.00099), rows 87021", "top-1 M3": "0.810"}
LAB3_T1_SEED = 20261201


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- guards (run before any data file is read)
def guard_once(path: Path, what="the real run is run once, A3 item 10"):
    if path.exists():
        sys.exit(f"refusing to run: {path} already exists ({what})")


def real_gate(leak=LEAK_LOG, selftest=SELFTEST, igap=IGAP_LOG, results=RESULTS, started=STARTED) -> list:
    """--real, before any data file is read: run-once guard, then every PASS log (A3 item 13i, A6 items 13 and 15)."""
    for p in (started, results):
        if p.exists():
            sys.exit(f"refusing to run --real: {p} already exists; an addendum must record what happened before another run")
    need = [(leak, "LEAK CHECK: PASS"), *[(p, f"SELF-TEST {k}: PASS") for k, p in selftest.items()], (igap, IGAP_DONE)]
    lines, bad = [], []
    for p, s in need:
        if s is IGAP_DONE:  # the descriptive I_gap report: its last non-blank line must be the completion line
            ok = p.exists() and [x for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()][-1:] == [s]
            lines.append(f"- {p.name} last non-blank line is {s!r}: {'PASS' if ok else 'FAIL'}")
        else:
            ok = p.exists() and s in p.read_text(encoding="utf-8", errors="replace")
            lines.append(f"- {p.name} contains {s!r}: {'PASS' if ok else 'FAIL'}")
        bad += [] if ok else [p.name]
    if bad:
        sys.exit("refusing to run --real (no data file read): missing or failed precondition log(s): " + ", ".join(bad))
    return lines


def time_gate(time_log=TIME_LOG, script=SCRIPT) -> str:
    """A6 item 10: the 200-seed null study needs the timing log, recording this script's current on-disk SHA-256."""
    if not time_log.exists():
        sys.exit(f"refusing to run the null study: {time_log.name} does not exist (A6 item 10; run --time first)")
    m = re.search(r"script sha256 ([0-9a-f]{64})", time_log.read_text(encoding="utf-8"))
    cur = sha(script)
    if not m or m.group(1) != cur:
        sys.exit(f"refusing to run the null study: {time_log.name} records {m.group(1) if m else 'no'} script sha256, "
                 f"current {script.name} is {cur} (A6 item 10; rerun --time)")
    return cur


def table_hashes(real: bool) -> bool:
    """Stop rule 6 (A3 item 6, A6 item 4). Lab 1's content hash needs its outcome column, so in fake modes its recorded
    file sha256 is checked instead, as lab 3 did ("fake-mode content-hash recompute of lab 1's table: skipped")."""
    ok = True
    if real:
        h = b3.content_hash(pd.read_parquet(b3.BASE))
        say(f"- lab 1 table content sha256 {h}: {'PASS' if h == l3.BASE_CONTENT else 'FAIL'}")
        ok &= h == l3.BASE_CONTENT
    else:
        h = b3.sha(b3.BASE)
        say(f"- lab 1 table: file sha256 {h} {'matches' if h == b3.BASE_FILE_SHA else 'DIFFERS from'} the recorded file; content-hash "
            "recompute skipped in fake mode (it needs the outcome column; as lab 3)")
        ok &= h == b3.BASE_FILE_SHA
    for name, path, want in (("lab 3", b3.OUT_PARQUET, l3.H3_CONTENT), ("lab 2", b2.OUT_PARQUET, LAB2_CONTENT)):
        h = b3.content_hash(pd.read_parquet(path))
        say(f"- {name} table content sha256 {h}: {'PASS' if h == want else 'FAIL'}")
        ok &= h == want
    return ok


# ---------------------------------------------------------------- data and outcomes
def load(real: bool) -> pd.DataFrame:
    """Lab 3's load (lab 1's guarded table + lab 3 counts), plus the lab 2 counts on (game_pk, id)."""
    X = l3.load(real)
    L2 = pd.read_parquet(b2.OUT_PARQUET).drop(columns=["date", "season", "opp_sp"])
    assert "h" not in L2
    X = X.merge(L2, on=["game_pk", "id"], how="left", validate="one_to_one", suffixes=("", "_lab2"))
    assert X.L.notna().all() and len(X) == 174348
    assert [c for c in X.columns if c.endswith("_lab2")] == ["c_hand_lab2"]
    assert ((X.c_hand == X.c_hand_lab2) | (X.c_hand.isna() & X.c_hand_lab2.isna())).all(), "c_hand differs"
    return X.drop(columns="c_hand_lab2")


_GEN = {}


def outcomes(X, kind, seed, prep=None):
    """null: lab 1's Bernoulli(0.75). planted / planted-null (A3 item 4): logit p = logit(0.75) + 0.30*[slot <= 4]
    - 0.40*z(a_k) + 0.50*z(a_hab) + c*z(I1), c = 0.10 / 0; the models' own shrunk columns from the train-2022-24 fold."""
    if kind == "null":
        return ba.fake_outcomes(X, "null", seed)
    c = {"planted": 0.10, "planted-null": 0.0}[kind]
    if "z" not in _GEN:
        p = prep if prep is not None else prepare(X, *WF[1])
        z = design(p["F"], p["tr"], ["a_k", "a_hab", "I1"])[:, 1:]  # I1 has 0 on I1_miss rows, as M2's own column
        z[p["F"].I1_miss.to_numpy() == 1, 2] = 0.0  # A3 item 4: rows with I1_miss = 1 get z(I1) = 0
        _GEN["z"] = z
    z = _GEN["z"]
    lp = np.log(3) + 0.30 * X.slot.le(4).to_numpy() - 0.40 * z[:, 0] + 0.50 * z[:, 1] + c * z[:, 2]
    y = (np.random.default_rng(seed).random(len(X)) < 1 / (1 + np.exp(-lp))).astype(float)
    y[X.ab.to_numpy() < 1] = np.nan
    return y, (f"Bernoulli(p), logit p = logit(0.75) + 0.30*[slot 1-4] - 0.40*z(a_k) + 0.50*z(a_hab) + {c:.2f}*z(I1 shrunk); "
               f"shrunk columns and z from the train-2022-24 fold, applied to all rows; I1_miss rows z(I1) = 0; seed {seed}")


# ---------------------------------------------------------------- folds: M1 and M3 (imported), I1..I4 (btslab2_features)
def prepare(X, train, test):
    """Outcome-free fold preparation: lab 3's prepare (M1 design Z, M3 design Z3) plus I1 and the exploratory columns."""
    p = l3.prepare(X, train, test)
    tr, F = p["tr"], p["F"]
    m2, mlog = b2.fold_m(X, tr)
    A = b2.apply_m(X, m2)
    malt, _ = b2.fold_m(X, tr, "_alt")
    Aalt = b2.apply_m(X, malt, "_alt")
    miss = A.I1_miss.to_numpy() == 1
    u, rb, rp = (A[[f"{c}_{f}" for f in F3]].to_numpy() for c in ("u", "r_b", "r_p"))
    F["I1"], F["I1_miss"], F["I1_alt"] = A.I1.to_numpy(), miss.astype(float), Aalt.I1.to_numpy()
    F["I2"] = np.where(miss, np.nan, (u * rb).sum(1))  # A6 item 7: NaN on I1_miss rows, filled by design()
    F["I3"] = np.where(miss, np.nan, (u * rp).sum(1))
    F["I4"] = np.where(miss, np.nan, (u * (rp - 1) * rb).max(1))
    p.update(m2=m2, m_alt=malt, m2log=mlog, A=A, Z2=design(F, tr, M2_COLS), Z4=design(F, tr, M4_COLS))
    return p


def fit(X, y, p, Z):
    return ba.fit_fold(X, y, {**p, "Z": Z}, exploratory=False)  # lab 1's inner CV, refit, CV AUC vs B1, wiring


# ---------------------------------------------------------------- placebo P (A6 item 2)
def placebo_units(X) -> dict:
    """Eligible units over all four seasons: side_s known and the starter has three-family pitches to side s (us_n > 0)."""
    el = X.side_s.notna().to_numpy() & (X.us_n.to_numpy() > 0)
    U = X.loc[el, UNIT + ["season"]].drop_duplicates(UNIT).sort_values(UNIT, kind="mergesort").reset_index(drop=True)
    ru = np.full(len(X), -1)
    ru[el] = pd.MultiIndex.from_frame(U[UNIT]).get_indexer(pd.MultiIndex.from_frame(X.loc[el, UNIT]))
    assert (ru[el] >= 0).all() and not U.duplicated(UNIT).any()
    first = np.full(len(U), -1)
    rows = np.flatnonzero(el)
    _, fi = np.unique(ru[rows], return_index=True)
    first[:] = rows[fi]
    groups = [(k, g.index.to_numpy()) for k, g in U.groupby(["season", "side_s"], sort=True)]  # season asc, then L, R
    return {"U": U, "row_unit": ru, "first": first, "groups": groups}


def donor_map(PU, k) -> np.ndarray:
    """One donor mapping per k (seed 20261150 + k), shuffled within (season, side s), as the check's lines 89-93."""
    rng = np.random.default_rng(PERM_SEED + k)
    donor = np.arange(len(PU["U"]))
    for _, idx in PU["groups"]:
        donor[idx] = idx[rng.permutation(len(idx))]
    return donor


def unit_profile(A, PU) -> np.ndarray:
    """The fold's starter-side bundle per unit; asserted constant within each unit."""
    V = A[PROF].to_numpy(float)
    prof = V[PU["first"]]
    ru = PU["row_unit"]
    k = ru >= 0
    same = (V[k] == prof[ru[k]]) | (np.isnan(V[k]) & np.isnan(prof[ru[k]]))
    assert same.all(), "starter-side profile not constant within a unit"
    return prof


def placebo_I1(X, A, PU, prof, donor) -> np.ndarray:
    """Each recipient row: its own r_b, R_b and L_f with the donor unit's u(.|s), r_p(.|s), R_p(s); I1_miss rows keep 0."""
    keep = A.I1_miss.to_numpy() == 0
    ru = PU["row_unit"]
    assert (ru[keep] >= 0).all()
    D = prof[donor[np.maximum(ru, 0)]]
    Lf = X[[f"L_{f}" for f in F3]].to_numpy(float)
    rb = A[[f"r_b_{f}" for f in F3]].to_numpy(float)
    return np.where(keep, pc.i1(D[:, 0:3], Lf, rb, D[:, 3:6], A.R_b.to_numpy(float), D[:, 6]), 0.0)


def placebo(X, y, wf, PU, donors, model, seed):
    """M2 placebo (vs M1) or M4 placebo (vs M3): refit at the fold's real C, no inner CV; per-row loss(placebo) - loss(ref)
    averaged over the permutations, date bootstrap (A3 item 7); real-vs-placebo I1 correlation on test rows (A6 item 2)."""
    cols, ref = (M2_COLS, "M1") if model == "M2" else (M4_COLS, "M3")
    j = 1 + cols.index("I1")
    dsum, rows, stats_k, corr = np.zeros(len(X)), np.zeros(len(X), bool), [], []
    for k, donor in enumerate(donors):
        dk = np.full(len(X), np.nan)
        for f in wf:
            p, te, ok = f["prep"], f["prep"]["te"], f[model]["ok"]
            I1p = placebo_I1(X, p["A"], PU, f["prof"], donor)
            Zp = p["Z2" if model == "M2" else "Z4"].copy()
            Zp[:, j] = design(pd.DataFrame({"I1": I1p}), p["tr"], ["I1"])[:, 1]  # design() is column-wise
            pp = pred(Zp, fit_l2(Zp[ok], y[ok], f[model]["C"]))
            kk = te & ~np.isnan(y)
            dk[kk] = ll(pp[kk], y[kk]) - ll(f[ref]["p_m1"][kk], y[kk])
            rows |= kk
            sel = te & (p["A"].I1_miss.to_numpy() == 0)
            corr.append({"fold test": p["test"], "k": k, "corr": float(np.corrcoef(p["F"].I1.to_numpy()[sel], I1p[sel])[0, 1])})
        dsum[rows] += dk[rows]
        stats_k.append(np.nanmean(dk))
    est, c = l3.ll_stat(X.date.to_numpy()[rows], dsum[rows] / len(donors), seed)
    return {"est": est, "ci": c, "stop": c[1] < 0, "perm_stats": np.array(stats_k), "corr": pd.DataFrame(corr)}


# ---------------------------------------------------------------- statistics
def t_rule(X, y, pairs, seed):
    """Mean over test rows of loss(new) - loss(old), paired by row, date bootstrap; Confirmed if the interval is below 0."""
    d, rows = np.full(len(X), np.nan), np.zeros(len(X), bool)
    for te, p_new, p_old in pairs:
        k = te & ~np.isnan(y)
        d[k] = ll(p_new[k], y[k]) - ll(p_old[k], y[k])
        rows |= k
    est, c = l3.ll_stat(X.date.to_numpy()[rows], d[rows], seed)
    return {"est": est, "ci": c, "confirmed": c[1] < 0, "d": d, "rows": rows}


def fmt(e, c, d=5):
    return l3.fmt_ci(e, c, d)


def sd_ratio(tf, to):
    """A6 item 8: per resample, sqrt(family between-unit variance) / sqrt(overall); 0 when the family variance is not
    positive (counted); undefined when the overall variance is not positive (left out, counted)."""
    tf, to = np.atleast_1d(np.asarray(tf, float)), np.atleast_1d(np.asarray(to, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(tf > 0, np.sqrt(np.maximum(tf, 0)), 0.0) / np.sqrt(to)
    r = np.where(to > 0, r, np.nan)
    return r, int((tf <= 0).sum()), int((to <= 0).sum())


def qualifying(n, mu, N, min_n):
    """The estimator's c (1 - n/N, or 1 without N) and its qualifying rows (btslab2_features.estimate_m_v lines 204-205)."""
    c = np.ones(n.shape) if N is None else 1 - n / np.where(N > 0, N, np.nan)
    return c, (n >= min_n) & np.isfinite(mu) & np.isfinite(c)


def kappa2(n, c):
    """sum(n c^2) / sum(n) over a cell's qualifying rows: tau^2 * kappa^2 is the variance of the family's deviation from the
    unit's overall rate, (family - overall) = c (family - rest). Unscored companion only."""
    return float((n * c ** 2).sum() / n.sum())


def tau2(x, n, mu, xx, N, W, min_n):
    """Between-unit variance of bf.estimate_m's moment line with w = the cell's pooled within-unit variance (A5 item 3).
    W None: the point estimate, in estimate_m_v's own expressions; else weight rows W (resample counts), one value per row."""
    c, k = qualifying(n, mu, N, min_n)
    x, n, mu, xx, c = x[k], n[k], mu[k], xx[k], c[k]
    if W is None:
        v = float((xx - x ** 2 / n).sum() / (n - 1).sum())  # as btslab2_features.est()
        w = np.full(n.shape, v)
        return np.array([((n * (x / n - mu) ** 2).sum() - (w * c).sum()) / (n * c ** 2).sum()]), np.array([v])
    W = W[:, k]
    v = (W @ (xx - x ** 2 / n)) / (W @ (n - 1))
    t = (W @ (n * (x / n - mu) ** 2) - v * (W @ c)) / (W @ (n * c ** 2))
    return t, v


def cell_min_n(n, mu, N):
    """A3 item 13a's fallback, applied per T5 cell: 100; if no row reaches it, 20; if none, no estimate."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.ones(n.shape) if N is None else 1 - n / np.where(N > 0, N, np.nan)
    ok = np.isfinite(mu) & np.isfinite(c)
    return b2.MIN_N if (ok & (n >= b2.MIN_N)).any() else b2.MIN_N_FALLBACK if (ok & (n >= b2.MIN_N_FALLBACK)).any() else None


def t5(X, seed):
    """T5 (plan Section 8; A5 item 8; A6 item 8): walk-forward 2022-24 training unit-season rows (A3 amendment 8)."""
    tr = X.season.isin(WF[1][0]).to_numpy()
    bat, _, sps = b2.units(X[tr])
    f = lambda d, c: d[c].to_numpy(float)
    specs = []  # (kind, cell, rows, player ids, x, n, xx, mu, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        specs.append(("batter", "overall", bat, f(bat, "b_x"), f(bat, "b_n"), f(bat, "b_xx"), f(bat, "L"), None))
        raw = f(bat, "b_x") / f(bat, "b_n")
        for ff in F3:
            specs.append(("batter", ff, bat, f(bat, f"b_x_{ff}"), f(bat, f"b_n_{ff}"), f(bat, f"b_xx_{ff}"),
                          raw * f(bat, f"L_{ff}") / f(bat, "L"), f(bat, "b_n")))
        for s in ("L", "R"):
            d = sps[sps.side_s.eq(s)]
            specs.append(("pitcher", f"overall/{s}", d, f(d, "p_x"), f(d, "p_n"), f(d, "p_xx"), f(d, "L"), None))
            raw = f(d, "p_x") / f(d, "p_n")
            for ff in F3:
                specs.append(("pitcher", f"{ff}/{s}", d, f(d, f"p_x_{ff}"), f(d, f"p_n_{ff}"), f(d, f"p_xx_{ff}"),
                              raw * f(d, f"L_{ff}") / f(d, "L"), f(d, "p_n")))
    players = {"batter": np.unique(bat.id.to_numpy()), "pitcher": np.unique(sps.opp_sp.to_numpy())}
    Wp = {kind: ba.boot_counts(len(ids), seed) for kind, ids in players.items()}  # same player resamples for every cell
    res = {}
    for kind, cell, d, x, n, xx, mu, N in specs:
        pid = np.searchsorted(players[kind], (d.id if kind == "batter" else d.opp_sp).to_numpy())
        mn = cell_min_n(n, mu, N)
        if mn is None:
            res[(kind, cell)] = None
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            t0, v0 = tau2(x, n, mu, xx, N, None, mn)
            tb, _ = tau2(x, n, mu, xx, N, Wp[kind][:, pid].astype(float), mn)
            m_eq = b2.estimate_m_v(x, n, mu, np.full(n.shape, v0[0]), min_n=mn, N=N)
        m_t = v0[0] / t0[0] if t0[0] > 0 else np.inf
        assert (np.isinf(m_t) and np.isinf(m_eq)) or abs(m_t - m_eq) <= 1e-12 * abs(m_eq), (cell, m_t, m_eq)
        c, kq = qualifying(n, mu, N, mn)
        res[(kind, cell)] = {"units": len(d), "min_n": mn, "rows": int(kq.sum()), "v": float(v0[0]), "tau2": float(t0[0]),
                             "boot": tb, "kappa2": kappa2(n[kq], c[kq])}
    rows = []
    for (kind, cell), r in res.items():
        if cell.startswith("overall") or r is None:
            continue
        o = res[(kind, "overall" if kind == "batter" else f"overall/{cell.split('/')[1]}")]
        pr, _, _ = sd_ratio(r["tau2"], o["tau2"])
        br, nf, no = sd_ratio(r["boot"], o["boot"])
        r.update(ratio=float(pr[0]), ratio_ci=ci(br), n_fam_nonpos=nf, n_den_nonpos=no, kind=kind,
                 dev_ratio=float(pr[0]) * np.sqrt(r["kappa2"]))  # unscored companion, point only
        rows.append(r | {"cell": cell})
    return res, rows


# ---------------------------------------------------------------- expectations (plan Section 9, A3 item 3; scoring A6 item 11)
def score_expectations(e) -> list:
    """e: t1, t1b (est, ci) of loss(new) - loss(old); coef (M2 I1 per SD); t3 (M2 - M1 top-1 est, ci); t3b2 (M2 - B2 ci);
    placebo ci; t6e (M1 + I1..I4 vs M2 ci); t5 rows (kind, tau2, ratio). Returns (label, met or None, numbers)."""
    inc0 = lambda c: c[0] <= 0 <= c[1]
    g1, g1b = -e["t1"][0], -e["t1b"][0]
    t5 = e["t5"]
    pit = [r["ratio"] for r in t5 if r["kind"] == "pitcher"]
    bat = [r["ratio"] for r in t5 if r["kind"] == "batter"]
    f_ok = (all(r["tau2"] > 0 for r in t5) and all(np.isfinite(r["ratio"]) and r["ratio"] < 0.5 for r in t5)
            and np.mean(pit) > np.mean(bat))
    return [("(a) loss(M1) − loss(M2) between 0 and 0.0004", 0 <= g1 <= 0.0004,
             f"{g1:+.6f}; interval excludes 0: {not inc0(e['t1'][1])} (the 35% chance is not scored, A6 item 11)"),
            ("(b) I1 coefficient positive and under 0.03 per SD", 0 < e["coef"] < 0.03, f"{e['coef']:+.4f}"),
            ("(c) top-1 M2 − M1 within ±0.02 and its 95% interval includes 0", abs(e["t3"][0]) <= 0.02 and inc0(e["t3"][1]),
             fmt(*e["t3"], 4)),
            ("(d) placebo (T4, placebo P) interval includes 0", inc0(e["placebo"]), f"({e['placebo'][0]:+.6f} to {e['placebo'][1]:+.6f})"),
            ("(e) M1 + I1..I4 vs M2 log-loss interval includes 0", inc0(e["t6e"]), f"({e['t6e'][0]:+.6f} to {e['t6e'][1]:+.6f})"),
            ("(f) every T5 cell variance > 0, every SD ratio < 0.5, mean pitcher ratio > mean batter ratio", bool(f_ok),
             f"variances {', '.join(f'{r['tau2']:.3g}' for r in t5)}; ratios {', '.join(f'{r['ratio']:.3f}' for r in t5)}; "
             f"mean pitcher {np.mean(pit):.3f}, batter {np.mean(bat):.3f}"),
            ("(g) M2 − B2 top-1 not confirmed by the T1-style rule", not e["t3b2"][0] > 0, f"({e['t3b2'][0]:+.4f} to {e['t3b2'][1]:+.4f})"),
            ("(h) loss(M3) − loss(M4) between 0 and 0.0003", 0 <= g1b <= 0.0003,
             f"{g1b:+.6f}; interval excludes 0: {not inc0(e['t1b'][1])} (the 25% chance is not scored, A6 item 11)"),
            ("(i) loss(M3) − loss(M4) ≤ loss(M1) − loss(M2)", g1b <= g1, f"{g1b:+.6f} vs {g1:+.6f}")]


# ---------------------------------------------------------------- reproduction (stop rule 7; --real only)
def repro_m3_values(X, y, fits, mask) -> dict:
    """The M3 numbers stop rule 7 compares, formatted at lab 3's printed precision."""
    folds = {}
    for (d, tr, te) in REPRO_M3_FOLDS:
        f = fits[(d, tr, te)]["M3"]
        folds[(d, tr, te)] = (f"{f['C']:.4g}" + (" (grid edge)" if f["C"] in (ba.CS[0], ba.CS[-1]) else ""),
                              f"{min(f['cv_ll']):.5f}", f"{f['auc_m1_cv']:.4f}", f"{f['auc_b1_cv']:.4f}", f"{f['wiring']:+.4f}")
    wf = [fits[("WF", tuple(tr), te)] for tr, te in WF]
    t = l3.gain_rule(X, y, [(f["prep"]["te"], f["M1"]["p_m1"]) for f in wf], [f["M3"]["p_m1"] for f in wf], LAB3_T1_SEED)
    k = t["rows"]
    pm1, pm3 = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for f in wf:
        te = f["prep"]["te"]
        pm1[te], pm3[te] = f["M1"]["p_m1"][te], f["M3"]["p_m1"][te]
    m3p = l3.picks(X, y, [{"prep": f["prep"], "p": f["M3"]["p_m1"]} for f in wf], "p", mask)
    got = {"test": f"batter-game log loss M1 {ll(pm1[k], y[k]).mean():.5f}, M3 {ll(pm3[k], y[k]).mean():.5f}; "
                   f"AUC M1 {ba.stats.auc(pm1[k], y[k]):.4f}, M3 {ba.stats.auc(pm3[k], y[k]):.4f}",
           "lab 3 T1": f"pooled 2024–25: {fmt(t['est'], t['ci'])}, rows {int(k.sum())}",
           "top-1 M3": f"{ba.rate(m3p.y.to_numpy(float))[0]:.3f}"}
    return {"folds": folds, "wf": got}


def repro_m3_compare(got, exp_folds=REPRO_M3_FOLDS, exp_wf=REPRO_M3_WF) -> list:
    """Exact string comparison with lab 3's printed values (A3 item 6); a missing or malformed value is a mismatch with
    a plain message, never an exception."""
    bad = []
    for key, exp in exp_folds.items():
        d, _, te = key
        g = got.get("folds", {}).get(key)
        if not (isinstance(exp, tuple) and len(exp) == 5 and all(isinstance(s, str) for s in exp)):
            bad.append(f"{d} test {te}: expected value missing or unparsable ({exp!r}); stop rule 7 cannot pass")
        elif g is None:
            bad.append(f"{d} test {te}: no computed value; stop rule 7 cannot pass")
        elif g != exp:
            bad.append(f"{d} test {te}: {g} vs {exp}")
    for kk, v in exp_wf.items():
        g = got.get("wf", {}).get(kk)
        if not isinstance(v, str):
            bad.append(f"{kk}: expected value missing or unparsable ({v!r}); stop rule 7 cannot pass")
        elif g is None:
            bad.append(f"{kk}: no computed value; stop rule 7 cannot pass")
        elif g != v:
            bad.append(f"{kk}: {g} vs {v}")
    return bad


def parse_lab3_m3(text) -> dict:
    """Unit checks only: lab 3's logged M3 values (analysis/out/lab3_results_log.txt) in repro_m3_compare's form; a line that
    is missing or does not parse gives None."""
    text = text.replace("\r\n", "\n")
    folds = {}
    for d, tr, te in REPRO_M3_FOLDS:
        m = re.search(rf"^\| {d} \| {'-'.join(map(str, tr))} \| {te} \| M3 \| (.+?) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$", text, re.M)
        folds[(d, tr, te)] = m.groups() if m else None
    pat = {"test": r"^batter-game log loss M1 \S+, M3 \S+; AUC M1 \S+, M3 \S+$", "lab 3 T1": r"^pooled 2024–25: .+, rows \d+$",
           "top-1 M3": r"^M3 (\d\.\d{3}) \("}
    wf = {}
    for k, p in pat.items():
        m = re.search(p, text, re.M)
        wf[k] = (m.group(1) if m.groups() else m.group(0)) if m else None
    return {"folds": folds, "wf": wf}


def repro_m3(X, y, fits, mask) -> bool:
    bad = repro_m3_compare(repro_m3_values(X, y, fits, mask))
    say(f"- Rule 7, M3 reproduction (6 folds; walk-forward test numbers; lab 3 T1 seed {LAB3_T1_SEED}): "
        f"{'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return not bad


# ---------------------------------------------------------------- the pipeline (one outcome vector)
def write_started(path: Path):
    """--real: on disk (flushed and fsynced) immediately before the first fit that uses the real outcome."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(f"lab 2 --real started\nUTC {now:%Y-%m-%d %H:%M:%S %z}\nlocal {now.astimezone():%Y-%m-%d %H:%M:%S %z}\n"
                 f"script {SCRIPT.name} on-disk sha256 {sha(SCRIPT)}\n")
        fh.flush()
        os.fsync(fh.fileno())


def build_fits(X, y, started=None):
    """M1, M2, M3 on lab 1's six folds and M4 on the two walk-forward folds; the shared 2022-24 training set is fitted once.
    started: --real's marker path, written before the first fit."""
    preps, fits = {}, {}
    for d, folds in (("WF", WF), ("LOSO", LOSO)):
        for tr, te in folds:
            key = (tuple(tr), te)
            if key not in preps:
                p = prepare(X, tr, te)
                if started is not None and not started.exists():
                    write_started(started)
                preps[key] = {"prep": p, "M1": fit(X, y, p, p["Z"]), "M3": fit(X, y, p, p["Z3"]), "M2": fit(X, y, p, p["Z2"])}
            if d == "WF" and "M4" not in preps[key]:
                p = preps[key]["prep"]
                preps[key]["M4"] = fit(X, y, p, p["Z4"])
            fits[(d, tuple(tr), te)] = preps[key]
    return fits


def run(X, y, real: bool, log: Path, started=None):
    assert started is not None or not real, "--real must write the started marker"
    t0 = time.perf_counter()
    mask = ba.pool(X)
    done = set()
    say(f"countable rows (AB >= 1): {int((~np.isnan(y)).sum())}; passes (AB = 0): {int(np.isnan(y).sum())}")

    def stop(msg):
        say(f"STOP: {msg}")
        if OPT["ignore_stops"] and not real:
            say("(--ignore-stops, fake outcomes: continuing so the rest of the pipeline is exercised)")
            return
        log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        sys.exit(2)

    fits = build_fits(X, y, started if real else None)
    done |= {2, 4, 5, 6}
    say("\n## Reproduction (stop rule 7: M1 then M3; A3 item 6)")
    if real:
        if not l3.repro_check(X, y, fits, mask):
            stop("M1 does not reproduce lab 1 (stop rule 7); M2 and M4 are not read.")
        if not repro_m3(X, y, fits, mask):
            stop("M3 does not reproduce lab 3 (stop rule 7); M2 and M4 are not read.")
    else:
        say("skipped on fake outcomes (lab 1's numbers are real-outcome numbers); logged as skipped (A6 item 5)")
    done |= {3}
    say("\n## Folds (inner CV: 5 contiguous date blocks, 12 C values 1e-3..1e2, mean log loss)")
    say(md_table(pd.DataFrame([{"design": d, "train": "-".join(map(str, tr)), "test": te, "model": mn,
                                "C": f"{f[mn]['C']:.4g}" + (" (grid edge)" if f[mn]["C"] in (ba.CS[0], ba.CS[-1]) else ""),
                                "inner-CV log loss": f"{min(f[mn]['cv_ll']):.5f}", "AUC (CV)": f"{f[mn]['auc_m1_cv']:.4f}",
                                "AUC B1 (CV)": f"{f[mn]['auc_b1_cv']:.4f}", "wiring 1-4 minus 9": f"{f[mn]['wiring']:+.4f}"}
                               for (d, tr, te), f in fits.items() for mn in ("M1", "M2", "M3", "M4") if mn in f])))
    wf = [fits[("WF", tuple(tr), te)] for tr, te in WF]
    for f in wf:
        say(f"lab 2 prior strengths m, test {f['prep']['test']}: " + ", ".join(f"{k}={v:.4g}" for k, v in f["prep"]["m2"].items())
            + "; alternative grouping: " + ", ".join(f"{k}={v:.4g}" for k, v in f["prep"]["m_alt"].items()))

    say("\n## Stop rules")
    say("- Rule 3, usage shares sum to 1 within 1e-9 on every row with history: PASS (asserted in apply_m for every fold)")
    s5a = all(f[mn]["auc_m1_cv"] > f[mn]["auc_b1_cv"] for f in wf for mn in ("M1", "M2", "M3", "M4"))
    s5b = all(f[mn]["wiring"] > 0 for f in wf for mn in ("M1", "M2", "M3", "M4"))
    say(f"- Rule 5a, model beats B1 on batter-game AUC in the training inner CV (M1-M4, each walk-forward fold): {'PASS' if s5a else 'FAIL'}")
    say(f"- Rule 5b, training mean P(hit) slots 1-4 minus slot 9 > 0 (M1-M4, each walk-forward fold): {'PASS' if s5b else 'FAIL'}")
    done |= {7}
    if not (s5a and s5b):
        stop("an inherited stop rule failed (rule 5).")

    # ---- placebos before any other result (stop rule 2; A3 item 6 order)
    PU = placebo_units(X)
    for f in wf:
        f["prof"] = unit_profile(f["prep"]["A"], PU)
    donors = [donor_map(PU, k) for k in range(OPT["nperm"])]
    moved = min(float((dn != np.arange(len(dn))).mean()) for dn in donors)
    pls = {}
    for model, sd in (("M2", SEED["T4"]), ("M4", SEED["M4 placebo"])):
        ref = "M1" if model == "M2" else "M3"  # the same mapping per k for both models and both folds
        say(f"\n## {'T4' if model == 'M2' else 'M4 placebo'} — placebo P (A6 item 2): {OPT['nperm']} permutations (seeds {PERM_SEED} + k), "
            f"{len(PU['U']):,} eligible units over 2022-2025, shuffled within (season, side s); {model} refitted at each fold's "
            f"real C; loss({model} placebo) − loss({ref}); date bootstrap seed {sd}")
        pl = placebo(X, y, wf, PU, donors, model, sd)
        pls[model] = pl
        q = np.quantile(pl["perm_stats"], [.025, .975])
        say(f"placebo statistic (per-row mean over permutations) {fmt(pl['est'], pl['ci'], 6)}; per-permutation statistics "
            f"2.5–97.5%: {q[0]:+.6f} to {q[1]:+.6f}; least share of units moved {moved:.4f}")
        say(f"- Stop rule 2, {model} placebo interval entirely below 0: {'YES - STOP' if pl['stop'] else 'no'}")
        if model == "M2":
            C = pl["corr"]
            say("Real vs placebo-P I1, Pearson correlation on test-season rows with I1_miss = 0 (descriptive, A6 item 2):")
            say(md_table(C.groupby("fold test")["corr"].agg(["mean", "min", "max"]).round(4).reset_index()))
            say(f"per k: " + "; ".join(f"{r['fold test']} k{r['k']} {r['corr']:.4f}" for _, r in C.iterrows()))
            if C["corr"].max() > 0.30:
                say("Limitation (A6 item 2): the maximum correlation exceeds 0.30; nothing else changes.")
        if pl["stop"]:
            stop(f"the {model} placebo is Confirmed (stop rule 2); no other result is read.")
    done |= {8, 9}

    # ---- T1, T1b
    pm = {mn: np.full(len(X), np.nan) for mn in ("M1", "M2", "M3", "M4")}
    for f in wf:
        te = f["prep"]["te"]
        for mn in pm:
            pm[mn][te] = f[mn]["p_m1"][te]
    t1 = t_rule(X, y, [(f["prep"]["te"], f["M2"]["p_m1"], f["M1"]["p_m1"]) for f in wf], SEED["T1"])
    t1b = t_rule(X, y, [(f["prep"]["te"], f["M4"]["p_m1"], f["M3"]["p_m1"]) for f in wf], SEED["T1b"])
    k = t1["rows"]
    say(f"\n## T1 — PRIMARY: walk-forward log loss(M2) − log loss(M1), paired by row (date bootstrap seed {SEED['T1']})")
    say(f"pooled 2024–25: {fmt(t1['est'], t1['ci'], 6)}, rows {int(k.sum())}")
    say("batter-game log loss " + ", ".join(f"{mn} {ll(pm[mn][k], y[k]).mean():.5f}" for mn in pm) + "; AUC "
        + ", ".join(f"{mn} {ba.stats.auc(pm[mn][k], y[k]):.4f}" for mn in pm) + f", B2 {ba.stats.auc(X.base_b2.to_numpy()[k], y[k]):.4f}")
    say(f"**T1 rule: {'Confirmed' if t1['confirmed'] else 'not confirmed'}** (Confirmed only if the 95% interval lies entirely below 0)")
    say(f"\n## T1b — log loss(M4) − log loss(M3), paired by row (date bootstrap seed {SEED['T1b']}; A3 item 3)")
    say(f"pooled 2024–25: {fmt(t1b['est'], t1b['ci'], 6)}, rows {int(t1b['rows'].sum())}")
    say(f"**T1b rule: {'Confirmed' if t1b['confirmed'] else 'not confirmed'}**; inclusion rule (A3 item 3): I1 "
        f"{'enters' if t1['confirmed'] and t1b['confirmed'] else 'stays out of'} the combined 2026 model")
    done |= {10, 11}

    # ---- T2
    lo = [fits[("LOSO", tuple(tr), te)] for tr, te in LOSO]
    t2 = t_rule(X, y, [(f["prep"]["te"], f["M2"]["p_m1"], f["M1"]["p_m1"]) for f in lo], SEED["T2"])
    say(f"\n## T2 — LOSO log loss(M2) − log loss(M1) (secondary, seed {SEED['T2']}): {fmt(t2['est'], t2['ci'], 6)}, rows "
        f"{int(t2['rows'].sum())}; T1's rule applied: {'Confirmed' if t2['confirmed'] else 'not confirmed'} (not confirmatory)")
    done |= {12}

    # ---- T3, T3b (no recalibration: A6 item 12)
    pk = {mn: l3.picks(X, y, [{"prep": f["prep"], "p": f[mn]["p_m1"]} for f in wf], "p", mask) for mn in ("M1", "M2", "M3", "M4")}
    pk["B2"] = l3.picks(X, y, [{"prep": f["prep"], "p": X.base_b2.to_numpy()} for f in wf], "p", mask)
    e21, c21, n21 = ba.paired(pk["M2"].y, pk["M1"].y, SEED["T3"])
    e2b, c2b, n2b = ba.paired(pk["M2"].y, pk["B2"].y, SEED["T3"])
    e43, c43, n43 = ba.paired(pk["M4"].y, pk["M3"].y, SEED["T3b"])
    say(f"\n## T3 — top-1 daily success (descriptive; walk-forward; seed {SEED['T3']}; no recalibration, A6 item 12)")
    say(", ".join(f"{mn} {ba.wil(pk[mn].y)}" for mn in ("M1", "M2", "B2")) + f"; M2 − M1 {fmt(e21, c21, 4)} ({n21} dates); "
        f"M2 − B2 {fmt(e2b, c2b, 4)} ({n2b} dates)")
    say(f"\n## T3b — top-1 M4 − M3 (descriptive; seed {SEED['T3b']}): M4 {ba.wil(pk['M4'].y)}, M3 {ba.wil(pk['M3'].y)}; "
        f"{fmt(e43, c43, 4)} ({n43} dates)")
    done |= {13, 14}

    # ---- T5
    _, t5rows = t5(X, SEED["T5"])
    say(f"\n## T5 — DIAGNOSTIC: between-unit variance of family-specific deviations (walk-forward 2022–24 training unit-season "
        f"rows; pooled within-unit variance per cell, A5 item 8; player resamples {ba.NBOOT}, seed {SEED['T5']}; A6 item 8)")
    say(md_table(pd.DataFrame([{"units": r["kind"], "cell": r["cell"], "unit-seasons": r["units"], "min_n": r["min_n"],
                                "qualifying rows": r["rows"], "within-unit variance": f"{r['v']:.5f}",
                                "tau^2, between-unit variance (unclipped)": f"{r['tau2']:+.3e}",
                                "95% interval": f"{np.quantile(r['boot'], .025):+.3e} to {np.quantile(r['boot'], .975):+.3e}",
                                "kappa^2": f"{r['kappa2']:.4f}",
                                "registered ratio": f"{r['ratio']:.3f}", "registered ratio 95%": f"{r['ratio_ci'][0]:.3f} to {r['ratio_ci'][1]:.3f}",
                                "resamples family var <= 0 (ratio 0)": r["n_fam_nonpos"],
                                "resamples overall var <= 0 (left out)": r["n_den_nonpos"],
                                "deviation-from-overall ratio (UNSCORED)": f"{r['dev_ratio']:.3f}"} for r in t5rows])))
    say("registered ratio = SD of (family minus rest-of-arsenal) differences / SD of overall skill (estimate_m's tau^2, A6 item 8); "
        "expectation (f) is scored on it alone.")
    say("kappa^2 = sum(n c^2) / sum(n) over the cell's qualifying rows, c = the estimator's 1 - n/N. deviation-from-overall ratio "
        "= registered point ratio * kappa: UNSCORED, point estimate only, feeds nothing.")
    say("Stated plainly (plan Section 8 T5): if the shares are near zero, there is nothing for the fit score to find.")
    done |= {15}

    # ---- T6 (exploratory; at M2's C, A6 item 7; also M1 + I1..I4 vs M2, A6 item 9)
    say(f"\n## T6 — EXPLORATORY comparisons (walk-forward; fitted at M2's C; date bootstrap seed {SEED['T6']}); "
        "Δ = loss(model) − loss(reference)")
    rows, t6e = [], None
    for name, cols in (("M1 + I1..I4", X1_COLS), ("M1 + I1 (alternative grouping)", XA_COLS)):
        pp = np.full(len(X), np.nan)
        for f in wf:
            Zx = design(f["prep"]["F"], f["prep"]["tr"], cols)
            ok = f["M2"]["ok"]
            pp[f["prep"]["te"]] = pred(Zx, fit_l2(Zx[ok], y[ok], f["M2"]["C"]))[f["prep"]["te"]]
        tp = l3.picks(X, y, [{"prep": f["prep"], "p": pp} for f in wf], "p", mask)
        for ref in (["M1", "M2"] if name == "M1 + I1..I4" else ["M1"]):
            e, c = l3.ll_stat(X.date.to_numpy()[k], ll(pp[k], y[k]) - ll(pm[ref][k], y[k]), SEED["T6"])
            et, ct, _ = ba.paired(tp.y, pk[ref].y, SEED["T6"])
            rows.append({"model": name, "vs": ref, "Δ log loss": fmt(e, c, 6), "Δ top-1 (model − ref)": fmt(et, ct, 4)})
            if ref == "M2":
                t6e = c
    say(md_table(pd.DataFrame(rows)))
    done |= {16}

    # ---- coefficients: M2 and M4, fit 2022-24, the same date resamples (plan Section 8; A3 item 3, P18)
    f = wf[1]
    ok = f["M2"]["ok"]
    _, di = np.unique(X.date.to_numpy()[ok], return_inverse=True)
    Wd = ba.boot_counts(di.max() + 1, SEED["coef"])
    say(f"\n## Standardised I1 coefficients, fit on 2022–24 (date bootstrap {len(Wd)} resamples, seed {SEED['coef']}, warm start)")
    coef = {}
    for mn, Z, cols in (("M2", f["prep"]["Z2"], M2_COLS), ("M4", f["prep"]["Z4"], M4_COLS)):
        assert (f[mn]["ok"] == ok).all()
        B = np.array([fit_l2(Z[ok], y[ok], f[mn]["C"], w[di].astype(float), f[mn]["beta"]) for w in Wd])
        for c in ("I1", "I1_miss"):
            j = 1 + cols.index(c)
            coef[(mn, c)] = (f[mn]["beta"][j], *np.quantile(B[:, j], [.025, .975]))
            say(f"{mn} {c}: {coef[(mn, c)][0]:+.4f} ({coef[(mn, c)][1]:+.4f} to {coef[(mn, c)][2]:+.4f}) at C {f[mn]['C']:.4g}")
    done |= {17}

    # ---- expectations
    exp = score_expectations({"t1": (t1["est"], t1["ci"]), "t1b": (t1b["est"], t1b["ci"]), "coef": coef[("M2", "I1")][0],
                              "t3": (e21, c21), "t3b2": c2b, "placebo": pls["M2"]["ci"], "t6e": t6e, "t5": t5rows})
    say("\n## Claude's pre-registered expectations (met / not met; no decision rule; scoring A6 item 11)")
    say(md_table(pd.DataFrame([{"expectation": a, "result": "met" if b else "not met", "numbers": c} for a, b, c in exp])))
    done |= {18}
    say(f"\nruntime {time.perf_counter() - t0:.0f} s")
    return {"t1": t1, "t1b": t1b, "placebo": pls, "coef": coef, "done": done}


# ---------------------------------------------------------------- planted self-tests (A3 item 4; A6 items 2-3)
def selftest_data(kind, seed, r) -> str:
    c = r["coef"][("M2", "I1")][0]
    say(f"\nM2 I1 coefficient (per SD, fit 2022–24): {c:+.6f}; T1 {'Confirmed' if r['t1']['confirmed'] else 'not confirmed'}; "
        f"T1b {'Confirmed' if r['t1b']['confirmed'] else 'not confirmed'}; M2 placebo "
        f"{'Confirmed' if r['placebo']['M2']['stop'] else 'not Confirmed'}; M4 placebo {'Confirmed' if r['placebo']['M4']['stop'] else 'not Confirmed'}")
    return (f"SELFTEST-DATA kind={kind} seed={seed} i1={c:+.6f} t1_confirmed={r['t1']['confirmed']} t1b_confirmed={r['t1b']['confirmed']} "
            f"placebo_stop={r['placebo']['M2']['stop']} placebo4_stop={r['placebo']['M4']['stop']}")


def parse_data(text) -> dict:
    m = re.findall(r"SELFTEST-DATA kind=(\S+) seed=(\d+) i1=(\S+) t1_confirmed=(\S+) t1b_confirmed=(\S+) placebo_stop=(\S+) "
                   r"placebo4_stop=(\S+)", text)
    assert len(m) == 1, f"expected one SELFTEST-DATA line, found {len(m)}"
    k, s, c, t, tb, p, p4 = m[0]
    return {"kind": k, "seed": int(s), "i1": float(c), "t1": t == "True", "t1b": tb == "True", "stop": p == "True", "stop4": p4 == "True"}


def verdict(d) -> bool:
    """A3 item 4 pass rules: planted I1: coefficient 0.05..0.15, T1 and T1b Confirmed, both placebos not Confirmed;
    planted null-I1: T1 and T1b not Confirmed. A6 item 3's predictions are scored separately (no rule)."""
    if d["kind"] == "planted":
        conds = [("M2 I1 coefficient in 0.05..0.15", 0.05 <= d["i1"] <= 0.15), ("T1 Confirmed", d["t1"]), ("T1b Confirmed", d["t1b"]),
                 ("M2 placebo not Confirmed", not d["stop"]), ("M4 placebo not Confirmed", not d["stop4"])]
        pred_ = [("both placebos not Confirmed", not d["stop"] and not d["stop4"])]
    else:
        conds = [("T1 not Confirmed", not d["t1"]), ("T1b not Confirmed", not d["t1b"])]
        pred_ = [("neither T1 nor T1b nor either placebo Confirmed", not (d["t1"] or d["t1b"] or d["stop"] or d["stop4"]))]
    for n, ok in conds:
        say(f"- {n}: {'yes' if ok else 'NO'}")
    for n, ok in pred_:
        say(f"- prediction (A6 item 3; no rule), {n}: {'met' if ok else 'not met'}")
    return all(ok for _, ok in conds)


# ---------------------------------------------------------------- null study (fake outcomes only)
_W = {}


def _null_init(nperm, nboot):
    ba.NBOOT = nboot
    X = load(False)  # never the real hit column
    PU = placebo_units(X)
    _W.update(X=X, preps=[prepare(X, tr, te) for tr, te in WF], PU=PU, donors=[donor_map(PU, k) for k in range(nperm)])


def _null_seed(seed):
    X, PU = _W["X"], _W["PU"]
    y, _ = outcomes(X, "null", seed)
    wf = []
    for p in _W["preps"]:
        wf.append({"prep": p, "M1": fit(X, y, p, p["Z"]), "M2": fit(X, y, p, p["Z2"]), "M3": fit(X, y, p, p["Z3"]),
                   "M4": fit(X, y, p, p["Z4"]), "prof": unit_profile(p["A"], PU)})
    t1 = t_rule(X, y, [(f["prep"]["te"], f["M2"]["p_m1"], f["M1"]["p_m1"]) for f in wf], SEED["T1"])
    t1b = t_rule(X, y, [(f["prep"]["te"], f["M4"]["p_m1"], f["M3"]["p_m1"]) for f in wf], SEED["T1b"])
    p2 = placebo(X, y, wf, PU, _W["donors"], "M2", SEED["T4"])
    p4 = placebo(X, y, wf, PU, _W["donors"], "M4", SEED["M4 placebo"])
    return {"seed": seed, "T1": t1["est"], "T1 lo": t1["ci"][0], "T1 hi": t1["ci"][1], "T1 Confirmed": t1["confirmed"],
            "T1b": t1b["est"], "T1b lo": t1b["ci"][0], "T1b hi": t1b["ci"][1], "T1b Confirmed": t1b["confirmed"],
            "M2 placebo": p2["est"], "M2 placebo hi": p2["ci"][1], "M2 placebo stop": p2["stop"],
            "M4 placebo": p4["est"], "M4 placebo hi": p4["ci"][1], "M4 placebo stop": p4["stop"],
            "C M1": wf[0]["M1"]["C"], "C M2": wf[0]["M2"]["C"], "C M3": wf[0]["M3"]["C"], "C M4": wf[0]["M4"]["C"]}


def null_study(seeds, procs, nperm, nboot, csv: Path):
    """Pass (plan Section 12; A3 item 5): false confirmations <= 8% (T1, T1b), false stops <= 8% (M2 and M4 placebos),
    mean D >= -0.25 SD(D) (T1, T1b). Predictions (A3 item 5) scored, no rule."""
    from multiprocessing import Pool
    t0 = time.perf_counter()
    with Pool(procs, initializer=_null_init, initargs=(nperm, nboot)) as pool_:
        R = pd.DataFrame(pool_.map(_null_seed, list(seeds), chunksize=1)).sort_values("seed")
    R.to_csv(csv, index=False)
    say(f"seeds {min(seeds)}-{max(seeds)} ({len(R)}), {procs} processes, {nboot} bootstrap resamples, {nperm} placebo "
        f"permutations per seed; per-seed rows: {csv.name}")
    checks, preds = [], []
    for t in ("T1", "T1b"):
        sd = R[t].std(ddof=1) if len(R) > 1 else np.nan
        checks += [(f"{t} false confirmation <= 8%", R[f"{t} Confirmed"].mean() <= 0.08, f"{R[f'{t} Confirmed'].mean():.3f}"),
                   (f"{t} no favourable bias: mean D >= -0.25 SD(D)", R[t].mean() >= -0.25 * sd, f"mean {R[t].mean():+.7f}, SD {sd:.7f}")]
        preds += [(f"{t} false confirmations 0-3%", R[f"{t} Confirmed"].mean() <= 0.03),
                  (f"{t} mean D between 0 and +0.00002", 0 < R[t].mean() <= 0.00002)]
    for m in ("M2", "M4"):
        checks.append((f"{m} placebo false stop <= 8%", R[f"{m} placebo stop"].mean() <= 0.08, f"{R[f'{m} placebo stop'].mean():.3f}"))
        preds.append((f"{m} placebo false stops 0-3%", R[f"{m} placebo stop"].mean() <= 0.03))
    for n, ok, v in checks:
        say(f"- {n}: {v} -> {'PASS' if ok else 'FAIL'}")
    passed = all(ok for _, ok, _ in checks)
    say("Predictions (A3 item 5; scored, no rule): " + "; ".join(f"{n}: {'met' if ok else 'not met'}" for n, ok in preds)
        + f"; the one-sided check passes: {'met' if all(ok for n, ok, _ in checks if 'bias' in n) else 'not met'}")
    say(f"runtime {time.perf_counter() - t0:.0f} s")
    return passed, R


# ---------------------------------------------------------------- unit checks (run by --smoke before the smoke run)
def unit_checks(X) -> list:
    import builtins
    import tempfile
    res = []
    out_before = set(os.listdir(OUT))

    def chk(name, cond, detail=""):
        res.append((name, bool(cond), detail))
        say(f"- {'PASS' if cond else 'FAIL'}: {name}{'; ' + detail if detail else ''}")

    # placebo P
    PU = placebo_units(X)
    d0, d0b, d1 = donor_map(PU, 0), donor_map(PU, 0), donor_map(PU, 1)
    chk("placebo P deterministic for a given seed (k = 0 twice identical; k = 1 differs)", (d0 == d0b).all() and (d0 != d1).any())
    perm_ok = all(np.array_equal(np.sort(d[idx]), idx) for d in (d0, d1) for _, idx in PU["groups"])
    chk("donor mapping is a permutation within each (season, side) group", perm_ok,
        f"{len(PU['groups'])} groups: " + ", ".join(f"{k[0]}/{k[1]} {len(i)}" for k, i in PU["groups"]))
    mv = [float((d != np.arange(len(d))).mean()) for d in (d0, d1)]
    chk("at least 99.9% of eligible units get a different donor", min(mv) >= 0.999, f"k=0 {mv[0]:.5f}, k=1 {mv[1]:.5f}; units {len(PU['U']):,}")
    order = [k for k, _ in PU["groups"]]
    rng, man = np.random.default_rng(PERM_SEED), np.arange(len(PU["U"]))
    rev, rng2 = np.arange(len(PU["U"])), np.random.default_rng(PERM_SEED)
    U = PU["U"]
    for s in sorted(U.season.unique()):
        for side in ("L", "R"):
            idx = U.index[(U.season == s) & (U.side_s == side)].to_numpy()
            man[idx] = idx[rng.permutation(len(idx))]
        for side in ("R", "L"):
            idx = U.index[(U.season == s) & (U.side_s == side)].to_numpy()
            rev[idx] = idx[rng2.permutation(len(idx))]
    chk("shuffle group order is season ascending, then L before R", order == sorted(order) and np.array_equal(man, d0)
        and not np.array_equal(rev, d0), f"groups {order}; a manual draw in that order equals the mapping, R-before-L does not")
    p = prepare(X, *WF[0])
    prof = unit_profile(p["A"], PU)
    real = p["F"].I1.to_numpy()
    ident = placebo_I1(X, p["A"], PU, prof, np.arange(len(PU["U"])))
    chk("identity donor mapping reproduces the real I1 exactly", np.array_equal(ident, real))
    plI = placebo_I1(X, p["A"], PU, prof, d0)
    miss = p["A"].I1_miss.to_numpy() == 1
    chk("I1_miss rows keep I1 = 0 and I1_miss = 1", (plI[miss] == 0).all() and (p["F"].I1_miss.to_numpy()[miss] == 1).all(),
        f"{int(miss.sum()):,} I1_miss rows")
    # the check's own procedure (test-season units, a generator per fold) reproduces its log's fold-1 k = 0 numbers
    te = p["te"]
    dck, rng = np.arange(len(U)), np.random.default_rng(PERM_SEED)
    for (s, side), idx in PU["groups"]:
        if s == p["test"]:
            dck[idx] = idx[rng.permutation(len(idx))]
    sel = te & ~miss
    plc = placebo_I1(X, p["A"], PU, prof, dck)
    r_p = float(np.corrcoef(real[sel], plc[sel])[0, 1])
    chk("placebo P code reproduces the placebo check's logged fold 2022-2023 -> 2024, k = 0 correlation 0.1030", f"{r_p:.4f}" == "0.1030",
        f"{r_p:.6f}")
    # estimators (A5)
    tr = X.season.isin(WF[1][0]).to_numpy()
    bat, _, sps = b2.units(X[tr])
    fz = lambda d, c: d[c].to_numpy(float)
    ok_eq = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for x, n, mu, N, xx in ((fz(bat, "b_x"), fz(bat, "b_n"), fz(bat, "L"), None, fz(bat, "b_xx")),
                                (fz(bat, "b_x_br"), fz(bat, "b_n_br"), fz(bat, "b_x") / fz(bat, "b_n") * fz(bat, "L_br") / fz(bat, "L"), fz(bat, "b_n"), fz(bat, "b_xx_br")),
                                (fz(sps, "p_x_os"), fz(sps, "p_n_os"), fz(sps, "p_x") / fz(sps, "p_n") * fz(sps, "L_os") / fz(sps, "L"), fz(sps, "p_n"), fz(sps, "p_xx_os"))):
            t, v = tau2(x, n, mu, xx, N, None, 100)
            # with v = mu(1 - mu) the moment line is bf.estimate_m's: compare m = mean(w) / tau2
            kk = (n >= 100) & np.isfinite(mu) & np.isfinite(np.ones(n.shape) if N is None else 1 - n / np.where(N > 0, N, np.nan))
            w = mu * (1 - mu)
            c = np.ones(n.shape) if N is None else 1 - n / np.where(N > 0, N, np.nan)
            tb = ((n * (x / n - mu) ** 2)[kk].sum() - (w * c)[kk].sum()) / (n * c ** 2)[kk].sum()
            m_bin, m_new = bf.estimate_m(x, n, mu, N=N), b2.estimate_m_v(x, n, mu, w, N=N)
            m_v = b2.estimate_m_v(x, n, mu, np.full(n.shape, v[0]), N=N)
            # A5 item 3: "both infinite, or a relative difference of at most 1e-12"; m = inf when tau2 is not positive
            eq = lambda a, b: (np.isinf(a) and np.isinf(b)) or abs(a - b) <= 1e-12 * abs(b)
            m_plain = w[kk].mean() / tb if tb > 0 else np.inf
            m_t = v[0] / t[0] if t[0] > 0 else np.inf
            ok_eq.append(eq(m_new, m_bin) and eq(m_plain, m_bin) and eq(m_t, m_v))
    chk("pooled-variance estimator equals lab 1's estimate_m when v = mu(1 - mu), and T5's tau2 gives estimate_m_v's m", all(ok_eq),
        "R_b, pooled r_b(br), r_p(os) cells on the 2022-24 training rows")
    # paired log loss and the date bootstrap, against plain Python
    import math
    r = np.random.default_rng(7)
    nd, nr = 12, 90
    dates = np.sort(r.integers(0, nd, nr))
    yy, pn, po = (r.random(nr) < 0.7).astype(float), r.uniform(.5, .9, nr), r.uniform(.5, .9, nr)
    Xs = pd.DataFrame({"date": dates})
    nb = ba.NBOOT
    ba.NBOOT = 50
    try:
        got = t_rule(Xs, yy, [(np.ones(nr, bool), pn, po)], 11)
    finally:
        ba.NBOOT = nb
    dd = [-(yy[i] * math.log(pn[i]) + (1 - yy[i]) * math.log(1 - pn[i])) + (yy[i] * math.log(po[i]) + (1 - yy[i]) * math.log(1 - po[i]))
          for i in range(nr)]
    ud = sorted(set(dates.tolist()))
    sums = {u: sum(dd[i] for i in range(nr) if dates[i] == u) for u in ud}
    cnts = {u: sum(1 for i in range(nr) if dates[i] == u) for u in ud}
    g = np.random.default_rng(11)
    bs = []
    for _ in range(50):
        draw = g.integers(0, len(ud), len(ud))
        bs.append(sum(sums[ud[j]] for j in draw) / sum(cnts[ud[j]] for j in draw))
    bs.sort()
    q = lambda a, p_: a[int(math.floor(p_ * (len(a) - 1)))] + (p_ * (len(a) - 1) - math.floor(p_ * (len(a) - 1))) * (
        a[min(int(math.floor(p_ * (len(a) - 1))) + 1, len(a) - 1)] - a[int(math.floor(p_ * (len(a) - 1)))])
    plain = (sum(dd) / nr, q(bs, .025), q(bs, .975))
    chk("paired log loss and the date bootstrap match a plain-Python recomputation", np.allclose([got["est"], *got["ci"]], plain, rtol=1e-12, atol=1e-15),
        f"{got['est']:+.8f} ({got['ci'][0]:+.8f} to {got['ci'][1]:+.8f}) vs {plain[0]:+.8f} ({plain[1]:+.8f} to {plain[2]:+.8f})")
    # T5 edge cases
    rr, nf, no = sd_ratio([-1.0, 0.04, 0.01, 0.02, 0.0], [0.04, 0.04, -0.01, 0.0, 0.01])
    k2 = kappa2(np.array([10.0, 30.0]), np.array([0.5, 0.25]))
    chk("T5 kappa^2 on constructed input n = [10, 30], c = [0.5, 0.25] is 0.109375", k2 == 0.109375, f"{k2!r}")
    chk("T5: family variance not positive gives ratio 0 and is counted; overall not positive is left out and counted",
        np.allclose(rr[[0, 1, 4]], [0, 1, 0]) and np.isnan(rr[2]) and np.isnan(rr[3]) and nf == 2 and no == 2
        and np.allclose(ci(rr), (np.quantile([0, 1, 0], .025), np.quantile([0, 1, 0], .975))),
        f"ratios {rr.tolist()}, family <= 0: {nf}, overall <= 0: {no}; interval over the 3 defined ratios")
    # expectation scoring
    base = {"t1": (-0.0002, (-0.0004, 0.0001)), "t1b": (-0.0001, (-0.0003, 0.0001)), "coef": 0.01, "t3": (0.019, (-0.01, 0.05)),
            "t3b2": (-0.02, 0.06), "placebo": (-0.0001, 0.0001), "t6e": (-0.0001, 0.0001),
            "t5": [{"kind": "batter", "tau2": 1e-4, "ratio": 0.2}, {"kind": "pitcher", "tau2": 1e-4, "ratio": 0.3}]}
    res_ = lambda e, i: score_expectations({**base, **e})[i][1]
    c_ok = (res_({}, 2) and not res_({"t3": (0.021, (-0.01, 0.05))}, 2) and not res_({"t3": (0.01, (0.001, 0.02))}, 2))
    f_ok = (res_({}, 5) and not res_({"t5": [{"kind": "batter", "tau2": -1e-5, "ratio": 0.0}, {"kind": "pitcher", "tau2": 1e-4, "ratio": 0.3}]}, 5)
            and not res_({"t5": [{"kind": "batter", "tau2": 1e-4, "ratio": 0.2}, {"kind": "pitcher", "tau2": 1e-4, "ratio": 0.5}]}, 5)
            and not res_({"t5": [{"kind": "batter", "tau2": 1e-4, "ratio": 0.3}, {"kind": "pitcher", "tau2": 1e-4, "ratio": 0.2}]}, 5))
    ah_ok = (res_({"t1": (-0.0002, (-0.0004, -0.0001))}, 0) == res_({"t1": (-0.0002, (-0.0004, 0.0001))}, 0) is True
             and res_({"t1b": (-0.0001, (-0.0003, -0.00001))}, 7) == res_({}, 7) is True and not res_({"t1": (0.0001, (0, 0.0002))}, 0))
    chk("expectation (c): within ±0.02 and interval includes 0", c_ok)
    chk("expectation (f): all three conditions", f_ok)
    chk("expectations (a) and (h): the percent-chance parts are not scored", ah_ok)
    # guards
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        opened, real_open = [], builtins.open
        real_rp = pd.read_parquet

        class DataRead(Exception):
            pass

        def spy_open(f_, *a, **k):
            if str(f_).endswith(".parquet"):
                opened.append(str(f_))
                raise DataRead(f_)
            return real_open(f_, *a, **k)

        def spy_rp(path, *a, **k):
            opened.append(str(path))
            raise DataRead(path)

        logs = {"leak": td / "leak.txt", "selftest": {k: td / f"{k}.txt" for k in SELFTEST}, "igap": td / "igap.txt", "results": td / "res.txt",
                "started": td / "started.txt"}
        real_pqf = ba.pq.ParquetFile

        def real_entry(**kw):  # the gate's own header lines are kept out of the smoke log
            n_log = len(ba.LOG)
            try:
                return globals()["real_entry"](**kw)
            finally:
                del ba.LOG[n_log:]

        builtins.open, pd.read_parquet, ba.pq.ParquetFile = spy_open, spy_rp, spy_rp
        try:
            try:
                real_entry(**logs)
                refused = None
            except SystemExit as e:
                refused = str(e)
            chk("--real with a missing PASS log refuses and reads no data file", refused and not opened, f"refused: \"{refused}\"; data files opened: {len(opened)}")
            logs["leak"].write_text("LEAK CHECK: PASS")
            for kk, pth in logs["selftest"].items():
                pth.write_text(f"SELF-TEST {kk}: PASS")
            logs["igap"].write_text(f"descriptive report\n{IGAP_DONE}\n\n")
            try:
                real_entry(**logs)
                reached = False
            except DataRead:
                reached = True
            chk("--real with every PASS log present (I_gap log ending in the completion line) passes the gate and only then "
                "reads data (stop rule 6)", reached and len(opened) == 1,
                f"first data read after the gate: {Path(opened[0]).name if opened else 'none'}")

            def gate_igap(text):
                if text is None:
                    logs["igap"].unlink()
                else:
                    logs["igap"].write_text(text)
                try:
                    real_gate(**logs)
                    return None
                except SystemExit as e:
                    return str(e)

            n0 = len(opened)
            g_ok, g_missing, g_other = (gate_igap(f"x\n{IGAP_DONE}\n"), gate_igap(None), gate_igap(f"{IGAP_DONE}\nsomething else\n"))
            chk("I_gap precondition: a log whose last non-blank line is the completion line passes", g_ok is None)
            chk("I_gap precondition: a missing log refuses", g_missing and "igap.txt" in g_missing, f"\"{g_missing}\"")
            chk("I_gap precondition: a log whose last non-blank line is anything else refuses", g_other and "igap.txt" in g_other,
                f"\"{g_other}\"; data files opened by these three gate calls: {len(opened) - n0}")
            logs["igap"].write_text(f"{IGAP_DONE}\n")
            # a stop rule 6 failure is a gate failure: it writes neither the marker nor the results log
            real_th = globals()["table_hashes"]
            globals()["table_hashes"] = lambda real: False
            try:
                real_entry(**logs)
                s6 = None
            except SystemExit as e:
                s6 = str(e)
            finally:
                globals()["table_hashes"] = real_th
            chk("a gate failure (a missing PASS log, or stop rule 6) writes neither the started marker nor the results log",
                refused and s6 and not logs["started"].exists() and not logs["results"].exists(), f"stop rule 6 failure: \"{s6}\"")
            logs["results"].write_text("x")
            n0 = len(opened)
            try:
                real_entry(**logs)
                ok2 = False
            except SystemExit as e:
                ok2 = "already exists" in str(e) and "addendum" in str(e)
            chk("a second --real refuses when the results log exists (run-once guard), reading no data file", ok2 and len(opened) == n0)
            logs["results"].unlink()
            logs["started"].write_text("x")
            try:
                real_entry(**logs)
                ok4 = False
            except SystemExit as e:
                ok4 = "lab2_real_started" in str(e) or "started.txt" in str(e)
            chk("--real refuses when the started marker exists, with 0 data files opened", ok4 and len(opened) == n0,
                f"data files opened: {len(opened) - n0}")
            logs["started"].unlink()
        finally:
            builtins.open, pd.read_parquet, ba.pq.ParquetFile = real_open, real_rp, real_pqf
        # simulated crash after the marker: the first fit raises; the marker stays and a second --real refuses
        class Crash(Exception):
            pass

        real_fit, n_log = globals()["fit"], len(ba.LOG)
        globals()["fit"] = lambda *a, **k: (_ for _ in ()).throw(Crash("simulated crash in the first fit"))
        try:
            run(X, ba.fake_outcomes(X, "null", SMOKE_SEED)[0], True, logs["results"], started=logs["started"])
            crashed = False
        except Crash:
            crashed = True
        finally:
            globals()["fit"] = real_fit
            del ba.LOG[n_log:]
        marker = logs["started"].read_text(encoding="utf-8") if logs["started"].exists() else ""
        try:
            real_gate(**logs)
            ok5 = False
        except SystemExit as e:
            ok5 = "started.txt" in str(e)
        chk("a crash after the marker (first fit patched to raise) leaves the marker, and a second --real then refuses",
            crashed and "UTC" in marker and "local" in marker and sha(SCRIPT) in marker and not logs["results"].exists() and ok5,
            "marker lines: " + " | ".join(marker.splitlines()))
        sc = td / "script.py"
        sc.write_text("print(1)\n")
        tl = td / "time.txt"
        tries = []
        for setup in (lambda: None, lambda: tl.write_text("script sha256 " + "0" * 64), lambda: tl.write_text(f"script sha256 {sha(sc)}")):
            setup()
            try:
                time_gate(tl, sc)
                tries.append(None)
            except SystemExit as e:
                tries.append(str(e))
        chk("null study refuses with no time log, and with a time log whose script hash does not match; accepts a match",
            tries[0] and tries[1] and tries[2] is None, f"no log: \"{tries[0]}\"; mismatch: \"{tries[1][:90]}...\"")
        reg = td / "btslab2_selftest_planted_log.txt"
        reg.write_text("x")
        try:
            guard_once(reg, "registered seed 3300 is run once")
            ok3 = False
        except SystemExit as e:
            ok3 = "already exists" in str(e)
        chk("a second run of a registered seed refuses (run-once guard)", ok3)
    # fake mode never touches the outcome column
    asked, real_rp = [], pd.read_parquet
    pd.read_parquet = lambda path, columns=None, **k: (asked.append((Path(path).name, columns)), real_rp(path, columns=columns, **k))[1]
    try:
        Xf = load(False)
    finally:
        pd.read_parquet = real_rp
    base_cols = [c for n, c in asked if n == bf.OUT_PARQUET.name][0]
    chk("fake mode never reads the real outcome column", "h" not in base_cols and "h" not in Xf.columns and "ab" in Xf.columns,
        f"lab 1 table read with {len(base_cols)} columns, 'h' not among them; 'h' absent from the loaded frame; lab 2/3 tables have no 'h'")
    # stop rule 7, M3: lab 3's logged values against A3's strings, and the comparison on constructed inputs
    lab3_log = OUT / "lab3_results_log.txt"
    txt = lab3_log.read_text(encoding="utf-8")
    parsed = parse_lab3_m3(txt)
    chk("stop rule 7 (M3): lab 3's logged M3 values, parsed from lab3_results_log.txt, equal A3's strings used by repro_m3",
        parsed["folds"] == REPRO_M3_FOLDS and parsed["wf"] == REPRO_M3_WF, f"{len(parsed['folds'])} fold rows, {len(parsed['wf'])} test strings")
    chk("stop rule 7 (M3): the exact logged values pass the comparison", repro_m3_compare(parsed) == [])
    pert = {"folds": {**parsed["folds"]}, "wf": {**parsed["wf"]}}
    key0 = next(iter(REPRO_M3_FOLDS))
    pert["folds"][key0] = (pert["folds"][key0][0], "0.65756", *pert["folds"][key0][2:])
    pert_wf = {"folds": parsed["folds"], "wf": {**parsed["wf"], "lab 3 T1": parsed["wf"]["lab 3 T1"].replace("+0.00069", "+0.00070")}}
    b1, b2_ = repro_m3_compare(pert), repro_m3_compare(pert_wf)
    chk("stop rule 7 (M3): one value changed in its last printed digit is a mismatch (rule 7 would fire)",
        len(b1) == 1 and "0.65756" in b1[0] and len(b2_) == 1 and "+0.00070" in b2_[0], f"{b1[0]} | {b2_[0]}")
    broken = parse_lab3_m3(txt.replace("pooled 2024–25: +0.00069", "pooled 2024-25 +0.00069").replace(
        "| WF | 2022-2023 | 2024 | M3 |", "| WF | 2022-2023 | 2024 | M3x |"))
    try:
        bm = repro_m3_compare(parsed, broken["folds"], broken["wf"])
        bm_ok = len(bm) == 2 and all("missing or unparsable" in s for s in bm)
    except Exception as e:  # noqa: BLE001
        bm, bm_ok = [repr(e)], False
    chk("stop rule 7 (M3): a missing or unparsable logged line gives a clear stop message, not an exception", bm_ok, " | ".join(bm))
    # the marker assertion's failing case; --real with --ignore-stops
    try:
        run(X, np.zeros(len(X)), True, Path(tempfile.gettempdir()) / "unused.txt")
        a_msg = None
    except AssertionError as e:
        a_msg = str(e)
    chk("run() under --real without a started marker path fails its assertion before anything else", a_msg == "--real must write the started marker",
        f"\"{a_msg}\"")
    argv = sys.argv
    sys.argv = [SCRIPT.name, "--real", "--ignore-stops"]
    try:
        main()
        ig = None
    except SystemExit as e:
        ig = str(e)
    finally:
        sys.argv = argv
    chk("--real together with --ignore-stops refuses before anything is read", ig == "--ignore-stops is not allowed with --real", f"\"{ig}\"")
    # real-branch rehearsal on fake outcomes
    t_r = time.perf_counter()
    rh = rehearsal(X)
    t_r = time.perf_counter() - t_r
    chk("real-branch rehearsal (fake outcomes, temp paths): gates pass, marker written, M1 reproduction fails, stop rule 7 fires, "
        "results log written, exit as a stop; a second --real refuses", rh["ok"], rh["detail"] + f"; {t_r:.0f} s")
    chk("after the unit checks, no real marker or results log exists in analysis/out", not STARTED.exists() and not RESULTS.exists(),
        f"{STARTED.name} exists: {STARTED.exists()}; {RESULTS.name} exists: {RESULTS.exists()}")
    new = sorted(set(os.listdir(OUT)) - out_before)
    chk("the unit checks wrote nothing into analysis/out", not new, f"new files: {new or 'none'}")
    return res


def rehearsal(X) -> dict:
    """Unit checks only: main()'s --real branch with the loader supplying fake outcomes from the no-h table (a fake 'h'
    column), stop rule 6 in its fake-mode form (no outcome column read) and every output path in a temporary directory."""
    import tempfile
    g = globals()
    names = ("real_entry", "table_hashes", "load", "RESULTS", "STARTED")
    saved = {k: g[k] for k in names}
    yf, _ = ba.fake_outcomes(X, "null", SMOKE_SEED)
    Xh = X.assign(h=np.where(np.isnan(yf), 0.0, yf))
    argv, opt, n_log = sys.argv, dict(OPT), len(ba.LOG)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        logs = {"leak": td / "leak.txt", "selftest": {k: td / f"{k}.txt" for k in SELFTEST}, "igap": td / "igap.txt",
                "results": td / "lab2_results_log.txt", "started": td / "lab2_real_started.txt"}
        logs["leak"].write_text("LEAK CHECK: PASS")
        for k, p in logs["selftest"].items():
            p.write_text(f"SELF-TEST {k}: PASS")
        logs["igap"].write_text(f"{IGAP_DONE}\n")
        g["real_entry"] = lambda: saved["real_entry"](**logs)
        g["table_hashes"] = lambda real: saved["table_hashes"](False)
        g["load"] = lambda real: Xh.copy()
        g["RESULTS"], g["STARTED"] = logs["results"], logs["started"]
        sys.argv = [SCRIPT.name, "--real"]
        try:
            try:
                main()
                code = "returned"
            except SystemExit as e:
                code = e.code
            marker = logs["started"].read_text(encoding="utf-8") if logs["started"].exists() else ""
            rlog = logs["results"].read_text(encoding="utf-8") if logs["results"].exists() else ""
            try:
                main()
                again = None
            except SystemExit as e:
                again = str(e)
        finally:
            for k, v in saved.items():
                g[k] = v
            sys.argv = argv
            OPT.clear()
            OPT.update(opt)
            del ba.LOG[n_log:]
    stop7 = "STOP: M1 does not reproduce lab 1 (stop rule 7); M2 and M4 are not read." in rlog
    m1fail = "- Rule 3, M1 reproduction (6 folds; walk-forward test numbers): FAIL" in rlog
    ok = (code == 2 and sha(SCRIPT) in marker and "UTC" in marker and stop7 and m1fail and "PASS" in rlog
          and again is not None and "already exists" in again and "addendum" in again)
    return {"ok": ok, "detail": f"exit code {code}; marker {'written' if marker else 'MISSING'}; results log "
                                f"{'written' if rlog else 'MISSING'}, M1 reproduction FAIL line {'present' if m1fail else 'absent'}, "
                                f"stop rule 7 STOP line {'present' if stop7 else 'absent'}; second --real: \"{again}\""}


def real_entry(leak=LEAK_LOG, selftest=SELFTEST, igap=IGAP_LOG, results=RESULTS, started=STARTED):
    """--real: guards and PASS logs first (no data read), then stop rule 6. A gate failure writes neither the started
    marker nor the results log."""
    lines = real_gate(leak, selftest, igap, results, started)
    say("# Beat the Streak lab 2 — RESULTS (real outcomes)")
    say(f"script {SCRIPT.name} on-disk sha256 {sha(SCRIPT)}")
    say("## Preconditions (A3 item 13i; A6 items 4, 13; before any data file is read)")
    for s in lines:
        say(s)
    say("## Stop rule 6, table identity (A3 item 6; A6 item 4)")
    if not table_hashes(True):
        sys.exit("STOP: stop rule 6 failed; no model is fitted; no marker or results log written")


# ---------------------------------------------------------------- timing (A3 item 9; A6 item 10)
def timing(procs):
    """One full null seed in-process (4 models x 2 folds with inner CV, 2 x 10 placebo refits x 2 folds, T1, T1b, 500
    resamples) plus the real-run pieces; projections for the null study and the real run."""
    ba.NBOOT = 500
    X = load(False)
    t = time.perf_counter()
    _null_init_local(X, 10)
    t_prep = time.perf_counter() - t
    t = time.perf_counter()
    r = _null_seed(SMOKE_SEED)
    t_seed = time.perf_counter() - t
    y, _ = outcomes(X, "null", SMOKE_SEED)
    p = _W["preps"][1]
    cv = {}
    for mn, Z in (("M1", p["Z"]), ("M2", p["Z2"]), ("M3", p["Z3"]), ("M4", p["Z4"])):
        t = time.perf_counter()
        f = fit(X, y, p, Z)
        cv[mn] = time.perf_counter() - t
    ok = f["ok"]
    t = time.perf_counter()
    fit_l2(p["Z4"][ok], y[ok], f["C"])
    t_fit = time.perf_counter() - t
    w = np.random.default_rng(1).poisson(1.0, ok.sum()).astype(float)
    t = time.perf_counter()
    fit_l2(p["Z4"][ok], y[ok], f["C"], w, f["beta"])
    t_boot = time.perf_counter() - t
    t = time.perf_counter()
    ba.NBOOT = 2000
    t5(X, SEED["T5"])
    t_t5 = time.perf_counter() - t
    return {"t_prep": t_prep, "t_seed": t_seed, "cv": cv, "t_fit": t_fit, "t_boot": t_boot, "t_t5": t_t5, "seed_row": r}


def _null_init_local(X, nperm):
    PU = placebo_units(X)
    _W.update(X=X, preps=[prepare(X, tr, te) for tr, te in WF], PU=PU, donors=[donor_map(PU, k) for k in range(nperm)])


# ---------------------------------------------------------------- output checklist (Step 1d review, as changed by A6)
CHECKLIST = [
    (1, "preconditions: PASS logs, then stop rule 6 table hashes incl. lab 2 table", "A3 item 13i; A3 item 6; A6 items 4, 13"),
    (2, "M1 fits, lab 1's six folds", "plan Section 7; A4 item 5e"), (3, "M1 reproduction (stop rule 7; skipped on fake outcomes)", "A3 item 6; A6 item 5"),
    (4, "M3 fits and reproduction (skipped on fake outcomes)", "A3 items 3, 6; A6 item 5"), (5, "M2 fits", "plan Section 7"),
    (6, "M4 fits", "A3 item 3"), (7, "stop rule 5 for M1-M4", "plan Section 11; A3 item 6"),
    (8, "T4 placebo P for M2, with real-vs-placebo correlation", "plan Section 8 T4; A3 item 7; A6 items 1-2"),
    (9, "M4 placebo (placebo P)", "A3 items 3, 7; A6 item 2"), (10, "T1", "plan Section 8"), (11, "T1b", "A3 item 3"),
    (12, "T2", "plan Section 8"), (13, "T3 (recalibration unused)", "plan Section 8; A6 item 12"),
    (14, "T3b (recalibration unused)", "A3 item 3; A6 item 12"), (15, "T5", "plan Section 8; A3 item 13e; A5 item 8; A6 item 8"),
    (16, "T6 incl. M1 + I1..I4 vs M2", "plan Section 8; A6 items 7, 9"), (17, "M2 and M4 I1 coefficients", "plan Section 8; A3 item 3 (P18)"),
    (18, "expectations (a)-(i)", "plan Section 9; A3 item 3; A6 item 11"), (19, "null study", "plan Section 12; A3 items 5, 7"),
    (20, "null-study predictions", "A3 item 5"), (21, "planted and planted-null (registered 3300, 3301)", "A3 item 4; A6 items 2-3"),
    (22, "dry runs 9101-9108 (--dry-criterion)", "A3 item 4"), (23, "runtime projection and time log", "A3 item 9; A6 item 10"),
    (24, "run-once guards (--real, registered seeds)", "A3 item 10"),
    (25, "confirmation null study 3401-3600: reserved, not built", "A3 item 8"),
    (26, "I_gap report: separate script; --real requires its log", "A4 item 1; A6 item 13")]


def checklist(done: set, mode: str):
    say(f"\n## Output checklist ({mode})")
    say(md_table(pd.DataFrame([{"#": n, "output": o, "required by": s, "this run": "done" if n in done else "not in this mode"}
                               for n, o, s in CHECKLIST])))


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true", help="the real run (real hit column), once")
    g.add_argument("--fake", choices=["planted", "planted-null"], help="registered self-test on seeded fake outcomes (3300 / 3301)")
    g.add_argument("--null-study", action="store_true", help="200 fake null seeds 3001-3200 in parallel")
    g.add_argument("--smoke", action="store_true", help="unit checks, then smoke seed 9109 with tiny settings")
    g.add_argument("--time", action="store_true", help="time one null seed and the real-run pieces; write the time log")
    g.add_argument("--dry-criterion", action="store_true", help="throwaway dry runs 9101-9108 (A3 item 4)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--ignore-stops", action="store_true", help="fake runs only: continue past stop rules")
    a = ap.parse_args()
    if not (a.real or a.fake or a.null_study or a.smoke or a.time or a.dry_criterion):
        sys.exit("refusing to run: choose --smoke, --time, --dry-criterion, --fake planted|planted-null, --null-study, "
                 "or --real (uses the real hit column)")
    if a.real and a.ignore_stops:
        sys.exit("--ignore-stops is not allowed with --real")
    print(f"{SCRIPT.name} on-disk sha256 {sha(SCRIPT)}", flush=True)
    if a.real:
        real_entry()  # guards and PASS logs before any data read, then stop rule 6; exits on failure, writing nothing
        log = RESULTS
    elif a.fake:
        log = SELFTEST[a.fake]
        guard_once(log, f"registered seed {REGISTERED[a.fake]} is run once")
    elif a.null_study:
        guard_once(SELFTEST["null"], "the registered null study is run once")
        guard_once(NULL_CSV, "the registered null study is run once")
        cur = time_gate()
    OPT["ignore_stops"] = a.ignore_stops
    ba._check_solver()
    bf._self_check()
    l3._check_m_normal()

    if a.time:
        T = timing(a.procs)
        n_real = (5 * (T["cv"]["M1"] + T["cv"]["M2"] + T["cv"]["M3"]) + 2 * T["cv"]["M4"] + 5 * T["t_prep"] / 2
                  + 2 * 2 * 50 * T["t_fit"] + 2 * 2 * T["t_fit"] + 2 * 2000 * T["t_boot"] + T["t_t5"])
        lines = [f"# Beat the Streak lab 2 — timing (A3 item 9; A6 item 10); fake null outcomes, seed {SMOKE_SEED}",
                 f"script sha256 {sha(SCRIPT)}",
                 f"processes for the projection: {a.procs} (os.cpu_count() {os.cpu_count()})",
                 f"fold preparation, 2 walk-forward folds (lab 1 + lab 3 + lab 2 features, placebo units, 10 mappings): {T['t_prep']:.1f} s",
                 f"one null seed in-process (M1-M4 inner CV on 2 folds, 2 x 10 placebo refits on 2 folds, T1, T1b, 500 resamples): "
                 f"{T['t_seed']:.1f} s",
                 "inner CV incl. refit, train 2022-24: " + ", ".join(f"{k} {v:.1f} s" for k, v in T["cv"].items()),
                 f"one M4 fit at fixed C: {T['t_fit']:.2f} s; one warm bootstrap refit: {T['t_boot']:.2f} s; T5 with 2,000 resamples: {T['t_t5']:.1f} s",
                 f"projection, null study 200 seeds: {200 * T['t_seed'] / 3600:.2f} h serial, {200 * T['t_seed'] / a.procs / 60:.0f} min on "
                 f"{a.procs} processes (plus about {T['t_prep']:.0f} s start-up per process; ignores contention between processes)",
                 f"projection, real run: {n_real / 60:.0f} min serial (5 distinct folds x M1, M2, M3 inner CV + 2 x M4 + 2 models x 50 "
                 f"placebo refits x 2 folds + 4 exploratory refits + 2 x 2,000 warm bootstrap refits + T5)"]
        for s in lines:
            say(s)
        TIME_LOG.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        print(f"wrote {TIME_LOG}")
        return

    if a.null_study:
        ba.LOG.clear()
        say("# Beat the Streak lab 2 — null study on FAKE outcomes (Bernoulli(0.75); real hits never loaded)")
        say(f"script sha256 {cur}; time log {TIME_LOG.name} records the same (A6 item 10)")
        say("## Stop rule 6, table identity")
        if not table_hashes(False):
            sys.exit("STOP: stop rule 6 failed")
        passed, _ = null_study(NULL_SEEDS, a.procs, 10, 500, NULL_CSV)
        say(f"SELF-TEST null: {'PASS' if passed else 'FAIL - stop, do not tune'}")
        checklist({19, 20, 24}, "null study")
        SELFTEST["null"].write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        sys.exit(0 if passed else 1)

    if a.smoke:
        OPT.update(nperm=2, ignore_stops=True)
        ba.NBOOT = 20
        say("# Beat the Streak lab 2 — SMOKE TEST ONLY (not a result): unit checks, then fake outcomes, throwaway seed 9109, "
            "20 bootstrap resamples, 2 placebo permutations. It shows the code runs end to end; nothing here is evidence.")
        say(f"script sha256 {sha(SCRIPT)}")
        say("\n## Stop rule 6, table identity")
        if not table_hashes(False):
            sys.exit("STOP: stop rule 6 failed")
        X = load(False)
        say("\n## Unit checks")
        res = unit_checks(X)
        if not all(ok for _, ok, _ in res):
            SMOKE_LOG.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
            sys.exit("STOP: a unit check failed; the smoke run is not started")
        _GEN.clear()
        y, desc = outcomes(X, "planted", SMOKE_SEED)
        say(f"\n## Smoke run, planted generator, seed {SMOKE_SEED}\noutcome: {desc}")
        r = run(X, y, False, SMOKE_LOG)
        say(line := selftest_data("planted", SMOKE_SEED, r))
        say("Pass rule on this throwaway seed (A3 item 4), for information only:")
        verdict(parse_data(line))
        say(f"\n## Null-study code path, 1 seed ({SMOKE_SEED}), 1 process, same tiny settings")
        null_study([SMOKE_SEED], 1, 2, 20, Path(os.environ.get("TEMP", ".")) / "btslab2_smoke_null_seed.csv")
        checklist(r["done"] | {24}, "smoke")
        SMOKE_LOG.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        print(f"wrote {SMOKE_LOG}")
        return

    if a.dry_criterion:
        return dry_criterion()

    if a.real:
        X = load(True)
        y = np.where(X.ab >= 1, (X.h >= 1).astype(float), np.nan)
        say("outcome: REAL box-score hits (H >= 1) on AB >= 1 rows")
        r = run(X, y, True, log, started=STARTED)
        checklist(r["done"] | {1, 24}, "--real")
    else:
        seed = REGISTERED[a.fake]
        say(f"# Beat the Streak lab 2 — SELF-TEST on FAKE outcomes: {a.fake} (seed {seed}; real hits never loaded)")
        say(f"script sha256 {sha(SCRIPT)}")
        say("## Stop rule 6, table identity")
        if not table_hashes(False):
            sys.exit("STOP: stop rule 6 failed")
        OPT["nperm"] = 10  # A6 item 2: registered planted runs use 10 permutations
        X = load(False)
        y, desc = outcomes(X, a.fake, seed)
        say(f"outcome: {desc}")
        r = run(X, y, False, log)
        say(line := selftest_data(a.fake, seed, r))
        say("\nPass rule (A3 item 4):")
        say(f"SELF-TEST {a.fake}: {'PASS' if verdict(parse_data(line)) else 'FAIL - stop, do not tune'}")
        checklist(r["done"] | {21, 24}, f"--fake {a.fake}")
    log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
    print(f"wrote {log}")


def dry_criterion():
    """Throwaway dry runs (A3 item 4): 9101-9102 planted I1, 9103-9108 planted null-I1; 20 resamples, 2 permutations;
    conditions fixed in advance; appended to the smoke log."""
    OPT.update(nperm=2, ignore_stops=True)
    ba.NBOOT = 20
    say("\n\n# DRY RUNS ONLY (not a result; A3 item 4): planted I1 9101, 9102; planted null-I1 9103-9108. 20 bootstrap "
        "resamples, 2 placebo permutations.")
    say(f"script sha256 {sha(SCRIPT)}")
    say("Conditions, fixed in advance: both planted seeds pass the rule; T1, and separately T1b, Confirmed on at most 1 of the 6 "
        "planted-null seeds.")
    say("## Stop rule 6, table identity")
    if not table_hashes(False):
        sys.exit("STOP: stop rule 6 failed")
    X = load(False)
    rows = []
    for kind, seed in DRY:
        y, desc = outcomes(X, kind, seed)
        say(f"\n## {kind}, seed {seed}: {desc}")
        r = run(X, y, False, SMOKE_LOG)
        d = parse_data(selftest_data(kind, seed, r))
        say("Pass rule on this seed:")
        rows.append({"kind": kind, "seed": seed, "M2 I1 coef": f"{d['i1']:+.4f}", "T1": "Confirmed" if d["t1"] else "not confirmed",
                     "T1b": "Confirmed" if d["t1b"] else "not confirmed", "M2 placebo": "Confirmed" if d["stop"] else "not Confirmed",
                     "M4 placebo": "Confirmed" if d["stop4"] else "not Confirmed", "pass rule": "PASS" if verdict(d) else "FAIL", "_d": d})
    ph = [r["_d"] for r in rows if r["kind"] == "planted"]
    pn = [r["_d"] for r in rows if r["kind"] == "planted-null"]
    c1 = all(r["pass rule"] == "PASS" for r in rows if r["kind"] == "planted")
    c2, c3 = sum(d["t1"] for d in pn) <= 1, sum(d["t1b"] for d in pn) <= 1
    say("\n## Dry runs: summary")
    say(md_table(pd.DataFrame([{k: v for k, v in r.items() if k != "_d"} for r in rows])))
    say(f"- Condition 1, both planted seeds pass: {'HOLDS' if c1 else 'FAILS'} ({len(ph)} seeds)")
    say(f"- Condition 2, T1 Confirmed on at most 1 of 6 planted-null seeds: {sum(d['t1'] for d in pn)} of 6 -> {'HOLDS' if c2 else 'FAILS'}")
    say(f"- Condition 3, T1b Confirmed on at most 1 of 6 planted-null seeds: {sum(d['t1b'] for d in pn)} of 6 -> {'HOLDS' if c3 else 'FAILS'}")
    say(f"DRY-RUN CONDITIONS: {'ALL HOLD' if c1 and c2 and c3 else 'NOT MET - write an addendum before any registered seed'}")
    checklist({22}, "--dry-criterion")
    with SMOKE_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(ba.LOG) + "\n")


if __name__ == "__main__":
    main()
