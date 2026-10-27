"""Beat the Streak lab, Step 2: the plan's models and tests (BEAT_THE_STREAK_LAB_PLAN.md f10661f + addendum A1 6d9cb3c).

Outcomes are FAKE unless --real is given; without --real the hit column is never loaded.
  python analysis/btslab_analysis.py --fake null       # self-test, writes analysis/out/btslab_selftest_null_log.txt
  python analysis/btslab_analysis.py --fake planted    # self-test, writes analysis/out/btslab_selftest_planted_log.txt
  python analysis/btslab_analysis.py --real            # the real run, writes analysis/out/btslab_results_log.txt
--ignore-stops (fake runs only) keeps going past the plan's stop rules so the whole pipeline is exercised.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab_features as bf  # noqa: E402
import stats  # noqa: E402
from v4_analysis import ci, ll, md_table, wilson_s  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SEED = 20261026  # + test number (plan Section 7)
WF = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
LOSO = [(tuple(s for s in bf.POOL_SEASONS if s != t), t) for t in bf.POOL_SEASONS]
CS = np.logspace(-3, 2, 12)
NS = [60, 80, 100, 130, None]  # None = all days
GROUPS = {"A": ["a_hab", "a_k", "a_bb", "a_iso", "a_plat", "A_miss", "A_plat_miss"],
          "B": [f"slot_{k}" for k in range(1, 10)],
          "C": ["c_hab", "c_k", "c_bb", "c_hand_L", "c_bf_exp", "C_miss", "C_hand_miss"],
          "D": ["d_pa_sp", "d_pa_bp"],
          "E": ["e_bp_hab", "e_fat_pitches", "e_fat_relievers", "E_miss"],
          "F": ["f_park", "f_temp", "f_indoor"],
          "G": ["g_hab14", "g_hab30"]}
M1_COLS = [c for g in GROUPS.values() for c in g]
MARKET, HCOLS = ["x_mkt_runs", "X_mkt_miss"], ["base_b2"]
LOG = []
NBOOT = 2000  # bootstrap resamples (plan); the null study uses 500


def boot_counts(C, seed):
    """Cluster bootstrap as resample counts per cluster; same draws as v4_analysis.boot_counts at NBOOT = 2000."""
    r = np.random.default_rng(seed)
    return np.array([np.bincount(r.integers(0, C, C), minlength=C) for _ in range(NBOOT)])


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- data and outcomes
def load(real: bool) -> pd.DataFrame:
    cols = pq.ParquetFile(bf.OUT_PARQUET).schema_arrow.names
    X = pd.read_parquet(bf.OUT_PARQUET, columns=[c for c in cols if real or c != "h"])
    assert real or "h" not in X, "guard: the real hit column must not be loaded without --real"
    return X.sort_values(["date", "game_pk", "id"]).reset_index(drop=True)


def fake_outcomes(X: pd.DataFrame, kind: str, seed=None) -> tuple[np.ndarray, str]:
    """Seeded fake hits on AB >= 1 rows only. Planted: logit p = logit 0.75 + 0.30*[slot<=4] - 0.40*z(K%) + 0.50*z(H/AB)."""
    seed = (777 if kind == "null" else 778) if seed is None else seed
    r = np.random.default_rng(seed)
    if kind == "null":
        p = np.full(len(X), 0.75)
        desc = f"independent Bernoulli(0.75), seed {seed}"
    else:
        z = lambda v: ((v - np.nanmean(v)) / np.nanstd(v)).fillna(0)
        k, hab = z(X.a_k / X.a_pa.where(X.a_pa > 0)), z(X.a_h / X.a_ab.where(X.a_ab > 0))
        lp = np.log(3) + 0.30 * X.slot.le(4) - 0.40 * k + 0.50 * hab
        p = 1 / (1 + np.exp(-lp.to_numpy()))
        desc = ("Bernoulli(p), logit p = logit(0.75) + 0.30*[slot 1-4] - 0.40*z(raw K%) + 0.50*z(raw H/AB); "
                f"z over all rows, missing -> 0; seed {seed}")
    y = (r.random(len(X)) < p).astype(float)
    y[X.ab.to_numpy() < 1] = np.nan
    return y, desc


# ---------------------------------------------------------------- fold-specific features (plan Section 4 common rules)
def derive(X: pd.DataFrame, tr: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Shrunk rates with m estimated on the training rows only; D split; flags. Missing values left NaN (filled later)."""
    T = X[tr]
    bat = T.drop_duplicates(["id", "season"], keep="last")  # one unit per player-season
    sp_ = T.drop_duplicates(["opp_sp", "season"], keep="last")
    ven = T.drop_duplicates(["venue_id", "season"], keep="last")
    f = lambda d, c: d[c].to_numpy(float)
    m = {"hab": bf.estimate_m(f(bat, "a_h"), f(bat, "a_ab"), f(bat, "lg_h_ab")),
         "k": bf.estimate_m(f(bat, "a_k"), f(bat, "a_pa"), f(bat, "lg_k_pa")),
         "bb": bf.estimate_m(f(bat, "a_bb"), f(bat, "a_pa"), f(bat, "lg_bb_pa")),
         "c_hab": bf.estimate_m(f(sp_, "c_h"), f(sp_, "c_ab"), f(sp_, "lg_h_ab")),
         "c_k": bf.estimate_m(f(sp_, "c_k"), f(sp_, "c_pa"), f(sp_, "lg_k_pa")),
         "c_bb": bf.estimate_m(f(sp_, "c_bb"), f(sp_, "c_pa"), f(sp_, "lg_bb_pa")),
         "bf": m_bf(T),
         "park": bf.estimate_m(f(ven, "f_venue_h"), f(ven, "f_venue_pa"), f(ven, "f_road_h") / f(ven, "f_road_pa"))}
    m["iso"] = m["hab"]  # ISO is not a proportion; it shares H/AB's AB units and prior strength (notes)
    sh = lambda x, n, mu, mm: bf.shrink(X[x].to_numpy(float), X[n].to_numpy(float), mu, mm)
    F = pd.DataFrame(index=X.index)
    F["a_hab"] = sh("a_h", "a_ab", X.lg_h_ab.to_numpy(), m["hab"])
    F["a_k"] = sh("a_k", "a_pa", X.lg_k_pa.to_numpy(), m["k"])
    F["a_bb"] = sh("a_bb", "a_pa", X.lg_bb_pa.to_numpy(), m["bb"])
    F["a_iso"] = sh("a_xb", "a_ab", X.lg_xb_ab.to_numpy(), m["iso"])
    own = F.a_hab.to_numpy()
    # platoon and recent form are subsets of the 365-day window: nested moment estimate against the raw window rate
    raw = lambda d: f(d, "a_h") / np.where(f(d, "a_ab") > 0, f(d, "a_ab"), np.nan)
    pl_ = T.drop_duplicates(["id", "season", "c_hand"], keep="last")
    m["plat"] = bf.estimate_m(f(pl_, "a_vs_h"), np.nan_to_num(f(pl_, "a_vs_ab")), raw(pl_), N=f(pl_, "a_ab"))
    # platoon and recent form enter as deviations from the batter's own H/AB, so their coefficients mean "beyond H/AB"
    F["a_plat"] = bf.shrink(X.a_vs_h.fillna(0).to_numpy(), X.a_vs_ab.fillna(0).to_numpy(), own, m["plat"]) - own
    F["A_miss"], F["A_plat_miss"] = X.a_pa.eq(0).astype(float), X.a_vs_ab.fillna(0).eq(0).astype(float)
    for k in range(1, 10):
        F[f"slot_{k}"] = X.slot.eq(k).astype(float)
    F["c_hab"] = sh("c_h", "c_ab", X.lg_h_ab.to_numpy(), m["c_hab"])
    F["c_k"] = sh("c_k", "c_pa", X.lg_k_pa.to_numpy(), m["c_k"])
    F["c_bb"] = sh("c_bb", "c_pa", X.lg_bb_pa.to_numpy(), m["c_bb"])
    F["c_hand_L"] = X.c_hand.map({"L": 1.0, "R": 0.0})
    F["c_bf_exp"] = sh("c_bf_last10", "c_starts_last10", X.lg_bf_per_start.to_numpy(), m["bf"])
    F["C_miss"], F["C_hand_miss"] = X.c_pa.eq(0).astype(float), X.c_hand.isna().astype(float)
    F["d_pa_sp"], F["d_pa_bp"] = bf.pa_split(X.d_exp_pa.to_numpy(), F.c_bf_exp.to_numpy(), X.slot.to_numpy())
    F["e_bp_hab"], F["e_fat_pitches"], F["e_fat_relievers"] = X.e_bp_hab, X.e_fat_pitches_3d.astype(float), X.e_fat_relievers_3d.astype(float)
    F["E_miss"] = X.e_bp_hab.isna().astype(float)
    road = (X.f_road_h / X.f_road_pa).to_numpy()
    F["f_park"] = np.where(X.f_road_pa > 0, bf.shrink(X.f_venue_h.to_numpy(float), X.f_venue_pa.to_numpy(float), road, m["park"]) / road, 1.0)
    F["f_temp"], F["f_indoor"] = X.f_temp_f, X.f_indoor.astype(float)
    m["g14"] = bf.estimate_m(f(bat, "g_h14"), f(bat, "g_ab14"), raw(bat), min_n=20, N=f(bat, "a_ab"))  # 100 AB unreachable
    m["g30"] = bf.estimate_m(f(bat, "g_h30"), f(bat, "g_ab30"), raw(bat), min_n=20, N=f(bat, "a_ab"))
    F["g_hab14"] = bf.shrink(X.g_h14.to_numpy(float), X.g_ab14.to_numpy(float), own, m["g14"]) - own
    F["g_hab30"] = bf.shrink(X.g_h30.to_numpy(float), X.g_ab30.to_numpy(float), own, m["g30"]) - own
    F["x_mkt_runs"], F["X_mkt_miss"] = X.x_mkt_team_runs, X.x_mkt_team_runs.isna().astype(float)
    F["base_b2"] = X.base_b2
    return F, m


def m_bf(T: pd.DataFrame) -> float:
    """Prior strength (in starts) for expected BF per start. Per-start variance w from consecutive starts of one
    pitcher with full 10-start windows (sum changes by bf_new - bf_old, variance 2w); between-pitcher variance =
    variance of 10-start means minus w/10. Pitchers with few starts differ in kind (openers), so 1/n regression fails."""
    d = T.drop_duplicates(["opp_sp", "date"]).sort_values(["opp_sp", "date"])
    d = d[d.c_starts_last10.eq(10)]
    same = d.opp_sp.eq(d.opp_sp.shift()) & d.season.eq(d.season.shift())
    w = d.c_bf_last10.diff()[same].var() / 2
    last = d.drop_duplicates(["opp_sp", "season"], keep="last")
    tau2 = (last.c_bf_last10 / 10).var() - w / 10
    return float(w / tau2) if tau2 > 0 else 1e6


def _check_m_bf():
    r = np.random.default_rng(0)
    rows = []
    for pid, mu in enumerate(r.normal(24, 3, 600)):
        bfs = r.normal(mu, 5, 40)
        rows += [{"opp_sp": pid, "date": t, "season": 2023, "c_starts_last10": 10, "c_bf_last10": bfs[t - 10:t].sum()} for t in range(10, 40)]
    m = m_bf(pd.DataFrame(rows))
    assert 1.8 < m < 4.2, m  # true 25 / 9 = 2.8


def design(F: pd.DataFrame, tr: np.ndarray, cols: list) -> np.ndarray:
    """Fill missing with the training mean, standardise with training mean/SD, prepend an intercept."""
    A = F[cols].to_numpy(float)
    mu = np.nanmean(A[tr], axis=0)
    A = np.where(np.isnan(A), mu, A)
    sd = A[tr].std(axis=0)
    sd[sd == 0] = 1.0
    return np.column_stack([np.ones(len(A)), (A - mu) / sd])


# ---------------------------------------------------------------- L2 logistic regression (sklearn's objective)
def fit_l2(Z, y, C, w=None, beta=None):
    """Minimise C * sum w*logloss + 0.5*|beta[1:]|^2 by Newton's method (intercept unpenalised)."""
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(Z.shape[1]) if beta is None else beta.copy()
    pen = np.full(Z.shape[1], 1 / C)
    pen[0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (w * (p - y)) + pen * beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def pred(Z, beta):
    return 1 / (1 + np.exp(-Z @ beta))


def _check_solver():
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(0)
    Z = np.column_stack([np.ones(3000), r.normal(size=(3000, 4))])
    y = (r.random(3000) < pred(Z, np.array([0.5, 1, -1, 0.3, 0]))).astype(float)
    b = fit_l2(Z, y, 0.05)
    sk = LogisticRegression(C=0.05, tol=1e-10, max_iter=10000).fit(Z[:, 1:], y)
    assert np.allclose(b, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-5), (b, sk.coef_)


# ---------------------------------------------------------------- one fold
def prepare_fold(X, train, test):
    """Everything about a fold that does not depend on the outcomes: features, design matrix, inner-CV blocks."""
    tr, te = X.season.isin(train).to_numpy(), X.season.eq(test).to_numpy()
    F, m = derive(X, tr)
    dates = X.date.to_numpy()
    blk = np.full(len(X), -1)
    for i, b in enumerate(np.array_split(np.sort(np.unique(dates[tr])), 5)):  # contiguous date blocks
        blk[np.isin(dates, b) & tr] = i
    return {"train": train, "test": test, "tr": tr, "te": te, "F": F, "m": m, "Z": design(F, tr, M1_COLS), "blk": blk}


def fit_fold(X, y, prep, exploratory=True):
    tr, Z, blk, F = prep["tr"], prep["Z"], prep["blk"], prep["F"]
    ok = tr & ~np.isnan(y)
    oof_by_c = [np.full(len(X), np.nan) for _ in CS]
    for i in range(5):
        fit, beta = ok & (blk != i), None
        for j, C in enumerate(CS):  # warm start along the C grid; same optimum, fewer Newton steps
            beta = fit_l2(Z[fit], y[fit], C, beta=beta)
            oof_by_c[j][blk == i] = pred(Z[blk == i], beta)
    cv_ll = [ll(o[ok], y[ok]).mean() for o in oof_by_c]
    j = int(np.argmin(cv_ll))
    C = CS[j]
    beta = fit_l2(Z[ok], y[ok], C)
    out = {**prep, "C": C, "cv_ll": cv_ll, "beta": beta, "ok": ok, "p_m1": pred(Z, beta), "oof": oof_by_c[j]}
    # stop rule 2: M1 beats B1 on batter-game AUC in the inner CV (rows where B1 is defined)
    b1 = X.base_b1.to_numpy()
    k = ok & ~np.isnan(b1)
    out["auc_m1_cv"], out["auc_b1_cv"] = stats.auc(out["oof"][k], y[k]), stats.auc(b1[k], y[k])
    # stop rule 3: training mean predicted P(hit), slots 1-4 minus slot 9
    sl = X.slot.to_numpy()
    out["wiring"] = out["p_m1"][tr & (sl <= 4)].mean() - out["p_m1"][tr & (sl == 9)].mean()
    if exploratory:
        from sklearn.ensemble import HistGradientBoostingClassifier
        for name, cols in (("M1 + market", M1_COLS + MARKET), ("M1 + H (B2)", M1_COLS + HCOLS)):
            Zx = design(F, tr, cols)
            out[name] = pred(Zx, fit_l2(Zx[ok], y[ok], C))
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                            l2_regularization=1.0, random_state=20261026)
        A = F[M1_COLS].to_numpy(float)
        out["gradient boosting"] = gb.fit(A[ok], y[ok]).predict_proba(A)[:, 1]
        out["drop"] = {}
        for g, gc in GROUPS.items():
            cols = [c for c in M1_COLS if c not in gc]
            Zd = design(F, tr, cols)
            out["drop"][g] = pred(Zd, fit_l2(Zd[ok], y[ok], C))
    return out


def run_fold(X, y, train, test, exploratory=True):
    return fit_fold(X, y, prepare_fold(X, train, test), exploratory)


# ---------------------------------------------------------------- picks
def pool(X):
    """Daily pick pool: a batter in both games of a doubleheader counts only in game 1 (plan Section 2)."""
    return ~(X.duplicated(["date", "id"], keep=False) & X.game_number.ne(1)).to_numpy()


def top(X, score, mask, y, k=1):
    """Top-k per date by score, ties to lower gamePk then lower player id. res: 1 hit, 0 loss, nan pass."""
    d = pd.DataFrame({"date": X.date, "game_pk": X.game_pk, "id": X.id, "s": score, "y": y, "season": X.season,
                      "row": np.arange(len(X))})[mask & ~np.isnan(score)]
    d = d.sort_values(["date", "s", "game_pk", "id"], ascending=[True, False, True, True]).groupby("date").head(k)
    d["rank"] = d.groupby("date").cumcount() + 1
    return d


def rate(res):
    res = res[~np.isnan(res)]
    return (res.sum() / len(res) if len(res) else np.nan), int(res.sum()), len(res)


def wil(res):
    _, k, n = rate(np.asarray(res, float))
    return wilson_s(k, n)


def paired(a: pd.Series, b: pd.Series, seed):
    """Mean of a - b over dates where both are decided, date bootstrap (2,000)."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (j.a - j.b).to_numpy()
    W = boot_counts(len(d), seed)
    return d.mean(), ci(W @ d / W.sum(1)), len(d)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recal_map(p, res):
    """Logistic recalibration a + b*logit(p) on decided picks, slope constrained to [0, 1] (addendum A2 a): an
    out-of-range slope is set to the nearer bound and the intercept refitted; b = 0 is a constant."""
    k = ~np.isnan(res)
    x, yy = logit(p[k]), res[k]
    a, b = fit_l2(np.column_stack([np.ones(len(x)), x]), yy, 1e10)
    if not 0 <= b <= 1:
        b, a = min(max(b, 0.0), 1.0), 0.0
        for _ in range(100):
            q = 1 / (1 + np.exp(-(a + b * x)))
            step = (q - yy).sum() / (q * (1 - q)).sum()
            a -= step
            if abs(step) < 1e-12:
                break
    return np.array([a, b])


def _check_recal():
    p = np.linspace(.70, .80, 400)
    y_rev = (p < .75).astype(float)  # outcomes reversed against p: unconstrained slope is negative
    a, b = recal_map(p, y_rev)
    assert b == 0 and np.isclose(1 / (1 + np.exp(-a)), y_rev.mean()), (a, b)
    r = np.random.default_rng(0)
    q = r.uniform(.6, .9, 5000)
    a, b = recal_map(q, (r.random(5000) < q).astype(float))
    assert 0.6 < b <= 1 and abs(a) < 0.3, (a, b)  # well calibrated: slope near 1


def apply_map(ab, p):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * logit(p))))


# ---------------------------------------------------------------- method bundle per fold
def methods(X, y, f, mask):
    """Daily top picks for M1 and B2 (test season) and their training-season counterparts (M1 out-of-fold, B2 own)."""
    te, tr = f["te"], f["tr"]
    res = {}
    for name, sc_te, sc_tr in (("M1", f["p_m1"], f["oof"]), ("B2", X.base_b2.to_numpy(), X.base_b2.to_numpy())):
        t = top(X, np.where(te, sc_te, np.nan), mask, y, 2)
        r = top(X, np.where(tr, sc_tr, np.nan), mask, y, 1)
        t1 = t[t["rank"].eq(1)].set_index("date")
        ab = recal_map(r.s.to_numpy(), r.y.to_numpy())
        r["cal"] = apply_map(ab, r.s.to_numpy())
        t1["cal"] = apply_map(ab, t1.s.to_numpy())
        t2 = t[t["rank"].eq(2)].set_index("date").reindex(t1.index)
        t1["cal2"] = apply_map(ab, t2.s.to_numpy())
        t1["y2"] = t2.y
        # secondary ("deployable"): training-year thresholds on the raw top-pick probability (plan T2)
        q = {lvl: np.quantile(r.s, lvl) for lvl in (0.8, 0.5)}
        t1["top20_thr"], t1["top50_thr"] = t1.s >= q[0.8], t1.s >= q[0.5]
        # primary (addendum A2 b): within-season rank by calibrated probability, ties by raw probability, then date
        t1["rk"] = t1.assign(d=t1.index).sort_values(["cal", "s", "d"], ascending=[False, False, True]).assign(
            rk=lambda d: np.arange(1, len(d) + 1)).rk.reindex(t1.index)
        D = len(t1)  # each fold's test set is one season
        t1["top20"], t1["top50"], t1["top10"] = t1.rk <= round(.2 * D), t1.rk <= round(.5 * D), t1.rk <= round(.1 * D)
        res[name] = {"t1": t1, "train": r, "map": ab, "q": q,
                     "days_train": r.groupby("season").size().mean()}
    t1b1 = top(X, np.where(te, X.base_b1.to_numpy(), np.nan), mask, y, 1).set_index("date")
    res["B1"] = {"t1": t1b1}
    return res


# ---------------------------------------------------------------- T4 simulation
def sim(p1, pass1, p2=None, pass2=None, dd=None, n=100_000, seed=20261030):
    """Share of n seasons with a streak >= 57. Day j: pass -> no change; else success w.p. p (both picks on DD days)."""
    rng = np.random.default_rng(seed)
    streak, done = np.zeros(n, np.int32), np.zeros(n, bool)
    for j in range(len(p1)):
        picks = [(p1[j], pass1[j])] + ([(p2[j], pass2[j])] if dd is not None and dd[j] else [])
        live = [p for p, ps in picks if not ps]  # a 0-AB pick is a pass; a DD day with one pass plays the other pick
        if not live:
            continue
        s = np.ones(n, bool)
        for p in live:
            s &= rng.random(n) < p
        streak = np.where(s, streak + len(live), 0)
        done |= streak >= 57
    return done.mean()


def _check_sim():
    one, no = np.ones(60), np.zeros(60, bool)
    assert sim(one[:57], no[:57], n=10) == 1 and sim(one[:56], no[:56], n=10) == 0
    assert sim(one[:40], no[:40], one[:40], no[:40], np.r_[np.ones(17, bool), np.zeros(23, bool)], n=10) == 1  # 17*2 + 23 = 57
    assert sim(one, np.r_[no[:59], True], n=10) == 1 and sim(np.full(60, .5), no, n=1000) == 0


def shift_to(p, target):
    """Shift logit(p) by one constant so mean(p) equals target (bisection)."""
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit(p) + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(logit(p) + (lo + hi) / 2)))


def t4_row(pl, ddm):
    p1, ps1 = pl.cal.to_numpy(), pl.y.isna().to_numpy()
    p2, ps2 = pl.cal2.to_numpy(), pl.y2.isna().to_numpy()
    _, kk, nn = rate(pl.y.to_numpy(float))
    lo, hi = stats.wilson(kk, nn) if nn else (np.nan, np.nan)
    zero = len(pl) < 57
    single = 0.0 if zero else sim(p1, ps1)
    rng_ = (0.0, 0.0) if zero else (sim(shift_to(p1, lo), ps1), sim(shift_to(p1, hi), ps1))
    dd = 0.0 if len(pl) + ddm.sum() < 57 else sim(p1, ps1, p2, ps2, ddm)
    return {"played days": len(pl), "mean recal p": f"{p1.mean():.4f}" if len(pl) else "–", "realised (Wilson)": wil(pl.y),
            "P(57)": one_in(single) + (" (zero by construction)" if zero else ""),
            "range (Wilson ends)": f"{one_in(rng_[0])} .. {one_in(rng_[1])}", "Double Down": one_in(dd)}


def t2_tier(m1, b2, col, seed):
    """T2 statistic: each method's success rate on its own tier days minus the other's; date bootstrap over the union."""
    dates = m1.index.union(b2.index)
    q1 = m1[col].reindex(dates, fill_value=False).to_numpy() & m1.y.reindex(dates).notna().to_numpy()
    q2 = b2[col].reindex(dates, fill_value=False).to_numpy() & b2.y.reindex(dates).notna().to_numpy()
    h1, h2 = m1.y.reindex(dates).fillna(0).to_numpy() * q1, b2.y.reindex(dates).fillna(0).to_numpy() * q2
    W = boot_counts(len(dates), seed)
    bt = (W @ h1) / (W @ q1) - (W @ h2) / (W @ q2)
    return {"est": h1.sum() / q1.sum() - h2.sum() / q2.sum(), "ci": ci(bt), "n1": int(m1[col].sum()), "n2": int(b2[col].sum()),
            "k1": int(h1.sum()), "d1": int(q1.sum()), "w1": wil(m1.y[m1[col]]), "w2": wil(b2.y[b2[col]]),
            "overlap": int((m1[col].reindex(dates, fill_value=False) & b2[col].reindex(dates, fill_value=False)).sum())}


def one_in(x):
    return "0 of 100,000 (< 1 in 100,000)" if x == 0 else f"1 in {1 / x:,.0f}"


# ---------------------------------------------------------------- null study (fake outcomes only)
_W = {}


def _null_init():
    global NBOOT
    NBOOT = 500
    X = load(False)  # never the real hit column
    _W.update(X=X, mask=pool(X), preps={(tuple(tr), te): prepare_fold(X, tr, te) for tr, te in dict.fromkeys(WF + LOSO)})


def _null_seed(seed):
    X, mask = _W["X"], _W["mask"]
    y, _ = fake_outcomes(X, "null", seed)
    fits = {k: fit_fold(X, y, pr, exploratory=False) for k, pr in _W["preps"].items()}
    out = {"seed": seed}
    for d, folds in (("WF", WF), ("LOSO", LOSO)):
        mm = [methods(X, y, fits[(tuple(tr), te)], mask) for tr, te in folds]
        m1, b2 = pd.concat([m["M1"]["t1"] for m in mm]), pd.concat([m["B2"]["t1"] for m in mm])
        est, c, _ = paired(m1.y, b2.y, SEED + 1)
        out.update({f"{d} M1 top-1": rate(m1.y.to_numpy(float))[0], f"{d} B2 top-1": rate(b2.y.to_numpy(float))[0],
                    f"{d} T1": est, f"{d} T1 excludes 0": not c[0] <= 0 <= c[1], f"{d} T1 Confirmed": c[0] > 0})
        for version, col in (("primary", "top20"), ("secondary", "top20_thr")):
            t = t2_tier(m1, b2, col, SEED + 2)
            k, n = t["k1"], t["d1"]
            out.update({f"{d} {version} M1 top-20% rate": k / n if n else np.nan, f"{d} {version} M1 top-20% days": n,
                        f"{d} {version} z": (k - 0.75 * n) / np.sqrt(n * 0.1875) if n else np.nan,
                        f"{d} {version} T2 excludes 0": not t["ci"][0] <= 0 <= t["ci"][1], f"{d} {version} T2 Confirmed": t["ci"][0] > 0})
    return out


def null_study(procs=8, seeds=range(1001, 1201)):
    from multiprocessing import Pool
    t0 = time.perf_counter()
    with Pool(procs, initializer=_null_init) as pool_:
        R = pd.DataFrame(pool_.map(_null_seed, list(seeds), chunksize=1)).sort_values("seed")
    R.to_csv(OUT / "btslab_null_study_seeds.csv", index=False)
    say("# Beat the Streak lab — null study on FAKE outcomes (independent Bernoulli(0.75); real hits never loaded)")
    say(f"seeds {min(seeds)}-{max(seeds)} ({len(R)}), {procs} processes, {NBOOT_STUDY} bootstrap resamples; skipped: coefficient "
        "bootstrap, T4, T5, exploratory models. Per-seed rows: analysis/out/btslab_null_study_seeds.csv")
    rows = []
    for d in ("WF", "LOSO"):
        for v in ("primary", "secondary"):
            z = R[f"{d} {v} z"]
            rows.append({"design": d, "tiers": v, "z mean": f"{z.mean():+.3f}", "z SD": f"{z.std(ddof=1):.3f}",
                         "M1 top-20% days (median)": int(R[f"{d} {v} M1 top-20% days"].median()),
                         "T2 Confirmed": f"{R[f'{d} {v} T2 Confirmed'].mean():.3f}", "T2 excludes 0": f"{R[f'{d} {v} T2 excludes 0'].mean():.3f}"})
    say("\n## M1 top-20% success: z-score against 0.75 (realised decided days), and T2")
    say(md_table(pd.DataFrame(rows)))
    say("\n## T1 and top-1 rates")
    say(md_table(pd.DataFrame([{"design": d, "M1 top-1 mean": f"{R[f'{d} M1 top-1'].mean():.4f}", "B2 top-1 mean": f"{R[f'{d} B2 top-1'].mean():.4f}",
                                "T1 mean": f"{R[f'{d} T1'].mean():+.4f}", "T1 SD": f"{R[f'{d} T1'].std(ddof=1):.4f}",
                                "T1 Confirmed": f"{R[f'{d} T1 Confirmed'].mean():.3f}", "T1 excludes 0": f"{R[f'{d} T1 excludes 0'].mean():.3f}"}
                               for d in ("WF", "LOSO")])))
    say("\n## Distribution of M1's top-20% success rate")
    qs = [0, .025, .1, .25, .5, .75, .9, .975, 1]
    say(md_table(pd.DataFrame([{"design / tiers": f"{d} {v}", **{f"q{q:g}": f"{R[f'{d} {v} M1 top-20% rate'].quantile(q):.3f}" for q in qs}}
                               for d in ("WF", "LOSO") for v in ("primary", "secondary")])))
    say("\n## Pass criteria (set before the study): z mean within ±0.15, z SD 0.85-1.15, false-confirmation rate <= 8%")
    checks = []
    for d in ("WF", "LOSO"):
        for v in ("primary", "secondary"):
            z = R[f"{d} {v} z"]
            checks += [(f"{d} {v} z mean", abs(z.mean()) <= 0.15), (f"{d} {v} z SD", 0.85 <= z.std(ddof=1) <= 1.15)]
    checks += [("WF T1 Confirmed rate", R["WF T1 Confirmed"].mean() <= 0.08),
               ("WF primary T2 Confirmed rate", R["WF primary T2 Confirmed"].mean() <= 0.08)]
    for name, ok in checks:
        say(f"- {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(ok for _, ok in checks)
    say(f"\nNULL STUDY: {'PASS' if passed else 'FAIL - stop, do not tune'}")
    say(f"runtime {time.perf_counter() - t0:.0f} s")
    (OUT / "btslab_null_study_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return passed


NBOOT_STUDY = 500


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true", help="use the real hit outcomes (the real run)")
    g.add_argument("--fake", choices=["null", "planted"], help="seeded fake outcomes on AB>=1 rows")
    g.add_argument("--null-study", action="store_true", help="200 fake null seeds (1001-1200), fast settings, in parallel")
    ap.add_argument("--procs", type=int, default=8, help="processes for --null-study")
    ap.add_argument("--ignore-stops", action="store_true", help="fake runs only: continue past stop rules")
    ap.add_argument("--fake-seed", type=int, help="fake runs only: generator seed (default 777 null, 778 planted)")
    a = ap.parse_args()
    if a.null_study:
        sys.exit(0 if null_study(a.procs) else 1)
    if not a.real and not a.fake:
        sys.exit("refusing to run: choose --fake null|planted (self-test), --null-study, or --real (uses the real hit column)")
    if a.real and (a.ignore_stops or a.fake_seed is not None):
        sys.exit("--ignore-stops and --fake-seed are not allowed with --real")
    t0 = time.perf_counter()
    sfx = "" if a.fake_seed is None else f"_seed{a.fake_seed}"
    log_path = OUT / ("btslab_results_log.txt" if a.real else f"btslab_selftest_{a.fake}{sfx}_log.txt")
    _check_solver()
    _check_sim()
    _check_m_bf()
    _check_recal()
    bf._self_check()
    X = load(a.real)
    if a.real:
        y = np.where(X.ab >= 1, (X.h >= 1).astype(float), np.nan)
        desc = "REAL box-score hits (H >= 1) on AB >= 1 rows"
    else:
        y, desc = fake_outcomes(X, a.fake, a.fake_seed)
    say(f"# Beat the Streak lab — {'RESULTS (real outcomes)' if a.real else 'SELF-TEST on FAKE outcomes: ' + a.fake}")
    say(f"plan f10661f + addendum A1 6d9cb3c; table {bf.OUT_PARQUET.name}: {len(X)} rows; outcome: {desc}")
    say(f"countable rows (AB >= 1): {int((~np.isnan(y)).sum())}; passes (AB = 0): {int(np.isnan(y).sum())}")
    mask = pool(X)

    # ---- folds
    say("\n## Folds (penalty by inner CV: 5 contiguous date blocks, 12 C values 1e-3..1e2, mean log loss)")
    folds = {"WF": [run_fold(X, y, tr, te) for tr, te in WF], "LOSO": [run_fold(X, y, tr, te, exploratory=False) for tr, te in LOSO]}
    rows = [{"design": k, "train": "-".join(map(str, f["train"])), "test": f["test"], "C": f"{f['C']:.4g}",
             "inner-CV log loss": f"{min(f['cv_ll']):.5f}", "AUC M1 (CV)": f"{f['auc_m1_cv']:.4f}",
             "AUC B1 (CV)": f"{f['auc_b1_cv']:.4f}", "wiring: slots 1-4 minus 9": f"{f['wiring']:+.4f}"}
            for k, fs in folds.items() for f in fs]
    say(md_table(pd.DataFrame(rows)))
    say("inner-CV mean log loss by C (walk-forward folds): " + " | ".join(
        f"test {f['test']}: " + ", ".join(f"{c_:.3g}:{v:.5f}" for c_, v in zip(CS, f["cv_ll"])) for f in folds["WF"]))
    say("shrinkage m per walk-forward fold: " + "; ".join(f"{f['test']}: " + ", ".join(f"{k}={v:.3g}" for k, v in f["m"].items())
                                                         for f in folds["WF"]))

    # ---- stop rules (Section 10)
    say("\n## Stop rules")
    s2 = all(f["auc_m1_cv"] > f["auc_b1_cv"] for f in folds["WF"])
    s3 = all(f["wiring"] > 0 for f in folds["WF"])
    say(f"- Rule 2, M1 beats B1 on batter-game AUC in the training inner CV, each walk-forward fold: {'PASS' if s2 else 'FAIL'}")
    say(f"- Rule 3, training mean P(hit) slots 1-4 minus slot 9 > 0, each walk-forward fold: {'PASS' if s3 else 'FAIL'}")
    if not (s2 and s3):
        if not a.ignore_stops:
            say("STOP: a stop rule failed. No test-season result is computed (plan Section 10).")
            log_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
            sys.exit(2)
        say("(--ignore-stops, fake outcomes: continuing so the rest of the pipeline is exercised)")

    M = {k: [methods(X, y, f, mask) for f in fs] for k, fs in folds.items()}
    cat = lambda design, meth, key="t1": pd.concat([mm[meth][key] for mm in M[design]])

    # ---- T1
    say("\n## T1 — overall top-1 success, M1 minus B2 (confirmatory; date bootstrap 2,000, seed 20261027)")
    t1 = {}
    for d in ("WF", "LOSO"):
        m1, b2, b1 = cat(d, "M1"), cat(d, "B2"), cat(d, "B1")
        t1[d] = paired(m1.y, b2.y, SEED + 1)
        rows = []
        for name, t in (("M1", m1), ("B2", b2), ("B1", b1)):
            for s in ["pooled"] + sorted(t.season.unique()):
                tt = t if s == "pooled" else t[t.season.eq(s)]
                rows.append({"design": d, "method": name, "set": s, "dates": len(tt), "passes": int(tt.y.isna().sum()),
                             "top-1 rate (Wilson)": wil(tt.y)})
        say(md_table(pd.DataFrame(rows)))
        est, c, n = t1[d]
        e1, c1, n1 = paired(m1.y, b1.y, SEED + 1)
        say(f"{d}: M1 − B2 = {est:+.4f} (95% {c[0]:+.4f} to {c[1]:+.4f}), {n} paired dates; M1 − B1 = {e1:+.4f} ({c1[0]:+.4f} to {c1[1]:+.4f})")
        for s in sorted(m1.season.unique()):
            es, cs, ns = paired(m1.y[m1.season.eq(s)], b2.y[b2.season.eq(s)], SEED + 1)
            say(f"   {s}: M1 − B2 = {es:+.4f} ({cs[0]:+.4f} to {cs[1]:+.4f}), n {ns}")
    est, c, _ = t1["WF"]
    t1_conf = c[0] > 0
    say(f"**T1 rule (walk-forward): {'Confirmed' if t1_conf else 'not confirmed'}** (Confirmed only if the interval lies entirely above 0)")

    # ---- T2
    say("\n## T2 — day tiers (confirmatory at top 20%, primary version; date bootstrap, seed 20261028)")
    say("Primary (addendum A2 b): exactly the top 20% / 50% of each season's days by within-season rank of the calibrated "
        "top-pick probability. Secondary (deployable, no rule): training-year thresholds applied unchanged.")
    t2 = {}
    for d in ("WF", "LOSO"):
        m1, b2 = cat(d, "M1"), cat(d, "B2")
        for version, sfx in (("primary", ""), ("secondary", "_thr")):
            for tier in ("top20", "top50"):
                r_ = t2_tier(m1, b2, tier + sfx, SEED + 2)
                t2[(d, version, tier)] = r_
                say(f"{d} {version} {tier}: M1 days {r_['n1']} of {len(m1)} ({r_['n1'] / len(m1):.2f}), rate {r_['w1']}; "
                    f"B2 days {r_['n2']} of {len(b2)} ({r_['n2'] / len(b2):.2f}), rate {r_['w2']}; overlap {r_['overlap']}; "
                    f"M1 − B2 = {r_['est']:+.4f} ({r_['ci'][0]:+.4f} to {r_['ci'][1]:+.4f})")
        say(f"{d} overall rates: M1 {wil(m1.y)}, B2 {wil(b2.y)}")
    e2, c2 = t2[("WF", "primary", "top20")]["est"], t2[("WF", "primary", "top20")]["ci"]
    t2_conf = c2[0] > 0
    say(f"**T2 rule (walk-forward, primary, top 20%): {'Confirmed' if t2_conf else 'not confirmed'}**")
    say("Multiplicity (plan): T1 and T2 form the confirmatory family; about 0.049 if independent, at most 0.05.")

    # ---- T3
    say("\n## T3 — calibration, cross-fitted recalibration (walk-forward)")
    rows = []
    for name in ("M1", "B2"):
        t = cat("WF", name)
        for tier, sel in (("all top-1", np.ones(len(t), bool)), ("top 50% days", t.top50.to_numpy()), ("top 20% days", t.top20.to_numpy())):
            tt = t[sel & t.y.notna().to_numpy()]
            rows.append({"method": name, "set": tier, "n decided": len(tt), "mean raw": f"{tt.s.mean():.4f}",
                         "mean recalibrated": f"{tt.cal.mean():.4f}", "realised (Wilson)": wil(tt.y)})
    say(md_table(pd.DataFrame(rows)))
    for f, mm in zip(folds["WF"], M["WF"]):
        say(f"recalibration map fold test {f['test']}: M1 a={mm['M1']['map'][0]:+.3f} b={mm['M1']['map'][1]:.3f}; "
            f"B2 a={mm['B2']['map'][0]:+.3f} b={mm['B2']['map'][1]:.3f}")
    te = np.zeros(len(X), bool)
    p_te = np.full(len(X), np.nan)
    for f in folds["WF"]:
        te |= f["te"]
        p_te[f["te"]] = f["p_m1"][f["te"]]
    k = te & ~np.isnan(y)
    edges = [0, .5, .6, .65, .7, .75, .8, .85, 1.0001]
    labels = ["<0.50", "0.50–0.60", "0.60–0.65", "0.65–0.70", "0.70–0.75", "0.75–0.80", "0.80–0.85", "≥0.85"]
    b = pd.cut(p_te[k], edges, right=False, labels=labels)
    cal = pd.DataFrame({"b": b, "p": p_te[k], "y": y[k]}).groupby("b", observed=False).agg(n=("y", "size"), pred=("p", "mean"), k=("y", "sum"))
    say("M1 batter-game calibration, test seasons, R2 bins:")
    say(md_table(pd.DataFrame([{"bin": i, "n": int(r.n), "mean predicted": f"{r.pred:.4f}" if r.n else "–",
                                "realised (Wilson)": wilson_s(int(r.k), int(r.n)) + ("" if r.n >= 30 else " (n<30)")}
                               for i, r in cal.iterrows()])))

    # ---- T4
    say("\n## T4 — chance of a 57-game streak in a season (100,000 seasons, seed 20261030; Streak Saver ignored)")
    for version in ("primary", "secondary"):
        say(f"\n{version}: " + ("exactly N days per season by within-season rank; Double Down on the top 10% of days by rank"
                                if version == "primary" else "training-year thresholds; realised N; below 57 is zero by construction"))
        rows = []
        for f, mm in zip(folds["WF"], M["WF"]):
            for name in ("M1", "B2"):
                t, r = mm[name]["t1"].sort_index(), mm[name]["train"]
                q10 = np.quantile(r.cal, 0.9)
                for N in NS:
                    if version == "primary":
                        pl = t if N is None else t[t.rk <= N]
                        ddm = pl.top10.to_numpy()
                    else:
                        thr = -np.inf if N is None else np.quantile(r.cal, 1 - N / mm[name]["days_train"])
                        pl = t[t.cal >= thr]
                        ddm = pl.cal.to_numpy() >= q10
                    rows.append({"test": f["test"], "method": name, "N": "all" if N is None else N, **t4_row(pl, ddm)})
        say(md_table(pd.DataFrame(rows)))
    say("Primary: N = 100, primary version.")

    # ---- T5 and exploratory models
    say("\n## T5 — drop-one-group value (EXPLORATORY; same C, date bootstrap 2,000, seed 20261031)")
    full = cat("WF", "M1")
    rows = []
    for g in GROUPS:
        tops, lls, lld = [], [], []
        for f in folds["WF"]:
            pdp = np.where(f["te"], f["drop"][g], np.nan)
            tops.append(top(X, pdp, mask, y, 1).set_index("date"))
            kk = f["te"] & ~np.isnan(y)
            lls.append(pd.DataFrame({"date": X.date[kk], "d": ll(f["drop"][g][kk], y[kk]) - ll(f["p_m1"][kk], y[kk])}))
        dt = pd.concat(tops)
        e, c, n = paired(dt.y, full.y, SEED + 5)
        L = pd.concat(lls).groupby("date").d.agg(["sum", "size"])
        W = boot_counts(len(L), SEED + 5)
        rows.append({"dropped group": g, "Δ top-1 rate (drop − full)": f"{e:+.4f} ({c[0]:+.4f} to {c[1]:+.4f})",
                     "Δ log loss (drop − full)": f"{L['sum'].sum() / L['size'].sum():+.6f} ({ci((W @ L['sum'].to_numpy()) / (W @ L['size'].to_numpy()))[0]:+.6f} "
                                                 f"to {ci((W @ L['sum'].to_numpy()) / (W @ L['size'].to_numpy()))[1]:+.6f})"})
    say(md_table(pd.DataFrame(rows)))
    say("\nEXPLORATORY models (walk-forward; seed 20261033 for their intervals):")
    rows = []
    for name in ("M1 + market", "M1 + H (B2)", "gradient boosting"):
        tops, l = [], []
        for f in folds["WF"]:
            tops.append(top(X, np.where(f["te"], f[name], np.nan), mask, y, 1).set_index("date"))
            kk = f["te"] & ~np.isnan(y)
            l.append(ll(f[name][kk], y[kk]))
        dt = pd.concat(tops)
        e, c, n = paired(dt.y, cat("WF", "B2").y, 20261033)
        pp = np.concatenate([f[name][f["te"] & ~np.isnan(y)] for f in folds["WF"]])
        yy = np.concatenate([y[f["te"] & ~np.isnan(y)] for f in folds["WF"]])
        rows.append({"model": name, "top-1 rate": wil(dt.y), "minus B2": f"{e:+.4f} ({c[0]:+.4f} to {c[1]:+.4f})",
                     "batter-game log loss": f"{np.concatenate(l).mean():.5f}", "batter-game AUC": f"{stats.auc(pp, yy):.4f}"})
    kk = te & ~np.isnan(y)
    rows.append({"model": "M1 (reference)", "top-1 rate": wil(full.y), "minus B2": f"{t1['WF'][0]:+.4f}",
                 "batter-game log loss": f"{ll(p_te[kk], y[kk]).mean():.5f}", "batter-game AUC": f"{stats.auc(p_te[kk], y[kk]):.4f}"})
    b2v = X.base_b2.to_numpy()
    rows.append({"model": "B2 (comparator)", "top-1 rate": wil(cat("WF", "B2").y), "minus B2": "–",
                 "batter-game log loss": f"{ll(b2v[kk], y[kk]).mean():.5f}", "batter-game AUC": f"{stats.auc(b2v[kk], y[kk]):.4f}"})
    say(md_table(pd.DataFrame(rows)))
    say("Not run: the R2 simulator P(hit) (exploratory H) is not in the feature table (Step 1 notes, item 21).")

    # ---- coefficients with date-bootstrap intervals, full walk-forward training fit 2022-24
    say("\n## M1 standardised coefficients, fit on 2022–24 (date bootstrap 2,000, seed 20261032)")
    f = folds["WF"][1]
    ok, Z, beta = f["ok"], f["Z"], f["beta"]
    _, di = np.unique(X.date.to_numpy()[ok], return_inverse=True)
    Wd = boot_counts(di.max() + 1, SEED + 6)
    Zo, yo = Z[ok], y[ok]
    bs = np.array([fit_l2(Zo, yo, f["C"], w[di].astype(float), beta) for w in Wd])
    coef = pd.DataFrame({"coef": beta[1:], "lo": np.quantile(bs[:, 1:], .025, axis=0), "hi": np.quantile(bs[:, 1:], .975, axis=0)}, index=M1_COLS)
    coef["group"] = [g for g, gc in GROUPS.items() for _ in gc]
    say(md_table(coef.reset_index().rename(columns={"index": "feature"}).assign(
        coef=lambda d: d.coef.map("{:+.4f}".format), lo=lambda d: d.lo.map("{:+.4f}".format), hi=lambda d: d.hi.map("{:+.4f}".format))))
    inc0 = lambda c_: coef.loc[c_, "lo"] <= 0 <= coef.loc[c_, "hi"]
    slot_max = coef.loc[GROUPS["B"], "coef"].abs().idxmax()

    # ---- William's hypotheses
    say("\n## William's hypotheses (no decision rules)")
    dsel = te & ~np.isnan(y)
    by_slot = [f"{s}: {wilson_s(int(np.nansum(y[dsel & (X.slot.to_numpy() == s)])), int((dsel & (X.slot.to_numpy() == s)).sum()))}"
               for s in range(1, 10)]
    say("W1 slots 1-4 hit more. Slot coefficients: " + ", ".join(f"{c_[-1]} {coef.loc[c_, 'coef']:+.3f}" for c_ in GROUPS["B"])
        + ". Realised hit rate by slot (test seasons): " + "; ".join(by_slot))
    say(f"W2 more walks help (Claude: small or negative). BB% coefficient {coef.loc['a_bb', 'coef']:+.4f} "
        f"({coef.loc['a_bb', 'lo']:+.4f} to {coef.loc['a_bb', 'hi']:+.4f})")
    say(f"W3 power helps, K-heavy power hurts (Claude: contact and low K% dominate, ISO little). ISO {coef.loc['a_iso', 'coef']:+.4f} "
        f"({coef.loc['a_iso', 'lo']:+.4f} to {coef.loc['a_iso', 'hi']:+.4f}); K% {coef.loc['a_k', 'coef']:+.4f} "
        f"({coef.loc['a_k', 'lo']:+.4f} to {coef.loc['a_k', 'hi']:+.4f}); group A drop in T5 above")
    say(f"W4 hitter-friendly parks. Park-factor coefficient {coef.loc['f_park', 'coef']:+.4f} ({coef.loc['f_park', 'lo']:+.4f} to {coef.loc['f_park', 'hi']:+.4f})")
    sel = np.zeros(len(X), bool)
    for ff in folds["WF"]:
        Fs = ff["F"]
        q_sp, q_bp = Fs.c_hab[ff["te"]].quantile(.75), Fs.e_bp_hab[ff["te"]].quantile(.75)
        sel |= ff["te"] & (Fs.c_hab >= q_sp).to_numpy() & (Fs.e_bp_hab >= q_bp).to_numpy()
    say(f"W5 bad starter AND bad bullpen (top quartile of both, per test season): realised {wilson_s(int(np.nansum(y[dsel & sel])), int((dsel & sel).sum()))} "
        f"vs others {wilson_s(int(np.nansum(y[dsel & ~sel])), int((dsel & ~sel).sum()))}")
    say(f"W6 bullpen usage. Fatigue pitches {coef.loc['e_fat_pitches', 'coef']:+.4f} ({coef.loc['e_fat_pitches', 'lo']:+.4f} to "
        f"{coef.loc['e_fat_pitches', 'hi']:+.4f}); relievers {coef.loc['e_fat_relievers', 'coef']:+.4f} "
        f"({coef.loc['e_fat_relievers', 'lo']:+.4f} to {coef.loc['e_fat_relievers', 'hi']:+.4f}); group E drop in T5 above")
    say(f"W7 hot hitter (Claude: small). 14-day {coef.loc['g_hab14', 'coef']:+.4f} ({coef.loc['g_hab14', 'lo']:+.4f} to {coef.loc['g_hab14', 'hi']:+.4f}); "
        f"30-day {coef.loc['g_hab30', 'coef']:+.4f} ({coef.loc['g_hab30', 'lo']:+.4f} to {coef.loc['g_hab30', 'hi']:+.4f}); group G drop in T5 above")

    # ---- Claude's expectations
    say("\n## Claude's pre-registered expectations (met / not met; no decision rule)")
    m1w = cat("WF", "M1")
    r_all, r20 = rate(m1w.y.to_numpy(float))[0], rate(m1w.y[m1w.top20].to_numpy(float))[0]
    t20 = m1w[m1w.top20 & m1w.y.notna()]
    top3 = coef.assign(a=coef.coef.abs()).drop(index=[c_ for c_ in GROUPS["B"] if c_ != slot_max]).a.nlargest(3).index
    exp = [("(a) M1 − B2 within −0.015..+0.020 and T1 not confirmed", -0.015 <= est <= 0.020 and not t1_conf,
            f"M1 − B2 {est:+.4f}; T1 {'Confirmed' if t1_conf else 'not confirmed'}"),
           ("(b) M1 top-20% rate ≥ overall + 0.05", r20 >= r_all + 0.05, f"top-20% {r20:.4f} vs overall {r_all:.4f}"),
           ("(b) M1 − B2 on top-20% days in 0..+0.04", 0 <= e2 <= 0.04, f"{e2:+.4f} ({c2[0]:+.4f} to {c2[1]:+.4f}); interval excludes 0: {not (c2[0] <= 0 <= c2[1])} (the ~15% is a probability, not scored)"),
           ("(c) top three |coef| are K%, H/AB and slot", set(top3) == {"a_k", "a_hab", slot_max}, f"top three: {', '.join(top3)}"),
           ("(c) ISO interval includes 0", inc0("a_iso"), f"ISO {coef.loc['a_iso', 'coef']:+.4f} ({coef.loc['a_iso', 'lo']:+.4f} to {coef.loc['a_iso', 'hi']:+.4f})"),
           ("(c) BB% small", None, f"BB% {coef.loc['a_bb', 'coef']:+.4f}; the plan sets no threshold, so not scored"),
           ("(d) each G coef |.| < 0.05", all(abs(coef.loc[c_, "coef"]) < 0.05 for c_ in GROUPS["G"]),
            ", ".join(f"{c_} {coef.loc[c_, 'coef']:+.4f}" for c_ in GROUPS["G"])),
           ("(e) both fatigue intervals include 0", inc0("e_fat_pitches") and inc0("e_fat_relievers"), "see W6"),
           ("(f) M1 top-20% raw predicted − realised ≥ 0.03", t20.s.mean() - t20.y.mean() >= 0.03,
            f"{t20.s.mean():.4f} − {t20.y.mean():.4f} = {t20.s.mean() - t20.y.mean():+.4f}")]
    say(md_table(pd.DataFrame([{"expectation": e_, "result": "–" if r_ is None else ("met" if r_ else "not met"), "numbers": n_}
                               for e_, r_, n_ in exp])))
    say(f"\nruntime {time.perf_counter() - t0:.0f} s")
    log_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
