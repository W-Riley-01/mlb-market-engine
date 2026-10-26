"""Run the frozen BACKTEST_V4_ANALYSIS_PLAN.md (amended) on the six v4 ledgers; writes analysis/BACKTEST_V4_RESULTS.md.
Outcomes come from the cached box scores, baselines from data/pitch_matrix.parquet (game_date < D). Read-only.
Run v4_integrity.py first."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stats
from v4_integrity import ROOT, box, load_ledgers

PLAN = ROOT / "analysis" / "BACKTEST_V4_ANALYSIS_PLAN.md"
PLAN_SHA = "4a57fcaab100b14b714dd4bdb04b96f16316a527d8758845080b2d909b27152a"
RESULTS = ROOT / "analysis" / "BACKTEST_V4_RESULTS.md"
TITLE, HASH_LINE = "# Full v4 backtest — results", "sha256:"
TRUE_STARTERS_ONLY = False  # R2 plan: drop lineup entries whose box battingOrder is not slot x 100
KEY = {}  # headline numbers and frames, read by v4r2_analysis.py
EXTRA = None  # callable(md) -> closing sections; default is an empty EXPLORATORY section
B, SEED, EPS = 2000, 20261022, 1e-6
H_EV = {"single", "double", "triple", "home_run"}
NON_AB = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"}
K_EV = {"strikeout", "strikeout_double_play"}
SEASONS = [2022, 2023, 2024, 2025]


# ---------------------------------------------------------------- helpers
def boot_counts(C, seed):
    """Cluster bootstrap as resample counts per cluster (2,000 draws)."""
    r = stats.rng(seed)
    return np.array([np.bincount(r.integers(0, C, C), minlength=C) for _ in range(B)])


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, .025), np.quantile(v, .975)


def implied(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logit(X, y, w=None):
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (w * mu * (1 - mu))[:, None]), X.T @ (w * (y - mu)))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def wauc(g, y, w, n_groups):
    """AUC with row weights; g = rank group of the score (ties share a group)."""
    wp = np.bincount(g, w * y, n_groups)
    wn = np.bincount(g, w * (1 - y), n_groups)
    below = np.cumsum(wn) - wn
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def auc_boot(p, y, cluster, seed):
    """AUC with a cluster bootstrap (clusters = games)."""
    _, g = np.unique(np.asarray(p), return_inverse=True)
    y = np.asarray(y, float)
    _, ci_idx = np.unique(np.asarray(cluster), return_inverse=True)
    n = g.max() + 1
    est = wauc(g, y, np.ones(len(y)), n)
    W = boot_counts(ci_idx.max() + 1, seed)
    return est, ci([wauc(g, y, w[ci_idx], n) for w in W])


def wcorr(x, y, w):
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def wilson_s(k, n):
    lo, hi = stats.wilson(k, n) if n else (np.nan, np.nan)
    return f"{k / n:.3f} ({lo:.3f}–{hi:.3f})" if n else "–"


def p_streak(p, n=180, run=57):
    """P(at least one run of >= `run` successes in n independent trials)."""
    st = np.zeros(run)
    st[0], done = 1.0, 0.0
    for _ in range(n):
        new = np.zeros(run)
        new[0] = st.sum() * (1 - p)
        new[1:] = st[:-1] * p
        done += st[-1] * p
        st = new
    return done


def f3(x):
    return f"{x:.3f}"


def fci(est, c, d=3):
    return f"{est:.{d}f} ({c[0]:.{d}f} to {c[1]:.{d}f})"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------- data
def outcomes(L):
    games, bats = [], []
    for r in L.itertuples():
        bt = box(r.game_pk)["teams"]
        g = {"game_pk": r.game_pk}
        for s in ("away", "home"):
            g[f"{s}_runs_box"] = bt[s]["teamStats"]["batting"]["runs"]
            sa = getattr(r, f"{s}_starter_actual")
            g[f"{s}_sp_k_box"] = bt[s]["players"][f"ID{int(sa)}"]["stats"]["pitching"].get("strikeOuts")
        games.append(g)
        for b in json.loads(r.batters):
            st = bt[b["side"]]["players"][f"ID{b['id']}"]["stats"].get("batting", {})
            bats.append({"game_pk": r.game_pk, "side": b["side"], "slot": b["slot"], "id": b["id"],
                         "p_1h": b["p_1h"], "p_1hr": b["p_1hr"],
                         "bo": bt[b["side"]]["players"][f"ID{b['id']}"].get("battingOrder"), "ab": st.get("atBats", 0), "h": st.get("hits", 0),
                         "pa": st.get("plateAppearances", 0), "hr": st.get("homeRuns", 0)})
    return pd.DataFrame(games), pd.DataFrame(bats)


class Window:
    """Sums over the trailing window [D - 365 days, D) of a daily table."""

    def __init__(self, df, keys):
        self.df = df.sort_values("game_date").reset_index(drop=True)
        self.d = self.df.game_date.to_numpy()
        self.keys = keys

    def at(self, D):
        lo = np.searchsorted(self.d, np.datetime64(D - pd.Timedelta(days=365)), "left")
        hi = np.searchsorted(self.d, np.datetime64(D), "left")
        sub = self.df.iloc[lo:hi].drop(columns="game_date")
        return sub.groupby(self.keys).sum() if self.keys else sub.sum()


def baseline_tables():
    pm = pd.read_parquet(ROOT / "data" / "pitch_matrix.parquet",
                         columns=["game_pk", "game_date", "inning_topbot", "at_bat_number", "pitcher", "batter", "events"])
    E = pm[pm.events.notna()].copy()  # one row per plate appearance (non-null events)
    E["game_date"] = pd.to_datetime(E.game_date)
    E["bat_side"] = np.where(E.inning_topbot.eq("Bot"), "home", "away")
    E = E.sort_values(["game_pk", "at_bat_number"])
    E["hit"] = E.events.isin(H_EV).astype(int)
    E["ab"] = (~E.events.isin(NON_AB)).astype(int)
    E["k"] = E.events.isin(K_EV).astype(int)
    E["pa"] = 1
    E["vs_sp"] = (E.pitcher == E.groupby(["game_pk", "bat_side"]).pitcher.transform("first")).astype(int)
    first = E.drop_duplicates(["game_pk", "bat_side", "batter"]).copy()
    first["slot"] = first.groupby(["game_pk", "bat_side"]).cumcount() + 1
    first = first[first.slot <= 9][["game_pk", "bat_side", "batter", "slot"]]
    S = E.merge(first, on=["game_pk", "bat_side", "batter"])
    slot_day = S.groupby(["game_date", "slot", "bat_side"]).agg(
        pa=("pa", "sum"), vs=("vs_sp", "sum"), games=("game_pk", "nunique")).reset_index()
    sp = E[E.vs_sp.eq(1)].groupby(["game_pk", "bat_side"]).agg(
        pitcher=("pitcher", "first"), game_date=("game_date", "first"), k=("k", "sum")).reset_index()
    start_day = sp.assign(starts=1).groupby(["pitcher", "game_date"])[["starts", "k"]].sum().reset_index()
    return {
        "bat": Window(E.groupby(["batter", "game_date"])[["pa", "hit", "ab"]].sum().reset_index(), ["batter"]),
        "pit": Window(E.groupby(["pitcher", "game_date"])[["pa", "hit"]].sum().reset_index(), ["pitcher"]),
        "lg": Window(E.groupby("game_date")[["pa", "hit"]].sum().reset_index(), []),
        "slot": Window(slot_day, ["slot", "bat_side"]),
        "start": Window(start_day, ["pitcher"]),
    }


def add_baselines(P, K, T):
    """B1, B2 for every batter-game row in P; K-per-start baseline for every start row in K."""
    outP, outK = [], []
    for ds in sorted(set(P.date) | set(K.date)):
        D = pd.Timestamp(ds)
        lg = T["lg"].at(D)
        Lr = lg.hit / lg.pa
        bat, pit, slot, st = T["bat"].at(D), T["pit"].at(D), T["slot"].at(D), T["start"].at(D)
        p = P[P.date.eq(ds)].copy()
        bb = bat.reindex(p.id.to_numpy())
        p["bat_pa"], p["bat_h"], p["bat_ab"] = bb.pa.fillna(0).to_numpy(), bb.hit.fillna(0).to_numpy(), bb.ab.fillna(0).to_numpy()
        pp = pit.reindex(p.opp_sp.to_numpy())
        bf, ha = pp.pa.fillna(0).to_numpy(), pp.hit.fillna(0).to_numpy()
        sl = slot.reindex(pd.MultiIndex.from_arrays([p.slot.to_numpy(), p.side.to_numpy()]))
        b = (p.bat_h + 200 * Lr) / (p.bat_pa + 200)
        q = (ha + 400 * Lr) / (bf + 400)
        x = (b * q / Lr) / (b * q / Lr + (1 - b) * (1 - q) / (1 - Lr))
        E_ = sl.pa.to_numpy() / sl.games.to_numpy()
        s = sl.vs.to_numpy() / sl.pa.to_numpy()
        p["b2"] = 1 - (1 - x) ** (s * E_) * (1 - b) ** ((1 - s) * E_)
        p["b1"] = np.where(p.bat_pa >= 200, p.bat_h / p.bat_ab.where(p.bat_ab > 0), np.nan)
        p["league_hpa"] = Lr
        outP.append(p)
        k = K[K.date.eq(ds)].copy()
        if len(k):
            tot = st.sum()
            ss = st.reindex(k.pitcher.to_numpy())
            n_st, n_k = ss.starts.fillna(0).to_numpy(), ss.k.fillna(0).to_numpy()
            k["base"] = np.where(n_st >= 3, n_k / np.maximum(n_st, 1), tot.k / tot.starts)
            outK.append(k)
    return pd.concat(outP, ignore_index=True), pd.concat(outK, ignore_index=True)


# ---------------------------------------------------------------- tests
def test1(G, md):
    M = G[G.odds_match.isin(["unique", "score_disambiguated"]) & G.odds_away_ml_close.notna()
          & G.odds_home_ml_close.notna()].copy()
    ih, ia = implied(M.odds_home_ml_close), implied(M.odds_away_ml_close)
    M["overround"] = ih + ia - 1
    n_bad = int((~M.overround.between(0, 0.10)).sum())
    M = M[M.overround.between(0, 0.10)].reset_index(drop=True)
    ih, ia = implied(M.odds_home_ml_close), implied(M.odds_away_ml_close)
    M["p_mkt"] = ih / (ih + ia)
    M["p_model"] = 1 - M.model_away_win
    X = np.column_stack([np.ones(len(M)), logit(M.p_mkt), logit(M.p_model)])
    y = M.y.to_numpy().astype(float)
    folds = []
    M["p_blend"] = np.nan
    for s in SEASONS:
        tr, te = M.season.ne(s).to_numpy(), M.season.eq(s).to_numpy()
        b = fit_logit(X[tr], y[tr])
        M.loc[te, "p_blend"] = 1 / (1 + np.exp(-X[te] @ b))
        folds.append((s, int(te.sum()), b))
    M["dll"] = ll(M.p_blend, y) - ll(M.p_mkt, y)
    W = boot_counts(len(M), SEED + 1)
    d = M.dll.to_numpy()
    est, c = d.mean(), ci(W @ d / W.sum(1))
    per = {}
    for s in SEASONS:
        m = M.season.eq(s).to_numpy()
        Ws = boot_counts(m.sum(), SEED + 1)
        per[s] = (d[m].mean(), ci(Ws @ d[m] / Ws.sum(1)), int(m.sum()))
    n_neg = sum(per[s][0] < 0 for s in SEASONS)
    confirmed = c[1] < 0 and n_neg >= 3
    beta = fit_logit(X, y)
    bt = np.array([fit_logit(X, y, w.astype(float)) for w in W])
    KEY["t1"] = dict(est=est, ci=c, n=len(M), n_neg=n_neg, b_model=beta[2], b_model_ci=ci(bt[:, 2]), outcome=confirmed)

    rows = []
    for s, n, b in folds:
        m = M.season.eq(s)
        yy = y[m.to_numpy()]
        r = {"held-out season": s, "n": n, "b_market": f3(b[1]), "b_model": f3(b[2])}
        for name, col in (("market", "p_mkt"), ("model raw", "p_model"), ("blend", "p_blend")):
            r[f"LL {name}"] = f"{ll(M.loc[m, col], yy).mean():.5f}"
            r[f"Brier {name}"] = f"{((M.loc[m, col] - yy) ** 2).mean():.5f}"
        rows.append(r)

    # ROI, descriptive: larger-edge side, bet if edge >= 0.04, 1 unit at closing price
    e = (M.p_blend - M.p_mkt).to_numpy()
    bet = np.abs(e) >= 0.04
    home = e > 0
    odds = np.where(home, M.odds_home_ml_close, M.odds_away_ml_close).astype(float)
    won = np.where(home, y == 1, y == 0)
    payout = np.where(odds > 0, odds / 100, 100 / np.abs(odds))
    profit = np.where(bet, np.where(won, payout, -1.0), 0.0)
    roi_rows = []
    for name, m in [("pooled", np.ones(len(M), bool))] + [(str(s), M.season.eq(s).to_numpy()) for s in SEASONS]:
        nb = int((bet & m).sum())
        Wm = boot_counts(m.sum(), SEED + 1)
        pr, st = profit[m], bet[m].astype(float)
        roi = pr.sum() / st.sum() if nb else np.nan
        rc = ci((Wm @ pr) / np.maximum(Wm @ st, 1)) if nb else (np.nan, np.nan)
        roi_rows.append({"set": name, "games": int(m.sum()), "bets": nb,
                         "win rate (Wilson)": wilson_s(int((won & bet & m).sum()), nb), "ROI (bootstrap)": fci(roi, rc)})

    md += ["## Test 1 — Sides (moneyline) — confirmatory", "",
           f"- **Data:** analysis-set games with a matched closing moneyline on both sides. Dropped for overround outside 0–0.10: {n_bad}. n = {len(M)} games "
           f"({', '.join(f'{s}: {per[s][2]}' for s in SEASONS)}). Mean overround {M.overround.mean():.3f}.",
           "- **Statistic:** mean per-game log loss of p_blend minus log loss of p_mkt, pooled over the four held-out seasons "
           "(leave-one-season-out cross-fit).",
           f"- **Result:** **{est:.5f}**, 95% interval **{c[0]:.5f} to {c[1]:.5f}** (game bootstrap, 2,000 resamples, seed {SEED + 1}).", "",
           md_table(pd.DataFrame([{"held-out season": s, "n": per[s][2], "dLL (blend − market)": f"{per[s][0]:.5f}",
                                   "95% interval": f"{per[s][1][0]:.5f} to {per[s][1][1]:.5f}"} for s in SEASONS])), "",
           f"Seasons with a negative point estimate: {n_neg} of 4.", "",
           "**Rule (plan, verbatim):** \"**\"Information confirmed\"** only if the pooled 95% interval is entirely below 0 **and** "
           "the held-out point estimate is negative in at least 3 of the 4 seasons. Otherwise **\"not confirmed\"**.\"", "",
           f"**Outcome: {'Information confirmed' if confirmed else 'not confirmed'}.**", "",
           "**Also reported (descriptive).**", "",
           f"- Full-sample coefficients (n = {len(M)}, game bootstrap 2,000, seed {SEED + 1}): "
           f"logit(p_model) {fci(beta[2], ci(bt[:, 2]))}; logit(p_market) {fci(beta[1], ci(bt[:, 1]))}; intercept {beta[0]:.3f}.",
           "", "Per-fold coefficients, log loss and Brier score:", "", md_table(pd.DataFrame(rows)), "",
           "ROI at closing prices, larger cross-fitted edge side, edge ≥ 0.04 (descriptive only, no rule):", "",
           md_table(pd.DataFrame(roi_rows)), ""]
    return confirmed


def test2(G, md):
    T = G[G.odds_over_close.notna()].reset_index(drop=True)
    x = (T.model_total_mean - T.odds_over_close).to_numpy()
    yv = (T.away_runs_box + T.home_runs_box - T.odds_over_close).to_numpy().astype(float)

    def fit(m, seed):
        Xt = np.column_stack([np.ones(m.sum()), x[m]])
        c = np.linalg.lstsq(Xt, yv[m], rcond=None)[0]
        W = boot_counts(m.sum(), seed)
        sl = []
        for w in W:
            sw = np.sqrt(w)
            sl.append(np.linalg.lstsq(Xt * sw[:, None], yv[m] * sw, rcond=None)[0][1])
        return c, ci(sl)

    rows, res = [], {}
    for name, m in [("pooled", np.ones(len(T), bool))] + [(str(s), T.season.eq(s).to_numpy()) for s in SEASONS]:
        c, sc = fit(m, SEED + 2)
        res[name] = (c, sc)
        rows.append({"set": name, "n": int(m.sum()), "intercept": f3(c[0]), "slope": f3(c[1]),
                     "slope 95% interval": f"{sc[0]:.3f} to {sc[1]:.3f}", "mean (model − line)": f3(x[m].mean()),
                     "SD (model − line)": f3(x[m].std(ddof=1)), "mean (actual − line)": f3(yv[m].mean())})
    c, sc = res["pooled"]
    confirmed = sc[0] > 0
    KEY["t2"] = dict(est=c[1], ci=sc, n=len(T), outcome=confirmed)
    md += ["## Test 2 — Totals — confirmatory", "",
           f"- **Data:** analysis-set games with `odds_over_close`; n = {len(T)}. Model value `model_total_mean`; outcome = box-score total runs.",
           "- **Statistic:** OLS slope of (actual total − closing line) on (`model_total_mean` − closing line).",
           f"- **Result:** slope **{c[1]:.3f}**, 95% interval **{sc[0]:.3f} to {sc[1]:.3f}** (game bootstrap, 2,000, seed {SEED + 2}).", "",
           md_table(pd.DataFrame(rows)), "",
           "**Rule (plan, verbatim):** \"**\"Confirmed\"** only if the slope's 95% interval lies entirely above 0. Otherwise "
           "**\"not confirmed\"**. A negative slope is \"not confirmed\".\"", "",
           f"**Outcome: {'Confirmed' if confirmed else 'not confirmed'}.**", ""]
    return confirmed


def picks(df, col, k):
    d = df[df[col].notna()].sort_values(["date", col, "game_pk", "id"], ascending=[True, False, True, True])
    d = d.groupby("date").head(k).copy()
    d["rank"] = d.groupby("date").cumcount() + 1
    return d


def test3(P, md):
    Q = P[~(P.duplicated(["date", "id"], keep=False) & P.game_number.ne(1))].copy()  # doubleheader: game 1 only
    Q["res"] = np.where(Q.h >= 1, "hit", np.where(Q.ab >= 1, "loss", "pass"))
    meth = {"simulator": "p_1h", "B1": "b1", "B2": "b2"}
    top1 = {m: picks(Q, c, 1).set_index("date") for m, c in meth.items()}

    rate_rows, rates = [], {}
    for m in meth:
        t = top1[m]
        for name, sel in [("pooled", np.ones(len(t), bool))] + [(str(s), t.season.eq(s).to_numpy()) for s in SEASONS]:
            tt = t[sel]
            k, n = int(tt.res.eq("hit").sum()), int(tt.res.ne("pass").sum())
            if name == "pooled":
                rates[m] = (k, n)
            rate_rows.append({"method": m, "set": name, "dates": len(tt), "passes": int(tt.res.eq("pass").sum()),
                              "decided": n, "hits": k, "success rate (Wilson)": wilson_s(k, n)})

    def paired(base, sel_season=None):
        j = top1["simulator"][["res", "season"]].join(top1[base][["res"]], rsuffix="_b", how="inner")
        j = j[j.res.ne("pass") & j.res_b.ne("pass")]
        if sel_season:
            j = j[j.season.eq(sel_season)]
        d = (j.res.eq("hit").astype(int) - j.res_b.eq("hit").astype(int)).to_numpy()
        W = boot_counts(len(d), SEED + 3)
        return d.mean(), ci(W @ d / W.sum(1)), len(d)

    pair = {b: paired(b) for b in ("B2", "B1")}
    pair_rows = [{"comparison": f"simulator − {b}", "set": name, "paired dates": r[2], "mean difference": f3(r[0]),
                  "95% interval": f"{r[1][0]:.3f} to {r[1][1]:.3f}"}
                 for b in ("B2", "B1") for name, r in [("pooled", pair[b])] + [(str(s), paired(b, s)) for s in SEASONS]]
    est, c, n = pair["B2"]
    adds = c[0] > 0

    # calibration of p_1h, pool batters with atBats >= 1
    C = Q[Q.ab >= 1]
    edges = [0, .5, .6, .65, .7, .75, .8, .85, 1.0001]
    labels = ["<0.50", "0.50–0.60", "0.60–0.65", "0.65–0.70", "0.70–0.75", "0.75–0.80", "0.80–0.85", "≥0.85"]
    C = C.assign(bin=pd.cut(C.p_1h, edges, right=False, labels=labels))
    cal = [{"bin": b, "n": len(g), "mean predicted": f3(g.p_1h.mean()) if len(g) else "–",
            "realized (Wilson)": wilson_s(int((g.h >= 1).sum()), len(g)) + ("" if len(g) >= 30 else " (n<30, not interpreted)")}
           for b, g in C.groupby("bin", observed=False)]

    t1_rows = []
    for m, pcol in (("simulator", "p_1h"), ("B2", "b2")):
        t = top1[m][top1[m].res.ne("pass")]
        t1_rows.append({"method": m, "decided picks": len(t), "mean predicted P(1+ hit)": f3(t[pcol].mean()),
                        "realized": wilson_s(int(t.res.eq("hit").sum()), len(t))})

    two_rows = []
    for m, c_ in meth.items():
        t2 = picks(Q, c_, 2)
        g = t2.groupby("date").agg(n=("rank", "size"), dec=("ab", lambda a: (a >= 1).all()), both=("h", lambda h: (h >= 1).all()))
        g = g[g.n.eq(2)]
        two_rows.append({"method": m, "dates": len(g), "excluded (a pick with 0 AB)": int((~g.dec).sum()),
                         "both-hit rate (Wilson)": wilson_s(int((g.both & g.dec).sum()), int(g.dec.sum()))})

    st_rows = []
    for m in meth:
        k, n_ = rates[m]
        lo, hi = stats.wilson(k, n_)
        st_rows.append({"method": m, "top-1 rate": f3(k / n_), "P(57) at rate": f"{p_streak(k / n_):.2e}",
                        "at lower end": f"{lo:.3f} → {p_streak(lo):.2e}", "at upper end": f"{hi:.3f} → {p_streak(hi):.2e}"})

    # 3b secondary
    P2 = P.assign(y=(P.h >= 1).astype(float))
    P2["dll"] = ll(P2.p_1h, P2.y) - ll(P2.b2, P2.y)
    s = P2.groupby("date").agg(sd=("dll", "sum"), n=("dll", "size"))
    W = boot_counts(len(s), SEED + 3)
    ll_est, ll_ci = s.sd.sum() / s.n.sum(), ci((W @ s.sd.to_numpy()) / (W @ s.n.to_numpy()))
    five = {}
    for m in ("simulator", "B2"):
        t5 = picks(Q, meth[m], 5)
        five[m] = t5.groupby("date").agg(h=("res", lambda r: r.eq("hit").sum()), d=("res", lambda r: r.ne("pass").sum()))
    f5 = five["simulator"].join(five["B2"], rsuffix="_b", how="inner")
    W5 = boot_counts(len(f5), SEED + 3)
    r_s, r_b = f5.h.sum() / f5.d.sum(), f5.h_b.sum() / f5.d_b.sum()
    bt5 = (W5 @ f5.h.to_numpy()) / (W5 @ f5.d.to_numpy()) - (W5 @ f5.h_b.to_numpy()) / (W5 @ f5.d_b.to_numpy())
    KEY["t3"] = dict(est=est, ci=c, n=n, outcome=adds, rates=rates, top1=top1,
                     passes={m: int(top1[m].res.eq("pass").sum()) for m in meth})
    KEY["t3b"] = dict(ll=ll_est, ll_ci=ll_ci, ll_n=len(P2), top5=r_s - r_b, top5_ci=ci(bt5), top5_n=len(f5))

    md += ["## Test 3 — Beat the Streak — confirmatory", "",
           f"- **Pool:** {len(Q)} starting-lineup batter-games in analysis-set games (doubleheader rule removed {len(P) - len(Q)}), {Q.date.nunique()} dates. "
           f"B1-eligible (≥200 PA): {int(Q.b1.notna().sum())}.",
           "- **Statistic:** top-1 success rate = hits ÷ (hits + losses), with Wilson intervals.", "",
           md_table(pd.DataFrame(rate_rows)), "",
           f"- **Paired comparison** (dates where both picks were decided; date bootstrap, 2,000, seed {SEED + 3}):", "",
           md_table(pd.DataFrame(pair_rows)), "",
           f"**Primary result:** simulator − B2 = **{est:.3f}**, 95% interval **{c[0]:.3f} to {c[1]:.3f}**, n = {n} paired dates.", "",
           "**Rule (plan, verbatim):** \"The simulator **\"adds value\"** only if its paired difference against **B2** has a 95% interval "
           "entirely above 0. Otherwise **\"does not add value\"**.\"", "",
           f"**Outcome: {'adds value' if adds else 'does not add value'}.**", "",
           "**Also reported.**", "", f"Calibration of `p_1h` (pool batters with atBats ≥ 1, n = {len(C)}):", "", md_table(pd.DataFrame(cal)), "",
           "Top-1 picks, mean predicted vs realized (selection inflates the prediction, as the plan notes):", "", md_table(pd.DataFrame(t1_rows)), "",
           "Top-2 both-hit rate:", "", md_table(pd.DataFrame(two_rows)), "",
           "Chance of a 57-game streak in 180 independent daily top-1 picks (rate and Wilson interval ends):", "", md_table(pd.DataFrame(st_rows)), "",
           "## Section 3b — Beat the Streak — secondary (no decision rule)", "",
           f"- **Log loss of P(1+ hit), simulator − B2**, all {len(P2)} lineup batter-games (both doubleheader games): "
           f"**{ll_est:.5f}**, 95% interval {ll_ci[0]:.5f} to {ll_ci[1]:.5f} (date bootstrap, 2,000, seed {SEED + 3}). "
           f"Mean log loss: simulator {ll(P2.p_1h, P2.y).mean():.5f}, B2 {ll(P2.b2, P2.y).mean():.5f}.",
           f"- **Top-5 pooled success rate**, {len(f5)} dates: simulator {r_s:.3f} ({int(f5.h.sum())}/{int(f5.d.sum())}), "
           f"B2 {r_b:.3f} ({int(f5.h_b.sum())}/{int(f5.d_b.sum())}); difference **{r_s - r_b:.3f}**, 95% interval {ci(bt5)[0]:.3f} to {ci(bt5)[1]:.3f} "
           f"(paired by date, date bootstrap, 2,000, seed {SEED + 3}).", ""]
    return adds


def test4(K, md):
    K = K.reset_index(drop=True)
    pred, base, act, oth = (K[c].to_numpy(float) for c in ("pred", "base", "actual", "other_actual"))
    _, gi = np.unique(K.game_pk, return_inverse=True)

    def stats_for(m, seed):
        _, g = np.unique(gi[m], return_inverse=True)
        W = boot_counts(g.max() + 1, seed)
        out = {}
        for name, f in (("bias_model", lambda w: wmean(pred[m] - act[m], w)), ("bias_base", lambda w: wmean(base[m] - act[m], w)),
                        ("mae_model", lambda w: wmean(np.abs(pred[m] - act[m]), w)), ("mae_base", lambda w: wmean(np.abs(base[m] - act[m]), w)),
                        ("dmae", lambda w: wmean(np.abs(pred[m] - act[m]) - np.abs(base[m] - act[m]), w)),
                        ("r_model", lambda w: wcorr(pred[m], act[m], w)), ("r_base", lambda w: wcorr(base[m], act[m], w)),
                        ("r_other", lambda w: wcorr(pred[m], oth[m], w))):
            out[name] = (f(np.ones(m.sum())), ci([f(w[g].astype(float)) for w in W]))
        return out

    res = {}
    rows = []
    for name, m in [("pooled", np.ones(len(K), bool))] + [(str(s), K.season.eq(s).to_numpy()) for s in SEASONS]:
        r = res[name] = stats_for(m, SEED + 4)
        rows.append({"set": name, "starts": int(m.sum()), **{k: fci(*v, d=3) for k, v in r.items()}})
    est, c = res["pooled"]["dmae"]
    outcome = "Model better" if c[1] < 0 else "Baseline better" if c[0] > 0 else "No difference shown"
    KEY["t4"] = dict(est=est, ci=c, n=len(K), outcome=outcome)
    md += ["## Test 4 — Starter strikeouts — confirmatory", "",
           f"- **Data:** analysis-set starts where the predicted starter started; n = {len(K)} starts in {K.game_pk.nunique()} games. "
           f"Baseline fell back to the league mean for {int(K.base_league.sum())} starts (fewer than 3 starts in the window).",
           "- **Statistic:** ΔMAE = MAE(model `model_*_k_med`) − MAE(baseline, trailing-365-day K per start), paired on the same starts.",
           f"- **Result:** ΔMAE **{est:.3f}**, 95% interval **{c[0]:.3f} to {c[1]:.3f}** (game bootstrap, 2,000, seed {SEED + 4}).", "",
           "All statistics with 95% intervals (bias = predicted − actual; r_other = model K vs the other starter's actual Ks):", "",
           md_table(pd.DataFrame(rows)), "",
           "**Rule (plan's table, cells verbatim):** **\"Model better\"** | 95% interval entirely below 0; **\"Baseline better\"** | "
           "interval entirely above 0; **\"No difference shown\"** | otherwise.", "",
           f"**Outcome: {outcome}.**", ""]
    return outcome


def wmean(x, w):
    return (w * x).sum() / w.sum()


def test5(L, G, P, md):
    rows = []
    for setname, D in (("analysis set", G), ("all ledger rows", L)):
        for name, m in [("pooled", np.ones(len(D), bool))] + [(str(s), D.season.eq(s).to_numpy()) for s in SEASONS]:
            d = D[m]
            est, c = auc_boot(1 - d.model_away_win.to_numpy(), d.y.to_numpy(), d.game_pk.to_numpy(), SEED + 5)
            rows.append({"games": setname, "set": name, "n": len(d), "home-win AUC": fci(est, c)})
    H = P[P.pa >= 1].assign(y=lambda d: (d.hr >= 1).astype(int))
    hr_rows = []
    for name, m in [("pooled", np.ones(len(H), bool))] + [(str(s), H.season.eq(s).to_numpy()) for s in SEASONS]:
        d = H[m]
        est, c = auc_boot(d.p_1hr.to_numpy(), d.y.to_numpy(), d.game_pk.to_numpy(), SEED + 5)
        KEY.setdefault("t5_hr", dict(est=est, ci=c, n=len(d)))  # first row is pooled
        hr_rows.append({"set": name, "batter-games": len(d), "HRs": int(d.y.sum()), "HR AUC": fci(est, c)})
    H = H.assign(dec=pd.qcut(H.p_1hr.rank(method="first"), 10, labels=False) + 1)
    cal = [{"decile": int(k), "n": len(g), "predicted range": f"{g.p_1hr.min():.3f}–{g.p_1hr.max():.3f}",
            "mean predicted": f3(g.p_1hr.mean()), "realized (Wilson)": wilson_s(int(g.y.sum()), len(g))} for k, g in H.groupby("dec")]
    md += ["## Section 5 — Descriptive only (no decision rules)", "",
           f"Home-win AUC, P(home) = 1 − `model_away_win`; game bootstrap, 2,000, seed {SEED + 5}:", "", md_table(pd.DataFrame(rows)), "",
           f"HR props (starters with plateAppearances ≥ 1), `p_1hr` vs homeRuns ≥ 1; game-cluster bootstrap, 2,000, seed {SEED + 5}:", "",
           md_table(pd.DataFrame(hr_rows)), "", "HR calibration, 10 equal-count bins (pooled):", "", md_table(pd.DataFrame(cal)), ""]


# ---------------------------------------------------------------- main
def main():
    sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert sha == PLAN_SHA, f"plan hash {sha} != frozen {PLAN_SHA}"
    L = load_ledgers()
    og, ob = outcomes(L)
    L = L.merge(og, on="game_pk")
    L["y"] = (L.home_runs_box > L.away_runs_box).astype(int)
    fb = L.away_starter_source.ne("probable") | L.home_starter_source.ne("probable")
    G = L[~fb].reset_index(drop=True)

    auc0 = stats.auc(1 - G.model_away_win, G.y)
    print(f"orientation: home-win AUC on analysis set = {auc0:.4f} (n={len(G)})", flush=True)
    if auc0 < 0.5:
        sys.exit("STOP: home-win AUC < 0.5 on the analysis set (possible wiring regression). Nothing swapped, no tests run.")

    # batter-games (analysis set) and starts
    gi = G[["game_pk", "date", "season", "game_number", "away_starter_pred", "home_starter_pred"]]
    P = ob.merge(gi, on="game_pk")
    P["opp_sp"] = np.where(P.side.eq("away"), P.home_starter_pred, P.away_starter_pred).astype(int)
    n_nonstart = int(P.bo.astype(str).ne((P.slot * 100).astype(str)).sum())
    if TRUE_STARTERS_ONLY:
        P = P[P.bo.astype(str).eq((P.slot * 100).astype(str))].reset_index(drop=True)
    ks = []
    for s, o in (("away", "home"), ("home", "away")):
        m = G[f"{s}_starter_pred"].eq(G[f"{s}_starter_actual"])
        ks.append(pd.DataFrame({"game_pk": G.game_pk[m], "date": G.date[m], "season": G.season[m],
                                "pitcher": G[f"{s}_starter_actual"][m].astype(int), "pred": G[f"model_{s}_k_med"][m],
                                "actual": G[f"{s}_sp_k_box"][m], "other_actual": G[f"{o}_sp_k_box"][m]}))
    K = pd.concat(ks, ignore_index=True)
    T = baseline_tables()
    print("baseline tables built", flush=True)
    P, K = add_baselines(P, K, T)
    K["base_league"] = False
    for ds, g in K.groupby("date"):
        st = T["start"].at(pd.Timestamp(ds)).reindex(g.pitcher.to_numpy()).starts.fillna(0).to_numpy()
        K.loc[g.index, "base_league"] = st < 3
    assert P.b2.between(0, 1).all() and K.base.notna().all() and K.actual.notna().all() and K.pred.notna().all()

    md = [TITLE, "",
          f"Plan: `analysis/{PLAN.name}`, SHA-256 `{sha}` (matches the last `{HASH_LINE}` line of `analysis/out/PLAN_HASH.txt`). "
          "Run by `analysis/v4_analysis.py` after `analysis/v4_integrity.py` (log: `analysis/out/v4_integrity_log.txt`). "
          "Outcomes from cached box scores; baselines from `data/pitch_matrix.parquet` with game_date < D.", "",
          "## Integrity gate and analysis set", "",
          f"- Ledger rows {len(L)} (2022 {int(L.season.eq(2022).sum())}, 2023 {int(L.season.eq(2023).sum())}, 2024 {int(L.season.eq(2024).sum())}, "
          f"2025 {int(L.season.eq(2025).sum())}); no duplicate gamePks; every eligible scheduled game present; every gamePk has a cached box score; "
          "all batter and starter ids joined (0 unjoined). Integrity: PASS.",
          f"- Excluded for `actual_fallback`: {int(fb.sum())} games → analysis set {len(G)} games.",
          *([f"- Lineup entries in analysis-set games whose box `battingOrder` is not slot × 100, excluded from the pool: {n_nonstart}."]
            if TRUE_STARTERS_ONLY else []),
          f"- Orientation check: home-win AUC on the analysis set = {auc0:.4f} (n = {len(G)}) ≥ 0.5 → continue; nothing swapped.", ""]
    r1 = test1(G, md)
    print("test 1 done", flush=True)
    r2 = test2(G, md)
    print("test 2 done", flush=True)
    r3 = test3(P, md)
    print("test 3 done", flush=True)
    r4 = test4(K, md)
    print("test 4 done", flush=True)
    test5(L, G, P, md)
    md[md.index("## Test 1 — Sides (moneyline) — confirmatory"):md.index("## Test 1 — Sides (moneyline) — confirmatory")] = [
        "## Confirmatory outcomes (plan's words)", "",
        md_table(pd.DataFrame([{"test": "1 Sides", "outcome": "Information confirmed" if r1 else "not confirmed"},
                               {"test": "2 Totals", "outcome": "Confirmed" if r2 else "not confirmed"},
                               {"test": "3 Beat the Streak", "outcome": "adds value" if r3 else "does not add value"},
                               {"test": "4 Starter strikeouts", "outcome": r4}])), "",
        "Multiplicity (plan): if all four nulls are true, the chance of at least one false favourable outcome is about 10% "
        "(1 − 0.975⁴ = 0.096 if independent; at most 0.10 under any dependence).", ""]
    md += ["## Implementation notes", "",
           "- Seeds: 20261022 + test number (1–5). Section 3b sits under test 3 and uses seed 20261022 + 3; the plan gives no separate number.",
           "- Per-season intervals use the same seed and settings as the pooled interval.",
           "- Test 3: the 57-game streak inputs use each method's pooled top-1 rate and its Wilson interval ends.",
           "- Test 4's rule wording is quoted from the plan's table.", ""]
    KEY.update(L=L, G=G, P=P, K=K)
    md += EXTRA(md) if EXTRA else ["## EXPLORATORY", "", "None. Nothing outside the plan was computed.", ""]
    RESULTS.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
