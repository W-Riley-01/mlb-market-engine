"""EXPLORATORY follow-ups (2026-10-23) to BACKTEST_V4_RESULTS.md. Not part of the frozen plan; prints only.
Rebuilds the same analysis set as v4_analysis.py and reuses its helpers."""
import json

import numpy as np
import pandas as pd

import stats
import v4_analysis as va
from v4_integrity import box, load_ledgers

SEED = 20261023  # exploratory seed, not one of the plan's


def setup():
    L = load_ledgers()
    og, ob = va.outcomes(L)
    L = L.merge(og, on="game_pk")
    L["y"] = (L.home_runs_box > L.away_runs_box).astype(int)
    G = L[L.away_starter_source.eq("probable") & L.home_starter_source.eq("probable")].reset_index(drop=True)
    P = ob.merge(G[["game_pk", "date", "season", "game_number", "away_starter_pred", "home_starter_pred"]], on="game_pk")
    P["opp_sp"] = np.where(P.side.eq("away"), P.home_starter_pred, P.away_starter_pred).astype(int)
    return G, P


def q1(G):
    M = G[G.odds_match.isin(["unique", "score_disambiguated"]) & G.odds_away_ml_close.notna() & G.odds_home_ml_close.notna()].copy()
    ih, ia = va.implied(M.odds_home_ml_close), va.implied(M.odds_away_ml_close)
    M = M[(ih + ia - 1 >= 0) & (ih + ia - 1 <= 0.10)].reset_index(drop=True)
    ih, ia = va.implied(M.odds_home_ml_close), va.implied(M.odds_away_ml_close)
    M["p_mkt"], M["p_model"] = ih / (ih + ia), 1 - M.model_away_win
    y = M.y.to_numpy().astype(float)
    X = np.column_stack([np.ones(len(M)), va.logit(M.p_mkt), va.logit(M.p_model)])
    print(f"Q1 n={len(M)}")
    print(f"  market AUC {stats.auc(M.p_mkt, y):.4f}, LL {va.ll(M.p_mkt, y).mean():.5f}; "
          f"model raw AUC {stats.auc(M.p_model, y):.4f}, LL {va.ll(M.p_model, y).mean():.5f}")
    b = va.fit_logit(X, y)
    print(f"  full-sample y ~ 1 + logit(p_mkt) + logit(p_model): a={b[0]:.4f} b_mkt={b[1]:.4f} b_model={b[2]:.4f}")
    bm = va.fit_logit(X[:, [0, 2]], y)
    print(f"  model alone y ~ 1 + logit(p_model): a={bm[0]:.4f} slope={bm[1]:.4f}")
    print(f"  SD logit(p_model) {X[:, 2].std():.3f}, SD logit(p_mkt) {X[:, 1].std():.3f}, r(logits) {np.corrcoef(X[:, 1], X[:, 2])[0, 1]:.3f}")
    # cross-fitted blend, as in test 1, to see which side the ROI bets take
    M["p_blend"] = np.nan
    for s in va.SEASONS:
        tr, te = M.season.ne(s).to_numpy(), M.season.eq(s).to_numpy()
        M.loc[te, "p_blend"] = 1 / (1 + np.exp(-X[te] @ va.fit_logit(X[tr], y[tr])))
    e = M.p_blend - M.p_mkt
    bet = e.abs() >= 0.04
    model_side = np.sign(M.p_model - M.p_mkt)
    agree = (np.sign(e[bet]) == model_side[bet])
    print(f"  ROI bets {int(bet.sum())}: bet side agrees with model's side vs market in {int(agree.sum())}, opposes in {int((~agree).sum())}")


def q2(G):
    T = G[G.odds_over_close.notna()]
    line = T.odds_over_close.to_numpy()
    x = T.model_total_mean.to_numpy() - line
    r = (T.away_runs_box + T.home_runs_box).to_numpy() - line
    c = np.linalg.lstsq(np.column_stack([np.ones(len(T)), line]), r, rcond=None)[0]
    W = va.boot_counts(len(T), SEED)
    sl = [np.linalg.lstsq(np.column_stack([np.ones(len(T)), line]) * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)[0][1] for w in W]
    print(f"Q2 n={len(T)}: mean(model_total_mean - line) {x.mean():.3f} (SD {x.std(ddof=1):.3f}); r(model - line, line) {np.corrcoef(x, line)[0, 1]:.3f}")
    print(f"  (actual - line) ~ line: intercept {c[0]:.3f} slope {c[1]:.3f} ({va.ci(sl)[0]:.3f} to {va.ci(sl)[1]:.3f}); "
          f"mean line {line.mean():.2f}, mean model_total_mean {T.model_total_mean.mean():.2f}, SD model_total_mean {T.model_total_mean.std():.3f}, SD line {line.std():.3f}")
    print(f"  r(model_total_mean, line) {np.corrcoef(T.model_total_mean, line)[0, 1]:.3f}")


def starts(G):
    ks = []
    for s, o in (("away", "home"), ("home", "away")):
        m = G[f"{s}_starter_pred"].eq(G[f"{s}_starter_actual"])
        ks.append(pd.DataFrame({"game_pk": G.game_pk[m], "date": G.date[m], "season": G.season[m], "side": s,
                                "pitcher": G[f"{s}_starter_actual"][m].astype(int), "pred": G[f"model_{s}_k_med"][m],
                                "k_dist": G[f"model_{s}_k_dist"][m], "actual": G[f"{s}_sp_k_box"][m]}))
    return pd.concat(ks, ignore_index=True)


def q3(P):
    Q = P[~(P.duplicated(["date", "id"], keep=False) & P.game_number.ne(1))].copy()
    Q["res"] = np.where(Q.h >= 1, "hit", np.where(Q.ab >= 1, "loss", "pass"))
    Q["non_ab_share"] = 1 - Q.bat_ab / Q.bat_pa.where(Q.bat_pa > 0)
    print(f"Q3 pool {len(Q)}, dates {Q.date.nunique()}")
    tops = {}
    for m, c in (("simulator", "p_1h"), ("B1", "b1"), ("B2", "b2")):
        t = va.picks(Q, c, 1)
        tops[m] = t
        k, n = int(t.res.eq("hit").sum()), int(t.res.ne("pass").sum())
        lo, hi = stats.wilson(k, n)
        print(f"  {m}: dates {len(t)}, decided {n}, passes {int(t.res.eq('pass').sum())}, hits {k}, rate {k / n:.3f} (Wilson {lo:.3f}-{hi:.3f}); "
              f"picks: mean slot {t.slot.mean():.2f}, mean game PA {t.pa.mean():.2f}, mean trailing non-AB share of PA {t.non_ab_share.mean():.3f}, "
              f"mean trailing PA {t.bat_pa.mean():.0f}, distinct players {t.id.nunique()}")
    print(f"  pool mean trailing non-AB share {Q.non_ab_share.mean():.3f}")
    sp = tops["simulator"][tops["simulator"].res.eq("pass")]
    rows = []
    for r in sp.itertuples():
        st = box(r.game_pk)["teams"][r.side]["players"][f"ID{r.id}"]["stats"]["batting"]
        rows.append({k: st.get(k, 0) for k in ("plateAppearances", "atBats", "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies", "sacBunts", "catchersInterference")})
    R = pd.DataFrame(rows)
    print(f"  simulator passes ({len(R)}): PA total {R.plateAppearances.sum()}, BB {R.baseOnBalls.sum()} (IBB {R.intentionalWalks.sum()}), HBP {R.hitByPitch.sum()}, "
          f"SF {R.sacFlies.sum()}, SH {R.sacBunts.sum()}, CI {R.catchersInterference.sum()}; picks with 0 PA {int(R.plateAppearances.eq(0).sum())}")
    print(f"  simulator pass picks: {sp.id.nunique()} distinct players; most frequent: {sp.id.value_counts().head(3).to_dict()}; "
          f"their mean trailing non-AB share {sp.non_ab_share.mean():.3f}")


def expected_bf(threshold=85):
    """Model-implied starter batters faced: PAs until the pitch count first exceeds `threshold` (same for every starter)."""
    inc, p = np.arange(1, 8), np.array([0.11, 0.15, 0.18, 0.19, 0.17, 0.12, 0.08])
    visit = np.zeros(threshold + 1)
    visit[0] = 1.0
    for c in range(1, threshold + 1):
        visit[c] = sum(p[i] * visit[c - inc[i]] for i in range(7) if c - inc[i] >= 0)
    return visit.sum(), (inc * p).sum()


def ip_outs(s):
    w, f = str(s).split(".")
    return int(w) * 3 + int(f)


def q4(K):
    bf, ppa = [], []
    for r in K.itertuples():
        st = box(r.game_pk)["teams"][r.side]["players"][f"ID{r.pitcher}"]["stats"]["pitching"]
        bf.append(st.get("battersFaced"))
        ppa.append(ip_outs(st.get("inningsPitched", "0.0")))
    K = K.assign(bf=bf, outs=ppa)
    ebf, mppa = expected_bf()
    kmean = K.k_dist.map(lambda s: np.mean(json.loads(s)))
    print(f"Q4 starts {len(K)}: actual BF/start {K.bf.mean():.2f} (SD {K.bf.std():.2f}), actual IP/start {K.outs.mean() / 3:.2f}; "
          f"model-implied BF/start {ebf:.2f} for every starter (pitches/PA mean {mppa:.2f}, pull when count > 85)")
    print(f"  predicted K: median-based mean {K.pred.mean():.3f}, distribution mean {kmean.mean():.3f}; actual mean {K.actual.mean():.3f}; "
          f"baseline mean {K.base.mean():.3f}; r(pred, baseline) {np.corrcoef(K.pred, K.base)[0, 1]:.3f}; "
          f"r(dist mean, baseline) {np.corrcoef(kmean, K.base)[0, 1]:.3f}")
    print(f"  actual K per BF {K.actual.sum() / K.bf.sum():.4f}; implied model K per BF (dist mean / {ebf:.2f}) {kmean.mean() / ebf:.4f}")


def q5(G, P, K):
    va.H_EV = {"home_run"}  # rebuild B2's machinery with HR in place of hits
    T = va.baseline_tables()
    P, _ = va.add_baselines(P.drop(columns=[c for c in ("b1", "b2", "bat_pa", "bat_h", "bat_ab", "league_hpa") if c in P]), K[["date", "pitcher"]], T)
    H = P[P.pa >= 1].reset_index(drop=True)
    y = (H.hr >= 1).to_numpy(float)
    _, gi = np.unique(H.game_pk, return_inverse=True)
    out = {}
    for name, col in (("simulator p_1hr", "p_1hr"), ("HR baseline", "b2")):
        _, g = np.unique(H[col].to_numpy(), return_inverse=True)
        out[name] = (g, g.max() + 1)
    W = va.boot_counts(gi.max() + 1, SEED)
    ests = {k: va.wauc(g, y, np.ones(len(y)), n) for k, (g, n) in out.items()}
    bs = {k: np.array([va.wauc(g, y, w[gi].astype(float), n) for w in W]) for k, (g, n) in out.items()}
    d = bs["simulator p_1hr"] - bs["HR baseline"]
    print(f"Q5 batter-games {len(H)}, HR games {int(y.sum())}")
    for k in ests:
        print(f"  {k}: AUC {ests[k]:.4f} ({va.ci(bs[k])[0]:.4f} to {va.ci(bs[k])[1]:.4f})")
    dd = ests["simulator p_1hr"] - ests["HR baseline"]
    print(f"  difference simulator - baseline {dd:.4f} ({va.ci(d)[0]:.4f} to {va.ci(d)[1]:.4f}), paired game bootstrap 2,000, seed {SEED}")
    print(f"  mean predicted: simulator {H.p_1hr.mean():.4f}, baseline {H.b2.mean():.4f}, realized {y.mean():.4f}; "
          f"log loss simulator {va.ll(H.p_1hr, y).mean():.5f}, baseline {va.ll(H.b2, y).mean():.5f}")
    return dict(diff=dd, diff_ci=va.ci(d), n=len(H), hr=int(y.sum()), auc=ests, auc_ci={k: va.ci(v) for k, v in bs.items()})


def main():
    G, P = setup()
    q1(G)
    q2(G)
    K = starts(G)
    T = va.baseline_tables()
    P, K = va.add_baselines(P, K, T)
    q3(P)
    q4(K)
    q5(G, P, K)


if __name__ == "__main__":
    main()
