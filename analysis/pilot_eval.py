"""Pilot evaluation (BACKTEST_V4_PLAN.md): v4 leak-free pilot vs the old leaky ledger on the same games."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats

ROOT = Path(__file__).parents[1]
B = 1000


def boot(df, fn, key="game_pk", seed=1):
    games = df[key].unique()
    idx = df.groupby(key).indices
    r = stats.rng(seed)
    v = []
    for _ in range(B):
        pick = r.choice(games, len(games))
        v.append(fn(df.iloc[np.concatenate([idx[g] for g in pick])]))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, .025), np.quantile(v, .975)


def fmt(est, ci):
    return f"{est:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"


def main():
    P = pd.read_csv(ROOT / "analysis" / "out" / "ledger_v4_pilot.csv")
    L = pd.read_csv(ROOT / "data" / "backtest_ledger.csv")
    L = L[L.game_id.isin(P.game_pk)].rename(columns={"game_id": "game_pk"})
    print(f"pilot games: {len(P)} ({P.date.min()}..{P.date.max()}); also in old ledger: {len(L)}")
    print("starter_source:", pd.concat([P.away_starter_source, P.home_starter_source]).value_counts().to_dict(),
          "| predicted starter actually started:",
          int((P.away_starter_pred == P.away_starter_actual).sum() + (P.home_starter_pred == P.home_starter_actual).sum()),
          "of", 2 * len(P))
    print("odds_match:", P.odds_match.value_counts().to_dict(), "| name collisions:", int(P.name_collision.sum()))
    print(f"seconds/game: sim mean {P.sim_seconds.mean():.2f} (median {P.sim_seconds.median():.2f}); "
          f"wall mean {P.wall_seconds.mean():.2f}")
    n24_25 = 2 * 2430
    print(f"projected 2024–2025 regular season ({n24_25} games) at {int(P.iterations.iloc[0])} iterations: "
          f"{n24_25 * P.wall_seconds.mean() / 3600:.1f} h (wall)")
    print("bullpen arms per team (mean):", round(pd.concat([P.away_bullpen_arms, P.home_bullpen_arms]).mean(), 1),
          "| teams with empty bullpen pool:", int((P.away_bullpen_arms.eq(0) | P.home_bullpen_arms.eq(0)).sum()))

    # --- home-win AUC ---
    P["hw"] = (P.home_runs > P.away_runs).astype(int)
    L["hw"] = (L.actual_home_score > L.actual_away_score).astype(int)
    rows = [("v4 pilot (leak-free)", P, 1 - P.model_away_win),
            ("old ledger, as stored", L, 1 - L.model_away_win),
            ("old ledger, swapped", L, L.model_away_win)]
    print("\n=== home-win AUC ===")
    for name, d, p in rows:
        d = d.assign(p=p.values)
        f = lambda s: stats.auc(s.p, s.hw)
        print(f"{name:24s} n={len(d)}  AUC {fmt(f(d), boot(d, f))}")

    # --- starter K: predicted median K vs that starter's actual K, and vs the other starter's ---
    print("\n=== starter-K correlation (predicted median K vs actual K) ===")
    rows = []
    for s, o in (("away", "home"), ("home", "away")):
        ok = P[f"{s}_starter_pred"].eq(P[f"{s}_starter_actual"]) & P[f"{o}_starter_pred"].eq(P[f"{o}_starter_actual"])
        rows.append(pd.DataFrame({"game_pk": P.game_pk[ok], "pred": P[f"model_{s}_k_med"][ok],
                                  "own": P[f"{s}_starter_actual_k"][ok], "other": P[f"{o}_starter_actual_k"][ok]}))
    K = pd.concat(rows).dropna().reset_index(drop=True)
    old = []
    for _, x in L.iterrows():
        ks = json.loads(x.actual_pitcher_ks)
        a, h = ks.get(str(int(x.away_starter_id))), ks.get(str(int(x.home_starter_id)))
        old.append({"game_pk": x.game_pk, "away_pred": x.model_away_k_med, "home_pred": x.model_home_k_med, "a": a, "h": h})
    O = pd.DataFrame(old).dropna()
    Ko = pd.concat([pd.DataFrame({"game_pk": O.game_pk, "pred": O.away_pred, "own": O.a, "other": O.h}),
                    pd.DataFrame({"game_pk": O.game_pk, "pred": O.home_pred, "own": O.h, "other": O.a})]).reset_index(drop=True)
    for name, d in (("v4 pilot (predicted starter started)", K), ("old ledger, as stored", Ko)):
        fo = lambda s: np.corrcoef(s.pred, s.own.astype(float))[0, 1]
        ft = lambda s: np.corrcoef(s.pred, s.other.astype(float))[0, 1]
        print(f"{name:38s} starts={len(d)}  r(own) {fmt(fo(d), boot(d, fo))}   r(other) {fmt(ft(d), boot(d, ft))}")

    # --- props: own vs opposite-side same-slot outcome ---
    print("\n=== batter props: AUC vs own outcome vs opposite-side same-slot outcome ===")
    bat = pd.concat([pd.DataFrame(json.loads(b)).assign(game_pk=g) for g, b in zip(P.game_pk, P.batters)])
    bat = bat[bat.plateAppearances.fillna(0) > 0]
    bat["y_1h"], bat["y_1hr"] = (bat.hits >= 1).astype(int), (bat.homeRuns >= 1).astype(int)
    # old ledger props are a name->probs dict in lineup order (away 1-9, then home 1-9)
    orows = []
    for _, x in L.iterrows():
        props = list(json.loads(x.model_player_props).items())
        if len(props) != 18:
            continue
        for i, (nm, pr) in enumerate(props):
            orows.append({"game_pk": x.game_pk, "side": "away" if i < 9 else "home", "slot": i % 9 + 1, "old_name": nm,
                          "p_1h": pr["1+ Hits"], "p_1hr": pr["1+ HR"]})
    Ob = pd.DataFrame(orows).merge(bat[["game_pk", "side", "slot", "name", "y_1h", "y_1hr"]], on=["game_pk", "side", "slot"])
    print(f"old-ledger slot/name agreement with pilot lineups: {(Ob.old_name == Ob.name).mean():.3f} (n={len(Ob)})")

    def opp(d):
        o = d[["game_pk", "side", "slot", "y_1h", "y_1hr"]].copy()
        o["side"] = o.side.map({"away": "home", "home": "away"})
        return d.merge(o, on=["game_pk", "side", "slot"], suffixes=("", "_opp"))

    for name, d in (("v4 pilot", opp(bat).reset_index(drop=True)), ("old ledger, as stored", opp(Ob).reset_index(drop=True))):
        for prop in ("1h", "1hr"):
            fo = lambda s: stats.auc(s[f"p_{prop}"], s[f"y_{prop}"])
            ft = lambda s: stats.auc(s[f"p_{prop}"], s[f"y_{prop}_opp"])
            print(f"{name:22s} 1+{prop[1:].upper():2s} n={len(d)}  AUC own {fmt(fo(d), boot(d, fo))}   "
                  f"AUC opposite-slot {fmt(ft(d), boot(d, ft))}")


if __name__ == "__main__":
    main()
