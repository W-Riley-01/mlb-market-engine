"""Lineup defect checks (BACKTEST_V4_ADDENDUM_2026-10-23.md). Read-only; no re-simulation.
1. Box field for the true starting nine (battingOrder codes ending in 00), verified against Statcast first-nine order.
2. EXPLORATORY: per-game substitute count vs test 1/2 quantities.
3. Is the defect in the pilot ledger and in data/backtest_ledger.csv?"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import v4_analysis as va
import v4_exploratory as ve
from v4_integrity import API, ROOT, load_ledgers

SEED = 20261023


def box(pk):
    return json.loads((API / "box" / f"{pk}.json").read_text(encoding="utf-8"))


def starters(team: dict) -> dict:
    """{slot: player id} for players whose box battingOrder code is slot*100 (the starting nine)."""
    return {int(p["battingOrder"]) // 100: p["person"]["id"] for p in team["players"].values()
            if str(p.get("battingOrder", "")).endswith("00")}


def statcast_first_nine():
    pm = pd.read_parquet(ROOT / "data" / "pitch_matrix.parquet", columns=["game_pk", "inning_topbot", "at_bat_number", "batter", "events"])
    E = pm[pm.events.notna()].sort_values(["game_pk", "at_bat_number"])
    E["side"] = np.where(E.inning_topbot.eq("Bot"), "home", "away")
    f = E.drop_duplicates(["game_pk", "side", "batter"]).copy()
    f["slot"] = f.groupby(["game_pk", "side"]).cumcount() + 1
    f = f[f.slot <= 9]
    return {k: dict(zip(g.slot, g.batter)) for k, g in f.groupby(["game_pk", "side"])}


def double_switch_games():
    """Games in the fetched live feeds where an incoming pitcher took a non-pitcher's batting slot."""
    out = set()
    for f in glob.glob(str(API / "feed" / "*.json")):
        feed = json.loads(Path(f).read_text(encoding="utf-8"))
        for pl in feed["liveData"]["plays"]["allPlays"]:
            for e in pl.get("playEvents", []):
                d = e.get("details", {}).get("description", "")
                if e.get("details", {}).get("eventType") == "pitching_substitution" and re.search(r"batting \w+, replacing (?!pitcher)", d):
                    out.add(int(Path(f).stem))
    return sorted(out)


def part_field(L, SC):
    print("=== Box field for the starting nine: players[*].battingOrder code == slot*100 ===")
    dh2 = L[L.game_number.eq(2)].sample(5, random_state=SEED)
    g1 = L[L.game_number.eq(1)].merge(dh2[["date", "away_code", "home_code"]], on=["date", "away_code", "home_code"])
    dh = sorted(set(dh2.game_pk) | set(g1.game_pk))
    ds = [pk for pk in double_switch_games() if pk in set(L.game_pk)][:8]
    rnd = list(L.game_pk.sample(8, random_state=SEED))
    rows = []
    for kind, pks in (("double switch", ds), ("doubleheader", dh), ("random", rnd)):
        for pk in pks:
            b = box(pk)["teams"]
            for s in ("away", "home"):
                st = starters(b[s])
                sc = SC.get((pk, s), {})
                rows.append({"kind": kind, "game_pk": pk, "side": s, "n_starters": len(st), "slots_1_9": sorted(st) == list(range(1, 10)),
                             "matches_statcast": st == sc, "subs_in_battingOrder": sum(1 for pid in b[s]["battingOrder"] if pid not in st.values())})
    V = pd.DataFrame(rows)
    print(V.groupby("kind").agg(games=("game_pk", "nunique"), team_games=("side", "size"), exactly_nine=("n_starters", lambda n: int((n == 9).sum())),
                                slots_1_9=("slots_1_9", "sum"), matches_statcast=("matches_statcast", "sum"),
                                subs_in_battingOrder_list=("subs_in_battingOrder", "sum")).to_string())
    print("verification games:", {k: list(g.game_pk.unique()) for k, g in V.groupby("kind")})
    full = []
    for pk in L.game_pk:
        b = box(pk)["teams"]
        for s in ("away", "home"):
            st = starters(b[s])
            full.append((len(st) == 9 and sorted(st) == list(range(1, 10)), st == SC.get((pk, s))))
    F = np.array(full)
    print(f"all {len(F)} team-games: exactly nine starters in slots 1-9 {int(F[:, 0].sum())} ({F[:, 0].mean():.4%}); "
          f"identical to Statcast first-nine order {int(F[:, 1].sum())} ({F[:, 1].mean():.4%})")
    return V


def sub_counts(L):
    rows = []
    for r in L.itertuples():
        b = box(r.game_pk)["teams"]
        st = {s: set(starters(b[s]).values()) for s in ("away", "home")}
        bat = json.loads(r.batters)
        rows.append({"game_pk": r.game_pk, "subs": sum(x["id"] not in st[x["side"]] for x in bat),
                     "subs_away": sum(x["id"] not in st["away"] for x in bat if x["side"] == "away"),
                     "subs_home": sum(x["id"] not in st["home"] for x in bat if x["side"] == "home")})
    return pd.DataFrame(rows)


def part_diag(G):
    print("\n=== EXPLORATORY: substitute count per game ===")
    G = G.merge(sub_counts(G), on="game_pk")
    print(f"games {len(G)}; substitutes per game: mean {G.subs.mean():.2f}, distribution {G.subs.value_counts().sort_index().to_dict()}")
    # totals (test 2 rows)
    T = G[G.odds_over_close.notna()].reset_index(drop=True)
    x = (T.model_total_mean - T.odds_over_close).to_numpy()
    yv = (T.away_runs_box + T.home_runs_box - T.odds_over_close).to_numpy(float)
    s = T.subs.to_numpy(float)
    print(f"(a) n={len(T)}: r(subs, model_total_mean - line) {np.corrcoef(s, x)[0, 1]:.3f}; r(subs, actual - line) {np.corrcoef(s, yv)[0, 1]:.3f}")
    X = np.column_stack([np.ones(len(T)), x, s])
    c = np.linalg.lstsq(X, yv, rcond=None)[0]
    W = va.boot_counts(len(T), SEED)
    bt = np.array([np.linalg.lstsq(X * np.sqrt(w)[:, None], yv * np.sqrt(w), rcond=None)[0] for w in W])
    print("(c) test 2 + subs: (actual - line) ~ 1 + (model - line) + subs: " +
          "; ".join(f"{n} {c[i]:.3f} ({va.ci(bt[:, i])[0]:.3f} to {va.ci(bt[:, i])[1]:.3f})" for i, n in enumerate(("intercept", "slope(model-line)", "subs"))))
    # moneyline (test 1 rows)
    M = G[G.odds_match.isin(["unique", "score_disambiguated"]) & G.odds_away_ml_close.notna() & G.odds_home_ml_close.notna()].copy()
    ih, ia = va.implied(M.odds_home_ml_close), va.implied(M.odds_away_ml_close)
    M = M[(ih + ia - 1 >= 0) & (ih + ia - 1 <= 0.10)].reset_index(drop=True)
    ih, ia = va.implied(M.odds_home_ml_close), va.implied(M.odds_away_ml_close)
    pm, pmod = ih / (ih + ia), 1 - M.model_away_win.to_numpy()
    y = M.y.to_numpy(float)
    s = M.subs.to_numpy(float)
    print(f"(b) n={len(M)}: r(subs, logit model - logit market) {np.corrcoef(s, va.logit(pmod) - va.logit(pm))[0, 1]:.3f}; "
          f"r(subs, home win - p_market) {np.corrcoef(s, y - pm)[0, 1]:.3f}")
    sd = (M.subs_home - M.subs_away).to_numpy(float)
    print(f"    home-minus-away substitutes: r with (logit model - logit market) {np.corrcoef(sd, va.logit(pmod) - va.logit(pm))[0, 1]:.3f}; "
          f"r with (home win - p_market) {np.corrcoef(sd, y - pm)[0, 1]:.3f}")
    for name, cov in (("subs", s), ("home-minus-away subs", sd)):
        X = np.column_stack([np.ones(len(M)), va.logit(pm), va.logit(pmod), cov])
        b = va.fit_logit(X, y)
        bt = np.array([va.fit_logit(X, y, w.astype(float)) for w in va.boot_counts(len(M), SEED)])
        print(f"(c) test 1 + {name}: y ~ 1 + logit(p_mkt) + logit(p_model) + {name}: " +
              "; ".join(f"{n} {b[i]:.4f} ({va.ci(bt[:, i])[0]:.4f} to {va.ci(bt[:, i])[1]:.4f})"
                        for i, n in enumerate(("intercept", "b_market", "b_model", name))))


def part_other_ledgers():
    print("\n=== Same defect elsewhere? ===")
    for f in ("ledger_v4_pilot.csv", "ledger_v4_pilot_azath_fixed.csv"):
        P = pd.read_csv(ROOT / "analysis" / "out" / f)
        n = sub = 0
        for r in P.itertuples():
            b = box(r.game_pk)["teams"]
            st = {s: set(starters(b[s]).values()) for s in ("away", "home")}
            for x in json.loads(r.batters):
                n += 1
                sub += x["id"] not in st[x["side"]]
        print(f"{f}: games {len(P)}, lineup entries {n}, substitutes {sub} ({sub / n:.1%})")
    O = pd.read_csv(ROOT / "data" / "backtest_ledger.csv")
    have = O[O.game_id.map(lambda pk: (API / "box" / f"{int(pk)}.json").exists())]
    n = sub = unmatched = 0
    for r in have.itertuples():
        b = box(int(r.game_id))["teams"]
        props = list(json.loads(r.model_player_props))
        if len(props) != 18:
            continue
        for i, nm in enumerate(props):
            team = b["away" if i < 9 else "home"]
            code = [str(p.get("battingOrder", "")) for p in team["players"].values() if p["person"]["fullName"] == nm]
            if len(code) != 1 or not code[0]:
                unmatched += 1
                continue
            n += 1
            sub += not code[0].endswith("00")
    print(f"data/backtest_ledger.csv: rows {len(O)} ({O.date.min()}..{O.date.max()}); rows with a cached box {len(have)}; "
          f"lineup names matched to a box batting-order code {n} (unmatched {unmatched}); substitutes {sub} ({sub / max(n, 1):.1%})")


def main():
    pd.set_option("display.width", 200)
    L = load_ledgers()
    SC = statcast_first_nine()
    part_field(L, SC)
    G, _ = ve.setup()
    part_diag(G)
    part_other_ledgers()


if __name__ == "__main__":
    main()
