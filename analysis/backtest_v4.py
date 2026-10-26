"""Leak-free walk-forward backtest with the fixed (v4) simulator (plan: BACKTEST_V4_PLAN.md).
Writes a NEW ledger (default analysis/out/ledger_v4_pilot.csv); never touches data/backtest_ledger.csv.
API responses are cached under analysis/cache/api/ so re-runs and resumes are cheap.

  python analysis/backtest_v4.py --start 2025-04-01 --end 2025-04-30 --max-games 300
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from advanced_simulator import run_advanced_monte_carlo  # noqa: E402
from backtester import team_abbr  # noqa: E402  (API team name -> pipeline code)
from odds_ingestion import OddsBook  # noqa: E402
from resolver import MatchupResolver  # noqa: E402

API = ROOT / "analysis" / "cache" / "api"
OUT = ROOT / "analysis" / "out" / "ledger_v4_pilot.csv"
PIPE_TO_VAULT = {"ARI": "AZ", "OAK": "ATH"}  # Statcast codes in the vault
# same list as MatchupResolver._count_terminal_pa
TERMINAL = ['strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run', 'field_out',
            'grounded_into_dp', 'fielders_choice', 'force_out', 'sac_fly', 'line_out', 'pop_out', 'sac_bunt',
            'intent_walk']


def get_json(url: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
            return json.loads(data)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def league_baseline_fn(pqm: pd.DataFrame, cqm: pd.DataFrame):
    """Expanding-window league rates using only rows with game_date < D."""
    term = pqm[pqm["events"].isin(TERMINAL)]
    p = term.groupby("game_date").agg(pa=("events", "size"),
                                      k=("events", lambda e: (e == "strikeout").sum()),
                                      bb=("events", lambda e: e.isin(["walk", "intent_walk"]).sum())).cumsum()
    c = cqm.groupby("game_date")["events"].agg(
        n="size", single=lambda e: (e == "single").sum(), double=lambda e: (e == "double").sum(),
        triple=lambda e: (e == "triple").sum(), hr=lambda e: (e == "home_run").sum()).cumsum()

    def at(D: pd.Timestamp) -> dict:
        pr = p[p.index < D].iloc[-1]
        cr = c[c.index < D].iloc[-1]
        return {"k_rate": pr.k / pr.pa, "bb_rate": pr.bb / pr.pa, "single": cr.single / cr.n,
                "double": cr.double / cr.n, "triple": cr.triple / cr.n, "hr": cr.hr / cr.n}
    return at


def relief_usage(pqm: pd.DataFrame) -> pd.DataFrame:
    """Relief pitches per (pitching team, game_date, pitcher). Starter = team's first pitcher of the game."""
    d = pqm[["game_pk", "game_date", "home_team", "away_team", "inning_topbot", "at_bat_number",
             "pitch_number", "pitcher"]].copy()
    d["team"] = np.where(d.inning_topbot.eq("Top"), d.home_team, d.away_team)
    d = d.sort_values(["game_pk", "team", "at_bat_number", "pitch_number"])
    starter = d.groupby(["game_pk", "team"])["pitcher"].transform("first")
    r = d[d.pitcher.ne(starter)]
    return r.groupby(["team", "game_date", "pitcher"]).size().rename("pitches").reset_index()


def bullpen_for(usage: pd.DataFrame, team: str, D: pd.Timestamp, exclude) -> dict:
    u = usage[usage.team.eq(team) & (usage.game_date >= D - pd.Timedelta(days=30)) & (usage.game_date < D)]
    w = u.groupby("pitcher")["pitches"].sum()
    return {int(k): float(v) for k, v in w.items() if k not in exclude}


def match_odds(odds: pd.DataFrame, d: str, away: str, home: str, a_runs: int, h_runs: int):
    # the odds file switched to AZ/ATH in 2025 while the pipeline uses ARI/OAK; accept either
    alias = lambda c: {c, PIPE_TO_VAULT.get(c, c)}
    m = odds[(odds.date == d) & odds.away_team.isin(alias(away)) & odds.home_team.isin(alias(home))]
    if len(m) == 1:
        return m.iloc[0], "unique"
    if len(m) > 1:
        s = m[(m.away_score == a_runs) & (m.home_score == h_runs)]
        return (s.iloc[0], "score_disambiguated") if len(s) == 1 else (None, "ambiguous")
    return None, "none"


def starting_nine(box_team: dict) -> list:
    """Starter ids in slot order. box battingOrder (the team list) holds each slot's END-of-game occupant;
    a player's own battingOrder code is slot*100 only for the starter (verified in v4_lineup_check.py)."""
    st = sorted((int(p["battingOrder"]), p["person"]["id"]) for p in box_team["players"].values()
                if str(p.get("battingOrder", "")).endswith("00"))
    return [pid for _, pid in st]


def batter_line(box_team: dict, pid: int) -> dict:
    s = box_team["players"].get(f"ID{pid}", {}).get("stats", {}).get("batting", {})
    return {k: s.get(k) for k in ("plateAppearances", "hits", "totalBases", "homeRuns", "rbi", "stolenBases")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2025-04-30")
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--contact-matrix", default=None,
                    help="contact matrix parquet (default: the resolver's own default, data/contact_matrix_env.parquet)")
    ap.add_argument("--game-pks", default=None, help="comma-separated gamePks: simulate only these")
    a = ap.parse_args()
    only = {int(x) for x in a.game_pks.split(",")} if a.game_pks else None
    out = Path(a.out)
    done = set(pd.read_csv(out, usecols=["game_pk"]).game_pk) if out.exists() else set()

    t_load = time.time()
    resolver = MatchupResolver(cqm_path=a.contact_matrix) if a.contact_matrix else MatchupResolver()
    master_pqm, master_cqm = resolver.pqm, resolver.cqm
    master_pqm["game_date"] = pd.to_datetime(master_pqm["game_date"])
    master_cqm["game_date"] = pd.to_datetime(master_cqm["game_date"])
    base_at = league_baseline_fn(master_pqm, master_cqm)
    usage = relief_usage(master_pqm)
    odds = OddsBook()._index.reset_index()
    print(f"[setup] {time.time() - t_load:.0f}s; already done: {len(done)}", flush=True)

    n_done, skipped = len(done), {}
    D = date.fromisoformat(a.start)
    while D <= date.fromisoformat(a.end) and n_done < a.max_games:
        ds = D.isoformat()
        sched = get_json(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={ds}&hydrate=probablePitcher,linescore",
                         API / "schedule" / f"{ds}.json")
        games = [g for day in sched.get("dates", []) for g in day["games"]]
        Dts = pd.Timestamp(ds)
        t_slice = time.time()
        resolver.pqm = master_pqm[master_pqm["game_date"] < Dts]
        resolver.cqm = master_cqm[master_cqm["game_date"] < Dts]
        resolver.league_avg = base_at(Dts)
        slice_s = time.time() - t_slice
        for g in games:
            if n_done >= a.max_games:
                break
            pk, st = g["gamePk"], g["status"]
            if only is not None and pk not in only:
                continue
            reason = None
            if g.get("gameType") != "R":
                reason = "not_regular_season"
            elif g.get("resumedFrom") or g.get("resumeDate") or "Suspend" in st.get("detailedState", "") \
                    or g.get("officialDate", ds) != ds:
                reason = "suspended_or_resumed"
            elif st.get("abstractGameState") != "Final" or st.get("detailedState") in ("Postponed", "Cancelled"):
                reason = "not_final"
            elif pk in done:
                continue
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            t0 = time.time()
            box = get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", API / "box" / f"{pk}.json")
            bt = box["teams"]
            runs = {s: bt[s]["teamStats"]["batting"]["runs"] for s in ("away", "home")}
            if runs["away"] == runs["home"]:
                skipped["tied_score"] = skipped.get("tied_score", 0) + 1
                continue
            nine = {s: starting_nine(bt[s]) for s in ("away", "home")}
            if any(len(nine[s]) != 9 for s in ("away", "home")):
                skipped["no_full_lineup"] = skipped.get("no_full_lineup", 0) + 1
                continue
            name = lambda s, pid: bt[s]["players"].get(f"ID{pid}", {}).get("person", {}).get("fullName", str(pid))
            lineups = {s: [{"id": pid, "name": name(s, pid)} for pid in nine[s]] for s in ("away", "home")}
            row = {"date": ds, "game_pk": pk, "game_number": g.get("gameNumber"), "double_header": g.get("doubleHeader"),
                   "iterations": a.iterations}
            sp, bp = {}, {}
            for s in ("away", "home"):
                tname = g["teams"][s]["team"]["name"]
                code = team_abbr(tname)
                actual = bt[s]["pitchers"][0] if bt[s]["pitchers"] else None
                prob = g["teams"][s].get("probablePitcher", {}).get("id")
                sp[s] = prob or actual
                bp[s] = bullpen_for(usage, PIPE_TO_VAULT.get(code, code), Dts, {sp[s]})
                pk_line = bt[s]["players"].get(f"ID{actual}", {}).get("stats", {}).get("pitching", {})
                row.update({f"{s}_team": tname, f"{s}_code": code, f"{s}_starter_pred": sp[s],
                            f"{s}_starter_actual": actual, f"{s}_starter_source": "probable" if prob else "actual_fallback",
                            f"{s}_starter_actual_k": pk_line.get("strikeOuts"),
                            f"{s}_bullpen_arms": len(bp[s]), f"{s}_bullpen_pitches": sum(bp[s].values()),
                            f"{s}_runs": runs[s]})
            names = [p["name"] for s in ("away", "home") for p in lineups[s]]
            row["name_collision"] = len(set(names)) < len(names)
            np.random.seed(pk % (2 ** 32))
            t_sim = time.time()
            res = run_advanced_monte_carlo(resolver, lineups["away"], lineups["home"], sp["away"], sp["home"],
                                           bp["away"], bp["home"], density_ratio=1.0, iterations=a.iterations)
            row["sim_seconds"] = time.time() - t_sim
            ls = g.get("linescore", {}).get("innings", [])
            f5 = [sum((i.get(s, {}).get("runs") or 0) for i in ls[:5]) for s in ("away", "home")]
            row.update({
                "model_away_win": res["away_win_prob"], "model_f5_away": res["f5_away_win_prob"],
                "model_f5_tie": res["f5_tie_prob"], "model_nrfi": res["nrfi_prob"], "model_total_mean": res["total_mean"],
                "model_median_total": res["median_total"], "model_total_std": res["total_std"],
                "model_rl_home": res["run_line_home_minus_1_5"], "model_rl_away": res["run_line_away_minus_1_5"],
                "model_away_k_med": res["away_pitcher_median_k"], "model_home_k_med": res["home_pitcher_median_k"],
                "model_away_k_dist": json.dumps([int(x) for x in res["away_k_dist"]]),
                "model_home_k_dist": json.dumps([int(x) for x in res["home_k_dist"]]),
                "league_avg": json.dumps({k: round(v, 5) for k, v in resolver.league_avg.items()}),
                "actual_f5_away": f5[0], "actual_f5_home": f5[1],
                "actual_nrfi": bool(ls) and all((ls[0].get(s, {}).get("runs") or 0) == 0 for s in ("away", "home")),
                "batters": json.dumps([{"side": s, "slot": i + 1, "id": p["id"], "name": p["name"],
                                        **{k: res["player_props"][p["name"]][v] for k, v in
                                           (("p_1h", "1+ Hits"), ("p_2tb", "2+ TB"), ("p_1hr", "1+ HR"),
                                            ("p_1rbi", "1+ RBI"), ("p_1sb", "1+ SB"))},
                                        **batter_line(bt[s], p["id"])}
                                       for s in ("away", "home") for i, p in enumerate(lineups[s])]),
            })
            o, how = match_odds(odds, ds, row["away_code"], row["home_code"], runs["away"], runs["home"])
            row["odds_match"] = how
            for k in ("away_ml_close", "home_ml_close", "over_close", "over_close_odds", "under_close_odds"):
                row[f"odds_{k}"] = None if o is None else o.get(k)
            row["wall_seconds"] = time.time() - t0 + slice_s / max(len(games), 1)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            n_done += 1
            print(f"[{ds}] {n_done}/{a.max_games} {pk} sim {row['sim_seconds']:.1f}s", flush=True)
        D += timedelta(days=1)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
