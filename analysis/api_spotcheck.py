"""Part A2: compare DB outcomes to MLB Stats API box scores for 25 random analysis games
(stratified by version). Read-only DB; public API GETs; box JSON cached under analysis/cache/box/."""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

import lib

BOX = lib.CACHE / "box"


def box(pk):
    f = BOX / f"{pk}.json"
    if not f.exists():
        with urllib.request.urlopen(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", timeout=30) as r:
            f.write_bytes(r.read())
        time.sleep(0.3)
    return json.loads(f.read_text())


def main():
    BOX.mkdir(parents=True, exist_ok=True)
    g = lib.analysis_games()
    rng = np.random.default_rng(2026)
    pick = pd.concat([g[g.version.eq("v2")].sample(13, random_state=rng.integers(1e9)),
                      g[g.version.eq("v3")].sample(12, random_state=rng.integers(1e9))])
    p = lib.load_players()
    issues, rows = [], []
    for _, x in pick.iterrows():
        b = box(int(x.game_id))
        for side in ("away", "home"):
            t = b["teams"][side]
            api_team = t["team"]["name"]
            api_runs = t["teamStats"]["batting"]["runs"]
            sp = t["pitchers"][0]
            sp_k = t["players"][f"ID{sp}"]["stats"]["pitching"].get("strikeOuts")
            db_team = x[f"{side}_team"]
            db_runs = x[f"{side}_score"]
            db_sp, db_k = x[f"{side}_starter_id"], x[f"{side}_starter_actual_k"]
            checks = {"team": (db_team, api_team), "runs": (db_runs, api_runs),
                      "starter_id": (db_sp, sp), "starter_K": (db_k, sp_k)}
            # 3 batters per side: DB player rows on this side vs the API box of the same side
            pr = p[(p.gp_id == x.gp_id) & (p.side == side) & p.p_status.eq("final")].head(3)
            for _, r in pr.iterrows():
                pl = t["players"].get(f"ID{r.player_id}")
                api_h = pl["stats"]["batting"].get("hits") if pl else "NOT IN THIS SIDE'S BOX"
                checks[f"hits {r.player_name}"] = (r.hits, api_h)
            for k, (dbv, apiv) in checks.items():
                ok = str(dbv) == str(apiv) or (pd.notna(dbv) and str(int(dbv)) == str(apiv))
                rows.append({"gp_id": x.gp_id, "game_pk": x.game_id, "version": x.version, "side": side,
                             "field": k, "db": dbv, "api": apiv, "match": ok})
    R = pd.DataFrame(rows)
    R.to_csv(lib.OUT / "api_spotcheck.csv", index=False)
    print(f"games: {R.gp_id.nunique()} (v2 {pick.version.eq('v2').sum()}, v3 {pick.version.eq('v3').sum()})  "
          f"comparisons: {len(R)}  mismatches: {int((~R.match).sum())}")
    print(R.groupby(R.field.str.split(" ").str[0]).match.agg(["size", "sum"]).rename(columns={"sum": "matches"}))
    print(R[~R.match].to_string() if (~R.match).any() else "no mismatches")

    # A3 support: DB team labels vs schedule for every analysis game
    s = pd.read_csv(lib.CACHE / "schedule_2026.csv").drop_duplicates("game_pk", keep="last").set_index("game_pk")
    lab = g[["game_id", "away_team", "home_team"]].join(s[["away_team", "home_team"]], on="game_id", rsuffix="_sched")
    print("\nanalysis games whose DB away/home team differs from the schedule:",
          int(((lab.away_team != lab.away_team_sched) | (lab.home_team != lab.home_team_sched)).sum()),
          "of", len(lab))


if __name__ == "__main__":
    main()
