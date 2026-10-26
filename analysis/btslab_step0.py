"""Beat the Streak lab Step 0: data inventory, counts only. Read-only (DB via analysis/lib.query, read-only txn)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
import lib  # noqa: E402
import stats  # noqa: E402
from backtest_v4 import match_odds, team_abbr  # noqa: E402
from odds_ingestion import OddsBook  # noqa: E402
from v4_integrity import API, box, schedule_games  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
SEASONS = [2022, 2023, 2024, 2025]


def show(title, df):
    print(f"\n### {title}\n{df.to_string()}", flush=True)


# ---- schedule (2022-2025 regular season, as the R2 backtest defined eligibility) + box scores
S = schedule_games()
sched = {}
for f in sorted((API / "schedule").glob("*.json")):
    for day in json.loads(f.read_text()).get("dates", []):
        for g in day["games"]:
            sched[g["gamePk"]] = g
E = S[S.final & ~S.suspended].copy()
E["has_box"] = E.game_pk.map(lambda pk: (API / "box" / f"{pk}.json").exists())

games, bats = [], []
for r in E[E.has_box].itertuples():
    b, g = box(r.game_pk), sched[r.game_pk]
    info = {i["label"]: i.get("value", "") for i in b.get("info", [])}
    row = {"game_pk": r.game_pk, "date": r.date, "season": r.season, "venue_id": g["venue"]["id"], "venue": g["venue"]["name"],
           "box_venue_info": info.get("Venue", "").rstrip("."), "weather_txt": info.get("Weather", "").rstrip(".")}
    for s in ("away", "home"):
        t = b["teams"][s]
        row[f"{s}_code"] = team_abbr(g["teams"][s]["team"]["name"])
        row[f"{s}_team_id"] = g["teams"][s]["team"]["id"]
        row[f"{s}_runs"] = t["teamStats"]["batting"]["runs"]
        row[f"{s}_prob"] = g["teams"][s].get("probablePitcher", {}).get("id")
        row[f"{s}_actual_sp"] = t["pitchers"][0] if t["pitchers"] else None
        for p in t["players"].values():
            bo = str(p.get("battingOrder", ""))
            if bo.isdigit() and int(bo) % 100 == 0 and 1 <= int(bo) // 100 <= 9:  # starting_nine() rule
                st = p["stats"].get("batting", {})
                bats.append({"game_pk": r.game_pk, "date": r.date, "season": r.season, "id": p["person"]["id"],
                             "slot": int(bo) // 100, "ab": st.get("atBats", 0)})
    games.append(row)
G, P = pd.DataFrame(games), pd.DataFrame(bats)
nine = P.groupby(["game_pk"]).size()
assert (nine == 18).all(), "a game without 9 true starters per side"

# ---- pitch matrix
pm = pd.read_parquet(ROOT / "data" / "pitch_matrix.parquet", columns=["game_pk", "game_date", "game_year", "batter", "events"])
pm["game_date"] = pd.to_datetime(pm.game_date)
pm_by_year = pm.groupby("game_year").agg(pitch_rows=("game_pk", "size"), games=("game_pk", "nunique"),
                                         first=("game_date", "min"), last=("game_date", "max"))
in_rs = pm[pm.game_pk.isin(E.game_pk)].groupby("game_year").agg(pitch_rows_in_eligible_games=("game_pk", "size"),
                                                                 eligible_games_with_pitches=("game_pk", "nunique"))

# ---- odds
odds = OddsBook(str(ROOT / "data" / "MLB_Basic.csv"))._index.reset_index()
om = [match_odds(odds, r.date, r.away_code, r.home_code, r.away_runs, r.home_runs) for r in G.itertuples()]
G["odds_match"] = [h for _, h in om]
for k in ("away_ml_close", "home_ml_close", "over_close", "away_ml_open", "home_ml_open"):
    G[f"o_{k}"] = [None if o is None else o.get(k) for o, _ in om]
G["has_close"] = G.o_away_ml_close.notna() & G.o_home_ml_close.notna() & G.o_over_close.notna()

# ---- item 1
t1 = pd.DataFrame({
    "eligible_games (final, R, not suspended)": E.groupby("season").size(),
    "with_cached_box": G.groupby("season").size(),
    "true_starter_rows": P.groupby("season").size(),
    "rows_AB>=1": P[P.ab >= 1].groupby("season").size(),
    "rows_AB=0 (passes)": P[P.ab == 0].groupby("season").size(),
    "games_with_box_weather_text": G[G.weather_txt.ne("")].groupby("season").size(),
    "games_with_closing_ML_and_total": G[G.has_close].groupby("season").size(),
})
t1["odds_match_rate"] = (t1["games_with_closing_ML_and_total"] / t1["with_cached_box"]).round(4)
t1 = t1.join(in_rs.rename_axis("season"))
show("1. Per season, 2022-2025 (schedule + box-score cache + pitch_matrix + odds file)", t1)
show("1b. pitch_matrix.parquet by game_year (all game types; the file has no game-type column)", pm_by_year)
wl = pd.read_parquet(ROOT / "data" / "game_weather_log.parquet")
wl["season"] = pd.to_datetime(wl.game_date).dt.year
show("1c. data/game_weather_log.parquet rows by year", wl.groupby("season").agg(rows=("home_team", "size"), dates=("game_date", "nunique")))
show("1d. odds_match values by season", G.groupby(["season", "odds_match"]).size().unstack(fill_value=0))

# ---- probable starters (schedule cached after the games)
pr = pd.concat([G[["season", f"{s}_prob", f"{s}_actual_sp"]].set_axis(["season", "prob", "actual"], axis=1) for s in ("away", "home")])
show("1e. probable starter (schedule cache) vs actual starter (box), team-games",
     pr.assign(has_prob=pr.prob.notna(), prob_eq_actual=pr.prob.eq(pr.actual)).groupby("season")[["has_prob", "prob_eq_actual"]]
     .agg(["sum", "size"]))

# ---- item 2: prior PA in pitch_matrix for 2022 true starters
pa = pm[pm.events.notna()].astype({"batter": "int64"}).groupby(["batter", "game_date"]).size().rename("pa").reset_index().sort_values("game_date")
pa["cum"] = pa.groupby("batter").pa.cumsum()
P22 = P[P.season.eq(2022)].assign(date=lambda d: pd.to_datetime(d.date).astype("datetime64[ns]")).sort_values("date")
pa["game_date"] = pa.game_date.astype("datetime64[ns]")
prior = pd.merge_asof(P22, pa.rename(columns={"batter": "id", "game_date": "date"})[["id", "date", "cum"]],
                      on="date", by="id", allow_exact_matches=False)  # strictly before D
prior["prior_pa_all"] = prior.cum.fillna(0)
win = []
for d, g in prior.groupby("date"):
    w = pa[(pa.game_date >= d - pd.Timedelta(days=365)) & (pa.game_date < d)].groupby("batter").pa.sum()
    win.append(g.id.map(w).fillna(0))
prior["prior_pa_365"] = pd.concat(win)
open22 = prior.date.min()
prior["day"] = (prior.date - open22).dt.days + 1
d2 = prior.groupby("day").agg(rows=("id", "size"), median_prior_pa_all=("prior_pa_all", "median"),
                              median_prior_pa_365=("prior_pa_365", "median"),
                              share_ge100_all=("prior_pa_all", lambda x: (x >= 100).mean()),
                              share_ge100_365=("prior_pa_365", lambda x: (x >= 100).mean()))
print(f"\n### 2. History. pitch_matrix earliest game_date {pm.game_date.min().date()}; DB earliest game_date "
      f"{lib.query('select min(game_date) d from game_predictions').d[0]}. 2022 opening day {open22.date()}.")
for col in ("median_prior_pa_all", "median_prior_pa_365"):
    hit = d2.index[d2[col] >= 100]
    print(f"first day of 2022 with {col} >= 100: day {hit.min() if len(hit) else 'never'}")
show("2b. 2022 true starters, prior PA strictly before D (selected days)", d2.loc[[d for d in (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60) if d in d2.index]].round(3))

# ---- item 3: 2026
q = lib.query
print("\n### 3. 2026")
print(q("select count(distinct game_id) games, min(game_date), max(game_date), count(*) filter (where weather_temp_f is not null) with_temp, "
        "count(*) filter (where away_starter_id is not null and home_starter_id is not null) with_both_starters from game_predictions").to_string(index=False))
print(q("select count(distinct o.game_id) games_with_outcome, count(*) filter (where o.status is not null) rows from game_outcomes o").to_string(index=False))
print(q("select count(*) rows, count(*) filter (where at_bats>=1) ab_ge1, count(*) filter (where at_bats=0) ab0, count(*) filter (where at_bats is null) ab_null from player_outcomes").to_string(index=False))
print(q("select status, count(*) from player_outcomes group by 1 order by 2 desc").to_string(index=False))
print(f"pitch_matrix 2026 rows: {int((pm.game_year == 2026).sum())}; odds file 2026 games: {int(odds.date.str.startswith('2026').sum())}; "
      f"game_weather_log 2026 rows: {int(wl.season.eq(2026).sum())}; schedule-cache 2026 files: {len(list((API / 'schedule').glob('2026-*.json')))}; "
      f"box-cache files: {len(list((API / 'box').glob('*.json')))}, of which 2022-2025 eligible games: {int(E.has_box.sum())}")

# ---- item 4: venues and roofs
print("\n### 4. Venues")
print(f"schedule venue name vs box info 'Venue' text agree: {int(G.venue.eq(G.box_venue_info).sum())} of {len(G)}")
ath = G[G.season.eq(2025) & G.home_team_id.eq(133)]
show("4a. Athletics (team id 133) 2025 home games by recorded venue", ath.groupby(["venue_id", "venue"]).size().rename("games").to_frame())
G["roof"] = np.select([G.weather_txt.str.contains("Roof Closed", case=False), G.weather_txt.str.contains("Dome", case=False),
                       G.weather_txt.eq("")], ["roof_closed", "dome", "missing"], "open_or_other")
rv = G.groupby(["venue_id", "venue", "roof"]).size().unstack(fill_value=0)
show("4b. Venues where any game's box weather says 'Roof Closed' or 'Dome', 2022-2025", rv[(rv.get("roof_closed", 0) + rv.get("dome", 0)) > 0])
small = G.groupby(["venue_id", "venue"]).size()
show("4c. Venues with fewer than 30 games in 2022-2025 (neutral or temporary sites)", small[small < 30].rename("games").to_frame())
print(q("select weather_is_dome, count(*) from game_predictions group by 1").to_string(index=False))

# ---- item 5: odds gaps
print("\n### 5. Odds file: columns", list(pd.read_csv(ROOT / "data" / "MLB_Basic.csv", nrows=0).columns))
miss = G[~G.has_close].copy()
miss["month"] = miss.date.str[:7]
print(f"2022-2025 eligible games without closing ML+total: {len(miss)} of {len(G)}")
show("5a. by season and odds_match", miss.groupby(["season", "odds_match"]).size().rename("games").to_frame())
show("5b. by month", miss.groupby("month").size().rename("games").to_frame())
show("5c. by team (away or home)", pd.concat([miss.away_code, miss.home_code]).value_counts().rename("games").to_frame())

# ---- added check: market AUC
print("\n### Added check: AUC of de-vigged market home-win probability vs actual home win (box runs), 2022-2025")
rows = []
for kind in ("close", "open"):
    M = G[G[f"o_away_ml_{kind}"].notna() & G[f"o_home_ml_{kind}"].notna()]
    ih, ia = (np.where(o > 0, 100 / (o + 100), -o / (-o + 100)) for o in (M[f"o_home_ml_{kind}"].astype(float), M[f"o_away_ml_{kind}"].astype(float)))
    p, y = ih / (ih + ia), (M.home_runs > M.away_runs).astype(int).to_numpy()
    for name, m in [("pooled", np.ones(len(M), bool))] + [(str(s), M.season.eq(s).to_numpy()) for s in SEASONS]:
        rows.append({"line": kind, "set": name, "games": int(m.sum()), "AUC": round(stats.auc(p[m], y[m]), 4)})
show("market AUC", pd.DataFrame(rows))

# ---- item 6: fields
print("\n### 6. Fields")
pmcols = set(pq.ParquetFile(ROOT / "data" / "pitch_matrix.parquet").schema_arrow.names)
b0 = box(int(G.game_pk.iloc[0]))["teams"]["home"]
pit = b0["players"][f"ID{b0['pitchers'][0]}"]["stats"]["pitching"]
dbcols = set(q("select column_name from information_schema.columns where table_schema='public'").column_name)
for what, checks in [("batter handedness", [("pitch_matrix.stand", "stand" in pmcols)]),
                     ("pitcher handedness", [("pitch_matrix.p_throws", "p_throws" in pmcols)]),
                     ("batters faced per start", [("box pitching.battersFaced", "battersFaced" in pit),
                                                  ("pitch_matrix (count PA rows of the first pitcher)", {"pitcher", "events", "at_bat_number"} <= pmcols)]),
                     ("per-reliever pitch counts by game date", [("box pitching.numberOfPitches / pitchesThrown", "numberOfPitches" in pit),
                                                                 ("pitch_matrix (rows per pitcher per game_date)", {"pitcher", "game_date"} <= pmcols)]),
                     ("any of these in the DB", [("DB columns stand/p_throws/batters_faced/pitches", bool(dbcols & {"stand", "p_throws", "batters_faced", "pitches"}))])]:
    for src, ok in checks:
        print(f"{what:40s} {src:55s} {'present' if ok else 'absent'}")
