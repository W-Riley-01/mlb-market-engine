"""Beat the Streak lab, Step 1: as-of feature table (plan BEAT_THE_STREAK_LAB_PLAN.md + addendum A1).

One row per true starter per game, 2022-2025 regular season, neutral-site games excluded.
Every history feature uses only games dated strictly before the row's date; each group carries
<group>_max_date, the latest game date it used. Stored values are as-of counts and league rates;
fold-specific shrinkage and the D split are applied at fit time by derive() (plan Section 4, common rules).
Choices the plan leaves open are listed in analysis/out/btslab_features_notes.txt.

  python analysis/btslab_features.py      # writes analysis/cache/btslab_features.parquet
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
from backtest_v4 import match_odds, team_abbr  # noqa: E402
from odds_ingestion import OddsBook  # noqa: E402

API = ROOT / "analysis" / "cache" / "api"
OUT_PARQUET = ROOT / "analysis" / "cache" / "btslab_features.parquet"
H_EV = {"single", "double", "triple", "home_run"}
TB = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NON_AB = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"}
K_EV, BB_EV = {"strikeout", "strikeout_double_play"}, {"walk", "intent_walk"}
POOL_SEASONS = (2022, 2023, 2024, 2025)
# addendum A1 change 2: the 18 neutral-site games of 2022-2025, by venue id
NEUTRAL_POOL = {5381: "London Stadium", 5340: "Estadio Alfredo Harp Helu", 3949: "Rickwood Field", 5445: "Field of Dreams",
                2735: "Williamsport (Muncy/Journey Bank Ballpark)", 2397: "Tokyo Dome", 5150: "Gocheok Sky Dome"}
# park-factor exclusions: the above plus other one-off sites in the 2019-2025 schedules (fixed list, see notes file)
NEUTRAL_PARK = set(NEUTRAL_POOL) | {2701, 5365, 6130}  # Monterrey 2019, Omaha 2019, Bristol 2025


# ---------------------------------------------------------------- raw inputs
def load_schedule() -> pd.DataFrame:
    """One row per regular-season gamePk: effective date = last date it appears as Final (conservative for
    suspended/postponed games), plus venue, teams, game number and pool eligibility (as in the v4 backtest)."""
    rows = []
    for f in sorted((API / "schedule").glob("*.json")):
        for day in json.loads(f.read_bytes()).get("dates", []):
            for g in day["games"]:
                if g.get("gameType") != "R":
                    continue
                st = g["status"]
                final = st.get("abstractGameState") == "Final" and st.get("detailedState") not in ("Postponed", "Cancelled")
                rows.append({"game_pk": g["gamePk"], "file_date": f.stem, "official": g.get("officialDate", f.stem),
                             "final": final, "suspended": bool(g.get("resumedFrom") or g.get("resumeDate"))
                             or "Suspend" in st.get("detailedState", ""),
                             "venue_id": g["venue"]["id"], "venue": g["venue"]["name"], "game_number": g.get("gameNumber"),
                             "home_id": g["teams"]["home"]["team"]["id"], "away_id": g["teams"]["away"]["team"]["id"],
                             "home_name": g["teams"]["home"]["team"]["name"], "away_name": g["teams"]["away"]["team"]["name"]})
    S = pd.DataFrame(rows)
    fin = S[S.final].sort_values("file_date")
    G = fin.groupby("game_pk").last()  # last Final appearance: its date is when the whole game was available
    G["date"] = pd.to_datetime(G.file_date).astype("datetime64[ns]")
    # pool eligibility as in the v4 backtest: Final on its official date, never suspended or resumed
    on_official = fin[fin.official.eq(fin.file_date)].game_pk.unique()
    G["pool_eligible"] = G.index.isin(on_official) & ~S.groupby("game_pk").suspended.any().reindex(G.index)
    G["season"] = G.date.dt.year
    return G.reset_index()


def load_box(sched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From the cached box scores: pool candidates (true starters), per-pitcher pitching lines, game info."""
    bats, pits, info = [], [], []
    for r in sched.itertuples():
        f = API / "box" / f"{r.game_pk}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_bytes())
        inf = {i["label"]: i.get("value", "") for i in b.get("info", [])}
        info.append({"game_pk": r.game_pk, "weather_txt": inf.get("Weather", "").rstrip("."), "wind_txt": inf.get("Wind", "").rstrip("."),
                     "away_runs": b["teams"]["away"]["teamStats"]["batting"]["runs"], "home_runs": b["teams"]["home"]["teamStats"]["batting"]["runs"]})
        for s in ("away", "home"):
            t = b["teams"][s]
            for i, pid in enumerate(t["pitchers"]):
                st = t["players"][f"ID{pid}"]["stats"]["pitching"]
                pits.append({"game_pk": r.game_pk, "side": s, "pitcher": pid, "order": i,
                             "bf": st.get("battersFaced"), "pitches": st.get("numberOfPitches")})
            for p in t["players"].values():
                bo = str(p.get("battingOrder", ""))
                if bo.isdigit() and int(bo) % 100 == 0 and 1 <= int(bo) // 100 <= 9:  # starting_nine() rule
                    st = p["stats"].get("batting", {})
                    bats.append({"game_pk": r.game_pk, "side": s, "slot": int(bo) // 100, "id": p["person"]["id"],
                                 "ab": st.get("atBats", 0), "h": st.get("hits", 0)})
    return pd.DataFrame(bats), pd.DataFrame(pits), pd.DataFrame(info)


def load_pitch_matrix() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "pitch_matrix.parquet",
                           columns=["game_pk", "game_date", "inning_topbot", "at_bat_number", "pitch_number", "pitcher",
                                    "batter", "p_throws", "events"])


def raw_tables(pm: pd.DataFrame, sched: pd.DataFrame, boxp: pd.DataFrame) -> dict:
    """History tables, regular-season games only, each dated by the game's effective date."""
    gd = sched.set_index("game_pk")
    pm = pm[pm.game_pk.isin(gd.index)].copy()
    pm["date"] = pm.game_pk.map(gd.date)
    pm["bat_side"] = np.where(pm.inning_topbot.eq("Bot"), "home", "away")
    pm["pit_side"] = np.where(pm.inning_topbot.eq("Top"), "home", "away")
    pm["pit_team"] = np.where(pm.pit_side.eq("home"), pm.game_pk.map(gd.home_id), pm.game_pk.map(gd.away_id))
    pm = pm.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first_p = pm.groupby(["game_pk", "pit_side"]).pitcher.transform("first")
    pm["is_sp"] = pm.pitcher.eq(first_p)

    E = pm[pm.events.notna()].astype({"batter": "int64", "pitcher": "int64"}).copy()  # one row per PA
    E["h"] = E.events.isin(H_EV).astype(int)
    E["ab"] = (~E.events.isin(NON_AB)).astype(int)
    E["k"] = E.events.isin(K_EV).astype(int)
    E["bb"] = E.events.isin(BB_EV).astype(int)
    E["xb"] = E.events.map(TB).fillna(0).astype(int) - E.h
    E["pa"] = 1
    cnt = ["pa", "h", "ab", "k", "bb", "xb"]

    # starts: first pitcher of each team-game; BF from the box score when cached, else counted PAs (addendum A1 change 7)
    sp = E[E.is_sp].groupby(["game_pk", "pit_side"]).agg(pitcher=("pitcher", "first"), date=("date", "first"),
                                                       bf_pm=("pa", "sum")).reset_index()
    bsp = boxp[boxp.order.eq(0)].rename(columns={"side": "pit_side"})[["game_pk", "pit_side", "pitcher", "bf"]]
    sp = sp.merge(bsp, on=["game_pk", "pit_side", "pitcher"], how="left")
    sp["bf"] = sp.bf.fillna(sp.bf_pm).astype(int)
    # relief pitches per (game, team, pitcher): box numberOfPitches when cached, else pitch rows
    rp = pm[~pm.is_sp].groupby(["game_pk", "pit_side", "pitcher"]).agg(date=("date", "first"), team=("pit_team", "first"),
                                                                      pitches_pm=("pitch_number", "size")).reset_index()
    brp = boxp[boxp.order.gt(0)].rename(columns={"side": "pit_side"})[["game_pk", "pit_side", "pitcher", "pitches"]]
    rp = rp.astype({"pitcher": "int64"}).merge(brp, on=["game_pk", "pit_side", "pitcher"], how="left")
    rp["pitches"] = rp.pitches.fillna(rp.pitches_pm).astype(int)
    # lineup slots: first 9 distinct batters of each team-game (as B2)
    first = E.drop_duplicates(["game_pk", "bat_side", "batter"]).copy()
    first["slot"] = first.groupby(["game_pk", "bat_side"]).cumcount() + 1
    first = first[first.slot <= 9].rename(columns={"bat_side": "side"})
    sl = E.rename(columns={"bat_side": "side"}).merge(first[["game_pk", "side", "batter", "slot"]], on=["game_pk", "side", "batter"])
    slot_daily = sl.groupby(["slot", "side", "date"]).agg(pa=("pa", "sum"), games=("game_pk", "nunique")).reset_index()
    # park: per game H and PA (both teams). Excluded: one-off sites, and 2020 games away from the home team's
    # 2019 and 2021 usual venue (relocations; a new 2020 park such as Globe Life Field is its 2021 venue)
    pk_ = E.groupby("game_pk")[["h", "pa"]].sum().join(gd[["date", "season", "venue_id", "home_id", "away_id"]], how="inner")
    usual = {y: gd[gd.season.eq(y)].groupby("home_id").venue_id.agg(lambda v: v.mode().iloc[0]) for y in (2019, 2021)}
    home20 = pk_.venue_id.eq(pk_.home_id.map(usual[2019])) | pk_.venue_id.eq(pk_.home_id.map(usual[2021]))
    pk_ = pk_[~pk_.venue_id.isin(NEUTRAL_PARK) & (pk_.season.ne(2020) | home20)].reset_index()

    daily = lambda df, keys, cols: df.groupby(keys + ["date"])[cols].sum().reset_index()
    return {
        "bat": daily(E, ["batter"], cnt),
        "bat_vs": daily(E, ["batter", "p_throws"], ["h", "ab"]),
        "pit": daily(E, ["pitcher"], cnt),
        "lg": daily(E.assign(one=1), ["one"], cnt),
        "hand": E.groupby(["pitcher", "date"]).p_throws.last().reset_index(),
        "starts": sp.sort_values(["pitcher", "date"]).reset_index(drop=True),
        "lg_starts": daily(sp.assign(one=1, n=1), ["one"], ["bf", "n"]),
        "relief": rp.sort_values(["team", "date"]).reset_index(drop=True),
        "slot": slot_daily,
        "park_venue": daily(pk_, ["venue_id"], ["h", "pa"]),
        "park_road": daily(pk_.rename(columns={"away_id": "team"}), ["team"], ["h", "pa"]),
    }


# ---------------------------------------------------------------- windows
def window(daily: pd.DataFrame, keys: list, q: pd.DataFrame, qkeys: list, cols: list, lo: pd.Series) -> pd.DataFrame:
    """Sum of `cols` over [lo, D) per query row (D = q.date), and the latest contributing date (NaT if none)."""
    d = daily.sort_values("date").copy()
    c = d.groupby(keys)[cols].cumsum()
    d[cols] = c
    d["src"] = d.date
    q = q[qkeys + ["date"]].rename(columns=dict(zip(qkeys, keys))).assign(_i=np.arange(len(q)), _lo=lo.to_numpy().astype("datetime64[ns]"))
    hi = pd.merge_asof(q.sort_values("date"), d, on="date", by=keys, allow_exact_matches=False).sort_values("_i")
    lo_ = pd.merge_asof(q.sort_values("_lo").drop(columns="date").rename(columns={"_lo": "date"}), d.drop(columns="src"),
                        on="date", by=keys, allow_exact_matches=False).sort_values("_i")
    out = pd.DataFrame({k: hi[k].fillna(0).to_numpy() - lo_[k].fillna(0).to_numpy() for k in cols})
    out["src"] = hi.src.to_numpy()
    out.loc[out[cols[0]].eq(0), "src"] = pd.NaT  # nothing in the window
    return out


def build(raw: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Feature columns for `rows` (date, id, slot, side, opp_sp, opp_team, bat_team, home_id, venue_id, season)."""
    D = rows.date
    y365 = D - pd.Timedelta(days=365)
    F = pd.DataFrame(index=rows.index)
    cnt = ["pa", "h", "ab", "k", "bb", "xb"]

    # A. batter, trailing 365 days; platoon vs the opposing starter's hand
    w = window(raw["bat"], ["batter"], rows, ["id"], cnt, y365)
    F[[f"a_{c}" for c in cnt]] = w[cnt].to_numpy()
    hand = pd.merge_asof(rows[["date", "opp_sp"]].assign(_i=np.arange(len(rows))).sort_values("date"),
                         raw["hand"].rename(columns={"pitcher": "opp_sp", "date": "hdate"}).assign(date=lambda x: x.hdate).sort_values("date"),
                         on="date", by="opp_sp", allow_exact_matches=False).sort_values("_i")
    F["c_hand"] = hand.p_throws.to_numpy()
    vs = window(raw["bat_vs"], ["batter", "p_throws"], rows.assign(p_throws=F.c_hand.fillna("?").to_numpy()),
                ["id", "p_throws"], ["h", "ab"], y365)
    F["a_vs_h"], F["a_vs_ab"] = vs.h.to_numpy(), vs.ab.to_numpy()
    F.loc[F.c_hand.isna(), ["a_vs_h", "a_vs_ab"]] = np.nan
    F["A_max_date"] = pd.concat([w.src, vs.src], axis=1).max(axis=1).to_numpy()

    # league rates in the same window (shrinkage targets)
    one = rows.assign(one=1)
    lg = window(raw["lg"], ["one"], one, ["one"], cnt, y365)
    for name, num, den in (("lg_h_ab", "h", "ab"), ("lg_k_pa", "k", "pa"), ("lg_bb_pa", "bb", "pa"), ("lg_xb_ab", "xb", "ab")):
        F[name] = (lg[num] / lg[den]).to_numpy()
    lgs = window(raw["lg_starts"], ["one"], one, ["one"], ["bf", "n"], y365)
    F["lg_bf_per_start"] = (lgs.bf / lgs.n).to_numpy()
    F["L_max_date"] = pd.concat([lg.src, lgs.src], axis=1).max(axis=1).to_numpy()

    # C. opposing (actual) starter, trailing 365 days; last 10 starts' BF
    pw = window(raw["pit"], ["pitcher"], rows, ["opp_sp"], cnt, y365)
    F[[f"c_{c}" for c in cnt]] = pw[cnt].to_numpy()
    st = raw["starts"].copy()
    st["k_idx"] = st.groupby("pitcher").cumcount() + 1
    st["cum_bf"] = st.groupby("pitcher").bf.cumsum()
    k = pd.merge_asof(rows[["date", "opp_sp"]].assign(_i=np.arange(len(rows))).sort_values("date"),
                      st[["pitcher", "date", "k_idx", "cum_bf"]].rename(columns={"pitcher": "opp_sp"}).assign(sdate=lambda x: x.date)
                      .sort_values("date"),
                      on="date", by="opp_sp", allow_exact_matches=False).sort_values("_i")
    kk = k.k_idx.fillna(0).astype(int).to_numpy()
    back = st.set_index(["pitcher", "k_idx"]).cum_bf
    prev = back.reindex(pd.MultiIndex.from_arrays([rows.opp_sp.to_numpy(), kk - 10])).fillna(0).to_numpy()
    F["c_starts_last10"] = np.minimum(kk, 10)
    F["c_bf_last10"] = k.cum_bf.fillna(0).to_numpy() - np.where(kk > 10, prev, 0)
    F["C_max_date"] = pd.concat([pw.src, k.sdate.reset_index(drop=True), hand.hdate.reset_index(drop=True)], axis=1).max(axis=1).to_numpy()

    # D. expected PA by slot and side (league, trailing 365 days); the split uses C's BF at fit time (derive)
    sw = window(raw["slot"], ["slot", "side"], rows, ["slot", "side"], ["pa", "games"], y365)
    F["d_exp_pa"] = (sw.pa / sw.games).to_numpy()
    F["D_max_date"] = sw.src.to_numpy()

    # E. opposing bullpen: relievers used in [D-30, D), pitch-weighted pooled H/AB allowed (trailing 365);
    #    fatigue = relief pitches and distinct relievers in [D-3, D)
    R = raw["relief"]
    by_team = {t: g for t, g in R.groupby("team")}
    tq = rows[["opp_team", "date", "opp_sp"]].drop_duplicates().reset_index(drop=True)
    pairs, fat = [], []
    for t, d, s in tq.itertuples(index=False):
        g = by_team.get(t)
        if g is None:
            fat.append((0, 0, pd.NaT))
            continue
        dd = g.date.to_numpy()
        a, b = np.searchsorted(dd, np.datetime64(d - pd.Timedelta(days=30))), np.searchsorted(dd, np.datetime64(d))
        u = g.iloc[a:b]
        u = u[u.pitcher.ne(s)].groupby("pitcher").agg(w=("pitches", "sum"), rdate=("date", "max")).reset_index()
        pairs.append(u.assign(opp_team=t, date=d, opp_sp=s))
        a3 = np.searchsorted(dd, np.datetime64(d - pd.Timedelta(days=3)))
        f3 = g.iloc[a3:b]
        fat.append((int(f3.pitches.sum()), f3.pitcher.nunique(), f3.date.max() if len(f3) else pd.NaT))
    pr = pd.concat(pairs, ignore_index=True)
    rw = window(raw["pit"], ["pitcher"], pr, ["pitcher"], ["h", "ab"], pr.date - pd.Timedelta(days=365))
    pr["wh"], pr["wab"], pr["psrc"] = pr.w * rw.h.to_numpy(), pr.w * rw.ab.to_numpy(), rw.src.to_numpy()
    # keyed by starter too: a doubleheader gives the team two lookups that day, each counted once
    agg = pr.groupby(["opp_team", "date", "opp_sp"]).agg(wh=("wh", "sum"), wab=("wab", "sum"), n=("pitcher", "size"),
                                               src=("rdate", "max"), psrc=("psrc", "max")).reset_index()
    tq = pd.concat([tq, pd.DataFrame(fat, columns=["e_fat_pitches_3d", "e_fat_relievers_3d", "fsrc"])], axis=1)
    tq = tq.merge(agg, on=["opp_team", "date", "opp_sp"], how="left")
    tq["e_bp_hab"] = tq.wh / tq.wab.where(tq.wab > 0)
    tq["e_bp_relievers_30d"] = tq.n.fillna(0)
    tq["E_max_date"] = pd.concat([tq.src, tq.psrc, tq.fsrc], axis=1).max(axis=1)
    m = rows[["opp_team", "date", "opp_sp"]].merge(tq, on=["opp_team", "date", "opp_sp"], how="left")
    for c in ("e_bp_hab", "e_bp_relievers_30d", "e_fat_pitches_3d", "e_fat_relievers_3d", "E_max_date"):
        F[c] = m[c].to_numpy()

    # F. park: venue H/PA and the home team's road H/PA, previous three seasons + current season before D
    lo = pd.to_datetime((rows.season - 3).astype(str) + "-01-01")
    pv = window(raw["park_venue"], ["venue_id"], rows, ["venue_id"], ["h", "pa"], lo)
    prd = window(raw["park_road"], ["team"], rows, ["home_id"], ["h", "pa"], lo)
    F["f_venue_h"], F["f_venue_pa"], F["f_road_h"], F["f_road_pa"] = pv.h.to_numpy(), pv.pa.to_numpy(), prd.h.to_numpy(), prd.pa.to_numpy()
    F["F_max_date"] = pd.concat([pv.src, prd.src], axis=1).max(axis=1).to_numpy()

    # G. recent form: batter H/AB over [D-14, D) and [D-30, D)
    g14 = window(raw["bat"], ["batter"], rows, ["id"], ["h", "ab"], D - pd.Timedelta(days=14))
    g30 = window(raw["bat"], ["batter"], rows, ["id"], ["h", "ab"], D - pd.Timedelta(days=30))
    F["g_h14"], F["g_ab14"], F["g_h30"], F["g_ab30"] = g14.h.to_numpy(), g14.ab.to_numpy(), g30.h.to_numpy(), g30.ab.to_numpy()
    F["G_max_date"] = pd.concat([g14.src, g30.src], axis=1).max(axis=1).to_numpy()
    return F


# ---------------------------------------------------------------- same-day attributes (plan Section 3 exceptions)
def weather(info: pd.DataFrame) -> pd.DataFrame:
    """Box-score game-time weather (addendum A1 change 6): roof closed / dome -> no temperature + indoor flag."""
    w = info[["game_pk"]].copy()
    w["f_indoor"] = info.weather_txt.str.contains("Roof Closed|Dome", case=False).astype(int)
    t = pd.to_numeric(info.weather_txt.str.extract(r"(-?\d+)\s*degrees", flags=re.I)[0], errors="coerce")
    w["f_temp_f"] = t.where(w.f_indoor.eq(0))
    w["x_wind_mph"] = pd.to_numeric(info.wind_txt.str.extract(r"(\d+)\s*mph", flags=re.I)[0], errors="coerce")  # exploratory
    return w


def market(G: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Exploratory 'M1 + market' (addendum A1 change 4): team implied runs from the closing total and moneyline (4B)."""
    odds = OddsBook(str(ROOT / "data" / "MLB_Basic.csv"))._index.reset_index()
    g = G.merge(info[["game_pk", "away_runs", "home_runs"]], on="game_pk")
    out = []
    for r in g.itertuples():
        o, _ = match_odds(odds, r.date.strftime("%Y-%m-%d"), team_abbr(r.away_name), team_abbr(r.home_name), r.away_runs, r.home_runs)
        row = {"game_pk": r.game_pk, "x_mkt_home_runs": np.nan, "x_mkt_away_runs": np.nan}
        if o is not None and pd.notna(o.get("away_ml_close")) and pd.notna(o.get("home_ml_close")) and pd.notna(o.get("over_close")):
            ih, ia = (100 / (v + 100) if v > 0 else -v / (-v + 100) for v in (o["home_ml_close"], o["away_ml_close"]))
            p = ih / (ih + ia)
            ratio = (p / (1 - p)) ** (1 / 1.83)
            row["x_mkt_home_runs"] = o["over_close"] * ratio / (1 + ratio)
            row["x_mkt_away_runs"] = o["over_close"] / (1 + ratio)
        out.append(row)
    return pd.DataFrame(out)


def pool_rows(sched: pd.DataFrame, bats: pd.DataFrame, boxp: pd.DataFrame) -> pd.DataFrame:
    G = sched[sched.pool_eligible & sched.season.isin(POOL_SEASONS)]
    P = bats.merge(G[["game_pk", "date", "season", "game_number", "venue_id", "home_id", "away_id"]], on="game_pk")
    sp = boxp[boxp.order.eq(0)].set_index(["game_pk", "side"]).pitcher
    opp = np.where(P.side.eq("away"), "home", "away")
    P["opp_sp"] = sp.reindex(pd.MultiIndex.from_arrays([P.game_pk, opp])).to_numpy().astype("int64")
    P["opp_team"] = np.where(P.side.eq("away"), P.home_id, P.away_id)
    P["bat_team"] = np.where(P.side.eq("away"), P.away_id, P.home_id)
    return P


def baselines(rows: pd.DataFrame, pm: pd.DataFrame, T: dict) -> pd.DataFrame:
    """Comparators B1 and B2 exactly as in R2 (v4_analysis.add_baselines on its own baseline_tables, i.e. all
    pitch_matrix rows with game_date < D), with the actual starter (addendum A1 change 3). B2 is also exploratory H."""
    import v4_analysis as va
    P = rows[["id", "slot", "side", "opp_sp"]].assign(date=rows.date.dt.strftime("%Y-%m-%d"), _i=np.arange(len(rows)))
    K = P[["date", "opp_sp"]].drop_duplicates().rename(columns={"opp_sp": "pitcher"})
    out = va.add_baselines(P, K, T)[0].sort_values("_i")
    # latest pitch_matrix date those windows used: batter, starter and league rows in [D-365, D)
    e = pm[pm.events.notna()].astype({"batter": "int64", "pitcher": "int64"}).assign(date=lambda d: pd.to_datetime(d.game_date).astype("datetime64[ns]"), pa=1)
    lo = rows.date - pd.Timedelta(days=365)
    src = [window(e.groupby([k, "date"]).pa.sum().reset_index(), [k], rows, [q], ["pa"], lo).src
           for k, q in (("batter", "id"), ("pitcher", "opp_sp"))]
    src.append(window(e.assign(one=1).groupby(["one", "date"]).pa.sum().reset_index(), ["one"], rows.assign(one=1), ["one"], ["pa"], lo).src)
    return pd.DataFrame({"base_b1": out.b1.to_numpy(), "base_b2": out.b2.to_numpy(),
                         "BASE_max_date": pd.concat(src, axis=1).max(axis=1).to_numpy()}, index=rows.index)


def full_build() -> tuple[pd.DataFrame, dict]:
    import v4_analysis as va
    sched = load_schedule()
    G = sched[sched.pool_eligible & sched.season.isin(POOL_SEASONS)]
    bats, boxp, info = load_box(G)
    P = pool_rows(sched, bats, boxp)
    n_all = len(P)
    P = P[~P.venue_id.isin(NEUTRAL_POOL)].reset_index(drop=True)
    pm = load_pitch_matrix()
    F = pd.concat([build(raw_tables(pm, sched, boxp), P), baselines(P, pm, va.baseline_tables())], axis=1)
    X = pd.concat([P, F], axis=1).merge(weather(info), on="game_pk", how="left").merge(market(G, info), on="game_pk", how="left")
    X["x_mkt_team_runs"] = np.where(X.side.eq("home"), X.x_mkt_home_runs, X.x_mkt_away_runs)
    X = X.drop(columns=["x_mkt_home_runs", "x_mkt_away_runs"])
    meta = {"eligible_games": len(G), "games_with_box": int(info.game_pk.nunique()), "rows_before_neutral": n_all,
            "neutral_rows": n_all - len(P), "neutral_games": int(G.venue_id.isin(NEUTRAL_POOL).sum())}
    return X, meta


# ---------------------------------------------------------------- fit-time derivation (per training fold)
def estimate_m(x: np.ndarray, n: np.ndarray, mu, min_n: float = 100, N=None) -> float:
    """Beta-binomial method-of-moments prior strength m for the rate x/n shrunk toward mu, rows with n >= min_n:
    E[n (x/n - mu)^2] = n c^2 tau2 + mu(1-mu) c, so tau2 = (sum n d2 - sum mu(1-mu) c) / sum n c^2 and m = mean mu(1-mu) / tau2.
    c = 1, or c = 1 - n/N when x/n is a subset of a window of N trials whose raw rate is mu (platoon, recent form)."""
    mu = np.broadcast_to(np.asarray(mu, float), n.shape)
    c = np.ones(n.shape) if N is None else 1 - n / np.where(N > 0, N, np.nan)
    k = (n >= min_n) & np.isfinite(mu) & np.isfinite(c)
    x, n, mu, c = x[k], n[k], mu[k], c[k]
    w = mu * (1 - mu)
    tau2 = ((n * (x / n - mu) ** 2).sum() - (w * c).sum()) / (n * c ** 2).sum()
    return float(w.mean() / tau2) if tau2 > 0 else np.inf  # no between-unit spread: shrink fully


def shrink(x, n, mu, m):
    if np.isinf(m):
        return np.broadcast_to(np.asarray(mu, float), np.shape(x)).copy()
    return (x + m * mu) / (n + m)


def pa_split(exp_pa, bf_exp, slot):
    vs_sp = np.minimum(exp_pa, np.maximum(0, (bf_exp - slot) / 9 + 1))
    return vs_sp, exp_pa - vs_sp


def _self_check():
    rng = np.random.default_rng(0)
    p = rng.beta(30, 70, 2000)  # true rates, variance 0.3*0.7/101
    n = rng.integers(100, 600, 2000)
    m = estimate_m(rng.binomial(n, p).astype(float), n.astype(float), 0.3)
    assert 60 < m < 160, m  # true prior strength 100
    N = rng.integers(400, 600, 4000)  # nested: subset of n of N trials, true subset deviation SD 0.02
    tot = rng.uniform(.22, .3, 4000)
    nn = (N * rng.uniform(.25, .4, 4000)).astype(int)
    xs = rng.binomial(nn, np.clip(tot + rng.normal(0, .02, 4000), 0, 1))
    xa = xs + rng.binomial(N - nn, tot)
    mn = estimate_m(xs.astype(float), nn.astype(float), xa / N, min_n=1, N=N.astype(float))
    assert 300 < mn < 800, mn  # true about 0.19 / 0.0004 = 480
    assert np.isclose(pa_split(np.array([4.3]), np.array([24.0]), np.array([1]))[0][0], 23 / 9 + 1)  # leadoff, 24 BF
    assert np.isclose(pa_split(np.array([3.8]), np.array([10.0]), np.array([9]))[0][0], 10 / 9)    # 9th, short start
    assert pa_split(np.array([4.0]), np.array([40.0]), np.array([1]))[0][0] == 4.0                # capped at E_PA


def coverage(X: pd.DataFrame) -> None:
    """Counts only: rows, passes, exclusions, missing share per feature, thin history. No outcome is related to a feature."""
    pd.set_option("display.width", 200)
    sched = load_schedule()
    S = sched[sched.season.isin(POOL_SEASONS)]
    print("## Rows by season\n" + pd.DataFrame({
        "final regular-season games": S.groupby("season").size(),
        "pool-eligible games": S[S.pool_eligible].groupby("season").size(),
        "neutral-site games excluded": S[S.pool_eligible & S.venue_id.isin(NEUTRAL_POOL)].groupby("season").size(),
        "games in table": X.groupby("season").game_pk.nunique(), "rows": X.groupby("season").size(),
        "rows AB>=1": X[X.ab >= 1].groupby("season").size(), "rows AB=0 (passes)": X[X.ab == 0].groupby("season").size(),
    }).fillna(0).astype(int).to_string())
    miss = {  # "missing" = no history in the window (zero denominator) or no value
        "A batter PA, 365d (a_pa = 0)": X.a_pa.eq(0), "A platoon AB vs starter hand (none or hand unknown)": X.a_vs_ab.fillna(0).eq(0),
        "C starter hand unknown": X.c_hand.isna(), "C starter BF, 365d (c_pa = 0)": X.c_pa.eq(0),
        "C no prior start (last-10 BF)": X.c_starts_last10.eq(0), "D expected PA by slot": X.d_exp_pa.isna(),
        "E bullpen H/AB, 30d relievers": X.e_bp_hab.isna(), "F venue PA (prior seasons + this season)": X.f_venue_pa.eq(0),
        "F home team road PA": X.f_road_pa.eq(0), "F temperature (indoor or not recorded)": X.f_temp_f.isna(),
        "F of which indoor (roof closed / dome)": X.f_indoor.eq(1), "G batter AB, 14d": X.g_ab14.eq(0), "G batter AB, 30d": X.g_ab30.eq(0),
        "league rates": X.lg_h_ab.isna(), "comparator B1 (needs >= 200 PA)": X.base_b1.isna(), "comparator B2": X.base_b2.isna(),
        "exploratory market runs": X.x_mkt_team_runs.isna(), "exploratory wind": X.x_wind_mph.isna(),
    }
    M = pd.DataFrame({k: v.groupby(X.season).mean() for k, v in miss.items()}).T
    M.insert(0, "all", [v.mean() for v in miss.values()])
    print("\n## Share of rows missing, by feature\n" + M.round(4).to_string())
    print("\nE fatigue (3-day relief pitches) is a count; zero is a value, not missing: share zero "
          f"{X.e_fat_pitches_3d.eq(0).mean():.4f}.")
    thin = pd.DataFrame({"batter < 100 PA, 365d": X.a_pa.lt(100).groupby(X.season).mean(),
                         "starter < 100 BF, 365d": X.c_pa.lt(100).groupby(X.season).mean(),
                         "rows": X.groupby("season").size()})
    thin.loc["all"] = [X.a_pa.lt(100).mean(), X.c_pa.lt(100).mean(), len(X)]
    print("\n## Thin history (share of rows)\n" + thin.round(4).to_string())


if __name__ == "__main__":
    if "--coverage" in sys.argv:
        coverage(pd.read_parquet(OUT_PARQUET))
        sys.exit()
    _self_check()
    X, meta = full_build()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PARQUET, index=False)
    print(meta)
    print(f"wrote {OUT_PARQUET} rows {len(X)} cols {X.shape[1]}")
