"""Lab 2 (pitch-type matchups), Step 0: read-only inventory of the pitch data. Counts only; no features, no models,
no outcome analysis, nothing fetched.

  python analysis/lab2_step0.py > analysis/out/lab2_step0_log.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
import btslab_features as bf  # noqa: E402

PM, VAULT = ROOT / "data" / "pitch_matrix.parquet", ROOT / "data" / "master_physics_vault.parquet"
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)

DESC = {  # one line per column (Statcast field meanings)
    "game_pk": "MLB game id", "game_date": "local game date", "game_year": "season", "game_type": "R regular, S spring, F/D/L/W postseason, E exhibition",
    "home_team": "home team code", "away_team": "away team code", "inning": "inning number", "inning_topbot": "Top or Bot half",
    "outs_when_up": "outs before the pitch", "on_1b": "runner id on 1B (null if empty)", "on_2b": "runner id on 2B", "on_3b": "runner id on 3B",
    "at_bat_number": "PA sequence number within the game", "pitch_number": "pitch number within the PA", "pitcher": "pitcher id", "batter": "batter id",
    "stand": "batter side for this pitch (L/R)", "p_throws": "pitcher hand (L/R)", "events": "PA outcome, on the PA's last pitch only",
    "description": "pitch result (ball, called_strike, swinging_strike, foul, hit_into_play, ...)", "type": "B ball / S strike / X in play",
    "post_away_score": "away score after the pitch", "post_home_score": "home score after the pitch", "away_score": "away score before",
    "home_score": "home score before", "pitch_type": "pitch type code (Statcast)", "release_speed": "release speed, mph",
    "pfx_x": "horizontal movement, ft", "pfx_z": "vertical movement, ft", "plate_x": "horizontal location at the plate, ft",
    "plate_z": "vertical location at the plate, ft", "balls": "balls before the pitch", "strikes": "strikes before the pitch",
    "zone": "Statcast zone 1-14", "launch_speed": "exit velocity, mph (balls in play)", "launch_angle": "launch angle, degrees",
    "hc_x": "hit coordinate x", "hc_y": "hit coordinate y"}
FAMILY = {"FF": "fastball", "SI": "fastball", "FA": "fastball", "FC": "fastball (cutter, ambiguous)",
          "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking", "KC": "breaking", "CS": "breaking",
          "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
          "KN": "other", "EP": "other", "PO": "other", "IN": "other", "AB": "other", "UN": "other"}


def say(title, df=None):
    print(f"\n### {title}" + ("" if df is None else "\n" + (df if isinstance(df, str) else df.to_string())), flush=True)


def main():
    # ---- 1. columns
    pm_cols = dict(zip(pq.ParquetFile(PM).schema_arrow.names, map(str, pq.ParquetFile(PM).schema_arrow.types)))
    v_cols = dict(zip(pq.ParquetFile(VAULT).schema_arrow.names, map(str, pq.ParquetFile(VAULT).schema_arrow.types)))
    say("1. Columns (pitch_matrix has 33; master_physics_vault has the same rows plus 5 more)", pd.DataFrame(
        [{"column": c, "type": t, "in pitch_matrix": c in pm_cols, "in vault": c in v_cols, "holds": DESC.get(c, "?")} for c, t in v_cols.items()]
    ).to_string(index=False))
    need = {"pitch type": ["pitch_type"], "pitch result": ["description", "type"], "PA outcome": ["events"],
            "launch speed": ["launch_speed"], "launch angle": ["launch_angle"],
            "expected stats (estimated_ba/woba_using_speedangle)": ["estimated_woba_using_speedangle", "estimated_ba_using_speedangle"],
            "plate location": ["plate_x", "plate_z", "zone"], "ball-strike count": ["balls", "strikes"],
            "batter / pitcher hand": ["stand", "p_throws"], "game id and date": ["game_pk", "game_date"], "pitch number in PA": ["pitch_number"],
            "game type": ["game_type"]}
    say("1b. Needed fields", pd.DataFrame([{"field": k, "pitch_matrix": "present" if all(c in pm_cols for c in v) else "absent",
                                            "master_physics_vault": "present" if all(c in v_cols for c in v) else "absent", "columns": ", ".join(v)}
                                           for k, v in need.items()]).to_string(index=False))

    # ---- 2. coverage
    V = pd.read_parquet(VAULT, columns=["game_pk", "game_date", "game_year", "game_type", "pitch_type", "release_speed", "type",
                                        "events", "launch_speed", "launch_angle", "pitcher", "batter", "stand", "p_throws"])
    V["date"] = pd.to_datetime(V.game_date).astype("datetime64[ns]")
    bip = V.type.eq("X")
    cov = V.groupby("game_year").agg(rows=("game_pk", "size"), games=("game_pk", "nunique"))
    cov["regular-season rows"] = V[V.game_type.eq("R")].groupby("game_year").size()
    cov["non-null pitch_type"] = V.pitch_type.notna().groupby(V.game_year).mean().round(4)
    cov["non-null release_speed"] = V.release_speed.notna().groupby(V.game_year).mean().round(4)
    cov["balls in play"] = bip.groupby(V.game_year).sum()
    cov["BIP with launch_speed"] = V.launch_speed.notna()[bip].groupby(V.game_year[bip]).mean().round(4)
    cov["BIP with launch_angle"] = V.launch_angle.notna()[bip].groupby(V.game_year[bip]).mean().round(4)
    cov["expected-stat columns"] = "absent"
    say("2. Coverage by season (master_physics_vault; same rows as pitch_matrix)", cov)
    say("2b. Game types by season", V.groupby(["game_year", "game_type"]).size().unstack(fill_value=0))
    cm = pq.ParquetFile(ROOT / "data" / "contact_matrix_env_fixed.parquet")
    print(f"\nThe project's own xwOBA estimate (xwoba_est, from launch speed/angle via data/xwoba_lookup.parquet) exists only in "
          f"data/contact_matrix_env_fixed.parquet ({cm.metadata.num_rows:,} balls in play); it is not Statcast's expected stats.")

    # ---- 3. vocabulary
    voc = V.groupby(["pitch_type", "game_year"]).size().unstack(fill_value=0)
    voc.insert(0, "family (proposed)", voc.index.map(lambda c: FAMILY.get(c, "UNMAPPED")))
    say("3. Pitch-type codes, pitches per season (all game types)", voc)
    first = (voc.drop(columns="family (proposed)") > 0).idxmax(axis=1)
    say("3b. First season each code appears", first.to_frame("first season").T)
    sl = voc.loc[[c for c in ("SL", "ST", "SV", "CU", "KC", "CS") if c in voc.index]].drop(columns="family (proposed)")
    say("3c. Breaking-ball codes as a share of all breaking balls, by season (the 2023 sweeper/slurve split)",
        (sl / sl.sum()).round(3))

    # ---- 4. sample sizes, 2024 batter-games and starts, trailing 365 days, regular season only
    R = V[V.game_type.eq("R") & V.pitch_type.notna()].astype({"batter": "int64", "pitcher": "int64"})
    R["fam"] = R.pitch_type.map(lambda c: FAMILY.get(c, "other")).str.split(" ").str[0]
    R["end"] = R.events.notna().astype(int)
    R["n"] = 1
    X = pd.read_parquet(bf.OUT_PARQUET, columns=["date", "id", "opp_sp", "season"])  # 2024 pool rows only, no outcomes
    X = X[X.season.eq(2024)].reset_index(drop=True)
    lo = X.date - pd.Timedelta(days=365)
    bat_side = R.groupby(["batter", "stand"]).n.sum().unstack(fill_value=0)
    X["primary_stand"] = X.id.map(bat_side.idxmax(axis=1))  # side with most pitches seen, 2019-2025 (descriptive split only)
    rows = []
    for fam in ("fastball", "breaking", "offspeed"):
        Rf = R[R.fam.eq(fam)]
        daily = Rf.groupby(["batter", "stand", "date"])[["n", "end"]].sum().reset_index()
        w = bf.window(daily, ["batter", "stand"], X.assign(stand=X.primary_stand), ["id", "stand"], ["n", "end"], lo)
        for side in ("L", "R"):
            k = (X.primary_stand == side).to_numpy()
            for what, col in (("pitches seen", "n"), ("PA-ending pitches", "end")):
                v = w[col].to_numpy()[k]
                rows.append({"unit": "batter-game (batter's main side)", "side": side, "family": fam, "count": what, "rows": int(k.sum()),
                             "p10": int(np.percentile(v, 10)), "median": int(np.median(v)), "p90": int(np.percentile(v, 90))})
        dp = Rf.groupby(["pitcher", "stand", "date"])[["n", "end"]].sum().reset_index()
        S = X.drop_duplicates(["date", "opp_sp"]).reset_index(drop=True)
        for side in ("L", "R"):
            w = bf.window(dp, ["pitcher", "stand"], S.assign(stand=side), ["opp_sp", "stand"], ["n", "end"], S.date - pd.Timedelta(days=365))
            for what, col in (("pitches thrown", "n"), ("PA-ending pitches", "end")):
                v = w[col].to_numpy()
                rows.append({"unit": "starter-start (vs batter side)", "side": side, "family": fam, "count": what, "rows": len(S),
                             "p10": int(np.percentile(v, 10)), "median": int(np.median(v)), "p90": int(np.percentile(v, 90))})
    say("4. 2024 sample sizes per family, trailing 365 days strictly before the game date, regular season (counts only)",
        pd.DataFrame(rows).to_string(index=False))

    # ---- 5. 2026
    print("\n### 5. 2026 pitch-level data")
    print(f"vault 2026 rows: {int(V.game_year.eq(2026).sum())}; pitch_matrix max date: {pd.read_parquet(PM, columns=['game_date']).game_date.max().date()}; "
          f"statcast_chunks 2026 files: {len(list((ROOT / 'data' / 'statcast_chunks').glob('statcast_2026-*')))}")
    print("Postgres (Step 0 of the BTS lab): four app-log tables, no pitch-level table. pg_dump/*.dump are dumps of that database.")

    # ---- 6. PAs with no pitch row: box-score PA vs pitch_matrix PA rows, 2022-2025 eligible games
    S = bf.load_schedule()
    G = S[S.pool_eligible & S.season.isin(bf.POOL_SEASONS)]
    box = []
    for r in G.itertuples():
        b = json.loads((bf.API / "box" / f"{r.game_pk}.json").read_bytes())["teams"]
        box.append({"game_pk": r.game_pk, "season": r.season,
                    "box_pa": sum(b[s]["teamStats"]["batting"].get("plateAppearances", 0) for s in ("away", "home")),
                    "box_ibb": sum(b[s]["teamStats"]["batting"].get("intentionalWalks", 0) for s in ("away", "home")),
                    "box_bb": sum(b[s]["teamStats"]["batting"].get("baseOnBalls", 0) for s in ("away", "home"))})
    B = pd.DataFrame(box)
    pmpa = V[V.events.notna()].groupby("game_pk").size().rename("pitch_rows_pa")
    pmw = V[V.events.eq("walk")].groupby("game_pk").size().rename("pitch_rows_walk")
    pmt = V[V.events.eq("truncated_pa")].groupby("game_pk").size().rename("truncated_pa")
    B = B.join(pmpa, on="game_pk").join(pmw, on="game_pk").join(pmt, on="game_pk").fillna(0)
    t = B.groupby("season")[["box_pa", "pitch_rows_pa", "truncated_pa", "box_ibb", "box_bb", "pitch_rows_walk"]].sum()
    # box PA = pitch-row PA - truncated_pa (baserunning out ends the inning; not a PA) + IBB (no pitch) + PAs whose pitches were dropped
    t["box PA - (pitch-row PA - truncated) - IBB = PAs lost"] = t.box_pa - (t.pitch_rows_pa - t.truncated_pa) - t.box_ibb
    t["walks: box BB - IBB - pitch-row walks"] = t.box_bb - t.box_ibb - t.pitch_rows_walk
    say("6. Plate appearances with no pitch row: box score vs pitch data, 2022-2025 eligible games", t.astype(int))
    print("events present in pitch rows:", sorted(V.events.dropna().unique()))
    print("intent_walk rows:", int(V.events.eq("intent_walk").sum()))
    d = B.box_pa - (B.pitch_rows_pa - B.truncated_pa) - B.box_ibb
    print(f"games where PAs are lost (d > 0): {int((d > 0).sum())} of {len(B)}; d < 0: {int((d < 0).sum())}; "
          f"distribution of d: {d.value_counts().sort_index().to_dict()}")
    print(f"eligible games with no pitch rows at all: {int(B.pitch_rows_pa.eq(0).sum())}")


if __name__ == "__main__":
    main()
