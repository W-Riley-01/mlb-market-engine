"""Lab 3 data collection: player bio (MLB Stats API people endpoint) and Savant sprint speed by season.
Collection only: no batter-game outcome is read (plate appearances are counted as distinct game_pk + at_bat_number).

  python analysis/lab3_data.py > analysis/out/lab3_data_log.txt
writes analysis/cache/player_bio.parquet and analysis/cache/sprint_speed_by_season.parquet (raw people responses
cached under analysis/cache/api/people/).
"""
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
from backtest_v4 import API, get_json  # noqa: E402

VAULT = ROOT / "data" / "master_physics_vault.parquet"
CURRENT = ROOT / "data" / "statcast_current_2025_2026.parquet"
BIO = ROOT / "analysis" / "cache" / "player_bio.parquet"
SPRINT = ROOT / "analysis" / "cache" / "sprint_speed_by_season.parquet"
BATCH = 100
SEASONS = range(2019, 2026)
MIN_OPP = 0  # statcast_sprint_speed passes min_opp straight to Savant; 0 is the lowest value


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print(f"Lab 3 data collection log ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z})")
    print(f"python {sys.version.split()[0]}; pandas {pd.__version__}; pyarrow {version('pyarrow')}; pybaseball {version('pybaseball')}; "
          f"requests {version('requests')}")

    # ---- 1. player bio
    cols = ["game_pk", "game_year", "game_type", "at_bat_number", "batter", "pitcher"]
    V = pd.read_parquet(VAULT, columns=cols)
    C = pd.read_parquet(CURRENT, columns=cols)
    ids = sorted(set(V.batter) | set(V.pitcher) | set(C.batter) | set(C.pitcher))
    print(f"\n## 1. Player bio (https://statsapi.mlb.com/api/v1/people?personIds=..., batches of {BATCH})")
    print(f"distinct ids: vault 2019-2025 {len(set(V.batter) | set(V.pitcher)):,}; current pull 2025-2026 "
          f"{len(set(C.batter) | set(C.pitcher)):,}; union requested {len(ids):,}")
    people, new = [], 0
    for i in range(0, len(ids), BATCH):
        b = ids[i:i + BATCH]
        f = API / "people" / f"batch_{b[0]}_{b[-1]}_{len(b)}.json"
        new += not f.exists()
        people += get_json(f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, b))}", f)["people"]
    bio = pd.DataFrame([{"mlbam_id": p["id"], "full_name": p.get("fullName"), "birth_date": p.get("birthDate"),
                         "bat_side": p.get("batSide", {}).get("code"), "pitch_hand": p.get("pitchHand", {}).get("code")}
                        for p in people]).drop_duplicates("mlbam_id")
    bio["birth_date"] = pd.to_datetime(bio.birth_date)
    bio = bio.sort_values("mlbam_id").reset_index(drop=True)
    bio.to_parquet(BIO, index=False)
    print(f"batches {-(-len(ids) // BATCH)} (newly fetched {new}); ids returned {len(bio):,}; "
          f"requested but not returned {len(set(ids) - set(bio.mlbam_id))} {sorted(set(ids) - set(bio.mlbam_id))[:20]}")
    nob = bio[bio.birth_date.isna()]
    print(f"ids missing a birth date: {len(nob)} {nob.mlbam_id.tolist()[:20]}")
    print(f"missing bat_side {int(bio.bat_side.isna().sum())}; missing pitch_hand {int(bio.pitch_hand.isna().sum())}; "
          f"bat_side codes {bio.bat_side.value_counts().to_dict()}; pitch_hand codes {bio.pitch_hand.value_counts().to_dict()}")
    pa = V[V.game_year.between(2022, 2025) & V.game_type.eq("R")].drop_duplicates(["game_pk", "at_bat_number"])
    pa = pa.merge(bio, left_on="batter", right_on="mlbam_id", how="left")
    print(f"2022-25 regular-season plate appearances (distinct game_pk + at_bat_number in the vault): {len(pa):,}; "
          f"share whose batter has a birth date {pa.birth_date.notna().mean():.5f}; a bat side {pa.bat_side.notna().mean():.5f}; "
          f"both {(pa.birth_date.notna() & pa.bat_side.notna()).mean():.5f}")
    d0 = pd.Timestamp("2022-04-01")
    age = lambda s: (d0 - s).dt.days / 365.2425
    b22 = bio[bio.mlbam_id.isin(set(pa.batter) | set(V.pitcher[V.game_year.between(2022, 2025)]))]
    a_all, a_pa = age(bio.birth_date).dropna(), age(pa.drop_duplicates("batter").birth_date).dropna()
    print(f"age on 2022-04-01, all returned ids: min {a_all.min():.2f}, max {a_all.max():.2f} (n {len(a_all):,}); "
          f"2022-25 batters: min {a_pa.min():.2f}, median {a_pa.median():.2f}, max {a_pa.max():.2f} (n {len(a_pa):,}); "
          f"2022-25 batters and pitchers n {len(b22):,}")

    # ---- 2. sprint speed
    import pybaseball as pyb
    print(f"\n## 2. Sprint speed (pybaseball.statcast_sprint_speed(year, min_opp={MIN_OPP}); Savant sprint speed leaderboard)")
    rows = []
    for y in SEASONS:
        s = pyb.statcast_sprint_speed(y, min_opp=MIN_OPP)
        rows.append(pd.DataFrame({"mlbam_id": s.player_id.astype("int64"), "season": y, "sprint_speed": s.sprint_speed.astype(float),
                                  "competitive_runs": s.competitive_runs.astype("int64")}))
        print(f"{y}: rows {len(s):,}; competitive runs min {s.competitive_runs.min()}, max {s.competitive_runs.max()}; "
              f"columns {list(s.columns)}" if y == SEASONS[0] else f"{y}: rows {len(s):,}; competitive runs min {s.competitive_runs.min()}, "
              f"max {s.competitive_runs.max()}")
    sp = pd.concat(rows, ignore_index=True)
    assert not sp.duplicated(["mlbam_id", "season"]).any()
    sp.to_parquet(SPRINT, index=False)
    print(f"sprint speed: min {sp.sprint_speed.min():.1f}, median {sp.sprint_speed.median():.1f}, max {sp.sprint_speed.max():.1f} "
          f"(n {len(sp):,}; nulls {int(sp.sprint_speed.isna().sum())})")
    pa["prior"] = pa.game_year - 1
    hit = pa.merge(sp, left_on=["batter", "prior"], right_on=["mlbam_id", "season"], how="left", suffixes=("", "_sp"))
    has = hit.sprint_speed.notna()
    print("2022-25 plate appearances whose batter has a PRIOR-season sprint speed: "
          + "; ".join(f"{y} {has[hit.game_year.eq(y)].mean():.4f}" for y in range(2022, 2026)) + f"; all {has.mean():.4f}")
    print("same, batter has >= 10 prior-season competitive runs: "
          f"{(hit.competitive_runs >= 10).mean():.4f}")

    # ---- 3. files
    print(f"\n## 3. Files\n{BIO.relative_to(ROOT)}: {len(bio):,} rows, sha256 {sha(BIO)}\n"
          f"{SPRINT.relative_to(ROOT)}: {len(sp):,} rows, sha256 {sha(SPRINT)}")


if __name__ == "__main__":
    main()
