"""Lab 2, Step 1b: collect 2026 data (data collection only; no 2026 outcome, hit rate or model result is computed).

  python analysis/lab2_step1b.py sched                 # 2026 schedules + box scores into analysis/cache/api/
  python analysis/lab2_step1b.py validate pybaseball 2025-08-04 2025-08-10   # re-pull one week, compare with the vault
  python analysis/lab2_step1b.py pull pybaseball       # 2026 regular-season date range, monthly chunks, resumable

Writes only new files: data/statcast_2026.parquet, analysis/cache/statcast_2026_chunks/, analysis/cache/lab2_step1b_val_*.parquet,
and new 2026 files under analysis/cache/api/. Never writes the vault, data/statcast_chunks or any existing data/ file.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
from backtest_v4 import API, get_json  # noqa: E402

VAULT = ROOT / "data" / "master_physics_vault.parquet"
OUT_2026 = ROOT / "data" / "statcast_2026.parquet"
CHUNKS = ROOT / "analysis" / "cache" / "statcast_2026_chunks"
WEEK1 = ("2025-06-02", "2025-06-08")  # Monday-Sunday; week 2 (lab 2 addendum A2 c) is 2025-08-04 to 2025-08-10
FAMILY = {**dict.fromkeys(["FF", "SI", "FA", "FC"], "fastball"), **dict.fromkeys(["SL", "ST", "SV", "CU", "KC", "CS"], "breaking"),
          **dict.fromkeys(["CH", "FS", "FO", "SC"], "offspeed")}  # plan Section 4; other codes -> "other"
KEY = ["game_pk", "at_bat_number", "pitch_number"]
SCHEMA = pq.read_schema(VAULT)  # CORE_COLUMNS of data_ingestion.py, with the vault's types
CORE = SCHEMA.names
SCHED_RANGE = (date(2026, 3, 1), date(2026, 11, 5))  # as the 2019 and 2021 caches


def fetch(source: str, start: str, end: str) -> pd.DataFrame:
    if source == "pybaseball":
        import pybaseball as pyb
        return pyb.statcast(start_dt=start, end_dt=end, verbose=False)
    if source == "pybaseballstats":
        import pybaseballstats.statcast as pbs
        return pbs.pitch_by_pitch_data(start, end, force_collect=True, show_progress=False).to_pandas()
    if source == "savant":  # direct Baseball Savant CSV, one request per day (the endpoint caps rows per request)
        import io
        import requests
        out, d = [], date.fromisoformat(start)
        while d <= date.fromisoformat(end):
            r = requests.get("https://baseballsavant.mlb.com/statcast_search/csv", timeout=120, params={
                "all": "true", "type": "details", "player_type": "pitcher", "game_date_gt": d.isoformat(),
                "game_date_lt": d.isoformat(), "hfGT": "", "sort_col": "pitches", "sort_order": "desc"})
            r.raise_for_status()
            if r.content.strip():
                out.append(pd.read_csv(io.BytesIO(r.content), low_memory=False))
            d += timedelta(days=1)
        return pd.concat(out, ignore_index=True)
    raise SystemExit(f"source {source} not implemented")


def core(df: pd.DataFrame) -> pa.Table:
    """data_ingestion.py's column set and filter (drop null pitch_type or release_speed), cast to the vault schema."""
    missing = [c for c in CORE if c not in df.columns]
    assert not missing, f"columns missing from the source: {missing}"
    df = df[CORE].dropna(subset=["pitch_type", "release_speed"]).copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.drop_duplicates(KEY).sort_values(["game_date"] + KEY)
    return pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)


def season_games(year: int) -> pd.DataFrame:
    """Regular-season games from the cached daily schedules: one row per gamePk, last status seen."""
    rows = []
    for f in sorted((API / "schedule").glob(f"{year}-*.json")):
        for day in json.loads(f.read_bytes()).get("dates", []):
            for g in day["games"]:
                st = g["status"]
                rows.append({"game_pk": g["gamePk"], "file_date": f.stem, "game_type": g.get("gameType"),
                             "final": st.get("abstractGameState") == "Final"
                             and st.get("detailedState") not in ("Postponed", "Cancelled"),
                             "state": st.get("detailedState")})
    S = pd.DataFrame(rows)
    return S


def cmd_sched():
    d, n_new, fails = SCHED_RANGE[0], 0, []
    while d <= SCHED_RANGE[1]:
        ds = d.isoformat()
        f = API / "schedule" / f"{ds}.json"
        n_new += not f.exists()
        try:
            get_json(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={ds}&hydrate=probablePitcher,linescore", f)
        except Exception as e:
            fails.append((ds, type(e).__name__))
        d += timedelta(days=1)
    print(f"schedule days {SCHED_RANGE[0]} to {SCHED_RANGE[1]}: {(SCHED_RANGE[1] - SCHED_RANGE[0]).days + 1}; newly fetched {n_new}; "
          f"failed {len(fails)} {fails[:10]}")
    S = season_games(2026)
    print("schedule entries by gameType (one row per game-day listing):", S.game_type.value_counts().to_dict())
    R = S[S.game_type.eq("R")]
    fin = R[R.final].drop_duplicates("game_pk", keep="last")
    print(f"regular season: distinct gamePk listed {R.game_pk.nunique():,}; Final {fin.game_pk.nunique():,}; "
          f"dates with a Final game {fin.file_date.min()} to {fin.file_date.max()}")
    never = set(R.game_pk) - set(fin.game_pk)
    if never:
        print(f"regular-season gamePk never Final: {len(never)}; last states "
              f"{R[R.game_pk.isin(never)].drop_duplicates('game_pk', keep='last').state.value_counts().to_dict()}")
    n_new, bfails = 0, []
    for pk in sorted(fin.game_pk):
        f = API / "box" / f"{pk}.json"
        n_new += not f.exists()
        try:
            get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", f)
        except Exception as e:
            bfails.append((pk, type(e).__name__))
    print(f"box scores for Final regular-season games: {len(fin):,}; newly fetched {n_new:,}; failed {len(bfails)} {bfails[:20]}")


def cmd_validate(source: str, start: str, end: str):
    """Plan Section 10.1 as amended by lab 2 addendum A2 c. Pass: row count within 0.5%; per-pitch family agreement
    >= 99.5% of matched pitches; release speed and launch speed both equal on >= 99.5%. Exact label and events are descriptive."""
    tmp = ROOT / "analysis" / "cache" / f"lab2_step1b_val_{source}_{start}_{end}.parquet"
    legacy = ROOT / "analysis" / "cache" / f"lab2_step1b_val_{source}.parquet"  # week 1, cached before A2
    if (start, end) == WEEK1 and legacy.exists():
        tmp = legacy
    if not tmp.exists():
        pq.write_table(core(fetch(source, start, end)), tmp)
    new = pd.read_parquet(tmp)
    old = pd.read_parquet(VAULT, filters=[("game_date", ">=", pd.Timestamp(start)), ("game_date", "<=", pd.Timestamp(end))])
    rows_ok = abs(len(new) - len(old)) / len(old) <= 0.005
    print(f"validation week {start} to {end}, source {source} ({tmp.name}): vault rows {len(old):,}; re-pull rows {len(new):,}; "
          f"difference {(len(new) - len(old)) / len(old):+.4%} (limit +/-0.5%): {'pass' if rows_ok else 'FAIL'}")
    print(f"  by game_type: vault {old.game_type.value_counts().to_dict()}; re-pull {new.game_type.value_counts().to_dict()}")
    print(f"  distinct game_pk: vault {old.game_pk.nunique()}; re-pull {new.game_pk.nunique()}")
    m = old.merge(new, on=KEY, suffixes=("_v", "_n"), how="outer", indicator=True)
    print(f"  keys: matched {int(m._merge.eq('both').sum()):,}; vault only {int(m._merge.eq('left_only').sum()):,}; "
          f"re-pull only {int(m._merge.eq('right_only').sum()):,}")
    b = m[m._merge.eq("both")].copy()
    same = lambda c: (b[f"{c}_v"].eq(b[f"{c}_n"]) | (b[f"{c}_v"].isna() & b[f"{c}_n"].isna()))
    b["fam_v"], b["fam_n"] = b.pitch_type_v.map(FAMILY).fillna("other"), b.pitch_type_n.map(FAMILY).fillna("other")
    fam = b.fam_v.eq(b.fam_n)
    speed = same("release_speed") & same("launch_speed")
    fam_ok, speed_ok = fam.mean() >= 0.995, speed.mean() >= 0.995
    print(f"  per-pitch family agreement: {fam.mean():.5f} ({int((~fam).sum()):,} differ; limit >= 0.995): {'pass' if fam_ok else 'FAIL'}")
    print(f"  release speed and launch speed equal: {speed.mean():.5f} (release {same('release_speed').mean():.5f}, "
          f"launch {same('launch_speed').mean():.5f}; limit >= 0.995): {'pass' if speed_ok else 'FAIL'}")
    print("  descriptive only:")
    lab = same("pitch_type")
    print(f"    exact pitch_type agreement {lab.mean():.5f} ({int((~lab).sum()):,} differ: {int((~lab & fam).sum()):,} within family, "
          f"{int((~fam).sum()):,} across); events agreement {same('events').mean():.5f}")
    cnt = pd.DataFrame({"vault": old.pitch_type.map(FAMILY).fillna("other").value_counts(),
                        "re-pull": new.pitch_type.map(FAMILY).fillna("other").value_counts()}).fillna(0).astype(int)
    print("    family counts (the plan's former rule): " + "; ".join(
        f"{f} {r.vault:,} vs {r['re-pull']:,} ({r['re-pull'] / r.vault - 1:+.2%})" for f, r in cnt.iterrows()))
    bad = b[~lab]
    if len(bad):
        print("    relabels (vault -> re-pull): " + "; ".join(
            f"{a} -> {c} x{n}" for (a, c), n in bad.groupby(["pitch_type_v", "pitch_type_n"], dropna=False).size()
            .sort_values(ascending=False).items()))
        print(f"    relabelled pitchers {bad.pitcher_v.nunique()} of {b.pitcher_v.nunique()}; pitches per pitcher "
              f"{bad.groupby('pitcher_v').size().sort_values(ascending=False).tolist()}")
    ok = rows_ok and fam_ok and speed_ok
    print(f"VALIDATION {source} {start} to {end} (plan 10.1 as amended by lab 2 addendum A2): {'PASS' if ok else 'FAIL'}")


def cmd_pull(source: str):
    S = season_games(2026)
    fin = S[S.game_type.eq("R") & S.final]
    first, last = date.fromisoformat(fin.file_date.min()), date.fromisoformat(fin.file_date.max())
    CHUNKS.mkdir(parents=True, exist_ok=True)
    chunks, d = [], first
    while d <= last:
        end = min(date(d.year + (d.month == 12), d.month % 12 + 1, 1) - timedelta(days=1), last)
        chunks.append((d, end))
        d = end + timedelta(days=1)
    for a, b in chunks:
        f = CHUNKS / f"statcast_{a:%Y-%m}.parquet"
        if f.exists():
            print(f"  [skip] {f.name}")
            continue
        t = core(fetch(source, a.isoformat(), b.isoformat()))
        pq.write_table(t, f.with_suffix(".tmp"))
        f.with_suffix(".tmp").replace(f)  # a crash never leaves a partial chunk under the final name
        print(f"  [saved] {f.name}: {a} to {b}, {t.num_rows:,} rows")
    D = pd.concat([pd.read_parquet(CHUNKS / f"statcast_{a:%Y-%m}.parquet") for a, _ in chunks], ignore_index=True)
    D = D.drop_duplicates(KEY).sort_values(["game_date"] + KEY)
    assert not OUT_2026.exists() or OUT_2026.stat().st_size > 0
    pq.write_table(pa.Table.from_pandas(D, schema=SCHEMA, preserve_index=False), OUT_2026)
    print(f"\nwrote {OUT_2026.relative_to(ROOT)}: {len(D):,} rows; pull range {first} to {last}; "
          f"game_date {D.game_date.min().date()} to {D.game_date.max().date()}")
    print("rows by game_type:", D.game_type.value_counts().to_dict())
    print("distinct game_pk by game_type:", D.groupby("game_type").game_pk.nunique().to_dict(), f"; total {D.game_pk.nunique():,}")
    print("null share per column:\n" + D.isna().mean().round(4).to_string())
    sched_r = set(fin.game_pk)
    got_r = set(D.game_pk[D.game_type.eq("R")])
    print(f"regular-season games: schedule Final {len(sched_r):,} (expected 2,430); in statcast_2026 {len(got_r):,}; "
          f"schedule-only {len(sched_r - got_r)}; statcast-only {len(got_r - sched_r)} {sorted(got_r - sched_r)[:10]}")
    if sched_r - got_r:
        print("  schedule-only gamePk:", sorted(sched_r - got_r)[:20])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"sched": cmd_sched, "validate": cmd_validate, "pull": cmd_pull}[cmd](*rest)
