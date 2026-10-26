"""Option A: rebuild the env contact matrix with a LEFT join so Arizona/Athletics home batted balls
(home_team AZ / ATH, absent from the weather log) are kept with blank weather.

Reads data/contact_matrix.parquet + data/game_weather_log.parquet (same steps as enviroment_merger.py,
but how='left'), adds xwoba_est with the same lookup xwoba_lookup.py built from the existing env file,
adds weather_source ('log' / 'missing'). Writes data/contact_matrix_env_fixed.parquet only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import xwoba_lookup as xl  # noqa: E402

DATA = ROOT / "data"


def main():
    cqm = pd.read_parquet(DATA / "contact_matrix.parquet")
    old = pd.read_parquet(DATA / "contact_matrix_env.parquet")
    w = pd.read_parquet(DATA / "game_weather_log.parquet").rename(
        columns={"temperature_f": "temperature", "elevation_m": "elevation"})
    assert not w.duplicated(["game_date", "home_team"]).any(), "weather log not unique per game day"

    # --- same steps as enviroment_merger.py, but a LEFT join ---
    cqm["game_date"] = pd.to_datetime(cqm["game_date"]).dt.strftime("%Y-%m-%d")
    w["game_date"] = pd.to_datetime(w["game_date"]).dt.strftime("%Y-%m-%d")
    env = cqm.merge(w[["game_date", "home_team", "temperature", "elevation"]],
                    on=["game_date", "home_team"], how="left")
    env["density_ratio"] = 1.0 - env["elevation"] * 0.000035 - (env["temperature"] - 60) * 0.001
    env["weather_source"] = np.where(env["temperature"].notna(), "log", "missing")

    # --- xwoba_est: rebuild the lookup from the old file's rows, prove it reproduces them, apply to all ---
    fine, coarse, gmean = xl.build_lookup(xl._assign_woba_value(old))
    re_old = xl.apply_lookup(old.drop(columns=["xwoba_est"]), fine, coarse, gmean)
    assert np.allclose(re_old["xwoba_est"].to_numpy(), old["xwoba_est"].to_numpy()), "lookup does not reproduce old xwoba_est"
    env = xl.apply_lookup(env, fine, coarse, gmean)

    fixed = env[list(old.columns) + ["weather_source"]].copy()
    # int64 cannot hold blanks; nullable Int64 keeps integer values for the kept rows
    fixed["elevation"] = fixed["elevation"].astype("Int64")

    # --- verification ---
    assert len(fixed) == len(cqm) == 786_886, len(fixed)
    s_new = pd.to_datetime(fixed.game_date).dt.year.value_counts().sort_index()
    s_cqm = pd.to_datetime(cqm.game_date).dt.year.value_counts().sort_index()
    assert s_new.equals(s_cqm), "per-season counts differ from contact_matrix.parquet"
    dup_fixed, dup_cqm = int(fixed.duplicated().sum()), int(cqm.duplicated().sum())
    assert dup_fixed == dup_cqm, (dup_fixed, dup_cqm)
    kept = fixed[fixed.weather_source.eq("log")].drop(columns="weather_source").reset_index(drop=True)
    assert len(kept) == len(old) == 732_668
    oldr = old.reset_index(drop=True)
    pd.testing.assert_frame_equal(kept.drop(columns="elevation"), oldr.drop(columns="elevation"), check_dtype=True)
    assert (kept["elevation"].astype("int64") == oldr["elevation"]).all()
    miss = fixed[fixed.weather_source.eq("missing")]
    assert len(miss) == 54_218 and set(miss.home_team) == {"AZ", "ATH"}
    assert miss[["temperature", "elevation", "density_ratio"]].isna().all().all()

    out = DATA / "contact_matrix_env_fixed.parquet"
    fixed.to_parquet(out, engine="pyarrow", index=False)
    print(f"wrote {out} ({len(fixed):,} rows)")
    print("per season (fixed == contact_matrix.parquet):")
    t = pd.DataFrame({"contact_matrix": s_cqm, "fixed": s_new,
                      "recovered": pd.to_datetime(miss.game_date).dt.year.value_counts().sort_index()})
    print(t.to_string())
    print(f"duplicate rows: fixed {dup_fixed}, contact_matrix {dup_cqm}")
    print("732,668 previously kept rows identical to data/contact_matrix_env.parquet: same order and values in every column; same dtypes except elevation int64 -> Int64 (nullable): OK")
    print(f"recovered rows: {len(miss):,} (AZ {int(miss.home_team.eq('AZ').sum()):,}, ATH {int(miss.home_team.eq('ATH').sum()):,}); "
          f"weather columns all blank; xwoba_est filled for all: {bool(miss.xwoba_est.notna().all())}")


if __name__ == "__main__":
    main()
