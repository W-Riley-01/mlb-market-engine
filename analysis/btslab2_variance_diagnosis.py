"""Lab 2: per-trial variance of the per-AB measure x, descriptive (cited by addendum A5). No fit and no m.

Reads only regular-season vault pitches dated 2022-2023 (the training window of the first walk-forward fold), the
schedule (2026 files skipped, as lab 3's load_schedule) and the contact table. No outcome column, box-score file or
2026 data. x is formed exactly as btslab2_features.daily_tables() forms it.

  python analysis/btslab2_variance_diagnosis.py > analysis/out/btslab2_variance_diagnosis_log.txt
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab2_features as b2  # noqa: E402
from lab2_contact_table import ANGLE_EDGES, SPEED_EDGES, bins  # noqa: E402

LO, HI = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")


def x_values(ab: pd.DataFrame, P: np.ndarray) -> np.ndarray:
    """As in btslab2_features.daily_tables(): contact-table p_hit for launch-data balls in play, the actual result
    (1 for a hit) for balls in play with no launch data, 0 for every other AB-ending pitch."""
    launch = ab.launch_speed.notna() & ab.launch_angle.notna()
    bip = ab.type.eq("X")
    x = np.zeros(len(ab))
    li = (bip & launch).to_numpy()
    x[li] = P[bins(ab.launch_speed[li], SPEED_EDGES).to_numpy(int), bins(ab.launch_angle[li], ANGLE_EDGES).to_numpy(int)]
    x[(bip & ~launch).to_numpy()] = ab.events[bip & ~launch].isin(b2.bf.H_EV).to_numpy(float)
    return x


def cell(df: pd.DataFrame, unit: list) -> dict:
    """Units = unit-seasons with n >= 100 AB-ending pitches in the cell (20 if none reach 100, A3 item 13a)."""
    g = df.groupby(unit).x.agg(["size", "sum", lambda s: (s ** 2).sum()])
    g.columns = ["n", "sx", "sxx"]
    min_n = 100 if (g.n >= 100).any() else 20
    q = g[g.n >= min_n]
    within = float((q.sxx - q.sx ** 2 / q.n).sum() / (q.n - 1).sum())
    mu = df.x.mean()
    bound = mu * (1 - mu)
    return {"units": len(q), "min_n": min_n, "AB-ending pitches": len(df), "mean x": mu, "within-unit var": within,
            "league var": df.x.var(ddof=0), "binomial bound mu(1-mu)": bound, "within / bound": within / bound,
            "league / bound": df.x.var(ddof=0) / bound}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 100)
    print(f"Lab 2 variance diagnosis ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}); descriptive, no fit, no m")
    sched = b2.b3.load_schedule()
    gd = sched.set_index("game_pk").date
    v = pd.read_parquet(b2.VAULT, columns=b2.VAULT_COLS, filters=[("game_date", ">=", LO), ("game_date", "<", HI)])
    v = v[v.game_pk.isin(gd.index)].copy()
    v["date"] = v.game_pk.map(gd)
    v = v[(v.date >= LO) & (v.date < HI)]  # effective date in 2022-2023
    v["season"] = v.date.dt.year
    print(f"vault rows read (game_date 2022-2023) and kept as regular season with effective date 2022-2023: {len(v):,}; "
          f"dates {v.date.min():%Y-%m-%d} to {v.date.max():%Y-%m-%d}")
    v = v[v.events.ne("truncated_pa") | v.events.isna()]
    v = v[v.pitch_type.map(b2.FAM).notna()]
    ab = v[v.events.notna() & ~v.events.isin(b2.bf.NON_AB)].copy()
    ab["x"] = x_values(ab, b2.contact_array(pd.read_csv(b2.b3.CONTACT, comment="#")))
    ab["fam"] = ab.pitch_type.map(b2.FAM)
    print(f"AB-ending pitches (three families, truncated_pa dropped): {len(ab):,}")
    rows = [{"estimator": "R_b", "cell": "all", **cell(ab, ["batter", "season"])}]
    rows += [{"estimator": "R_p(s)", "cell": f"side {s}", **cell(ab[ab.stand.eq(s)], ["pitcher", "season"])} for s in ("L", "R")]
    rows += [{"estimator": "r_b(f)", "cell": f, **cell(ab[ab.fam.eq(f)], ["batter", "season"])} for f in b2.F3]
    rows += [{"estimator": "r_p(f|s)", "cell": f"{f}/{s}", **cell(ab[ab.fam.eq(f) & ab.stand.eq(s)], ["pitcher", "season"])}
             for f in b2.F3 for s in ("L", "R")]
    print("\nPer cell. units: unit-seasons (batter-season, or pitcher-season) with n >= min_n AB-ending pitches in the cell;")
    print("within-unit var: pooled, sum over units of (sum x^2 - n * mean^2) / sum over units of (n - 1);")
    print("league var: variance of x over all AB-ending pitches in the cell; binomial bound: mu(1 - mu) at the cell's mean x.")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda z: f"{z:.4f}"))


if __name__ == "__main__":
    main()
