"""Lab 2, Step 1a: the frozen contact-quality table P(hit | launch speed bin, launch angle bin)
(plan BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md Section 3, as amended by BEAT_THE_STREAK_LAB2_ADDENDUM_A1.md).
Fitted on 2020-2021 regular-season balls in play only. Checks: 1 (barrels vs weak contact) and the discrimination
rule on 2022 (log loss at least 0.10 below a constant rate), with a descriptive 2022 drift report. Descriptive: the two
oversized cells by season (launch data only; 2022 ball-in-play outcomes are read at league level only).

  python analysis/lab2_contact_table.py > analysis/out/lab2_contact_table_log.txt
writes analysis/lab2_contact_table.csv
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
import btslab_features as bf  # noqa: E402

VAULT = ROOT / "data" / "master_physics_vault.parquet"
OUT_CSV = ROOT / "analysis" / "lab2_contact_table.csv"
TRAIN = (2020, 2021)
TEST = 2022  # league-level ball-in-play outcomes only (addendum A1 d2)
HIT = {"single", "double", "triple", "home_run"}
SPEED_EDGES = [-np.inf] + list(range(60, 120, 5)) + [np.inf]   # <60, 60-65, ..., 110-115, >=115: 13 bins
ANGLE_EDGES = [-np.inf] + list(range(-30, 70, 10)) + [np.inf]  # <-30, -30--20, ..., 50-60, >=60: 11 bins
MIN_CELL = 100
HI_SB = SPEED_EDGES.index(105)  # 105-110, 110-115, >=115 mph: sparse cells borrow the next-lower speed (A1 d3)
ODD = {"80-85 mph, -30 to -20 deg": (5, 1), "80-85 mph, >= 60 deg": (5, 10)}  # (speed bin, angle bin)


def label(edges, i, unit):
    lo, hi = edges[i], edges[i + 1]
    return f"<{hi:g}{unit}" if lo == -np.inf else f">={lo:g}{unit}" if hi == np.inf else f"{lo:g}-{hi:g}{unit}"


def bins(v, edges):
    return pd.cut(v, edges, right=False, labels=False)  # left-closed: 60 belongs to 60-65


def fit_table(B: pd.DataFrame) -> pd.DataFrame:
    """143 cells; a cell with fewer than MIN_CELL balls takes its angle bin's pooled rate, except at >= 105 mph, where it
    takes the next-lower-speed cell in the same angle bin if that cell has >= MIN_CELL balls (not chained)."""
    cells = B.groupby(["ab", "sb"]).agg(n=("hit", "size"), hits=("hit", "sum"))
    full = pd.MultiIndex.from_product([range(11), range(13)], names=["ab", "sb"])
    T = cells.reindex(full, fill_value=0).reset_index()
    T = T.join(B.groupby("ab").agg(band_n=("hit", "size"), band_hits=("hit", "sum")), on="ab")
    T["p_raw"] = T.hits / T.n.where(T.n > 0)
    T["sparse"] = T.n < MIN_CELL
    T["p_pooled"] = T.band_hits / T.band_n
    lower_p, lower_n = T.groupby("ab").p_raw.shift(), T.groupby("ab").n.shift()  # rows are sb-ordered within ab
    borrow = T["sparse"] & T.sb.ge(HI_SB) & lower_n.ge(MIN_CELL)
    T["rule"] = np.where(~T["sparse"], "cell", np.where(borrow, "lower speed", "angle pool"))
    T["p_hit"] = np.where(~T["sparse"], T.p_raw, np.where(borrow, lower_p, T.p_pooled))
    return T


def lookup(T, B):
    return B[["ab", "sb"]].merge(T[["ab", "sb", "p_hit"]], on=["ab", "sb"], how="left").p_hit.to_numpy()


def main():
    assert len(SPEED_EDGES) - 1 == 13 and len(ANGLE_EDGES) - 1 == 11
    pd.set_option("display.width", 250, "display.max_columns", 20)
    sched = bf.load_schedule()
    rs = set(sched.game_pk)  # regular season per the schedule (gameType R), 2019-2025
    # launch data for every season (no outcome column); outcomes only for the training seasons
    L = pd.read_parquet(VAULT, columns=["game_pk", "game_year", "game_type", "type", "launch_speed", "launch_angle"])
    V = pd.read_parquet(VAULT, columns=["game_pk", "game_date", "game_year", "game_type", "type", "events", "launch_speed",
                                        "launch_angle"], filters=[("game_year", "in", [*TRAIN, TEST])])
    print(f"vault rows {TRAIN[0]}-{TEST}: {len(V):,}; in schedule regular-season games: {int(V.game_pk.isin(rs).sum()):,}; "
          f"vault game_type == 'R': {int(V.game_type.eq('R').sum()):,}; disagreements: "
          f"{int((V.game_pk.isin(rs) != V.game_type.eq('R')).sum()):,}")
    A = V[V.game_pk.isin(rs) & V.type.eq("X")].copy()
    A["hit"] = A.events.isin(HIT).astype(int)
    nolaunch = A.launch_speed.isna() | A.launch_angle.isna()
    print(f"balls in play without launch speed or angle (excluded), by season: {nolaunch.groupby(A.game_year).sum().to_dict()}")
    A = A[~nolaunch].copy()
    A["sb"], A["ab"] = bins(A.launch_speed, SPEED_EDGES), bins(A.launch_angle, ANGLE_EDGES)
    B, te = A[A.game_year.isin(TRAIN)].copy(), A[A.game_year.eq(TEST)].copy()
    dmin, dmax = pd.to_datetime(B.game_date).min(), pd.to_datetime(B.game_date).max()
    print(f"fitting balls in play: {len(B):,} (by season {B.game_year.value_counts().sort_index().to_dict()}); "
          f"game_date {dmin.date()} to {dmax.date()}")
    assert dmin >= pd.Timestamp("2020-01-01") and dmax < pd.Timestamp("2022-01-01"), "a row outside 2020-2021 in the fit"

    # ---- the frozen table, 2020-2021
    T = fit_table(B)
    T["speed_bin"] = [label(SPEED_EDGES, i, "mph") for i in T.sb]
    T["angle_bin"] = [label(ANGLE_EDGES, i, "deg") for i in T.ab]
    T["speed_lo"], T["speed_hi"] = [SPEED_EDGES[i] for i in T.sb], [SPEED_EDGES[i + 1] for i in T.sb]
    T["angle_lo"], T["angle_hi"] = [ANGLE_EDGES[i] for i in T.ab], [ANGLE_EDGES[i + 1] for i in T.ab]
    out = T[["angle_bin", "speed_bin", "angle_lo", "angle_hi", "speed_lo", "speed_hi", "n", "hits", "p_raw", "sparse",
             "p_pooled", "rule", "p_hit"]]
    with open(OUT_CSV, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# lab 2 contact-quality table, P(hit | launch speed bin, launch angle bin); built {date.today()}\n"
                 f"# training: {', '.join(map(str, TRAIN))} regular-season balls in play (schedule gameType R, type X), "
                 f"{len(B):,} with launch data; max game_date {dmax.date()}\n"
                 f"# bins left-closed; cells with n < {MIN_CELL} (sparse) take their angle bin's pooled rate, or at >= 105 mph "
                 f"the next-lower-speed cell's rate if that cell has n >= {MIN_CELL} (column rule); "
                 f"p_hit is the value to apply. Plan: BEAT_THE_STREAK_LAB2_ARSENAL_PLAN.md Section 3 + lab 2 addendum A1\n")
        out.to_csv(fh, index=False, float_format="%.6f", lineterminator="\n")
    print(f"wrote {OUT_CSV.name}: {len(out)} cells, {int(T['sparse'].sum())} sparse")
    lab_s = [label(SPEED_EDGES, i, "") for i in range(13)]
    lab_a = [label(ANGLE_EDGES, i, "") for i in range(11)]
    piv = lambda col: T.pivot(index="ab", columns="sb", values=col).set_axis(lab_a).set_axis(lab_s, axis=1)
    print("\n## Balls per cell (rows: launch angle, columns: launch speed mph)\n" + piv("n").to_string())
    print("\n## p_hit per cell (sparse cells show their angle bin's pooled rate)\n" + piv("p_hit").round(3).to_string())
    print(f"\nsparse cells (n < {MIN_CELL}): {int(T['sparse'].sum())} of {len(T)}; balls in sparse cells: {int(T.n[T['sparse']].sum()):,}")

    # ---- the high-speed sparse rule (A1 d3): before (angle pool) and after
    hi = T[T["sparse"] & T.sb.ge(HI_SB)]
    raw = T.set_index(["ab", "sb"]).p_raw
    assert all(r.p_hit == raw[(r.ab, r.sb - 1)] for r in hi[hi.rule.eq("lower speed")].itertuples())
    print("\n## Sparse cells at >= 105 mph: before (p_pooled, angle-bin pooled rate) and after (p_hit, A1 d3)\n"
          + hi[["speed_bin", "angle_bin", "n", "p_raw", "p_pooled", "rule", "p_hit"]].round(4).to_string(index=False))
    te["pred"] = lookup(T, te)
    te["pred_pooled"] = lookup(T.assign(p_hit=np.where(T["sparse"], T.p_pooled, T.p_raw)), te)
    h = te[te.sb.ge(HI_SB)].groupby("sb").agg(n=("hit", "size"), before=("pred_pooled", "mean"), after=("pred", "mean"),
                                              actual=("hit", "mean"))
    h.index = [label(SPEED_EDGES, i, " mph") for i in h.index]
    print(f"{TEST} balls at >= 105 mph, mean predicted before and after the rule vs actual:\n" + h.round(4).to_string())

    # ---- check 1 (plan Section 3, unchanged)
    B["p_hit"] = lookup(T, B)
    barrel = B[(B.launch_speed >= 98) & B.launch_angle.between(20, 35)]
    weak = B[B.launch_speed < 80]
    d = barrel.p_hit.mean() - weak.p_hit.mean()
    ok1 = d >= 0.25
    print(f"\n## Check 1, barrels vs weak contact: mean p_hit for speed >= 98 and angle 20-35 = {barrel.p_hit.mean():.4f} "
          f"(n {len(barrel):,}; raw hit rate {barrel.hit.mean():.4f}); speed < 80, all angles = {weak.p_hit.mean():.4f} "
          f"(n {len(weak):,}; raw {weak.hit.mean():.4f}); difference {d:+.4f}, required >= +0.25: {'PASS' if ok1 else 'FAIL'}")

    # ---- discrimination rule and drift report (lab 2 addendum A1 d2): the 2020-2021 table applied to 2022
    assert te.pred.notna().all()
    eps = 1e-6
    ll = lambda p, y: -(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))).mean()
    base = B.hit.mean()
    ll_t, ll_c = ll(te.pred.to_numpy(), te.hit.to_numpy()), ll(np.full(len(te), base), te.hit.to_numpy())
    ok2 = ll_c - ll_t >= 0.10
    print(f"\n## Discrimination rule: the 2020-2021 table applied to {TEST} ({len(te):,} regular-season balls in play with "
          f"launch data)\n{TEST} log loss: table {ll_t:.5f}; constant {base:.4f} (2020-2021 rate) {ll_c:.5f}; "
          f"improvement {ll_c - ll_t:.5f}, required >= 0.10: {'PASS' if ok2 else 'FAIL'}")
    g = te.groupby("sb").agg(n=("hit", "size"), predicted=("pred", "mean"), actual=("hit", "mean"))
    g.index = [lab_s[i] for i in g.index]
    g.loc["overall"] = [len(te), te.pred.mean(), te.hit.mean()]
    g["difference"] = g.predicted - g.actual
    print(f"\n## Drift report, {TEST} (descriptive, no pass/fail): mean predicted vs actual hit rate by speed bin (mph)\n"
          + g.astype({"n": int}).round(4).to_string())

    # ---- descriptive (lab 2 addendum A1 c): the two oversized cells, launch data only
    Lb = L[L.game_pk.isin(rs) & L.type.eq("X") & L.launch_speed.notna() & L.launch_angle.notna()].copy()
    Lb["sb"], Lb["ab"] = bins(Lb.launch_speed, SPEED_EDGES), bins(Lb.launch_angle, ANGLE_EDGES)
    rows = []
    for name, (s, a) in ODD.items():
        in_cell = Lb.sb.eq(s) & Lb.ab.eq(a)
        cnt = in_cell.groupby(Lb.game_year).sum()
        share = in_cell.groupby(Lb.game_year).mean()
        for y in sorted(Lb.game_year.unique()):
            rows.append({"cell": name, "season": y, "balls in play": int(cnt[y]), "share of season's BIP": f"{share[y]:.4f}"})
        top = Lb[in_cell].groupby(["launch_speed", "launch_angle"]).size().sort_values(ascending=False)
        print(f"\n{name}: most frequent recorded (speed, angle) pairs, all seasons: "
              + "; ".join(f"({sp:g}, {an:g}) x{n:,}" for (sp, an), n in top.head(3).items()))
    print("\n## Descriptive: the two oversized cells by season (regular-season balls in play, launch data only)")
    print(pd.DataFrame(rows).to_string(index=False))
    print("hit rates, fitting seasons 2020-2021 only: " + "; ".join(
        f"{name}: {B[B.sb.eq(s) & B.ab.eq(a)].hit.mean():.4f} (n {int((B.sb.eq(s) & B.ab.eq(a)).sum()):,})" for name, (s, a) in ODD.items()))

    print(f"\nCONTACT TABLE CHECKS (1 and discrimination): {'PASS' if ok1 and ok2 else 'FAIL - stop rule 4: do not build features'}")


if __name__ == "__main__":
    main()
