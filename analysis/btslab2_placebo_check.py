"""Lab 2 Step 1d: outcome-free check of the T4 placebo design (placebo U, A3 item 7 as written; placebo P, the
starter-side profile moved as a bundle). Reads only the lab 2 feature table (counts and league levels); no outcome
column, box-score file or 2026 data.

  python analysis/btslab2_placebo_check.py > analysis/out/btslab2_placebo_check_log.txt
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab2_features as b2  # noqa: E402

RULE = ("A placebo is usable only if the maximum Pearson correlation between real and placebo I1 over both folds and all "
        "five permutations is at most 0.30. Reason: the planted effect (c = 0.10) gives a log-loss gain of about 0.0009; a "
        "placebo sharing a fraction r^2 of the real feature's variance reproduces about r^2 of that gain, and r^2 of at most "
        "0.09 keeps it near 0.0001, the test's resolving power. Rule: if U is usable, T4 stays as written. Else if P is "
        "usable, P replaces U as T4 and the placebo for M4, with the other mechanics of A3 item 7 unchanged. Else neither "
        "is a usable stop-rule placebo: both are reported descriptively, and the Step 1d addendum drops the placebo stop "
        "rule and the planted test's 'placebo not Confirmed' condition, stating that plainly.")
THRESHOLD = 0.30
TABLE_CONTENT = "62445424bda0c468f7dab50225a5a1d7fb346206e113a5cb18ef7367b7beb8f7"
SEEDS = [20261150 + k for k in range(5)]
WF = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
UNIT = ["game_pk", "opp_sp", "side_s"]


def i1(u, Lf, rb, rp, Rb, Rp):
    """I1 = sum_f u(f|s) L_f (r_b(f) r_p(f|s) - R_b R_p(s)), summed in the builder's order (apply_m)."""
    out = np.zeros(len(Rb))
    for j in range(3):
        out += u[:, j] * Lf[:, j] * (rb[:, j] * rp[:, j] - Rb * Rp)
    return out


def main():
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Lab 2 placebo design check ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}); outcome-free")
    print("\nDECISION RULE (fixed before any computation):")
    print(RULE)
    print(f"\nrule written at {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}; computation starts now")

    # hashes: plan and A1-A5 against the latest PLAN_HASH lines, and the table's content hash
    rec = {}
    for line in (b2.ROOT / "analysis" / "out" / "PLAN_HASH.txt").read_text(encoding="utf-8").splitlines():
        mm = re.match(r"^(lab2(?: A\d)?) sha256: (\w+)", line)
        if mm:
            rec[mm.group(1)] = mm.group(2)
    files = {"lab2": "ARSENAL_PLAN", **{f"lab2 A{i}": f"ADDENDUM_A{i}" for i in range(1, 6)}}
    for lab, n in files.items():
        h = b2.b3.sha(b2.ROOT / "analysis" / f"BEAT_THE_STREAK_LAB2_{n}.md")
        assert h == rec[lab], f"stop: {lab} hash differs"
        print(f"{lab}: {h} matches PLAN_HASH.txt")
    F = pd.read_parquet(b2.OUT_PARQUET)
    ch = b2.b3.content_hash(F)
    assert ch == TABLE_CONTENT, f"stop: table content hash {ch}"
    print(f"table {b2.OUT_PARQUET.name}: content sha256 {ch} matches; rows {len(F):,}")

    rows = []
    for tr, te in WF:
        m, _ = b2.fold_m(F, F.season.isin(tr).to_numpy())
        A = b2.apply_m(F, m)
        T = pd.concat([F[["game_pk", "opp_sp", "side_s", "season", "us_n", "I1_miss"]], A.drop(columns="I1_miss")], axis=1)
        T = T[T.season.eq(te)]
        u = T[[f"u_{f}" for f in b2.F3]].to_numpy()
        rb, rp = T[[f"r_b_{f}" for f in b2.F3]].to_numpy(), T[[f"r_p_{f}" for f in b2.F3]].to_numpy()
        Lf = F.loc[T.index, [f"L_{f}" for f in b2.F3]].to_numpy()
        Rb, Rp = T.R_b.to_numpy(), T.R_p.to_numpy()
        keep = T.I1_miss.to_numpy() == 0
        real = np.where(keep, i1(u, Lf, rb, rp, Rb, Rp), 0.0)
        assert np.array_equal(real, T.I1.to_numpy()), "real I1 differs from the builder's apply_m output"
        # eligible units: starter has three-family pitches to side s; the starter-side profile is constant within a unit
        el = T.side_s.notna().to_numpy() & (T.us_n.to_numpy() > 0)
        prof = T.loc[el, UNIT + [f"u_{f}" for f in b2.F3] + [f"r_p_{f}" for f in b2.F3] + ["R_p"]]
        assert (prof.groupby(UNIT).nunique(dropna=False) <= 1).all().all(), "starter-side profile not constant within a unit"
        units = prof.drop_duplicates(UNIT).sort_values(UNIT, kind="mergesort").reset_index(drop=True)
        units["season"] = te
        print(f"\nfold train {'-'.join(map(str, tr))} -> test {te}: test rows {len(T):,}; with I1_miss = 0 {int(keep.sum()):,}; "
              f"eligible units {len(units):,} (side L {int(units.side_s.eq('L').sum()):,}, R {int(units.side_s.eq('R').sum()):,}); "
              f"real I1 equals the builder's I1: True; m = " + ", ".join(f"{k} {v:.4g}" for k, v in m.items()))
        ukey = pd.MultiIndex.from_frame(T[UNIT])
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            donor = np.arange(len(units))
            for (_, side), g in units.groupby(["season", "side_s"], sort=True):
                idx = g.index.to_numpy()
                donor[idx] = idx[rng.permutation(len(idx))]
            D = units.iloc[donor].reset_index(drop=True)
            D.index = pd.MultiIndex.from_frame(units[UNIT])
            pos = D.index.get_indexer(ukey)  # -1 for rows outside eligible units (those rows have I1_miss = 1)
            assert (pos[keep] >= 0).all()
            p = np.where(pos >= 0, pos, 0)
            du = D[[f"u_{f}" for f in b2.F3]].to_numpy()[p]
            drp = D[[f"r_p_{f}" for f in b2.F3]].to_numpy()[p]
            dRp = D.R_p.to_numpy()[p]
            plU = np.where(keep, i1(du, Lf, rb, rp, Rb, Rp), 0.0)
            plP = np.where(keep, i1(du, Lf, rb, drp, Rb, dRp), 0.0)
            moved = float((donor != np.arange(len(units))).mean())
            for name, pl in (("U", plU), ("P", plP)):
                r = float(np.corrcoef(real[keep], pl[keep])[0, 1])
                rows.append({"fold": f"{'-'.join(map(str, tr))} -> {te}", "placebo": name, "k": seed - 20261150, "seed": seed,
                             "corr": r, "SD ratio": float(pl[keep].std() / real[keep].std()), "units moved": moved})

    R = pd.DataFrame(rows)
    print("\n## Correlation between real and placebo I1, and SD(placebo I1) / SD(real I1); test-season rows with I1_miss = 0")
    print(R.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("\n## Per placebo, across both folds and k = 0..4")
    S = R.groupby("placebo")["corr"].agg(["mean", "min", "max"])
    print(S.to_string(float_format=lambda z: f"{z:.4f}"))
    usable = {p: bool(S.loc[p, "max"] <= THRESHOLD) for p in ("U", "P")}
    print(f"\nusable (max correlation <= {THRESHOLD}): U {usable['U']}, P {usable['P']}")
    if usable["U"]:
        branch = "U is usable: T4 stays as written."
    elif usable["P"]:
        branch = ("U is not usable; P is usable: P replaces U as T4 and the placebo for M4, with the other mechanics of "
                  "A3 item 7 unchanged.")
    else:
        branch = ("Neither U nor P is usable: both are reported descriptively, and the Step 1d addendum drops the placebo "
                  "stop rule and the planted test's 'placebo not Confirmed' condition, stating that plainly.")
    print(f"BRANCH OF THE RULE THAT APPLIES: {branch}")
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
