"""Step 0: reproduce misc/calibration_report.txt, then attribute every change
to a documented exclusion or the >= totals fix."""
from pathlib import Path

import pandas as pd

import lib

REPORT = Path(__file__).parents[1] / "misc" / "calibration_report.txt"
METRICS = ["n", "mean_predicted", "mean_actual", "brier", "baseline_brier", "skill_score", "ece", "bias"]


def parse_report() -> pd.DataFrame:
    lines = REPORT.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("market_type"))
    rows = []
    for l in lines[start + 1:]:
        parts = l.split()
        if len(parts) != 10:
            break
        rows.append([parts[0]] + [float(x) for x in parts[2:]])
    df = pd.DataFrame(rows, columns=["market_type"] + METRICS)
    df["n"] = df["n"].astype(int)
    return df.set_index("market_type")


def main():
    lib._self_check()
    lib.OUT.mkdir(exist_ok=True)
    games = lib.exclusion_flags(lib.load_games())
    players = lib.load_players()
    old = parse_report()

    # --- A. Legacy replication: calibration.py rules, as of the report ----
    window = (games["model_version"].eq("v2") & games["status"].eq("final")
              & games["iterations"].ge(5000)
              & games["game_date"].between("2026-05-01", "2026-05-26"))
    gw = games[window]
    lg = pd.concat([lib.long_games(gw, ge_fix=False),
                    lib.long_players(players, gw, void_zero_pa=False)])
    new = lib.market_summary(lg).set_index("market_type")
    print(f"Legacy replication: games={len(gw)} rows={len(lg)} markets={new.shape[0]} "
          f"(report: 285 games, 33,910 rows, 34 markets)")
    cmp_ = old.join(new, how="outer", rsuffix="_ours").rename(columns=dict(zip(METRICS, [m + "_report" for m in METRICS])))
    diffs = {m: (cmp_[f"{m}_ours"] - cmp_[f"{m}_report"]).abs().max() for m in METRICS}
    print("max |ours - report| per metric:", {k: round(v, 5) for k, v in diffs.items()})

    # gp 233 (rained out 05-09, made up 07-17) was unresolved when the report
    # ran; its outcome is now the makeup game's. Drop it and re-compare.
    asof = gw[~gw["x_postponed_attempt"]]
    la = pd.concat([lib.long_games(asof, ge_fix=False),
                    lib.long_players(players, asof, void_zero_pa=False)])
    new2 = lib.market_summary(la).set_index("market_type")
    cmp2 = old.join(new2, how="outer", rsuffix="_ours")
    d2 = {m: (cmp2[f"{m}_ours"] - cmp2[m]).abs().max() for m in METRICS}
    print(f"As-of-report (minus postponed attempt gp {gw.loc[gw['x_postponed_attempt'], 'gp_id'].tolist()}): "
          f"games={len(asof)} rows={len(la)}")
    print("max |ours - report| per metric:", {k: round(v, 5) for k, v in d2.items()},
          "(report prints 4 decimals; <=0.00005 is rounding)")
    cmp2.to_csv(lib.OUT / "step0_asof_vs_report.csv")

    # --- B. Attribute changes, one step at a time (same v2 window) -------
    steps = []
    excl_cols = ["x_postponed_attempt", "x_after_first_pitch"]
    g_ex = gw[~gw[excl_cols].any(axis=1)]
    print("\nGames removed from the report window by exclusion:")
    for c in excl_cols:
        print(f"  {c}: {int(gw[c].sum())}  gp_ids={sorted(gw.loc[gw[c], 'gp_id'].tolist())}")

    def summ(g, ge, void):
        return lib.market_summary(pd.concat([lib.long_games(g, ge_fix=ge),
                                             lib.long_players(players, g, void_zero_pa=void)]))

    stages = {
        "0_legacy": summ(gw, False, False),
        "1_+exclusions": summ(g_ex, False, False),
        "2_+void_zero_pa": summ(g_ex, False, True),
        "3_+ge_fix": summ(g_ex, True, True),
    }
    for name, s in stages.items():
        s["stage"] = name
        steps.append(s)
    st = pd.concat(steps)
    st.to_csv(lib.OUT / "step0_stages.csv", index=False)
    piv = st.pivot(index="market_type", columns="stage", values=["n", "bias"])
    pd.set_option("display.width", 200)
    print("\nPer-market n and bias by stage (only markets that change):")
    changed = piv[(piv["n"].nunique(axis=1) > 1) | (piv["bias"].round(6).nunique(axis=1) > 1)]
    print(changed.round(4).to_string())

    # --- C. Totals rung by rung, before/after the >= fix (same games) ---
    tot = (st[st["market_type"].str.startswith("over_total_")
              & st["stage"].isin(["2_+void_zero_pa", "3_+ge_fix"])]
           .pivot(index="market_type", columns="stage", values=["mean_predicted", "mean_actual", "bias"]))
    tot = tot.loc[sorted(tot.index, key=lambda m: float(m.rsplit("_", 1)[1]))]
    print("\nTotals ladder, v2 report window after exclusions: before (>) vs after (>=) fix")
    print(tot.round(4).to_string())


if __name__ == "__main__":
    main()
