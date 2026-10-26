"""Run the frozen R2 plan (BACKTEST_V4_ANALYSIS_PLAN_R2.md, amended) on the six ledger_v4r2 ledgers; writes
analysis/BACKTEST_V4_RESULTS_R2.md. Reuses v4_integrity.py, v4_analysis.py and v4_exploratory.q5. Read-only."""
import hashlib
import sys

import pandas as pd

import v4_integrity as vi

vi.PREFIX = "ledger_v4r2"
import v4_analysis as va  # noqa: E402
import v4_exploratory as ve  # noqa: E402

HASHES = {  # file -> PLAN_HASH.txt line it must match
    "BACKTEST_V4_ANALYSIS_PLAN.md": ("sha256:", "4a57fcaab100b14b714dd4bdb04b96f16316a527d8758845080b2d909b27152a"),
    "BACKTEST_V4_ADDENDUM_2026-10-23.md": ("addendum sha256:", "bf66bfe3d9876d26dbbc4fd65cf8cce21c7c793c004c79c8543858bf00f8f0e1"),
    "BACKTEST_V4_ANALYSIS_PLAN_R2.md": ("r2 amended sha256:", "ec6b21042fcb34bee58c51e0acb62b641d684514a5088c2076ea44a7afe4a86c"),
}
# R1 values from BACKTEST_V4_RESULTS.md (tests, Section 3b, EXPLORATORY E0 and Q5)
R1 = {"t1": "-0.00053 (-0.00143 to 0.00038); not confirmed", "t2": "-0.201 (-0.301 to -0.105); not confirmed",
      "t3": "-0.020 (-0.063 to 0.021); does not add value", "t3b_ll": "0.01138 (0.01077 to 0.01203)",
      "t3b_top5": "0.001 (-0.015 to 0.018)", "t4": "0.043 (0.029 to 0.057); Baseline better",
      "t5_hr": "0.613 (0.609 to 0.617)"}
R1_SLOPE, R1_BMODEL, R1_PASSES, R1_SUBS = -0.201, -0.391, {"simulator": 26, "B1": 1, "B2": 1}, 53
R1_RATES, R1_HR_AUC, R1_HR_DIFF = {"simulator": 0.727, "B1": 0.726, "B2": 0.749}, 0.613, 0.0025


def held(ok):
    return "**held** (met)" if ok else "**not held** (not met)"


def extra(md):
    K = va.KEY
    t1, t2, t3, t3b, t4, t5 = (K[k] for k in ("t1", "t2", "t3", "t3b", "t4", "t5_hr"))
    q = ve.q5(K["G"], K["P"], K["K"])
    va.H_EV = {"single", "double", "triple", "home_run"}  # q5 swaps it for home runs; restore
    fc = lambda e, c, d=3: f"{e:.{d}f} ({c[0]:.{d}f} to {c[1]:.{d}f})"
    head = pd.DataFrame([
        {"test": "1 Sides: dLL blend − market", "n": t1["n"], "R2 (95% interval)": fc(t1["est"], t1["ci"], 5),
         "R2 outcome": "Information confirmed" if t1["outcome"] else "not confirmed", "R1": R1["t1"]},
        {"test": "2 Totals: slope", "n": t2["n"], "R2 (95% interval)": fc(t2["est"], t2["ci"]),
         "R2 outcome": "Confirmed" if t2["outcome"] else "not confirmed", "R1": R1["t2"]},
        {"test": "3 BTS: top-1 simulator − B2 (paired dates)", "n": t3["n"], "R2 (95% interval)": fc(t3["est"], t3["ci"]),
         "R2 outcome": "adds value" if t3["outcome"] else "does not add value", "R1": R1["t3"]},
        {"test": "3b LL P(1+ hit) simulator − B2", "n": t3b["ll_n"], "R2 (95% interval)": fc(t3b["ll"], t3b["ll_ci"], 5),
         "R2 outcome": "no rule", "R1": R1["t3b_ll"]},
        {"test": "3b top-5 rate simulator − B2 (dates)", "n": t3b["top5_n"], "R2 (95% interval)": fc(t3b["top5"], t3b["top5_ci"]),
         "R2 outcome": "no rule", "R1": R1["t3b_top5"]},
        {"test": "4 Starter K: ΔMAE model − baseline", "n": t4["n"], "R2 (95% interval)": fc(t4["est"], t4["ci"]),
         "R2 outcome": t4["outcome"], "R1": R1["t4"]},
        {"test": "5 HR AUC `p_1hr` (pooled)", "n": t5["n"], "R2 (95% interval)": fc(t5["est"], t5["ci"]),
         "R2 outcome": "descriptive", "R1": R1["t5_hr"]},
    ])
    L = K["L"]
    md[2] = md[2].replace("Run by `analysis/v4_analysis.py` after `analysis/v4_integrity.py` (log: `analysis/out/v4_integrity_log.txt`)",
                          "Run by `analysis/v4r2_analysis.py` (reusing `v4_analysis.py`, `v4_integrity.py`) "
                          "(logs: `analysis/out/v4r2_integrity_log.txt`, `analysis/out/v4r2_analysis_log.txt`)")
    src = pd.concat([L.away_starter_source, L.home_starter_source]).value_counts().to_dict()
    i = md.index("## Confirmatory outcomes (plan's words)")
    md[i:i] = ["## R2 headline next to R1", "", va.md_table(head), "",
               "R2 governs tests 1, 2, 3 and 4 and Section 3b (plan, \"Which run governs\" and amendment 1). "
               "R1 outcomes stay reported in `BACKTEST_V4_RESULTS.md`.", "",
               f"Integrity extras: `starter_source` values (team-games) {src}; `odds_match` {L.odds_match.value_counts().to_dict()}; "
               f"`name_collision` games {int(L.name_collision.sum())}. Full gate output: `analysis/out/v4r2_integrity_log.txt`.", ""]

    top1 = t3["top1"]["simulator"]
    subs = int(top1.bo.astype(str).ne((top1.slot * 100).astype(str)).sum())
    rate = {m: k / n for m, (k, n) in t3["rates"].items()}
    ps = t3["passes"]
    e1 = abs(q["diff"]) <= 0.01
    e2 = abs(t5["est"] - R1_HR_AUC) <= 0.01
    rows = [
        {"expectation": "(a) totals slope", "rule": "−0.201 < R2 slope < 0", "R2": f"{t2['est']:.3f}", "R1": f"{R1_SLOPE}",
         "result": held(R1_SLOPE < t2["est"] < 0)},
        {"expectation": "(b) model coefficient", "rule": "R2 logit(p_model) coefficient < 0",
         "R2": fc(t1["b_model"], t1["b_model_ci"]), "R1": f"{R1_BMODEL}", "result": held(t1["b_model"] < 0)},
        {"expectation": "(c) passes", "rule": "simulator passes ≤ max(B1, B2 passes) + 3",
         "R2": f"sim {ps['simulator']}, B1 {ps['B1']}, B2 {ps['B2']}", "R1": "sim 26, B1 1, B2 1",
         "result": held(ps["simulator"] <= max(ps["B1"], ps["B2"]) + 3)},
        {"expectation": "(c) substitute picks", "rule": "simulator top-1 substitute picks = 0", "R2": str(subs),
         "R1": str(R1_SUBS), "result": held(subs == 0)},
        *[{"expectation": f"(d) top-1 rate, {m}", "rule": "abs(R2 − R1) ≤ 0.02", "R2": f"{rate[m]:.3f} ({t3['rates'][m][0]}/{t3['rates'][m][1]})",
           "R1": f"{R1_RATES[m]:.3f}", "result": held(abs(rate[m] - R1_RATES[m]) <= 0.02)} for m in ("simulator", "B1", "B2")],
        {"expectation": "(e) primary: HR AUC simulator − baseline", "rule": "abs(difference) ≤ 0.01",
         "R2": fc(q["diff"], q["diff_ci"], 4), "R1": f"{R1_HR_DIFF}", "result": held(e1)},
        {"expectation": "(e) second check: test 5 pooled HR AUC", "rule": "abs(R2 − 0.613) ≤ 0.01",
         "R2": f"{t5['est']:.4f} (Δ {t5['est'] - R1_HR_AUC:+.4f})", "R1": f"{R1_HR_AUC}", "result": held(e2)},
    ]
    auc = "; ".join(f"{k} {q['auc'][k]:.4f} ({q['auc_ci'][k][0]:.4f} to {q['auc_ci'][k][1]:.4f})" for k in q["auc"])
    return ["## Pre-registered expectations (a)–(e), R2 vs R1", "",
            "No decision rule; they do not change any test outcome (plan).", "", va.md_table(pd.DataFrame(rows)), "",
            f"(e) primary: {q['n']} true-starting-nine batter-games with plateAppearances ≥ 1 ({q['hr']} with a HR); AUC {auc}. "
            "Baseline and bootstrap as in `v4_exploratory.py` `q5` (paired game bootstrap, 2,000, seed 20261023, "
            "the seed q5 used for R1; the amendment names no seed).", "",
            "(c) substitute picks are counted on the simulator's top-1 picks by box `battingOrder` ≠ slot × 100.", "",
            "## EXPLORATORY", "", "None. Nothing outside the plan was computed.", ""]


def main():
    for f, (label, want) in HASHES.items():
        got = hashlib.sha256((vi.ROOT / "analysis" / f).read_bytes()).hexdigest()
        assert got == want, f"{f}: {got} != `{label}` {want}"
    print("hashes: OK", flush=True)
    if not vi.main():
        sys.exit("STOP: R2 integrity gate failed.")
    va.PLAN, va.PLAN_SHA = vi.ROOT / "analysis" / "BACKTEST_V4_ANALYSIS_PLAN_R2.md", HASHES["BACKTEST_V4_ANALYSIS_PLAN_R2.md"][1]
    va.RESULTS = vi.ROOT / "analysis" / "BACKTEST_V4_RESULTS_R2.md"
    va.TITLE, va.HASH_LINE = "# Full v4 backtest, corrected re-run (R2) — results", "r2 amended sha256:"
    va.TRUE_STARTERS_ONLY, va.EXTRA = True, extra
    va.main()


if __name__ == "__main__":
    main()
