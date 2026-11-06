"""Beat the Streak lab 3, Step 3: models and tests (plan BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md + addendum A1).

M1 is lab 1's model exactly (btslab_analysis: derive, design, fit_l2, fit_fold, boot_counts, top, pool); M3 = M1 + group H,
shrunk per training fold from the counts in analysis/cache/btslab3_features.parquet. Outcomes are FAKE unless --real is
given; without --real the hit column is never loaded (lab 1's load() guard).
  python analysis/btslab3_analysis.py --fake planted          # self-test, seed 4300 -> btslab3_selftest_planted_log.txt
  python analysis/btslab3_analysis.py --fake planted-null     # self-test, seed 4301 -> btslab3_selftest_planted_null_log.txt
  python analysis/btslab3_analysis.py --selftest-eval         # descriptive h_xhit coefficient difference, appended to both logs
  python analysis/btslab3_analysis.py --null-study            # seeds 4001-4200, parallel -> btslab3_null_study_log.txt
  python analysis/btslab3_analysis.py --smoke                 # guard checks + throwaway seeds 9001/9002 -> btslab3_smoke_log.txt
  python analysis/btslab3_analysis.py --time                  # times one M3 fit and one inner CV; projects runtimes
  python analysis/btslab3_analysis.py --real                  # the real run, once -> analysis/out/lab3_results_log.txt
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import btslab3_features as b3  # noqa: E402
import btslab_analysis as ba  # noqa: E402
import btslab_features as bf  # noqa: E402
from btslab_analysis import design, fit_l2, pred, say  # noqa: E402
from v4_analysis import ci, ll, md_table, wilson_s  # noqa: E402

OUT = ba.OUT
RESULTS = OUT / "lab3_results_log.txt"  # plan Section 11 step 3: --real refuses to run if it exists
SELFTEST = {"planted": OUT / "btslab3_selftest_planted_log.txt", "planted-null": OUT / "btslab3_selftest_planted_null_log.txt",
            "null": OUT / "btslab3_null_study_log.txt"}
SEED = 20261201  # + test number (plan Section 6)
PERM_SEED = 20261250  # + k
BASE_CONTENT = "e1fb5d01cf19f18eab46b561575a5975867348cca1f16f29656d203bf7600f9a"
H3_CONTENT = "d8dccfe335795fec26b7f5bceb138059c10209bbe9a557463072860f1889bfa1"  # addendum A1 item 2
WF, LOSO = ba.WF, ba.LOSO
T6 = [((2023,), 2024), ((2023, 2024), 2025)]
H = {"H1": ["h_hard", "h_xhit", "h_gb", "h_ld", "h_pu", "H1_miss"], "H2": ["h_whiff", "h_chase", "h_zcon", "H2_miss"],
     "H3": ["h_same", "h_same_miss", "h_park_hand"], "H4": ["h_speed", "h_speed_miss"],
     "H5": ["h_age", "h_age2", "sp_age", "sp_age2", "trend_hab", "trend_xhit", "H5_trend_miss"]}
H_COLS = [c for g in H.values() for c in g]
M3_COLS = ba.M1_COLS + H_COLS
# lab 1's printed numbers (analysis/out/btslab_results_log.txt) for the M1 reproduction check (plan Section 10 rule 3)
REPRO_FOLDS = {("WF", (2022, 2023), 2024): ("0.001", "0.65839", "0.5731", "0.5492", "+0.1185"),
               ("WF", (2022, 2023, 2024), 2025): ("0.001", "0.65864", "0.5726", "0.5480", "+0.1220"),
               ("LOSO", (2023, 2024, 2025), 2022): ("0.001", "0.65843", "0.5695", "0.5429", "+0.1249"),
               ("LOSO", (2022, 2024, 2025), 2023): ("0.001", "0.65973", "0.5711", "0.5471", "+0.1226"),
               ("LOSO", (2022, 2023, 2025), 2024): ("0.001", "0.65879", "0.5715", "0.5462", "+0.1211"),
               ("LOSO", (2022, 2023, 2024), 2025): ("0.001", "0.65864", "0.5726", "0.5480", "+0.1220")}
REPRO_WF = {"test log loss": "0.65947", "test AUC": "0.5740", "top-1 pooled": "0.783", "top-1 2024": "0.765",
            "top-1 2025": "0.801", "M1 - B2": "+0.0220 (95% -0.0220 to +0.0687)"}
OPT = {"nperm": 50, "ignore_stops": False}


# ---------------------------------------------------------------- data and outcomes
def load(real: bool) -> pd.DataFrame:
    """Lab 1's table through lab 1's guarded load() (no hit column unless real), plus the lab 3 counts on (game_pk, id)."""
    X = ba.load(real)
    Hf = pd.read_parquet(b3.OUT_PARQUET).drop(columns=["date", "season"])
    X = X.merge(Hf, on=["game_pk", "id"], how="left", validate="one_to_one", suffixes=("", "_lab3"))
    assert X.h1_bip.notna().all() and len(X) == 174348
    assert ((X.c_hand == X.c_hand_lab3) | (X.c_hand.isna() & X.c_hand_lab3.isna())).all(), "c_hand differs"  # only shared column
    X = X.drop(columns="c_hand_lab3")
    return X


_GEN = {}


def outcomes(X, kind, seed):
    """Fake hits on AB >= 1 rows. null: lab 1's Bernoulli(0.75). planted / planted-null (addendum A2 item 1): the models'
    own shrunk columns, logit p = logit(0.75) + 0.30*[slot <= 4] - 0.40*z(a_k) + 0.50*z(a_hab) + c*z(h_xhit), c = 0.10 / 0."""
    if kind == "null":
        return ba.fake_outcomes(X, "null", seed)
    c = {"planted": 0.10, "planted-null": 0.0}[kind]
    if "z" not in _GEN:
        # train-2022-24 fold (WF[1]): a_k and a_hab come from btslab_analysis.derive() (lab 1's M1 columns, via
        # ba.prepare_fold inside prepare()); h_xhit from derive_h() here (M3's column); each with that fold's prior
        # strengths m. Standardised by btslab_analysis.design() with that fold's training mean and SD, applied to all
        # rows; design() fills a missing value with the training mean, i.e. z = 0 ("missing -> 0").
        p = prepare(X, *WF[1])
        _GEN["z"] = design(p["F"], p["tr"], ["a_k", "a_hab", "h_xhit"])[:, 1:]
    z = _GEN["z"]
    lp = np.log(3) + 0.30 * X.slot.le(4).to_numpy() - 0.40 * z[:, 0] + 0.50 * z[:, 1] + c * z[:, 2]
    y = (np.random.default_rng(seed).random(len(X)) < 1 / (1 + np.exp(-lp))).astype(float)
    y[X.ab.to_numpy() < 1] = np.nan
    return y, (f"Bernoulli(p), logit p = logit(0.75) + 0.30*[slot 1-4] - 0.40*z(a_k) + 0.50*z(a_hab) + {c:.2f}*z(h_xhit shrunk); "
               f"shrunk columns and z from the train-2022-24 fold, applied to all rows; missing -> 0; seed {seed}")


# ---------------------------------------------------------------- group H at fit time (plan Section 4)
def m_normal(n, S, Q, mu, min_n=100):
    """Normal variance-components prior strength for a mean of per-ball values (h_xhit): s2w pooled within, tau2 between."""
    k = (n >= min_n) & np.isfinite(mu)
    n, S, Q, mu = n[k], S[k], Q[k], mu[k]
    s2w = (Q - S ** 2 / n).sum() / (n - 1).sum()
    tau2 = ((n * (S / n - mu) ** 2).sum() - len(n) * s2w) / n.sum()
    return float(s2w / tau2) if tau2 > 0 else np.inf


def _check_m_normal():
    r = np.random.default_rng(0)
    n = r.integers(100, 500, 3000).astype(float)
    true = r.normal(0.33, 0.02, 3000)  # between SD 0.02, within SD 0.25 -> m = 0.0625 / 0.0004 = 156
    S = np.array([r.normal(t, 0.25, int(k)).sum() for t, k in zip(true, n)])
    Q = S ** 2 / n + 0.0625 * (n - 1)
    m = m_normal(n, S, Q, np.full(3000, 0.33))
    assert 110 < m < 210, m


def derive_h(X, tr, F, m):
    """Adds the group H columns to lab 1's derived frame F (m estimated on training rows only, lab 1's units)."""
    T = X[tr]
    bat = T.drop_duplicates(["id", "season"], keep="last")
    f = lambda d, c: d[c].to_numpy(float)
    F, m = F.copy(), dict(m)
    for name, x, n, mu in (("h_hard", "h1_hard", "h1_bip", "lg_hard"), ("h_gb", "h1_gb", "h1_bip", "lg_gb"),
                           ("h_ld", "h1_ld", "h1_bip", "lg_ld"), ("h_pu", "h1_pu", "h1_bip", "lg_pu"),
                           ("h_whiff", "h2_miss", "h2_swing", "lg_whiff"), ("h_chase", "h2_out_swing", "h2_out", "lg_chase")):
        m[name] = bf.estimate_m(f(bat, x), f(bat, n), f(bat, mu))
        F[name] = bf.shrink(f(X, x), f(X, n), f(X, mu), m[name])
    zc = lambda d: d.h2_in_swing.to_numpy(float) - d.h2_in_miss.to_numpy(float)
    m["h_zcon"] = bf.estimate_m(zc(bat), f(bat, "h2_in_swing"), f(bat, "lg_zcon"))
    F["h_zcon"] = bf.shrink(zc(X), f(X, "h2_in_swing"), f(X, "lg_zcon"), m["h_zcon"])
    m["h_xhit"] = m_normal(f(bat, "h1_bip"), f(bat, "h1_S"), f(bat, "h1_Q"), f(bat, "lg_xhit"))
    F["h_xhit"] = bf.shrink(f(X, "h1_S"), f(X, "h1_bip"), f(X, "lg_xhit"), m["h_xhit"])
    for c in ("H1_miss", "H2_miss", "h_same", "h_same_miss", "h_speed_miss", "H5_trend_miss", "h_age", "h_age2", "sp_age", "sp_age2"):
        F[c] = X[c].to_numpy(float)
    F["h_speed"] = X.h_speed.to_numpy(float)  # not shrunk (Resolved choice 4); missing filled by design()
    # park by batter side: one m over venue-season-side units, target = that side's road rate (as derive() for park)
    ven = T[T.hp_side.notna()].drop_duplicates(["venue_id", "season", "hp_side"], keep="last")
    m["h_park_hand"] = bf.estimate_m(f(ven, "hp_venue_h"), f(ven, "hp_venue_pa"), f(ven, "hp_road_h") / f(ven, "hp_road_pa"))
    road = f(X, "hp_road_h") / f(X, "hp_road_pa")
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.where(f(X, "hp_road_pa") > 0, bf.shrink(f(X, "hp_venue_h"), f(X, "hp_venue_pa"), road, m["h_park_hand"]) / road, 1.0)
    F["h_park_hand"] = np.where(X.hp_side.notna(), ph, F.f_park)  # no side: lab 1's f_park
    # trend: first terms are a_hab and h_xhit; older window shrunk with the same fold m toward its own league rate
    F["trend_hab"] = F.a_hab - bf.shrink(f(X, "t_h_old"), f(X, "t_ab_old"), f(X, "lg_h_ab_old"), m["hab"])
    F["trend_xhit"] = F.h_xhit - bf.shrink(f(X, "t_S_old"), f(X, "t_bip_old"), f(X, "lg_xhit_old"), m["h_xhit"])
    Zs = design(F, tr, ["h_speed", "h_gb"])  # training mean fill and SD, as design()
    F["x_speed_gb"] = Zs[:, 1] * Zs[:, 2]  # exploratory only
    return F, m


def prepare(X, train, test):
    """Outcome-free fold preparation: lab 1's prepare_fold (M1 design, inner-CV blocks) plus group H and M3's design."""
    p = ba.prepare_fold(X, train, test)
    p["F"], p["m"] = derive_h(X, p["tr"], p["F"], p["m"])
    p["Z3"] = design(p["F"], p["tr"], M3_COLS)
    return p


def fit_m1(X, y, p):
    return ba.fit_fold(X, y, p, exploratory=False)  # inner CV, refit, CV AUC vs B1, wiring


def fit_m3(X, y, p):
    return ba.fit_fold(X, y, {**p, "Z": p["Z3"]}, exploratory=False)  # same procedure, its own inner CV


# ---------------------------------------------------------------- statistics
def ll_stat(dates, d, seed):
    """Mean of per-row d with a date-cluster bootstrap interval (ratio of resampled sums, as lab 1's T5)."""
    L = pd.DataFrame({"date": dates, "d": d}).groupby("date").d.agg(["sum", "size"])
    W = ba.boot_counts(len(L), seed)
    return float(d.mean()), ci((W @ L["sum"].to_numpy()) / (W @ L["size"].to_numpy()))


def gain_rule(X, y, folds_p1, folds_p3, seed):
    """T1 statistic (log loss M1 - M3, paired by row) over the folds' test rows; pooled and per test season; the rule."""
    rows = np.zeros(len(X), bool)
    d = np.full(len(X), np.nan)
    for (te, p1), p3 in zip(folds_p1, folds_p3):
        k = te & ~np.isnan(y)
        d[k] = ll(p1[k], y[k]) - ll(p3[k], y[k])
        rows |= k
    est, c = ll_stat(X.date.to_numpy()[rows], d[rows], seed)
    per = {s: ll_stat(X.date.to_numpy()[rows & X.season.eq(s).to_numpy()], d[rows & X.season.eq(s).to_numpy()], seed)
           for s in sorted(X.season[rows].unique())}
    conf = c[0] > 0 and all(e > 0 for e, _ in per.values())
    return {"est": est, "ci": c, "per": per, "confirmed": conf, "d": d, "rows": rows}


def permutations(X, nperm):
    """Within-date permutations of row indices, seeds 20261250 + k; X is sorted by date, so blocks are contiguous."""
    dates = X.date.to_numpy()
    return [np.lexsort((np.random.default_rng(PERM_SEED + k).random(len(X)), dates)) for k in range(nperm)]


def placebo(X, y, wf, perms, seed):
    """T4: H columns permuted within date (all sub-groups and flags together); M3 refitted at each fold's real M3 C."""
    dsum, stats_k = np.zeros(len(X)), []
    rows = np.zeros(len(X), bool)
    for perm in perms:
        dk = np.full(len(X), np.nan)
        for f in wf:
            p, te, ok = f["prep"], f["prep"]["te"], f["M3"]["ok"]
            Fp = p["F"].copy()
            Fp[H_COLS] = p["F"][H_COLS].to_numpy()[perm]
            Zp = design(Fp, p["tr"], M3_COLS)
            pp = pred(Zp, fit_l2(Zp[ok], y[ok], f["M3"]["C"]))
            k = te & ~np.isnan(y)
            dk[k] = ll(f["M1"]["p_m1"][k], y[k]) - ll(pp[k], y[k])
            rows |= k
        dsum[rows] += dk[rows]
        stats_k.append(np.nanmean(dk))
    dbar = dsum[rows] / len(perms)
    est, c = ll_stat(X.date.to_numpy()[rows], dbar, seed)
    return {"est": est, "ci": c, "stop": c[0] > 0, "perm_stats": np.array(stats_k)}


def picks(X, y, f_list, key, mask):
    """Top-1 per test date for predictions f[key] (lab 1's top(): ties to lower gamePk, then lower id)."""
    return pd.concat([ba.top(X, np.where(f["prep"]["te"], f[key], np.nan), mask, y, 1).set_index("date") for f in f_list])


def fmt_ci(e, c, d=5):
    return f"{e:+.{d}f} (95% {c[0]:+.{d}f} to {c[1]:+.{d}f})"


# ---------------------------------------------------------------- real-run preconditions (--real only)
def guard_once(path: Path):
    if path.exists():
        sys.exit(f"refusing to run: {path} already exists (the real run is run once, plan Section 11 step 3)")


def preconditions(selftest=SELFTEST, leak=OUT / "btslab3_leakcheck_log.txt", base=True) -> dict:
    """Stop rules 1, 2 and 4 before any model fit: leak checks passed, base content hash, self-tests passed.
    base=False skips rule 2 (it loads the outcome columns); only the unit test uses it."""
    ok = {"rule 1 leak checks": "LEAK CHECK: PASS" in leak.read_text(encoding="utf-8", errors="replace")}
    say(f"- Rule 1, leak checks ({leak.name} contains 'LEAK CHECK: PASS'): {'PASS' if ok['rule 1 leak checks'] else 'FAIL'}")
    if base:
        h = b3.content_hash(pd.read_parquet(b3.BASE))
        ok["rule 2 base content hash"] = h == BASE_CONTENT
        say(f"- Rule 2, lab 1 table content sha256 {h}: {'PASS' if h == BASE_CONTENT else 'FAIL'} (addendum A1 item 2)")
    h3 = b3.content_hash(pd.read_parquet(b3.OUT_PARQUET))
    ok["lab 3 table content hash"] = h3 == H3_CONTENT
    say(f"- lab 3 feature table content sha256 {h3}: {'matches' if h3 == H3_CONTENT else 'DIFFERS from'} addendum A1")
    ok["rule 4 self-tests"] = all(p.exists() and f"SELF-TEST {k}: PASS" in p.read_text(encoding="utf-8") for k, p in selftest.items())
    say(f"- Rule 4, self-tests passed ({', '.join(p.name for p in selftest.values())} contain "
        f"{', '.join(repr(f'SELF-TEST {k}: PASS') for k in selftest)}): {'PASS' if ok['rule 4 self-tests'] else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- planted self-tests (plan Section 11 step 4; addendum A2 item 1)
def selftest_data(kind, seed, r) -> str:
    c = r["coef"].loc["h_xhit", "coef"]
    say(f"\nM3 h_xhit coefficient (per SD of the shrunk h_xhit, fit 2022–24; descriptive on its own): {c:+.6f}; "
        f"T1 {'Confirmed' if r['t1']['confirmed'] else 'not confirmed'}; placebo {'Confirmed' if r['placebo']['stop'] else 'not Confirmed'}")
    return (f"SELFTEST-DATA kind={kind} seed={seed} h_xhit={c:+.6f} t1_confirmed={r['t1']['confirmed']} "
            f"placebo_stop={r['placebo']['stop']}")


def parse_data(text) -> dict:
    m = re.findall(r"SELFTEST-DATA kind=(\S+) seed=(\d+) h_xhit=(\S+) t1_confirmed=(\S+) placebo_stop=(\S+)", text)
    assert len(m) == 1, f"expected one SELFTEST-DATA line, found {len(m)}"
    k, s, c, t, p = m[0]
    return {"kind": k, "seed": int(s), "h_xhit": float(c), "t1": t == "True", "stop": p == "True"}


def verdict(d) -> bool:
    """The plan's pass rules (Section 11 step 4): planted-H: h_xhit coefficient in 0.05..0.15, T1 Confirmed, placebo not
    Confirmed; planted-null: T1 not Confirmed."""
    conds = ([("M3 h_xhit coefficient in 0.05..0.15", 0.05 <= d["h_xhit"] <= 0.15), ("T1 Confirmed", d["t1"]),
              ("placebo not Confirmed", not d["stop"])] if d["kind"] == "planted" else [("T1 not Confirmed", not d["t1"])])
    for n, ok in conds:
        say(f"- {n}: {'yes' if ok else 'NO'}")
    return all(ok for _, ok in conds)


def repro_check(X, y, fits, mask):
    """Stop rule 3: M1 refitted must print lab 1's numbers at lab 1's precision."""
    bad = []
    for (d, tr, te), exp in REPRO_FOLDS.items():
        f = fits[(d, tr, te)]["M1"]
        got = (f"{f['C']:.4g}", f"{min(f['cv_ll']):.5f}", f"{f['auc_m1_cv']:.4f}", f"{f['auc_b1_cv']:.4f}", f"{f['wiring']:+.4f}")
        bad += [f"{d} test {te}: {got} vs {exp}"] if got != exp else []
    wf = [fits[("WF", tuple(tr), te)] for tr, te in WF]
    te = np.any([f["prep"]["te"] for f in wf], axis=0)
    k = te & ~np.isnan(y)
    p = np.zeros(len(X))
    for f in wf:
        p[f["prep"]["te"]] = f["M1"]["p_m1"][f["prep"]["te"]]
    m1 = picks(X, y, [{"prep": f["prep"], "p": f["M1"]["p_m1"]} for f in wf], "p", mask)
    b2 = picks(X, y, [{"prep": f["prep"], "p": X.base_b2.to_numpy()} for f in wf], "p", mask)
    e, c, _ = ba.paired(m1.y, b2.y, 20261027)
    got = {"test log loss": f"{ll(p[k], y[k]).mean():.5f}", "test AUC": f"{ba.stats.auc(p[k], y[k]):.4f}",
           "top-1 pooled": f"{ba.rate(m1.y.to_numpy(float))[0]:.3f}",
           **{f"top-1 {s}": f"{ba.rate(m1.y[m1.season.eq(s)].to_numpy(float))[0]:.3f}" for s in (2024, 2025)},
           "M1 - B2": f"{e:+.4f} (95% {c[0]:+.4f} to {c[1]:+.4f})"}
    bad += [f"{kk}: {got[kk]} vs {v}" for kk, v in REPRO_WF.items() if got[kk] != v]
    say(f"- Rule 3, M1 reproduction (6 folds; walk-forward test numbers): {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return not bad


# ---------------------------------------------------------------- the pipeline
def run(X, y, real: bool, log: Path):
    t0 = time.perf_counter()
    mask = ba.pool(X)
    say(f"countable rows (AB >= 1): {int((~np.isnan(y)).sum())}; passes (AB = 0): {int(np.isnan(y).sum())}")

    def stop(msg):
        say(f"STOP: {msg}")
        if OPT["ignore_stops"] and not real:
            say("(--ignore-stops, fake outcomes: continuing so the rest of the pipeline is exercised)")
            return
        log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        sys.exit(2)

    # ---- M1 first (rule 3 before M3 is fitted)
    designs = [("WF", WF), ("LOSO", LOSO), ("T6", T6)]
    fits = {}
    for d, folds in designs:
        for tr, te in folds:
            p = prepare(X, tr, te)
            fits[(d, tuple(tr), te)] = {"prep": p, "M1": fit_m1(X, y, p)}
    say("\n## M1 reproduction (plan Section 10 rule 3)")
    if real:
        if not repro_check(X, y, fits, mask):
            stop("M1 does not reproduce lab 1 (rule 3); M3 is not fitted.")
    else:
        say("skipped on fake outcomes (lab 1's numbers are real-outcome numbers); logged as skipped")

    # ---- M3
    for f in fits.values():
        f["M3"] = fit_m3(X, y, f["prep"])
    say("\n## Folds (inner CV: 5 contiguous date blocks, 12 C values 1e-3..1e2, mean log loss)")
    say(md_table(pd.DataFrame([{"design": d, "train": "-".join(map(str, tr)), "test": te, "model": mn,
                                "C": f"{f[mn]['C']:.4g}" + (" (grid edge)" if f[mn]["C"] in (ba.CS[0], ba.CS[-1]) else ""),
                                "inner-CV log loss": f"{min(f[mn]['cv_ll']):.5f}", "AUC (CV)": f"{f[mn]['auc_m1_cv']:.4f}",
                                "AUC B1 (CV)": f"{f[mn]['auc_b1_cv']:.4f}", "wiring 1-4 minus 9": f"{f[mn]['wiring']:+.4f}"}
                               for (d, tr, te), f in fits.items() for mn in ("M1", "M3")])))
    wf = [fits[("WF", tuple(tr), te)] for tr, te in WF]
    for f in wf:
        say(f"group H shrinkage m, test {f['prep']['test']}: " + ", ".join(f"{k}={v:.3g}" for k, v in f["prep"]["m"].items()
                                                                            if k.startswith("h_")))

    # ---- stop rule 6 (inherited from lab 1), M1 and M3, each walk-forward fold
    say("\n## Stop rules")
    s6a = all(f[mn]["auc_m1_cv"] > f[mn]["auc_b1_cv"] for f in wf for mn in ("M1", "M3"))
    s6b = all(f[mn]["wiring"] > 0 for f in wf for mn in ("M1", "M3"))
    say(f"- Rule 6a, model beats B1 on batter-game AUC in the training inner CV (M1 and M3, each walk-forward fold): {'PASS' if s6a else 'FAIL'}")
    say(f"- Rule 6b, training mean P(hit) slots 1-4 minus slot 9 > 0 (M1 and M3, each walk-forward fold): {'PASS' if s6b else 'FAIL'}")
    if not (s6a and s6b):
        stop("an inherited stop rule failed (rule 6).")

    # ---- T4 placebo, before T1 is read (rule 5)
    say(f"\n## T4 — placebo: H columns permuted within date, {OPT['nperm']} permutations (seeds {PERM_SEED} + k), "
        f"M3 refitted at each fold's real C; date bootstrap seed {SEED + 3}")
    pl = placebo(X, y, wf, permutations(X, OPT["nperm"]), SEED + 3)
    q = np.quantile(pl["perm_stats"], [.025, .975])
    say(f"placebo gain (log loss M1 − permuted M3, per-row mean over permutations) {fmt_ci(pl['est'], pl['ci'])}; "
        f"per-permutation statistics 2.5–97.5%: {q[0]:+.5f} to {q[1]:+.5f}")
    say(f"- Rule 5, placebo interval entirely above 0: {'YES - STOP' if pl['stop'] else 'no'}")
    if pl["stop"]:
        stop("the placebo shows a gain (rule 5); no other result is read.")

    # ---- T1
    say(f"\n## T1 — PRIMARY: walk-forward log loss M1 − M3, paired by row (date bootstrap seed {SEED})")
    t1 = gain_rule(X, y, [(f["prep"]["te"], f["M1"]["p_m1"]) for f in wf], [f["M3"]["p_m1"] for f in wf], SEED)
    say(f"pooled 2024–25: {fmt_ci(t1['est'], t1['ci'])}, rows {int(t1['rows'].sum())}")
    for s, (e, c) in t1["per"].items():
        say(f"   {s}: {fmt_ci(e, c)}")
    k = t1["rows"]
    pm1, pm3 = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for f in wf:
        pm1[f["prep"]["te"]], pm3[f["prep"]["te"]] = f["M1"]["p_m1"][f["prep"]["te"]], f["M3"]["p_m1"][f["prep"]["te"]]
    say(f"batter-game log loss M1 {ll(pm1[k], y[k]).mean():.5f}, M3 {ll(pm3[k], y[k]).mean():.5f}; "
        f"AUC M1 {ba.stats.auc(pm1[k], y[k]):.4f}, M3 {ba.stats.auc(pm3[k], y[k]):.4f}")
    say(f"**T1 rule: {'Confirmed' if t1['confirmed'] else 'not confirmed'}** (95% interval entirely above 0 and the point "
        "estimate positive in 2024 and in 2025)")

    # ---- T2
    lo = [fits[("LOSO", tuple(tr), te)] for tr, te in LOSO]
    t2 = gain_rule(X, y, [(f["prep"]["te"], f["M1"]["p_m1"]) for f in lo], [f["M3"]["p_m1"] for f in lo], SEED + 1)
    say(f"\n## T2 — LOSO log loss M1 − M3 (secondary, seed {SEED + 1}): {fmt_ci(t2['est'], t2['ci'])}; "
        + "; ".join(f"{s} {fmt_ci(e, c)}" for s, (e, c) in t2["per"].items())
        + f". T1's rule applied: {'Confirmed' if t2['confirmed'] else 'not confirmed'} (not confirmatory)")

    # ---- T3
    m1p, m3p = picks(X, y, [{"prep": f["prep"], "p": f["M1"]["p_m1"]} for f in wf], "p", mask), \
        picks(X, y, [{"prep": f["prep"], "p": f["M3"]["p_m1"]} for f in wf], "p", mask)
    b2p = picks(X, y, [{"prep": f["prep"], "p": X.base_b2.to_numpy()} for f in wf], "p", mask)
    e31, c31, n31 = ba.paired(m3p.y, m1p.y, SEED + 2)
    e32, c32, n32 = ba.paired(m3p.y, b2p.y, SEED + 2)
    say(f"\n## T3 — top-1 daily success (descriptive; walk-forward; seed {SEED + 2})")
    say(f"M3 {ba.wil(m3p.y)}, M1 {ba.wil(m1p.y)}, B2 {ba.wil(b2p.y)}; M3 − M1 {fmt_ci(e31, c31, 4)} ({n31} dates); "
        f"M3 − B2 {fmt_ci(e32, c32, 4)} ({n32} dates)")

    # ---- T5 exploratory models
    say(f"\n## T5 — EXPLORATORY ablations (walk-forward; date bootstrap seed {SEED + 4}); gain = log loss(reference) − log loss(model)")
    rows, gains_add = [], {}
    for g, gc in H.items():
        for kind, cols, ref, Ckey in (("M1 + " + g, ba.M1_COLS + gc, "M1", "M1"),
                                      ("M3 − " + g, [c for c in M3_COLS if c not in gc], "M3", "M3")):
            pp = np.full(len(X), np.nan)
            for f in wf:
                Zx = design(f["prep"]["F"], f["prep"]["tr"], cols)
                ok = f[Ckey]["ok"]
                pp[f["prep"]["te"]] = pred(Zx, fit_l2(Zx[ok], y[ok], f[Ckey]["C"]))[f["prep"]["te"]]
            pref = pm1 if ref == "M1" else pm3
            e, c = ll_stat(X.date.to_numpy()[k], ll(pref[k], y[k]) - ll(pp[k], y[k]), SEED + 4)
            tp = picks(X, y, [{"prep": f["prep"], "p": pp} for f in wf], "p", mask)
            et, ct, _ = ba.paired(tp.y, (m1p if ref == "M1" else m3p).y, SEED + 4)
            if ref == "M1":
                gains_add[g] = e
            rows.append({"model": kind, "vs": ref, "log-loss gain": fmt_ci(e, c, 6), "Δ top-1 (model − ref)": fmt_ci(et, ct, 4)})
    say(md_table(pd.DataFrame(rows)))

    # ---- coefficients (M3 and M3 + x_speed_gb, fit 2022-24), date bootstrap, warm start at the full fit (lab 1)
    f = wf[1]
    p, ok, C3 = f["prep"], f["M3"]["ok"], f["M3"]["C"]
    _, di = np.unique(X.date.to_numpy()[ok], return_inverse=True)
    Wd = ba.boot_counts(di.max() + 1, SEED + 6)
    yo = y[ok]
    Zx = design(p["F"], p["tr"], M3_COLS + ["x_speed_gb"])
    bx = fit_l2(Zx[ok], yo, C3)
    B3 = np.array([fit_l2(p["Z3"][ok], yo, C3, w[di].astype(float), f["M3"]["beta"]) for w in Wd])
    Bx = np.array([fit_l2(Zx[ok], yo, C3, w[di].astype(float), bx) for w in Wd])
    coef = pd.DataFrame({"coef": f["M3"]["beta"][1:], "lo": np.quantile(B3[:, 1:], .025, 0), "hi": np.quantile(B3[:, 1:], .975, 0)}, index=M3_COLS)
    xs = (bx[-1], *np.quantile(Bx[:, -1], [.025, .975]))
    say(f"\n## M3 standardised coefficients, fit on 2022–24 at C {C3:.4g} (date bootstrap seed {SEED + 6}); group H rows")
    say(md_table(coef.loc[H_COLS].reset_index().rename(columns={"index": "feature"}).assign(
        coef=lambda d: d.coef.map("{:+.4f}".format), lo=lambda d: d.lo.map("{:+.4f}".format), hi=lambda d: d.hi.map("{:+.4f}".format))))
    say(f"M3 + x_speed_gb (exploratory, same C and resamples): x_speed_gb {xs[0]:+.4f} ({xs[1]:+.4f} to {xs[2]:+.4f})")
    tr = p["tr"]
    sd = {c: p["F"][c][tr].std(ddof=0) for c in ("h_age", "h_age2")}
    ia, ia2 = 1 + M3_COLS.index("h_age"), 1 + M3_COLS.index("h_age2")
    contrast = lambda b: b[ia] * (6 - 1) / sd["h_age"] + b[ia2] * (36 - 1) / sd["h_age2"]  # ages 34 vs 29 (h_age = age - 28)
    age = (contrast(f["M3"]["beta"]), *np.quantile([contrast(b) for b in B3], [.025, .975]))

    # ---- T6
    say(f"\n## T6 — DIAGNOSTICS (seed {SEED + 5})")
    t6f = [fits[("T6", tuple(tr_), te)] for tr_, te in T6]
    t6 = gain_rule(X, y, [(f_["prep"]["te"], f_["M1"]["p_m1"]) for f_ in t6f], [f_["M3"]["p_m1"] for f_ in t6f], SEED + 5)
    say(f"2022 dropped from training: {fmt_ci(t6['est'], t6['ci'])}; " + "; ".join(f"{s} {fmt_ci(e, c)}" for s, (e, c) in t6["per"].items()))
    for s in ("R", "L", "S"):
        kk = k & X.bat_side.eq(s).to_numpy()
        e, c = ll_stat(X.date.to_numpy()[kk], t1["d"][kk], SEED + 5)
        say(f"batter side {s}: {fmt_ci(e, c)}, rows {int(kk.sum())}")

    # ---- T7
    say("\n## T7 — hypothesis checks (proposed by Claude; kept by William; coefficients from the table above; no rule)")
    cf = lambda c: (coef.loc[c, "coef"], coef.loc[c, "lo"], coef.loc[c, "hi"])
    sgn = lambda v, want: f"{v[0]:+.4f} ({v[1]:+.4f} to {v[2]:+.4f}), sign {'as read for support' if (v[0] > 0) == (want > 0) else 'against'}"
    for hyp, items in (("V1", [("h_xhit", cf("h_xhit"), 1), ("h_hard", cf("h_hard"), 1)]),
                       ("V2", [("h_zcon", cf("h_zcon"), 1), ("h_whiff", cf("h_whiff"), -1)]),
                       ("V3", [("h_same", cf("h_same"), -1)]),
                       ("V4", [("h_speed", cf("h_speed"), 1), ("x_speed_gb", xs, 1)]),
                       ("V5", [("age contrast 34 vs 29 (logit)", age, -1), ("trend_hab", cf("trend_hab"), 1)]),
                       ("V6", [("h_chase", cf("h_chase"), -1)])):
        say(f"{hyp}: " + "; ".join(f"{n} {sgn(v, w)}" for n, v, w in items))

    # ---- Claude's expectations
    inc0 = lambda v: v[1] <= 0 <= v[2]
    t1d = m3p.assign(season=m3p.season)
    r_all = ba.rate(t1d.y.to_numpy(float))[0]
    top20 = []
    for s, g in t1d.groupby("season"):
        g = g.assign(d=g.index).sort_values(["s", "d"], ascending=[False, True])
        top20.append(g.head(round(0.2 * len(g))))
    r20 = ba.rate(pd.concat(top20).y.to_numpy(float))[0]
    exp = [("(a) T1 gain in +0.0002..+0.0010", 0.0002 <= t1["est"] <= 0.0010, f"{t1['est']:+.5f}; interval excludes 0: {not (t1['ci'][0] <= 0 <= t1['ci'][1])}"),
           ("(b) M1 + H1 has the largest gain of the five M1 + Hk", max(gains_add, key=gains_add.get) == "H1",
            ", ".join(f"{g} {v:+.6f}" for g, v in gains_add.items())),
           ("(c) h_same < 0, interval excludes 0", cf("h_same")[0] < 0 and not inc0(cf("h_same")), sgn(cf("h_same"), -1)),
           ("(d) h_whiff, h_chase, h_zcon intervals each include 0", all(inc0(cf(c)) for c in ("h_whiff", "h_chase", "h_zcon")), ""),
           ("(e) |h_speed| < 0.02 per SD", abs(cf("h_speed")[0]) < 0.02, f"{cf('h_speed')[0]:+.4f}"),
           ("(e) age contrast interval excludes 0", not inc0(age), f"{age[0]:+.4f} ({age[1]:+.4f} to {age[2]:+.4f})"),
           ("(e) trend_hab interval includes 0", inc0(cf("trend_hab")), sgn(cf("trend_hab"), 1)),
           ("(f) top-1 M3 − M1 in −0.015..+0.025", -0.015 <= e31 <= 0.025, f"{e31:+.4f}"),
           ("(g) placebo interval includes 0", pl["ci"][0] <= 0 <= pl["ci"][1], fmt_ci(pl["est"], pl["ci"])),
           ("(h) T6 2022-dropped same sign as T1", np.sign(t6["est"]) == np.sign(t1["est"]), f"{t6['est']:+.5f} vs {t1['est']:+.5f}"),
           ("(i) M3 top-20% days ≤ overall + 0.03", r20 - r_all <= 0.03, f"top-20% {r20:.4f} vs overall {r_all:.4f}")]
    say("\n## Claude's pre-registered expectations (met / not met; no decision rule)")
    say(md_table(pd.DataFrame([{"expectation": a, "result": "met" if b else "not met", "numbers": c} for a, b, c in exp])))
    say(f"\nruntime {time.perf_counter() - t0:.0f} s")
    return {"t1": t1, "placebo": pl, "coef": coef}


# ---------------------------------------------------------------- null study (fake outcomes only)
_W = {}


def _null_init(nperm, nboot):
    ba.NBOOT = nboot
    X = load(False)  # never the real hit column
    _W.update(X=X, preps=[prepare(X, tr, te) for tr, te in WF], perms=permutations(X, nperm))


def _null_seed(seed):
    X = _W["X"]
    y, _ = outcomes(X, "null", seed)
    wf = [{"prep": p, "M1": fit_m1(X, y, p), "M3": fit_m3(X, y, p)} for p in _W["preps"]]
    t1 = gain_rule(X, y, [(f["prep"]["te"], f["M1"]["p_m1"]) for f in wf], [f["M3"]["p_m1"] for f in wf], SEED)
    pl = placebo(X, y, wf, _W["perms"], SEED + 3)
    return {"seed": seed, "T1": t1["est"], "T1 lo": t1["ci"][0], "T1 hi": t1["ci"][1], "T1 Confirmed": t1["confirmed"],
            "placebo": pl["est"], "placebo lo": pl["ci"][0], "placebo stop": pl["stop"],
            "C M1": wf[0]["M1"]["C"], "C M3": wf[0]["M3"]["C"]}


def null_study(seeds, procs, nperm, nboot, log: Path, csv: Path):
    from multiprocessing import Pool
    t0 = time.perf_counter()
    with Pool(procs, initializer=_null_init, initargs=(nperm, nboot)) as pool_:
        R = pd.DataFrame(pool_.map(_null_seed, list(seeds), chunksize=1)).sort_values("seed")
    R.to_csv(csv, index=False)
    say(f"seeds {min(seeds)}-{max(seeds)} ({len(R)}), {procs} processes, {nboot} bootstrap resamples, {nperm} placebo "
        f"permutations per seed; per-seed rows: {csv.name}")
    sd = R.T1.std(ddof=1) if len(R) > 1 else np.nan
    checks = [("T1 false confirmation <= 8%", R["T1 Confirmed"].mean() <= 0.08, f"{R['T1 Confirmed'].mean():.3f}"),
              ("T4 false stop <= 8%", R["placebo stop"].mean() <= 0.08, f"{R['placebo stop'].mean():.3f}"),
              ("T1 centred: |mean| < 0.25 SD across seeds", abs(R.T1.mean()) < 0.25 * sd, f"mean {R.T1.mean():+.6f}, SD {sd:.6f}")]
    for n, ok, v in checks:
        say(f"- {n}: {v} -> {'PASS' if ok else 'FAIL'}")
    passed = all(ok for _, ok, _ in checks)
    say(f"runtime {time.perf_counter() - t0:.0f} s")
    return passed, R


# ---------------------------------------------------------------- checks of the guards and the new code
def _check_guards():
    """Fake mode never asks parquet for the hit column; the run-once guard refuses an existing log (temp path)."""
    import tempfile
    asked = []
    real_read = pd.read_parquet
    pd.read_parquet = lambda path, columns=None, **k: (asked.append((Path(path).name, columns)), real_read(path, columns=columns, **k))[1]
    try:
        X = load(False)
    finally:
        pd.read_parquet = real_read
    base_cols = [c for n, c in asked if n == bf.OUT_PARQUET.name][0]
    assert "h" not in base_cols and "h" not in X.columns and "ab" in X.columns
    with tempfile.TemporaryDirectory() as td:
        dummy = Path(td) / "lab3_results_log.txt"
        guard_once(dummy)  # absent: passes
        dummy.write_text("dummy")
        try:
            guard_once(dummy)
            raise AssertionError("run-once guard did not refuse")
        except SystemExit as e:
            refused = str(e)
    return base_cols, sorted(set(X.columns) - set(base_cols)), refused


def timing():
    """One M3 full fit, one M3 inner-CV selection and one warm-started bootstrap refit on the largest walk-forward fold."""
    X = load(False)
    y, _ = outcomes(X, "null", 9002)
    p = prepare(X, *WF[1])
    ok = p["tr"] & ~np.isnan(y)
    t = time.perf_counter(); b = fit_l2(p["Z3"][ok], y[ok], 0.001); t_fit = time.perf_counter() - t
    t = time.perf_counter(); f3 = fit_m3(X, y, p); t_cv3 = time.perf_counter() - t
    t = time.perf_counter(); f1 = fit_m1(X, y, p); t_cv1 = time.perf_counter() - t
    w = np.random.default_rng(1).poisson(1.0, ok.sum()).astype(float)
    t = time.perf_counter(); fit_l2(p["Z3"][ok], y[ok], 0.001, w, b); t_boot = time.perf_counter() - t
    return {"rows": int(ok.sum()), "cols M3": p["Z3"].shape[1], "M3 fit": t_fit, "M3 inner CV (incl. refit)": t_cv3,
            "M1 inner CV (incl. refit)": t_cv1, "warm bootstrap refit": t_boot}


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the log text has − and – (cp1252 console)
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true", help="the real run (real hit column), once")
    g.add_argument("--fake", choices=["planted", "planted-null"], help="self-test on seeded fake outcomes")
    g.add_argument("--null-study", action="store_true", help="200 fake null seeds 4001-4200 in parallel")
    g.add_argument("--smoke", action="store_true", help="smoke test: guard checks, seeds 9001/9002, tiny settings")
    g.add_argument("--time", action="store_true", help="time one fit and one inner CV; project runtimes")
    g.add_argument("--selftest-eval", action="store_true", help="descriptive coefficient difference of the planted runs (4300, 4301)")
    g.add_argument("--dry-criterion", action="store_true", help="smoke: second dry run, new generator, seeds 9001-9008")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--ignore-stops", action="store_true", help="fake runs only: continue past stop rules")
    a = ap.parse_args()
    if not (a.real or a.fake or a.null_study or a.smoke or a.time or a.selftest_eval or a.dry_criterion):
        sys.exit("refusing to run: choose --fake planted|planted-null, --selftest-eval, --null-study, --smoke, --dry-criterion, "
                 "--time, or --real (uses the real hit column)")
    if a.real and a.ignore_stops:
        sys.exit("--ignore-stops is not allowed with --real")
    OPT["ignore_stops"] = a.ignore_stops
    ba._check_solver()
    bf._self_check()
    _check_m_normal()

    if a.selftest_eval:
        return selftest_eval()
    if a.dry_criterion:
        return dry_criterion()

    if a.time:
        T = timing()
        print(T)
        n_seed = 2 * (T["M1 inner CV (incl. refit)"] + T["M3 inner CV (incl. refit)"] + 10 * T["M3 fit"])
        n_real = (2 * (T["M1 inner CV (incl. refit)"] + T["M3 inner CV (incl. refit)"] + 50 * T["M3 fit"] + 11 * T["M3 fit"])
                  + 6 * (T["M1 inner CV (incl. refit)"] + T["M3 inner CV (incl. refit)"]) + 2 * 2000 * T["warm bootstrap refit"])
        print(f"null study: per seed {n_seed:.0f} s serial (2 folds x (M1 and M3 inner CV incl. refits + 10 placebo refits)); "
              f"200 seeds {200 * n_seed / 3600:.1f} h serial, {200 * n_seed / a.procs / 3600:.1f} h on {a.procs} processes")
        print(f"real run: {n_real / 3600:.1f} h serial (2 WF folds x (inner CVs + 50 placebo + 11 exploratory refits) + "
              "6 more folds' inner CVs (4 LOSO, 2 T6) + 2 x 2,000 warm bootstrap refits)")
        return

    if a.null_study:
        ba.LOG.clear()
        say("# Beat the Streak lab 3 — null study on FAKE outcomes (Bernoulli(0.75); real hits never loaded)")
        passed, _ = null_study(range(4001, 4201), a.procs, 10, 500, SELFTEST["null"], OUT / "btslab3_null_study_seeds.csv")
        say(f"SELF-TEST null: {'PASS' if passed else 'FAIL - stop, do not tune'}")
        SELFTEST["null"].write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        sys.exit(0 if passed else 1)

    if a.smoke:
        log = OUT / "btslab3_smoke_log.txt"
        OPT.update(nperm=2, ignore_stops=True)
        ba.NBOOT = 20
        say("# Beat the Streak lab 3 — SMOKE TEST ONLY (not a result): fake outcomes, throwaway seeds 9001 and 9002, "
            "20 bootstrap resamples, 2 placebo permutations. It shows the code runs end to end; nothing here is evidence.")
        base_cols, added, refused = _check_guards()
        say(f"guard: fake-mode read of {bf.OUT_PARQUET.name} asked for {len(base_cols)} columns, 'h' not among them; "
            f"'ab' present; run-once guard on a temp dummy log refused: \"{refused}\"")
        X = load(False)
        y, desc = outcomes(X, "planted", 9001)
        say(f"outcome: {desc}")
        run(X, y, False, log)
        say("\n## Null-study code path, 1 seed (9002), same tiny settings")
        null_study([9002], 1, 2, 20, log, Path(os.environ.get("TEMP", ".")) / "btslab3_smoke_null_seed.csv")
        log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
        print(f"wrote {log}")
        return

    if a.real:
        guard_once(RESULTS)
        log = RESULTS
        say("# Beat the Streak lab 3 — RESULTS (real outcomes)")
        say("## Preconditions (plan Section 10 rules 1, 2, 4; before any model fit)")
        if not all(preconditions().values()):
            say("STOP: a precondition failed; no model is fitted.")
            log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
            sys.exit(2)
        X = load(True)
        y = np.where(X.ab >= 1, (X.h >= 1).astype(float), np.nan)
        say("outcome: REAL box-score hits (H >= 1) on AB >= 1 rows")
        run(X, y, True, log)
    else:
        log = SELFTEST[a.fake]
        seed = 4300 if a.fake == "planted" else 4301
        say(f"# Beat the Streak lab 3 — SELF-TEST on FAKE outcomes: {a.fake} (seed {seed}; real hits never loaded)")
        say("fake-mode content-hash recompute of lab 1's table: skipped (it needs the outcome columns; addendum A1 item 2)")
        X = load(False)
        y, desc = outcomes(X, a.fake, seed)
        say(f"outcome: {desc}")
        say(line := selftest_data(a.fake, seed, r=run(X, y, False, log)))
        say("\nPass rule (plan Section 11 step 4):")
        say(f"SELF-TEST {a.fake}: {'PASS' if verdict(parse_data(line)) else 'FAIL - stop, do not tune'}")
    log.write_text("\n".join(ba.LOG) + "\n", encoding="utf-8")
    print(f"wrote {log}")


def selftest_eval():
    """Descriptive only (addendum A2 item 1): the h_xhit coefficient difference planted-H − planted-null, appended to both logs."""
    dp, dn = (parse_data(SELFTEST[k].read_text(encoding="utf-8")) for k in ("planted", "planted-null"))
    assert (dp["kind"], dp["seed"], dn["kind"], dn["seed"]) == ("planted", 4300, "planted-null", 4301)
    line = (f"\nDescriptive (not a pass condition): M3 h_xhit coefficient planted-H (seed 4300) {dp['h_xhit']:+.6f} − "
            f"planted-null (seed 4301) {dn['h_xhit']:+.6f} = {dp['h_xhit'] - dn['h_xhit']:+.6f}\n")
    for k in ("planted", "planted-null"):
        with SELFTEST[k].open("a", encoding="utf-8") as fh:
            fh.write(line)
    print(line.strip())


def dry_criterion():
    """Second dry run, new generator (throwaway seeds; tiny settings), with the freeze conditions fixed in advance;
    appended to the smoke log."""
    OPT.update(nperm=2, ignore_stops=True)
    ba.NBOOT = 20
    runs = [("planted", s) for s in (9001, 9002)] + [("planted-null", s) for s in range(9003, 9009)]
    say("\n\n# SMOKE TEST ONLY (not a result): second dry run, new generator (addendum A2 item 1). Throwaway seeds, not "
        "registered: planted-H 9001, 9002; planted-null 9003-9008. 20 bootstrap resamples, 2 placebo permutations.")
    say("Freeze conditions, fixed in advance: both planted-H seeds have the h_xhit coefficient in 0.05..0.15, T1 Confirmed "
        "and the placebo not Confirmed; AND T1 is Confirmed on at most 1 of the 6 planted-null seeds.")
    X = load(False)
    rows = []
    for kind, seed in runs:
        y, desc = outcomes(X, kind, seed)
        say(f"\n## {kind}, seed {seed}: {desc}")
        r = run(X, y, False, OUT / "btslab3_smoke_log.txt")
        d = parse_data(selftest_data(kind, seed, r))
        say("Plan's pass rule on this seed:")
        rows.append({"kind": kind, "seed": seed, "h_xhit coef": f"{d['h_xhit']:+.4f}",
                     "T1": f"{'Confirmed' if d['t1'] else 'not confirmed'}, {fmt_ci(r['t1']['est'], r['t1']['ci'])}",
                     "placebo": f"{'Confirmed' if d['stop'] else 'not Confirmed'}, {fmt_ci(r['placebo']['est'], r['placebo']['ci'])}",
                     "pass rule": "PASS" if verdict(d) else "FAIL", "_d": d})
    ph = [r["_d"] for r in rows if r["kind"] == "planted"]
    pn = [r["_d"] for r in rows if r["kind"] == "planted-null"]
    c1 = all(0.05 <= d["h_xhit"] <= 0.15 and d["t1"] and not d["stop"] for d in ph)
    c2 = sum(d["t1"] for d in pn) <= 1
    say("\n## Second dry run, new generator: summary")
    say(md_table(pd.DataFrame([{k: v for k, v in r.items() if k != "_d"} for r in rows])))
    say(f"- Freeze condition 1, both planted-H seeds pass the plan's rule: {'HOLDS' if c1 else 'FAILS'}")
    say(f"- Freeze condition 2, T1 Confirmed on at most 1 of 6 planted-null seeds: {sum(d['t1'] for d in pn)} of 6 -> "
        f"{'HOLDS' if c2 else 'FAILS'}")
    say(f"FREEZE CONDITIONS: {'ALL HOLD' if c1 and c2 else 'NOT MET - do not freeze'}")
    with (OUT / "btslab3_smoke_log.txt").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(ba.LOG) + "\n")


if __name__ == "__main__":
    main()
