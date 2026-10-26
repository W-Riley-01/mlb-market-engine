"""Read-only loader, exclusion flags and metric functions for the analysis.

Standalone on purpose: does not import calibration.py / prediction_logger.py
(those require streamlit, boto3 and AWS env vars at import time).
Password comes from PGPASSWORD via libpq; never passed in code.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

DSN = "postgresql://postgres@localhost:5435/mlb_engine"
CACHE = Path(__file__).parent / "cache"
OUT = Path(__file__).parent / "out"
LOG_LOSS_EPS = 1e-15  # same clip as calibration_metrics.py
K_THRESHOLDS = (3, 4, 5, 6, 7)
PROPS = {"p_1h": "hit_1h", "p_2tb": "hit_2tb", "p_1hr": "hit_1hr",
         "p_1rbi": "hit_1rbi", "p_1sb": "hit_1sb"}
REG_SEASON_END = pd.Timestamp("2026-09-27")


def query(sql: str, params=None) -> pd.DataFrame:
    conn = psycopg2.connect(DSN, options="-c default_transaction_read_only=on")
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW default_transaction_read_only")
            assert cur.fetchone()[0] == "on", "connection is not read-only"
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    """One row per game_predictions row, LEFT-joined to its outcome (any status)."""
    df = query("""
        SELECT gp.id AS gp_id, gp.game_id, gp.game_date, gp.predicted_at,
               gp.game_datetime_utc, gp.model_version, gp.iterations,
               gp.away_team, gp.home_team, gp.away_starter_id, gp.home_starter_id,
               gp.weather_park, gp.weather_is_dome, gp.weather_label, gp.weather_temp_f,
               gp.weather_wind_out_mph, gp.weather_hr_score, gp.weather_carry_delta_ft,
               gp.away_win_prob, gp.home_win_prob, gp.f5_away_prob, gp.f5_home_prob,
               gp.f5_tie_prob, gp.nrfi_prob, gp.median_total, gp.total_mean, gp.total_std,
               gp.total_thresholds, gp.run_line_home_minus_1_5_prob,
               gp.run_line_away_minus_1_5_prob,
               gp.away_pitcher_median_k, gp.home_pitcher_median_k,
               gp.away_p_3k, gp.away_p_4k, gp.away_p_5k, gp.away_p_6k, gp.away_p_7k,
               gp.home_p_3k, gp.home_p_4k, gp.home_p_5k, gp.home_p_6k, gp.home_p_7k,
               gp.away_k_dist, gp.home_k_dist,
               go.status, go.away_score, go.home_score, go.away_won,
               go.f5_away_won, go.f5_tied, go.nrfi_hit, go.total_runs,
               go.away_starter_actual_k, go.away_starter_pitched,
               go.home_starter_actual_k, go.home_starter_pitched
        FROM game_predictions gp
        LEFT JOIN game_outcomes go ON go.game_prediction_id = gp.id
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_id"] = df["game_id"].astype(int)
    for c in [c for c in df.columns if c.endswith("_prob") or c.startswith(("away_p_", "home_p_"))
              or c in ("total_mean", "total_std", "median_total")]:
        df[c] = df[c].astype(float)
    return df


def load_players() -> pd.DataFrame:
    df = query("""
        SELECT pp.id AS pp_id, pp.game_prediction_id AS gp_id, pp.player_id, pp.player_name,
               pp.side, pp.batting_order, pp.p_1h, pp.p_2tb, pp.p_1hr, pp.p_1rbi, pp.p_1sb,
               po.status AS p_status, po.plate_appearances, po.at_bats, po.hits,
               po.total_bases, po.home_runs, po.rbi, po.stolen_bases,
               po.hit_1h, po.hit_2tb, po.hit_1hr, po.hit_1rbi, po.hit_1sb
        FROM player_predictions pp
        LEFT JOIN player_outcomes po ON po.player_prediction_id = pp.id
    """)
    for c in PROPS:
        df[c] = df[c].astype(float)
    return df


# ---------------------------------------------------------------------------
#  Exclusions (documented in the Phase 1 plan)
# ---------------------------------------------------------------------------
def exclusion_flags(games: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per exclusion reason and `keep` (= none apply).

    Uses analysis/cache/schedule_2026.csv and first_pitch.csv (fetch_schedule.py).
    """
    g = games.copy()
    sched = pd.read_csv(CACHE / "schedule_2026.csv")
    fp = pd.read_csv(CACHE / "first_pitch.csv")
    fp["first_pitch_utc"] = pd.to_datetime(fp["first_pitch_utc"], utc=True)

    # game type from the schedule (any listing of the gamePk carries it)
    gtype = sched.drop_duplicates("game_pk").set_index("game_pk")["game_type"]
    g["game_type"] = g["game_id"].map(gtype)
    g["x_v1"] = g["model_version"].eq("v1")
    g["x_allstar"] = g["game_type"].eq("A")
    g["x_postseason"] = g["game_type"].isin(["F", "D", "L", "W"])
    g["x_not_final"] = g["status"].ne("final")

    # game_date is the UTC date (prediction_logger._parse_game_date), so night
    # games sit one day after their ET schedule date. Match each prediction to
    # its schedule listing on game_date, else game_date - 1 day.
    sched["listed_date"] = pd.to_datetime(sched["listed_date"])
    lst = sched.set_index(["game_pk", "listed_date"])["detailed_state"]
    same = pd.MultiIndex.from_arrays([g["game_id"], g["game_date"]])
    prev = pd.MultiIndex.from_arrays([g["game_id"], g["game_date"] - pd.Timedelta(days=1)])
    on_same = same.isin(lst.index)
    g["sched_date"] = g["game_date"].where(on_same, g["game_date"] - pd.Timedelta(days=1))
    g["utc_date_shift"] = ~on_same & prev.isin(lst.index)
    g["sched_state"] = lst.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["sched_date"]])).to_numpy()
    assert g["sched_state"].notna().all(), "prediction with no schedule listing"

    # postponed/cancelled on the predicted date: the outcome row was scored
    # against the makeup game (record_outcomes resolves by game_id only)
    g["x_postponed_attempt"] = g["sched_state"].isin(["Postponed", "Cancelled"])

    # late: logged after ACTUAL first pitch (falls back to scheduled when unknown)
    actual = g["game_id"].map(fp.set_index("game_pk")["first_pitch_utc"])
    first_pitch = actual.fillna(g["game_datetime_utc"])
    g["first_pitch_utc"] = first_pitch
    g["x_after_first_pitch"] = g["predicted_at"] > first_pitch
    g["x_after_scheduled_only"] = (g["predicted_at"] > g["game_datetime_utc"]) & ~g["x_after_first_pitch"]

    reasons = ["x_v1", "x_allstar", "x_postseason", "x_not_final",
               "x_postponed_attempt", "x_after_first_pitch"]
    g["keep"] = ~g[reasons].any(axis=1)
    g["regular_season"] = g["game_type"].eq("R")
    return g


# ---------------------------------------------------------------------------
#  Long format
# ---------------------------------------------------------------------------
def long_games(w: pd.DataFrame, ge_fix: bool = True) -> pd.DataFrame:
    """Game markets in long format, mirroring calibration._melt_game_markets.

    ge_fix=True scores totals rung X as total >= X (what the writer stores,
    advanced_simulator.py: np.mean(totals_arr >= x)). ge_fix=False reproduces
    calibration.py's `total_runs > X`.
    """
    ctx = ["gp_id", "game_id", "game_date", "model_version", "iterations"]
    aw = w["away_won"].astype("boolean")
    f5a = w["f5_away_won"].astype("boolean")
    f5t = w["f5_tied"].astype("boolean")
    nrfi = w["nrfi_hit"].astype("boolean")
    hd = w["home_score"] - w["away_score"]
    chunks = []

    def add(name, p, y):
        c = w[ctx].copy()
        c["market_type"], c["predicted_prob"], c["actual_outcome"] = name, p.values, y.astype("Int64").values
        chunks.append(c)

    add("away_ml", w["away_win_prob"], aw)
    add("home_ml", w["home_win_prob"], ~aw)
    add("f5_away", w["f5_away_prob"], f5a & ~f5t)
    add("f5_home", w["f5_home_prob"], ~f5a & ~f5t)
    add("f5_tie", w["f5_tie_prob"], f5t)
    add("nrfi", w["nrfi_prob"], nrfi)
    add("run_line_home_-1.5", w["run_line_home_minus_1_5_prob"], hd >= 2)
    add("run_line_away_-1.5", w["run_line_away_minus_1_5_prob"], -hd >= 2)

    rows = []
    for r in w[ctx + ["total_thresholds", "total_runs"]].itertuples(index=False):
        th = r.total_thresholds
        if not isinstance(th, dict) or pd.isna(r.total_runs):
            continue
        for k, p in th.items():
            x = float(k)
            y = (r.total_runs >= x) if ge_fix else (r.total_runs > x)
            rows.append((*r[:len(ctx)], f"over_total_{k}", float(p), int(y)))
    chunks.append(pd.DataFrame(rows, columns=ctx + ["market_type", "predicted_prob", "actual_outcome"]))

    for side in ("away", "home"):
        s = w[w[f"{side}_starter_pitched"].eq(True) & w[f"{side}_starter_actual_k"].notna()]
        for t in K_THRESHOLDS:
            c = s[ctx].copy()
            c["market_type"] = f"pitcher_{t}k_{side}"
            c["predicted_prob"] = s[f"{side}_p_{t}k"].values
            c["actual_outcome"] = (s[f"{side}_starter_actual_k"] >= t).astype("Int64").values
            chunks.append(c)

    out = pd.concat(chunks, ignore_index=True)
    out = out[out["actual_outcome"].notna() & out["predicted_prob"].notna()].copy()
    out["actual_outcome"] = out["actual_outcome"].astype(int)
    return out


def long_players(p: pd.DataFrame, g: pd.DataFrame, void_zero_pa: bool = True) -> pd.DataFrame:
    """Player props in long format. Requires p_status='final'; optionally voids 0-PA rows."""
    ctx = ["gp_id", "game_id", "game_date", "model_version", "iterations"]
    m = p.merge(g[ctx], on="gp_id")
    m = m[m["p_status"].eq("final")]
    if void_zero_pa:
        m = m[m["plate_appearances"].fillna(0) > 0]
    chunks = []
    for pc, oc in PROPS.items():
        c = m[ctx + ["pp_id", "player_id", "side", "batting_order"]].copy()
        c["market_type"] = "player_" + pc[2:]
        c["predicted_prob"] = m[pc].values
        c["actual_outcome"] = m[oc].astype(int).values
        chunks.append(c)
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
#  Metrics (legacy-compatible definitions; see calibration_metrics.py)
# ---------------------------------------------------------------------------
def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y, eps=LOG_LOSS_EPS):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def ece_uniform(p, y, n_bins=10):
    """Legacy ECE: uniform bins, n-weighted |mean_pred - mean_actual|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.searchsorted(np.linspace(0, 1, n_bins + 1), p, side="right") - 1, 0, n_bins - 1)
    return float(sum(abs(p[idx == i].mean() - y[idx == i].mean()) * (idx == i).sum()
                     for i in range(n_bins) if (idx == i).any()) / len(p))


def market_summary(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mt, s in long.groupby("market_type"):
        p, y = s["predicted_prob"].to_numpy(), s["actual_outcome"].to_numpy()
        br, base = brier(p, y), y.mean() * (1 - y.mean())
        rows.append({"market_type": mt, "n": len(s), "mean_predicted": p.mean(),
                     "mean_actual": y.mean(), "brier": br, "baseline_brier": base,
                     "skill_score": 1 - br / base if base else np.nan,
                     "ece": ece_uniform(p, y), "bias": p.mean() - y.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Analysis frames (Phase 2b) — see SLICES.md
# ---------------------------------------------------------------------------
RUNS_PER_FT_CARRY = 0.05  # engine_runner.RUNS_PER_FT_CARRY
WIND_FT_PER_MPH = 3.8     # weather.WIND_FT_PER_MPH
REGIMES = [("v2", "2026-05-01", "2026-07-19"), ("v3a", "2026-07-20", "2026-08-04"),
           ("v3b", "2026-08-10", "2026-09-27")]
HALF_RUNGS = ["6.5", "7.5", "8.5", "9.5", "10.5", "11.5"]


def _bucket(x, edges, labels):
    return pd.cut(x, edges, right=False, labels=labels).astype(object)


def analysis_games() -> pd.DataFrame:
    """Kept games with ET date, regime and segment columns."""
    g = exclusion_flags(load_games())
    g = g[g["keep"]].copy()
    g["et_date"] = g["sched_date"]
    g["regime"] = None
    for name, lo, hi in REGIMES:
        g.loc[g["et_date"].between(lo, hi), "regime"] = name
    assert g["regime"].notna().all()
    g["version"] = g["model_version"]
    g["month"] = g["et_date"].dt.strftime("%b")
    et = g["game_datetime_utc"].dt.tz_convert("America/New_York")
    g["start"] = _bucket(et.dt.hour, [0, 16, 19, 21, 24], ["<4pm", "4-7pm", "7-9pm", "9pm+"])
    lead_h = (g["game_datetime_utc"] - g["predicted_at"]).dt.total_seconds() / 3600
    g["lead"] = _bucket(lead_h, [-99, 1, 2, 99], ["<1h", "1-2h", ">=2h"])
    g["weather_missing"] = g["weather_carry_delta_ft"].isna()
    g["wlabel"] = g["weather_label"].replace("", np.nan)
    g["shift"] = g["weather_carry_delta_ft"].fillna(0.0) * RUNS_PER_FT_CARRY
    g["abs_shift_b"] = _bucket(g["shift"].abs(), [0, 0.5000001, 1.5, 2.5, 99],
                               ["<=0.5", "0.5-1.5", "1.5-2.5", ">2.5"])
    g.loc[g["weather_missing"], "abs_shift_b"] = None
    g["park"] = g["home_team"]
    for name, lo, hi in [("v2", "2026-05-01", "2026-07-19"), ("v3", "2026-07-20", "2026-09-27")]:
        sub = g["version"].eq(name)
        med = g.loc[sub, "et_date"].median()
        g.loc[sub, "half"] = np.where(g.loc[sub, "et_date"] <= med, "H1", "H2")
    return g


SEG_COLS = ["version", "regime", "month", "start", "lead", "weather_missing", "wlabel",
            "abs_shift_b", "park", "half", "et_date"]


def target_rows(g: pd.DataFrame, p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long frame: one row per (target, scored unit). Columns: target, gp_id, p, y + segments."""
    base = g[["gp_id"] + SEG_COLS]
    out = []

    def add(name, prob, y, extra=None, mask=None):
        d = base.copy()
        d["target"], d["p"], d["y"] = name, prob.values, y.values
        if extra is not None:
            for k, v in extra.items():
                d[k] = v.values if hasattr(v, "values") else v
        if mask is not None:
            d = d[mask.values]
        out.append(d)

    hw = ~g["away_won"].astype(bool)
    add("home_win", g["home_win_prob"], hw)
    add("nrfi", g["nrfi_prob"], g["nrfi_hit"].astype(bool))
    f5a, f5t = g["f5_away_won"].astype(bool), g["f5_tied"].astype(bool)
    add("f5_home", g["f5_home_prob"], ~f5a & ~f5t)
    add("f5_away", g["f5_away_prob"], f5a & ~f5t)
    add("f5_tie", g["f5_tie_prob"], f5t)
    hd = g["home_score"] - g["away_score"]
    add("rl_home", g["run_line_home_minus_1_5_prob"], hd >= 2)
    add("rl_away", g["run_line_away_minus_1_5_prob"], -hd >= 2)
    for r in HALF_RUNGS:
        add(f"over_{r}", g["total_thresholds"].map(lambda t: t[r]), g["total_runs"] >= float(r) + 0.5)
    for side in ("away", "home"):
        ok = g[f"{side}_starter_pitched"].eq(True) & g[f"{side}_starter_actual_k"].notna()
        for t in K_THRESHOLDS:
            add(f"k{t}_{side}", g[f"{side}_p_{t}k"], g[f"{side}_starter_actual_k"].fillna(-1) >= t,
                extra={"side": side}, mask=ok)
            if t == 5:
                add("k5", g[f"{side}_p_{t}k"], g[f"{side}_starter_actual_k"].fillna(-1) >= t,
                    extra={"side": side}, mask=ok)
    if p is not None:
        m = player_frame(g, p)
        for pc, oc in PROPS.items():
            d = m[["gp_id"] + SEG_COLS + ["side", "slot", "bat_team", "player_id",
                                          "base_own_" + pc]].copy()
            d = d.rename(columns={"base_own_" + pc: "base_own"})
            d["target"], d["p"], d["y"] = pc, m[pc].values, m[oc].astype(bool).values
            d["base_league"] = m["league_" + pc].values
            d["base_slot"] = m["slot_" + pc].values
            out.append(d)
    t = pd.concat(out, ignore_index=True)
    t["y"] = t["y"].astype(int)
    return t


def player_frame(g: pd.DataFrame, p: pd.DataFrame, prior_games: int = 20) -> pd.DataFrame:
    """Analysis-set player rows with baselines (a) league, (b) slot, (c) own past rate."""
    allg = exclusion_flags(load_games())[["gp_id", "sched_date", "away_team", "home_team"]]
    hist = p.merge(allg, on="gp_id")
    hist = hist[hist["p_status"].eq("final") & hist["plate_appearances"].fillna(0).gt(0)]
    # name-collision rows (two players with one name in a game: props merged by name)
    dup = hist.duplicated(["gp_id", "player_name"], keep=False)
    hist = hist[~dup].copy()
    hist["bat_team"] = np.where(hist["side"].eq("home"), hist["home_team"], hist["away_team"])
    # (c) player's own past rate, strictly earlier ET dates, shrunk to past league rate
    hist = hist.sort_values(["sched_date", "pp_id"])
    for pc, oc in PROPS.items():
        y = hist[oc].astype(float)
        day = hist.groupby("sched_date")[oc].agg(["sum", "count"]).astype(float)
        cum = day.cumsum().shift(1).fillna(0)
        league_past = (cum["sum"] / cum["count"]).reindex(hist["sched_date"]).to_numpy()
        pday = hist.assign(_y=y).groupby(["player_id", "sched_date"])["_y"].agg(["sum", "count"])
        pcum = pday.groupby(level=0).cumsum() - pday  # strictly earlier dates
        pc_ = pcum.reindex(pd.MultiIndex.from_arrays([hist["player_id"], hist["sched_date"]]))
        k, n = pc_["sum"].to_numpy(), pc_["count"].to_numpy()
        hist["base_own_" + pc] = (k + prior_games * league_past) / (n + prior_games)
    m = hist.merge(g[["gp_id"] + SEG_COLS], on="gp_id")
    m["slot"] = _bucket(m["batting_order"], [1, 4, 7, 10], ["1-3", "4-6", "7-9"])
    for pc, oc in PROPS.items():  # (a), (b): in-sample per version
        yv = m[oc].astype(float)
        m["league_" + pc] = yv.groupby(m["version"]).transform("mean")
        m["slot_" + pc] = yv.groupby([m["version"], m["batting_order"]]).transform("mean")
    return m


def _self_check():
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert abs(brier([1, 0], [1, 0])) < 1e-12
    w = pd.DataFrame({"gp_id": [1], "game_id": [1], "game_date": [pd.Timestamp("2026-05-01")],
                      "model_version": ["v2"], "iterations": [5000], "total_runs": [8],
                      "total_thresholds": [{"7.5": 0.6, "8.0": 0.5, "8.5": 0.4}],
                      **{c: [np.nan] for c in ["away_won", "f5_away_won", "f5_tied", "nrfi_hit",
                                               "home_score", "away_score", "away_win_prob",
                                               "home_win_prob", "f5_away_prob", "f5_home_prob",
                                               "f5_tie_prob", "nrfi_prob",
                                               "run_line_home_minus_1_5_prob",
                                               "run_line_away_minus_1_5_prob"]
                                    + [f"{s}_p_{t}k" for s in ("away", "home") for t in K_THRESHOLDS]},
                      "away_starter_pitched": [False], "home_starter_pitched": [False],
                      "away_starter_actual_k": [np.nan], "home_starter_actual_k": [np.nan]})
    fixed = long_games(w, ge_fix=True).set_index("market_type")["actual_outcome"]
    legacy = long_games(w, ge_fix=False).set_index("market_type")["actual_outcome"]
    assert fixed["over_total_8.0"] == 1 and legacy["over_total_8.0"] == 0  # total == line
    assert fixed["over_total_7.5"] == legacy["over_total_7.5"] == 1


if __name__ == "__main__":
    _self_check()
    print("lib self-check ok")
