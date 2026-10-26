"""Phase 2a: pairing, coverage and exclusion tables + version step-change check.
Writes CSVs to analysis/out/ and prints the tables."""
import numpy as np
import pandas as pd

import lib

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string())


def main():
    lib.OUT.mkdir(exist_ok=True)
    g = lib.exclusion_flags(lib.load_games())
    p = lib.load_players()
    sched = pd.read_csv(lib.CACHE / "schedule_2026.csv")
    fp = pd.read_csv(lib.CACHE / "first_pitch.csv")

    # 1. version x iterations
    vi = pd.crosstab(g["model_version"], g["iterations"], margins=True)
    show("model_version x iterations (game_predictions rows)", vi)

    # 2. exclusion waterfall (sequential, games then player rows)
    order = ["x_v1", "x_allstar", "x_postseason", "x_not_final",
             "x_postponed_attempt", "x_after_first_pitch"]
    rows, alive = [], pd.Series(True, index=g.index)
    pp = p.merge(g[["gp_id"]], on="gp_id")
    for c in ["(all)"] + order:
        if c != "(all)":
            hit = alive & g[c]
            alive &= ~g[c]
        else:
            hit = pd.Series(False, index=g.index)
        row = {"step": c, "games_removed": int(hit.sum())}
        for v in ("v1", "v2", "v3"):
            row[f"{v}_left"] = int((alive & g["model_version"].eq(v)).sum())
        row["games_left"] = int(alive.sum())
        row["player_rows_left"] = int(pp["gp_id"].isin(g.loc[alive, "gp_id"]).sum())
        rows.append(row)
    kept_p = pp[pp["gp_id"].isin(g.loc[alive, "gp_id"])]
    for label, mask in [("player: did_not_play", kept_p["p_status"].eq("did_not_play")),
                        ("player: pending", kept_p["p_status"].eq("pending")),
                        ("player: final but 0 PA", kept_p["p_status"].eq("final")
                         & kept_p["plate_appearances"].fillna(0).eq(0))]:
        rows.append({"step": label, "player_rows_removed": int(mask.sum())})
    final_p = kept_p[kept_p["p_status"].eq("final") & kept_p["plate_appearances"].fillna(0).gt(0)]
    rows.append({"step": "ANALYSIS SET", "games_left": int(alive.sum()),
                 "player_rows_left": len(final_p),
                 **{f"{v}_left": int((alive & g["model_version"].eq(v)).sum()) for v in ("v1", "v2", "v3")}})
    wf = pd.DataFrame(rows)
    show("Exclusion waterfall (sequential; a game counts under its first reason)", wf.fillna(""))
    wf.to_csv(lib.OUT / "exclusion_waterfall.csv", index=False)

    # overlap: games hit by more than one reason
    multi = g[order].sum(axis=1)
    print(f"games with >1 exclusion reason: {int((multi > 1).sum())}")
    print("pending game:", g.loc[g["x_not_final"], ["gp_id", "game_id", "game_date", "model_version", "game_type"]].to_dict("records"))

    # starter-not-pitched voids (K markets only) in the analysis set
    a = g[alive]
    for side in ("away", "home"):
        n = int((~a[f"{side}_starter_pitched"].eq(True)).sum())
        print(f"{side} starter K markets voided (starter did not pitch / unknown): {n}")
    k_avail = a.apply(lambda r: bool(r["away_k_dist"]) and bool(r["home_k_dist"]), axis=1)
    print(f"analysis-set games without both K distributions (Poisson fallback): {int((~k_avail).sum())}")

    # 3. late predictions vs ACTUAL first pitch
    late = g[g["predicted_at"] > g["game_datetime_utc"]].copy()
    late = late.merge(fp[["game_pk", "first_pitch_utc", "delay_minutes"]],
                      left_on="game_id", right_on="game_pk", how="left", suffixes=("", "_fp"))
    late["mins_after_sched"] = (late["predicted_at"] - late["game_datetime_utc"]).dt.total_seconds() / 60
    late["mins_after_actual"] = (late["predicted_at"] - pd.to_datetime(late["first_pitch_utc_fp"], utc=True)).dt.total_seconds() / 60
    late["verdict"] = np.where(late["mins_after_actual"] > 0, "after actual first pitch -> EXCLUDE",
                               "delayed start; before actual first pitch -> keep")
    cols = ["gp_id", "game_id", "sched_date", "model_version", "mins_after_sched",
            "delay_minutes", "mins_after_actual", "verdict"]
    show("Predictions logged after SCHEDULED first pitch (n=%d)" % len(late),
         late[cols].sort_values("mins_after_actual").round(1))
    show("Late-prediction verdicts", late.groupby(["model_version", "verdict"]).size().to_frame("n"))
    late[cols].to_csv(lib.OUT / "late_predictions.csv", index=False)

    # 4. coverage vs schedule (regular season, games actually completed on the listed date)
    sched["listed_date"] = pd.to_datetime(sched["listed_date"])
    played = sched[sched["game_type"].eq("R") & sched["detailed_state"].isin(["Final", "Completed Early"])].copy()
    played = played.drop_duplicates("game_pk", keep="last")  # suspended-and-resumed counted once
    played = played[played["listed_date"] <= lib.REG_SEASON_END]
    played["et_hour"] = pd.to_datetime(played["scheduled_utc"]).dt.tz_convert("America/New_York").dt.hour
    predicted_any = set(g.loc[~g["x_postponed_attempt"], "game_id"])
    kept = set(g.loc[alive, "game_id"])
    played["predicted"] = played["game_pk"].isin(predicted_any)
    played["in_analysis"] = played["game_pk"].isin(kept)
    played["month"] = played["listed_date"].dt.strftime("%Y-%m")
    played["start_bucket"] = pd.cut(played["et_hour"], [0, 16, 19, 21, 24], right=False,
                                    labels=["<4pm ET", "4-7pm", "7-9pm", "9pm+"])

    def cov(by):
        t = played.groupby(by, observed=True).agg(scheduled=("game_pk", "size"),
                                                  predicted=("predicted", "sum"),
                                                  in_analysis=("in_analysis", "sum"))
        t["coverage"] = (t["in_analysis"] / t["scheduled"]).round(3)
        return t

    covm = cov("month")
    show("Coverage of completed regular-season games by month (from first logged day)", covm)
    covs = cov("start_bucket")
    show("Coverage by scheduled start time (ET)", covs)
    covd = cov("day_night")
    show("Coverage by day/night", covd)
    team = pd.concat([played.assign(team=played["home_team"]), played.assign(team=played["away_team"])])
    covt = team.groupby("team").agg(scheduled=("game_pk", "size"), in_analysis=("in_analysis", "sum"))
    covt["coverage"] = (covt["in_analysis"] / covt["scheduled"]).round(3)
    show("Coverage by team (min / median / max)", covt["coverage"].describe().round(3).to_frame().T)
    show("Lowest and highest team coverage", pd.concat([covt.nsmallest(3, "coverage"), covt.nlargest(3, "coverage")]))
    pd.concat({"month": covm, "start": covs, "day_night": covd}).to_csv(lib.OUT / "coverage.csv")
    covt.to_csv(lib.OUT / "coverage_by_team.csv")
    days_no_pred = sorted(set(played["listed_date"].dt.date) - set(g.loc[alive, "sched_date"].dt.date))
    print("\nregular-season dates with completed games but no analysis-set prediction (ET):",
          [str(d) for d in days_no_pred])

    # 5. step changes by sched_date (all versions, pre-exclusion, postponed attempts dropped)
    s = g[~g["x_postponed_attempt"] & ~g["x_allstar"]].copy()
    s["k_dist"] = s.apply(lambda r: bool(r["away_k_dist"]) and bool(r["home_k_dist"]), axis=1)
    s["weather_null"] = s["weather_label"].isna() | s["weather_label"].eq("")
    s["ladder_null"] = s["total_thresholds"].isna()
    pmean = p.groupby("gp_id")[["p_1h", "p_2tb", "p_1hr"]].mean()
    s = s.join(pmean, on="gp_id")

    def daily(lo, hi):
        d = s[s["sched_date"].between(lo, hi)]
        return d.groupby(d["sched_date"].dt.strftime("%m-%d")).agg(
            n=("gp_id", "size"),
            versions=("model_version", lambda x: ",".join(sorted(set(x)))),
            iters=("iterations", lambda x: ",".join(map(str, sorted(set(x))))),
            k_dist_pct=("k_dist", "mean"), weather_null_pct=("weather_null", "mean"),
            ladder_null_pct=("ladder_null", "mean"),
            mean_total=("total_mean", "mean"), mean_med_k=("away_pitcher_median_k", "mean"),
            mean_p_1h=("p_1h", "mean"), mean_p_2tb=("p_2tb", "mean")).round(3)

    show("Daily, around v1 -> v2 (sched_date 04-27 .. 05-08)", daily("2026-04-27", "2026-05-08"))
    show("Daily, around v2 -> v3 (sched_date 07-10 .. 07-27)", daily("2026-07-10", "2026-07-27"))
    wk = s.groupby([s["sched_date"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%m-%d"), "model_version"]).agg(
        n=("gp_id", "size"), k_dist_pct=("k_dist", "mean"), weather_null_pct=("weather_null", "mean"),
        mean_total=("total_mean", "mean"), sd_total=("total_std", "mean"),
        mean_med_k=("away_pitcher_median_k", "mean"),
        mean_p_1h=("p_1h", "mean"), mean_p_2tb=("p_2tb", "mean"), mean_p_1hr=("p_1hr", "mean")).round(3)
    show("Weekly series (week starting Monday) - look for undocumented steps", wk)
    wk.to_csv(lib.OUT / "weekly_inputs_series.csv")


if __name__ == "__main__":
    main()
