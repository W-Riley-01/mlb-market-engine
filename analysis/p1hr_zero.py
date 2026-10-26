"""POST-HOC (B9). Rows with p_1hr exactly 0. Because of the side swap (P1), the stored p_1hr of the
batter NAMED in a row was simulated for the opposite team's batter in the same lineup slot (the
"effective" batter), facing the NAMED batter's own starter. Both views are tabulated.
Handedness from data/master_physics_vault.parquet (modal stand / p_throws); profile presence from
data/batter_arsenal_profiles.parquet and data/recent_form_hitters.parquet."""
import pandas as pd
import pyarrow.parquet as pq

import lib

DATA = lib.CACHE.parents[1] / "data"


def main():
    g = lib.exclusion_flags(lib.load_games())
    p = lib.load_players()
    gi = g.set_index("gp_id")
    vault = pq.read_table(DATA / "master_physics_vault.parquet", columns=["batter", "pitcher", "stand", "p_throws", "game_date"]).to_pandas()
    stand = vault.groupby("batter")["stand"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    throws = vault.groupby("pitcher")["p_throws"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    vault_max = pd.to_datetime(vault["game_date"]).max()
    prof = pd.read_parquet(DATA / "batter_arsenal_profiles.parquet", columns=["batter", "as_of_date"]).drop_duplicates()
    prof["as_of_date"] = pd.to_datetime(prof["as_of_date"])
    rf = pd.read_parquet(DATA / "recent_form_hitters.parquet", columns=["batter", "as_of_date", "effective_n", "recent_hr"])
    rf["as_of_date"] = pd.to_datetime(rf["as_of_date"])
    print(f"vault max game_date {vault_max.date()}  | profile as_of_dates: {prof.as_of_date.nunique()} "
          f"({prof.as_of_date.min().date()}..{prof.as_of_date.max().date()})  | recent-form as_of_dates: "
          f"{rf.as_of_date.nunique()} ({rf.as_of_date.min().date()}..{rf.as_of_date.max().date()})")

    z = p[p.p_1hr.eq(0)].copy()
    opp = p[["gp_id", "side", "batting_order", "player_id", "player_name"]].copy()
    opp["side"] = opp.side.map({"away": "home", "home": "away"})
    z = z.merge(opp.rename(columns={"player_id": "eff_id", "player_name": "eff_name"}),
                on=["gp_id", "side", "batting_order"], how="left")
    z["as_of"] = z.gp_id.map(gi["sched_date"])  # as_of_date = run date (ET)
    z["et_date"] = z.gp_id.map(gi["sched_date"])
    z["version"] = z.gp_id.map(gi["model_version"])
    z["kept"] = z.gp_id.map(gi["keep"])
    # starter faced: named view = opposing starter; effective view = the named batter's own team's starter
    z["named_opp_sp"] = [gi.at[r.gp_id, "home_starter_id" if r.side == "away" else "away_starter_id"] for r in z.itertuples()]
    z["eff_opp_sp"] = [gi.at[r.gp_id, "away_starter_id" if r.side == "away" else "home_starter_id"] for r in z.itertuples()]
    z["named_bats"] = z.player_id.map(stand)
    z["eff_bats"] = z.eff_id.map(stand)
    z["named_sp_throws"] = z.named_opp_sp.map(throws)
    z["eff_sp_throws"] = z.eff_opp_sp.map(throws)
    pk = set(zip(prof.batter, prof.as_of_date))
    rk = rf.set_index(["batter", "as_of_date"])
    for v in ("named", "eff"):
        pid = z.player_id if v == "named" else z.eff_id
        z[f"{v}_profile_exact"] = [(b, d) in pk for b, d in zip(pid, z.as_of)]
        z[f"{v}_profile_any"] = pid.isin(set(prof.batter))
        z[f"{v}_in_vault"] = pid.isin(stand.index)
        z[f"{v}_recent_hr"] = [rk["recent_hr"].get((b, d)) if (b, d) in rk.index else None for b, d in zip(pid, z.as_of)]

    pd.set_option("display.width", 250)
    print(f"\nrows with p_1hr == 0: {len(z)} (analysis set: {int(z.kept.sum())}); versions {z.version.value_counts().to_dict()}")
    for c in ["named_bats", "named_sp_throws", "eff_bats", "eff_sp_throws", "named_profile_exact", "eff_profile_exact",
              "named_profile_any", "eff_profile_any", "named_in_vault", "eff_in_vault"]:
        print(f"{c}: {z[c].value_counts(dropna=False).to_dict()}")
    print("eff recent_hr at as_of_date:", z.eff_recent_hr.describe().round(4).to_dict())
    print("\nby ET date:")
    print(z.groupby(z.et_date.dt.strftime("%Y-%m-%d")).size().to_string())
    print("\nrows per game (all 18 batters of a game? or isolated):")
    print(z.groupby("gp_id").size().value_counts().sort_index().to_string())
    z.drop(columns=["as_of"]).to_csv(lib.OUT / "p1hr_zero_rows.csv", index=False)
    # comparison: same fields for all analysis-set player rows (base rates)
    allp = p.merge(opp.rename(columns={"player_id": "eff_id"})[["gp_id", "side", "batting_order", "eff_id"]],
                   on=["gp_id", "side", "batting_order"])
    allp["as_of"] = allp.gp_id.map(gi["sched_date"])
    allp = allp[allp.gp_id.map(gi["keep"])]
    print("\nall analysis rows: eff batter exact-date profile present:",
          round(pd.Series([(b, d) in pk for b, d in zip(allp.eff_id, allp.as_of)]).mean(), 3),
          "| eff batter in vault:", round(allp.eff_id.isin(stand.index).mean(), 3))


if __name__ == "__main__":
    main()
