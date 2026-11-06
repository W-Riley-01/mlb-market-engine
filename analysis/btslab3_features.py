"""Beat the Streak lab 3, Step 1: group H as-of feature table (plan BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md, Sections 2-4).

One row per row of lab 1's table, same order, keyed (game_pk, id). Only lab 1's key, date and id columns are read; the
base file is checked by its recorded file sha256, so no outcome column is ever loaded. Every history count uses games
with effective date strictly before D; each sub-group carries H<k>_max_date. As in lab 1 the table stores as-of COUNTS
and league rates; shrinkage, standardisation and fills are applied per training fold at fit time (plan Section 4).
No 2026 row is read: the vault, pitch_matrix and sprint file end in 2025, and 2026 schedule files are skipped.

  python analysis/btslab3_features.py > analysis/out/btslab3_features_build_log.txt
writes analysis/cache/btslab3_features.parquet
"""
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
import btslab_features as bf  # noqa: E402
from lab2_contact_table import ANGLE_EDGES, SPEED_EDGES, bins  # noqa: E402

BASE = bf.OUT_PARQUET
BASE_FILE_SHA = "14cf9d8b75516ce1349558a7e378e130a517fd145296b35ab68b7692e8f63497"  # content e1fb5d01..., lab2_step1b_log
BASE_COLS = ["game_pk", "id", "date", "season", "opp_sp", "venue_id", "home_id", "away_id"]
OUT_PARQUET = ROOT / "analysis" / "cache" / "btslab3_features.parquet"
VAULT = ROOT / "data" / "master_physics_vault.parquet"
BIO = ROOT / "analysis" / "cache" / "player_bio.parquet"
SPRINT = ROOT / "analysis" / "cache" / "sprint_speed_by_season.parquet"
CONTACT = ROOT / "analysis" / "lab2_contact_table.csv"
VAULT_COLS = ["game_pk", "game_date", "batter", "type", "description", "zone", "launch_speed", "launch_angle"]
SWING = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt", "missed_bunt", "bunt_foul_tip",
         "hit_into_play", "foul_pitchout", "swinging_pitchout"}
MISS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
H1C = ["bip", "hard", "gb", "ld", "pu", "S", "Q"]
H2C = ["pitch", "swing", "miss", "out", "out_swing", "in_swing", "in_miss"]
EMPTY_BOX = pd.DataFrame({"game_pk": pd.Series(dtype="int64"), "side": pd.Series(dtype="str"), "pitcher": pd.Series(dtype="int64"),
                          "order": pd.Series(dtype="int64"), "bf": pd.Series(dtype=float), "pitches": pd.Series(dtype=float)})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- inputs
def load_schedule() -> pd.DataFrame:
    """bf.load_schedule() without the 2026 day files; a 2019-2025 game's last Final appearance is never in them."""
    glob = Path.glob
    Path.glob = lambda self, pat: (p for p in glob(self, pat) if p.name < "2026")
    try:
        return bf.load_schedule()
    finally:
        Path.glob = glob


def load_base() -> pd.DataFrame:
    assert sha(BASE) == BASE_FILE_SHA, "stop (plan Section 10 rule 2): lab 1 table differs from the recorded file"
    return pd.read_parquet(BASE, columns=BASE_COLS)


def load_pm() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "pitch_matrix.parquet",
                           columns=["game_pk", "game_date", "inning_topbot", "at_bat_number", "pitch_number", "pitcher",
                                    "batter", "p_throws", "events", "stand"])


def load_contact() -> np.ndarray:
    """13 x 11 array of p_hit by (speed bin, angle bin), the bins of lab2_contact_table.py (left-closed)."""
    T = pd.read_csv(CONTACT, comment="#")
    P = np.full((len(SPEED_EDGES) - 1, len(ANGLE_EDGES) - 1), np.nan)
    for r in T.itertuples():
        i, j = SPEED_EDGES.index(r.speed_lo), ANGLE_EDGES.index(r.angle_lo)
        assert (SPEED_EDGES[i + 1], ANGLE_EDGES[j + 1]) == (r.speed_hi, r.angle_hi)
        P[i, j] = r.p_hit
    assert len(T) == 143 and not np.isnan(P).any()
    return P


def load_inputs() -> dict:
    return {"sched": load_schedule(), "pm": load_pm(), "vault": pd.read_parquet(VAULT, columns=VAULT_COLS),
            "bio": pd.read_parquet(BIO), "sprint": pd.read_parquet(SPRINT), "contact": load_contact()}


# ---------------------------------------------------------------- pitch-level daily tables (vault)
def vault_daily(vault: pd.DataFrame, sched: pd.DataFrame, P: np.ndarray) -> dict:
    """Per batter-day and league-day sums of the H1 (launch-data BIP) and H2 (all pitches) trial counts."""
    gd = sched.set_index("game_pk").date
    v = vault[vault.game_pk.isin(gd.index)]
    v = pd.DataFrame({"batter": v.batter.to_numpy(), "date": v.game_pk.map(gd).to_numpy(), "type": v.type.to_numpy(),
                      "desc": v.description.to_numpy(), "zone": v.zone.to_numpy(),
                      "ls": v.launch_speed.to_numpy(float), "la": v.launch_angle.to_numpy(float)})
    sw, ms = v.desc.isin(SWING), v.desc.isin(MISS)
    out, inz = v.zone.between(11, 14), v.zone.between(1, 9)  # null zone is neither
    v["pitch"], v["swing"], v["miss"] = 1, sw.astype(int), ms.astype(int)
    v["out"], v["out_swing"] = out.astype(int), (out & sw).astype(int)
    v["in_swing"], v["in_miss"] = (inz & sw).astype(int), (inz & ms).astype(int)
    b = v[v.type.eq("X") & v.ls.notna() & v.la.notna()].copy()
    b["bip"], b["hard"] = 1, (b.ls >= 95).astype(int)
    b["gb"], b["ld"], b["pu"] = (b.la < 10).astype(int), ((b.la >= 10) & (b.la < 25)).astype(int), (b.la > 50).astype(int)
    b["S"] = P[bins(b.ls, SPEED_EDGES).to_numpy(int), bins(b.la, ANGLE_EDGES).to_numpy(int)]
    b["Q"] = b.S ** 2
    daily = lambda df, keys, cols: df.groupby(keys + ["date"])[cols].sum().reset_index()
    return {"h1": daily(b, ["batter"], H1C), "h1_lg": daily(b.assign(one=1), ["one"], H1C),
            "h2": daily(v, ["batter"], H2C), "h2_lg": daily(v.assign(one=1), ["one"], H2C)}


# ---------------------------------------------------------------- build
def build(inp: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Group H columns for `rows` (game_pk, id, date, season, opp_sp, venue_id, home_id)."""
    sched, pm, bio, sprint = inp["sched"], inp["pm"], inp["bio"].set_index("mlbam_id"), inp["sprint"]
    rows = rows.reset_index(drop=True)
    D = rows.date
    y365, y730 = D - pd.Timedelta(days=365), D - pd.Timedelta(days=730)
    old = rows.assign(date=y365)  # older window [D-730, D-365): window() sums [lo, q.date)
    one, one_old = rows.assign(one=1), old.assign(one=1)
    vd = vault_daily(inp["vault"], sched, inp["contact"])
    raw = bf.raw_tables(pm.drop(columns="stand"), sched, EMPTY_BOX)
    F = rows[["game_pk", "id", "date", "season"]].copy()
    mx = lambda *s: pd.concat([x.reset_index(drop=True) for x in s], axis=1).max(axis=1).to_numpy()

    # H1 contact quality, [D-365, D)
    w1 = bf.window(vd["h1"], ["batter"], rows, ["id"], H1C, y365)
    l1 = bf.window(vd["h1_lg"], ["one"], one, ["one"], H1C, y365)
    for c in H1C:
        F[f"h1_{c}"] = w1[c].to_numpy()
    for c in ("hard", "gb", "ld", "pu"):
        F[f"lg_{c}"] = (l1[c] / l1.bip).to_numpy()
    F["lg_xhit"] = (l1.S / l1.bip).to_numpy()
    F["H1_miss"] = F.h1_bip.eq(0).astype(int)
    F["H1_max_date"] = mx(w1.src, l1.src)

    # H2 plate discipline, [D-365, D)
    w2 = bf.window(vd["h2"], ["batter"], rows, ["id"], H2C, y365)
    l2 = bf.window(vd["h2_lg"], ["one"], one, ["one"], H2C, y365)
    for c in H2C:
        F[f"h2_{c}"] = w2[c].to_numpy()
    F["lg_whiff"] = (l2.miss / l2.swing).to_numpy()
    F["lg_chase"] = (l2.out_swing / l2.out).to_numpy()
    F["lg_zcon"] = ((l2.in_swing - l2.in_miss) / l2.in_swing).to_numpy()
    F["H2_miss"] = F.h2_pitch.eq(0).astype(int)
    F["H2_max_date"] = mx(w2.src, l2.src)

    # H3 handedness: bio sides; switch hitter bats opposite the starter; park H/PA split by the PA-ending pitch's stand
    F["bat_side"] = rows.id.map(bio.bat_side).to_numpy()
    F["sp_hand"] = rows.opp_sp.map(bio.pitch_hand).to_numpy()
    hand = pd.merge_asof(rows[["date", "opp_sp"]].assign(_i=np.arange(len(rows))).sort_values("date"),
                         raw["hand"].rename(columns={"pitcher": "opp_sp", "date": "hdate"}).assign(date=lambda x: x.hdate).sort_values("date"),
                         on="date", by="opp_sp", allow_exact_matches=False).sort_values("_i")
    F["c_hand"] = hand.p_throws.to_numpy()  # lab 1's c_hand: the starter's last p_throws before D
    opp = {"R": "L", "L": "R"}
    sp_known = F.sp_hand.isin(["R", "L"])
    sw_side = np.where(sp_known, F.sp_hand.map(opp), F.c_hand.map(opp))
    F["hp_side"] = np.where(F.bat_side.eq("S"), sw_side, F.bat_side)
    F["hp_side"] = F.hp_side.where(F.hp_side.isin(["R", "L"]))  # NaN only for a switch hitter vs a switch pitcher, c_hand unknown
    F["h_same"] = (sp_known & F.hp_side.eq(F.sp_hand)).astype(int)
    F["h_same_miss"] = (~sp_known).astype(int)
    lo = pd.to_datetime((rows.season - 3).astype(str) + "-01-01")
    for c in ("hp_venue_h", "hp_venue_pa", "hp_road_h", "hp_road_pa"):
        F[c] = np.nan
    F["H3_max_date"] = pd.NaT
    for s in ("R", "L"):
        rs = bf.raw_tables(pm[pm.stand.eq(s)].drop(columns="stand"), sched, EMPTY_BOX)
        pv = bf.window(rs["park_venue"], ["venue_id"], rows, ["venue_id"], ["h", "pa"], lo)
        pr = bf.window(rs["park_road"], ["team"], rows, ["home_id"], ["h", "pa"], lo)
        k = F.hp_side.eq(s).to_numpy()
        F.loc[k, "hp_venue_h"], F.loc[k, "hp_venue_pa"] = pv.h.to_numpy()[k], pv.pa.to_numpy()[k]
        F.loc[k, "hp_road_h"], F.loc[k, "hp_road_pa"] = pr.h.to_numpy()[k], pr.pa.to_numpy()[k]
        F.loc[k, "H3_max_date"] = mx(pv.src, pr.src)[k]
    used_c = (F.bat_side.eq("S") & ~sp_known & F.hp_side.notna()).to_numpy()  # switch vs switch: side came from c_hand
    F.loc[used_c, "H3_max_date"] = mx(F.H3_max_date, hand.hdate)[used_c]

    # H4 speed: season s-1 only, not shrunk; max date = end of season s-1 (its last regular-season effective date)
    sp = sprint.rename(columns={"mlbam_id": "id", "season": "h_speed_season"})
    m4 = rows[["id", "season"]].assign(h_speed_season=rows.season - 1).merge(sp, on=["id", "h_speed_season"], how="left")
    F["h_speed"] = m4.sprint_speed.to_numpy()
    F["h_speed_season"] = np.where(F.h_speed.notna(), m4.h_speed_season, np.nan)
    F["h_speed_miss"] = F.h_speed.isna().astype(int)
    end = sched.groupby("season").date.max()
    F["H4_max_date"] = (rows.season - 1).map(end).where(F.h_speed.notna()).to_numpy()

    # H5 age (on D, centred at 28) and trend: older window [D-730, D-365) of lab 1's H/AB and of the contact-table xhit
    for c, who in (("h_age", rows.id), ("sp_age", rows.opp_sp)):
        b = who.map(bio.birth_date)
        assert b.notna().all(), f"{c}: birth date missing"
        F[c] = (D - b).dt.days.to_numpy() / 365.2425 - 28
        F[f"{c}2"] = F[c] ** 2
    t = bf.window(raw["bat"], ["batter"], old, ["id"], ["h", "ab"], y730)
    tl = bf.window(raw["lg"], ["one"], one_old, ["one"], ["h", "ab"], y730)
    tx = bf.window(vd["h1"], ["batter"], old, ["id"], ["bip", "S"], y730)
    txl = bf.window(vd["h1_lg"], ["one"], one_old, ["one"], ["bip", "S"], y730)
    new = bf.window(raw["bat"], ["batter"], rows, ["id"], ["h", "ab"], y365)  # a_hab's window (its counts stay in lab 1's table)
    newl = bf.window(raw["lg"], ["one"], one, ["one"], ["h", "ab"], y365)
    F["t_h_old"], F["t_ab_old"], F["lg_h_ab_old"] = t.h.to_numpy(), t.ab.to_numpy(), (tl.h / tl.ab).to_numpy()
    F["t_bip_old"], F["t_S_old"], F["lg_xhit_old"] = tx.bip.to_numpy(), tx.S.to_numpy(), (txl.S / txl.bip).to_numpy()
    F["H5_trend_miss"] = F.t_ab_old.eq(0).astype(int)
    F["H5_max_date"] = mx(t.src, tl.src, tx.src, txl.src, new.src, newl.src, F.H1_max_date)
    return F


def check(F: pd.DataFrame) -> None:
    for g in ("H1", "H2", "H3", "H4", "H5"):
        assert not (F[f"{g}_max_date"] >= F.date).any(), f"{g}_max_date not before D"
    k = F.h_speed.notna()
    assert (F.h_speed_season[k] == F.season[k] - 1).all(), "speed must come from season s-1"
    assert F.bat_side.isin(["R", "L", "S"]).all(), "bat side missing"


def describe(F: pd.DataFrame) -> pd.DataFrame:
    """Unshrunk window rates (shrinkage is fit-time) and the stored final columns: missing share, mean, SD, min, max."""
    r = lambda a, b: F[a] / F[b].where(F[b] > 0)
    road = r("hp_road_h", "hp_road_pa")
    cols = {"h_hard raw": r("h1_hard", "h1_bip"), "h_xhit raw": r("h1_S", "h1_bip"), "h_gb raw": r("h1_gb", "h1_bip"),
            "h_ld raw": r("h1_ld", "h1_bip"), "h_pu raw": r("h1_pu", "h1_bip"), "H1_miss": F.H1_miss,
            "h_whiff raw": r("h2_miss", "h2_swing"), "h_chase raw": r("h2_out_swing", "h2_out"),
            "h_zcon raw": (F.h2_in_swing - F.h2_in_miss) / F.h2_in_swing.where(F.h2_in_swing > 0), "H2_miss": F.H2_miss,
            "h_same": F.h_same, "h_same_miss": F.h_same_miss, "h_park_hand raw (venue/road H per PA)": r("hp_venue_h", "hp_venue_pa") / road,
            "h_speed": F.h_speed, "h_speed_miss": F.h_speed_miss, "h_age": F.h_age, "h_age2": F.h_age2, "sp_age": F.sp_age,
            "sp_age2": F.sp_age2, "H/AB older window raw (trend_hab 2nd term)": r("t_h_old", "t_ab_old"),
            "xhit older window raw (trend_xhit 2nd term)": r("t_S_old", "t_bip_old"),
            "trend_xhit raw": r("h1_S", "h1_bip") - r("t_S_old", "t_bip_old"), "H5_trend_miss": F.H5_trend_miss}
    for c in F.columns:
        if c.startswith(("h1_", "h2_", "lg_", "hp_", "t_")) and F[c].dtype.kind in "fi":
            cols[c] = F[c]
    return pd.DataFrame({k: {"missing": v.isna().mean(), "mean": v.mean(), "sd": v.std(), "min": v.min(), "max": v.max()}
                         for k, v in cols.items()}).T


def main():
    t0 = time.time()
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(f"Lab 3 group H feature build ({datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}); "
          f"python {sys.version.split()[0]}, pandas {pd.__version__}")
    print(f"plan sha256 {sha(ROOT / 'analysis' / 'BEAT_THE_STREAK_LAB3_PROFILE_PLAN.md')}")
    X = load_base()
    print(f"base {BASE.relative_to(ROOT)}: file sha256 matches the recorded {BASE_FILE_SHA[:8]}... (content e1fb5d01...); "
          f"columns read {BASE_COLS}; rows {len(X):,}")
    inp = load_inputs()
    v = inp["vault"]
    print(f"inputs: schedule games {len(inp['sched']):,} (seasons {sorted(inp['sched'].season.unique())}); vault rows {len(v):,} "
          f"({v.game_date.min():%Y-%m-%d} to {v.game_date.max():%Y-%m-%d}); pitch_matrix rows {len(inp['pm']):,} "
          f"({inp['pm'].game_date.min()} to {inp['pm'].game_date.max()}); bio {len(inp['bio']):,} (sha256 {sha(BIO)[:8]}...); "
          f"sprint {len(inp['sprint']):,} (sha256 {sha(SPRINT)[:8]}...); contact table sha256 {sha(CONTACT)[:8]}...")
    reg = v[v.game_pk.isin(inp["sched"].game_pk)]
    print(f"vault regular-season pitches {len(reg):,}; null zone {int(reg.zone.isna().sum())} "
          f"(by year {reg[reg.zone.isna()].game_date.dt.year.value_counts().to_dict()}); zone 10 {int(reg.zone.eq(10).sum())}; "
          f"pitch_matrix PA rows with null stand {int(inp['pm'][inp['pm'].events.notna()].stand.isna().sum())}")
    F = build(inp, X)
    check(F)
    assert len(F) == len(X) and (F.game_pk.to_numpy() == X.game_pk.to_numpy()).all() and (F.id.to_numpy() == X.id.to_numpy()).all()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET.relative_to(ROOT)}: rows {len(F):,}, cols {F.shape[1]}; same rows and order as the base table "
          f"(game_pk, id): True; duplicate keys {int(F.duplicated(['game_pk', 'id']).sum())}")
    print(f"content sha256 {content_hash(F)} (sha256 of pandas.util.hash_pandas_object(df, index=False)); file sha256 {sha(OUT_PARQUET)}")
    print("\n## Switch hitters and hands (row counts)")
    print(pd.crosstab(F.bat_side, F.sp_hand.fillna("missing")).to_string())
    print(f"rows with hp_side missing (switch hitter vs switch pitcher, c_hand unknown -> f_park at fit time): {int(F.hp_side.isna().sum())}; "
          f"switch vs switch with c_hand used: {int((F.bat_side.eq('S') & ~F.sp_hand.isin(['R', 'L']) & F.hp_side.notna()).sum())}")
    print("\n## Missing flags by season (share of rows)")
    print(F.groupby("season")[["H1_miss", "H2_miss", "h_same_miss", "h_speed_miss", "H5_trend_miss"]].mean().round(4).to_string())
    print("\n## Feature summary, all rows (raw = unshrunk window rate; shrinkage is applied per fold at fit time)")
    print(describe(F).to_string(float_format=lambda x: f"{x:.5g}"))
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
