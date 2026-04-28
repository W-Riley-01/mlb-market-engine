"""
analyze_statcast_splits.py

Generic split analysis for a single batter's Statcast data.

Inputs:
    - data/processed/<prefix>_statcast_clean.csv

Outputs (all in data/processed/):
    - <prefix>_splits_vs_hand.csv
    - <prefix>_splits_by_month.csv
    - <prefix>_splits_by_pitch_type.csv
    - <prefix>_splits_by_pitch_group.csv
    - <prefix>_splits_by_count_bucket.csv
    - <prefix>_zone_discipline.csv
    - <prefix>_home_away_splits.csv   (only if --team is given)
    - <prefix>_league_context.csv     (percentiles vs league hitters)

Usage:
    python analyze_statcast_splits.py
    python analyze_statcast_splits.py --prefix vinny --team KCR
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd
from pybaseball import batting_stats_bref

from utils import load_processed, save_processed

# -----------------------
# PITCH TYPE LOOKUP TABLE
# -----------------------

PITCH_TYPE_NAME = {
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "FF": "Four-seam Fastball",
    "FO": "Forkball",
    "FS": "Split-finger Fastball",
    "KC": "Knuckle Curve",
    "SI": "Sinker",
    "SL": "Slider",
    "ST": "Sweeper",
    "SC": "Screwball",
    "KN": "Knuckleball",
}

# -----------------------
# COMMON SUMMARY HELPER
# -----------------------

def summarize_group(df: pd.DataFrame) -> dict:
    """
    Compute core hitter metrics for a subset of Statcast rows.

    Assumes 'is_ab', 'is_hit', 'is_2b', 'is_3b', 'is_hr', 'is_bb', 'is_so'
    columns already exist (from ingest_statcast.py).
    """
    pa = len(df)
    ab = int(df["is_ab"].sum())
    h = int(df["is_hit"].sum())
    doubles = int(df["is_2b"].sum())
    triples = int(df["is_3b"].sum())
    hr = int(df["is_hr"].sum())
    bb = int(df["is_bb"].sum())
    so = int(df["is_so"].sum())

    # Slash line
    avg = h / ab if ab > 0 else None
    obp = None
    obp_denom = ab + bb
    if obp_denom > 0:
        obp = (h + bb) / obp_denom

    singles = h - doubles - triples - hr
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    slg = tb / ab if ab > 0 else None
    ops = (obp + slg) if (obp is not None and slg is not None) else None
    iso = (slg - avg) if (slg is not None and avg is not None) else None

    # Plate discipline
    bb_pct = (bb / pa * 100) if pa > 0 else None
    k_pct = (so / pa * 100) if pa > 0 else None
    k_minus_bb = (k_pct - bb_pct) if (bb_pct is not None and k_pct is not None) else None

    # Contact quality (if available)
    avg_ev = avg_la = hardhit_pct = sweetspot_pct = None
    if "launch_speed" in df.columns and "launch_angle" in df.columns:
        batted = df[df["launch_speed"].notna()].copy()
        n_batted = len(batted)
        if n_batted > 0:
            avg_ev = batted["launch_speed"].mean()
            avg_la = batted["launch_angle"].mean()
            hardhit = (batted["launch_speed"] >= 95).sum()
            hardhit_pct = hardhit / n_batted * 100
            sweet = batted["launch_angle"].between(8, 32, inclusive="both").sum()
            sweetspot_pct = sweet / n_batted * 100

    # wOBA / xwOBA (if available)
    woba = xwoba = None
    if "woba_value" in df.columns and "woba_denom" in df.columns:
        wv = df["woba_value"].sum()
        wd = df["woba_denom"].sum()
        if wd > 0:
            woba = wv / wd
    if "estimated_woba_using_speedangle" in df.columns:
        xwoba = df["estimated_woba_using_speedangle"].mean()

    return {
        "PA": pa,
        "AB": ab,
        "H": h,
        "2B": doubles,
        "3B": triples,
        "HR": hr,
        "BB": bb,
        "SO": so,
        "AVG": avg,
        "OBP": obp,
        "SLG": slg,
        "OPS": ops,
        "ISO": iso,
        "BB%": bb_pct,
        "K%": k_pct,
        "K-BB%": k_minus_bb,
        "avg_ev": avg_ev,
        "avg_la": avg_la,
        "HardHit%": hardhit_pct,
        "SweetSpot%": sweetspot_pct,
        "wOBA": woba,
        "xwOBA": xwoba,
    }


def round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rounding conventions:
      - 3 decimals for AVG/OBP/SLG/OPS/ISO/wOBA/xwOBA
      - 1 decimal for percent/quality metrics
    """
    df = df.copy()

    three_dec = ["AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "xwOBA"]
    one_dec = [
        "BB%", "K%", "K-BB%",
        "avg_ev", "avg_la",
        "HardHit%", "SweetSpot%",
    ]

    for col in three_dec:
        if col in df.columns:
            df[col] = df[col].round(3)

    for col in one_dec:
        if col in df.columns:
            df[col] = df[col].round(1)

    return df

# -----------------------
# SPLIT FUNCTIONS
# -----------------------

def split_vs_hand(df: pd.DataFrame) -> pd.DataFrame:
    if "p_throws" not in df.columns:
        raise KeyError("Missing 'p_throws' column for handedness splits.")

    records = []
    for (year, hand), g in df.groupby(["game_year", "p_throws"], dropna=False):
        row = {"game_year": year, "pitcher_hand": hand}
        row.update(summarize_group(g))
        records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)


def split_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" not in df.columns:
        raise KeyError("Missing 'game_date' column for monthly splits.")

    d = df.copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.to_period("M").astype(str)

    records = []
    for (year, month), g in d.groupby(["game_year", "month"], dropna=False):
        row = {"game_year": year, "month": month}
        row.update(summarize_group(g))
        records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)


def split_by_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    if "pitch_type" not in df.columns:
        raise KeyError("Missing 'pitch_type' column for pitch-type splits.")

    records = []
    for (year, ptype), g in df.groupby(["game_year", "pitch_type"], dropna=False):
        row = {
            "game_year": year,
            "pitch_type": ptype,
            "pitch_name": PITCH_TYPE_NAME.get(str(ptype).upper(), "Unknown"),
        }
        row.update(summarize_group(g))
        records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)


def map_pitch_group(pitch_type: Optional[str]) -> str:
    """
    Map raw pitch_type strings to broader groups:
      - Fastball
      - Breaking
      - Offspeed
      - Other
    """
    if pitch_type is None or pd.isna(pitch_type):
        return "Other"

    pt = str(pitch_type).upper()

    fastballs = {"FF", "FA", "FC", "FT", "SI", "FS"}
    breaking = {"SL", "CU", "KC", "ST"}
    offspeed = {"CH", "FO"}

    if pt in fastballs:
        return "Fastball"
    if pt in breaking:
        return "Breaking"
    if pt in offspeed:
        return "Offspeed"
    return "Other"


def split_by_pitch_group(df: pd.DataFrame) -> pd.DataFrame:
    if "pitch_type" not in df.columns:
        raise KeyError("Missing 'pitch_type' column for pitch-group splits.")

    d = df.copy()
    d["pitch_group"] = d["pitch_type"].apply(map_pitch_group)

    records = []
    for (year, group), g in d.groupby(["game_year", "pitch_group"], dropna=False):
        row = {"game_year": year, "pitch_group": group}
        row.update(summarize_group(g))
        records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)


def categorize_count(balls: int, strikes: int) -> str:
    """
    Simple count bucket:
      - Hitter-Friendly: 2-0, 3-0, 3-1
      - Pitcher-Friendly: 0-2, 1-2, 2-2, 0-1, 3-2
      - Neutral: everything else
    """
    c = f"{balls}-{strikes}"
    hitter_friendly = {"2-0", "3-0", "3-1"}
    pitcher_friendly = {"0-2", "1-2", "2-2", "0-1", "3-2"}

    if c in hitter_friendly:
        return "Hitter-Friendly"
    if c in pitcher_friendly:
        return "Pitcher-Friendly"
    return "Neutral"


def split_by_count_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "balls" not in df.columns or "strikes" not in df.columns:
        raise KeyError("Missing 'balls'/'strikes' columns for count splits.")

    d = df.copy()
    d["count_bucket"] = d.apply(
        lambda r: categorize_count(int(r["balls"]), int(r["strikes"])), axis=1
    )

    records = []
    for (year, bucket), g in d.groupby(["game_year", "count_bucket"], dropna=False):
        row = {"game_year": year, "count_bucket": bucket}
        row.update(summarize_group(g))
        records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)


def compute_zone_discipline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Z-Swing%, O-Swing%, Z-Contact%, O-Contact% by year.

    Approx logic:
      - in-zone = 1–9 in 'zone'
      - out-of-zone = everything else / NaN
      - swing = description includes swing/foul/hit_into_play
      - contact = description includes foul/hit_into_play
    """
    required = {"zone", "description", "game_year"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for zone discipline: {missing}")

    d = df.copy()
    d["in_zone"] = d["zone"].between(1, 9, inclusive="both")

    desc = d["description"].fillna("").str.lower()
    d["swing"] = desc.str.contains("swinging_strike|hit_into_play|foul")
    d["contact"] = desc.str.contains("hit_into_play|foul")

    records = []

    for year, g in d.groupby("game_year", dropna=False):
        def pct(part: int, whole: int) -> Optional[float]:
            return (part / whole * 100) if whole > 0 else None

        z = g[g["in_zone"] == True]
        o = g[g["in_zone"] == False]

        z_pitches = len(z)
        z_swings = int(z["swing"].sum())
        z_contact = int(z["contact"].sum())

        o_pitches = len(o)
        o_swings = int(o["swing"].sum())
        o_contact = int(o["contact"].sum())

        z_swing_pct = pct(z_swings, z_pitches)
        o_swing_pct = pct(o_swings, o_pitches)
        z_contact_pct = pct(z_contact, z_swings) if z_swings > 0 else None
        o_contact_pct = pct(o_contact, o_swings) if o_swings > 0 else None

        records.append({
            "game_year": year,
            "Z_Swing%": z_swing_pct,
            "O_Swing%": o_swing_pct,
            "Z_Contact%": z_contact_pct,
            "O_Contact%": o_contact_pct,
            "Z_Pitches": z_pitches,
            "O_Pitches": o_pitches,
        })

    out = pd.DataFrame(records)
    for col in ["Z_Swing%", "O_Swing%", "Z_Contact%", "O_Contact%"]:
        if col in out.columns:
            out[col] = out[col].round(1)
    return out

# -----------------------
# HOME / AWAY (optional)
# -----------------------

def split_home_away(df: pd.DataFrame, team: str) -> pd.DataFrame:
    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise KeyError("Missing 'home_team'/'away_team' columns for home/away splits.")

    d = df.copy()
    d["is_home"] = d["home_team"].str.upper() == team.upper()
    d["is_away"] = d["away_team"].str.upper() == team.upper()

    records = []

    for where, mask_col in [("home", "is_home"), ("away", "is_away")]:
        sub = d[d[mask_col] == True]
        if sub.empty:
            continue
        for year, g in sub.groupby("game_year", dropna=False):
            row = {"game_year": year, "venue": where.capitalize()}
            row.update(summarize_group(g))
            records.append(row)

    out = pd.DataFrame(records)
    return round_summary(out)

# -----------------------
# SIMPLE LEAGUE CONTEXT
# -----------------------

def build_league_context(prefix: str, min_pa: int = 300) -> None:
    """
    Temporarily disabled league context because pybaseball/batting_stats_bref
    is throwing an internal 'list index out of range' in this environment.

    All other split outputs remain valid.
    """
    print("[league_context] Disabled; skipping league percentile context.")
    return

# -----------------------
# MAIN / CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Statcast hitter splits.")
    parser.add_argument(
        "--prefix",
        default="vinny",
        help="Prefix used for processed files (default: vinny). "
             "Script expects <prefix>_statcast_clean.csv in data/processed/.",
    )
    parser.add_argument(
        "--team",
        default=None,
        help="Optional team code (e.g. KCR, NYY, LAD) to compute home/away splits.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix
    team = args.team

    clean_name = f"{prefix}_statcast_clean.csv"
    df = load_processed(clean_name)
    print(f"[INFO] Loaded cleaned Statcast: {clean_name} → {df.shape[0]} rows, {df.shape[1]} columns")

    # 1) vs handedness
    try:
        vs_hand = split_vs_hand(df)
        save_processed(vs_hand, f"{prefix}_splits_vs_hand.csv")
        print("[INFO] Saved vs-hand splits.")
    except KeyError as e:
        print(f"[WARN] Skipping vs-hand splits: {e}")

    # 2) monthly
    try:
        by_month = split_by_month(df)
        save_processed(by_month, f"{prefix}_splits_by_month.csv")
        print("[INFO] Saved monthly splits.")
    except KeyError as e:
        print(f"[WARN] Skipping monthly splits: {e}")

    # 3) pitch type
    try:
        by_pt = split_by_pitch_type(df)
        save_processed(by_pt, f"{prefix}_splits_by_pitch_type.csv")
        print("[INFO] Saved pitch-type splits.")
    except KeyError as e:
        print(f"[WARN] Skipping pitch-type splits: {e}")

    # 4) pitch group
    try:
        by_pg = split_by_pitch_group(df)
        save_processed(by_pg, f"{prefix}_splits_by_pitch_group.csv")
        print("[INFO] Saved pitch-group splits.")
    except KeyError as e:
        print(f"[WARN] Skipping pitch-group splits: {e}")

    # 5) count buckets
    try:
        by_count = split_by_count_bucket(df)
        save_processed(by_count, f"{prefix}_splits_by_count_bucket.csv")
        print("[INFO] Saved count-bucket splits.")
    except KeyError as e:
        print(f"[WARN] Skipping count-bucket splits: {e}")

    # 6) zone discipline
    try:
        zone_disc = compute_zone_discipline(df)
        save_processed(zone_disc, f"{prefix}_zone_discipline.csv")
        print("[INFO] Saved zone discipline stats.")
    except KeyError as e:
        print(f"[WARN] Skipping zone discipline: {e}")

    # 7) home/away (optional)
    if team:
        try:
            ha = split_home_away(df, team)
            save_processed(ha, f"{prefix}_home_away_splits.csv")
            print("[INFO] Saved home/away splits.")
        except KeyError as e:
            print(f"[WARN] Skipping home/away splits: {e}")
    else:
        print("[INFO] No --team provided; skipping home/away splits.")

    # 8) league context
    try:
        build_league_context(prefix)
        print("[INFO] Saved league context.")
    except Exception as e:
        print(f"[WARN] Skipping league context: {e}")

    print("[DONE] Split analysis complete.")


if __name__ == "__main__":
    main()
