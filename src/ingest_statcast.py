"""
ingest_statcast.py

Generic Statcast ingestion + hitter summary.

- Loads a raw Statcast CSV from data/raw/
- Cleans & flags pitch-level data
- Builds an advanced season-level offensive summary with grading
- Saves both to data/processed/

Defaults are set up for:
    raw file: vinny_statcast_2022_2025.csv
    prefix:   vinny

Use:
    python ingest_statcast.py
or:
    python ingest_statcast.py --raw trout_statcast_2018_2025.csv --prefix trout
"""

import argparse
from typing import Optional

import pandas as pd
from utils import load_raw, save_processed


# -----------------------
# GRADING HELPERS
# -----------------------

def grade_k_pct(k_pct: float | None) -> Optional[str]:
    """
    K% thresholds (lower is better):
      <=15   Elite
      15–18  Great
      18–23  Average
      23–28  Below Avg
      >28    Poor
    """
    if k_pct is None or pd.isna(k_pct):
        return None
    if k_pct <= 15:
        return "Elite"
    elif k_pct <= 18:
        return "Great"
    elif k_pct <= 23:
        return "Average"
    elif k_pct <= 28:
        return "Below Avg"
    else:
        return "Poor"


def grade_bb_pct(bb_pct: float | None) -> Optional[str]:
    """
    BB% thresholds (higher is better):
      >=12   Elite
      10–12  Great
      7–10   Average
      5–7    Below Avg
      <5     Poor
    """
    if bb_pct is None or pd.isna(bb_pct):
        return None
    if bb_pct >= 12:
        return "Elite"
    elif bb_pct >= 10:
        return "Great"
    elif bb_pct >= 7:
        return "Average"
    elif bb_pct >= 5:
        return "Below Avg"
    else:
        return "Poor"


def grade_iso(iso: float | None) -> Optional[str]:
    """
    ISO thresholds (power):
      >=.250 Elite
      .200–.250 Great
      .170–.200 Average
      .140–.170 Below Avg
      <.140     Poor
    """
    if iso is None or pd.isna(iso):
        return None
    if iso >= 0.250:
        return "Elite"
    elif iso >= 0.200:
        return "Great"
    elif iso >= 0.170:
        return "Average"
    elif iso >= 0.140:
        return "Below Avg"
    else:
        return "Poor"


def grade_hardhit_pct(hh_pct: float | None) -> Optional[str]:
    """
    HardHit% thresholds (EV >=95mph):
      >=45   Elite
      40–45  Great
      35–40  Average
      30–35  Below Avg
      <30    Poor
    """
    if hh_pct is None or pd.isna(hh_pct):
        return None
    if hh_pct >= 45:
        return "Elite"
    elif hh_pct >= 40:
        return "Great"
    elif hh_pct >= 35:
        return "Average"
    elif hh_pct >= 30:
        return "Below Avg"
    else:
        return "Poor"


def grade_woba(woba: float | None) -> Optional[str]:
    """
    wOBA thresholds:
      >=.380 Elite
      .350–.380 Great
      .320–.350 Average
      .300–.320 Below Avg
      <.300     Poor
    """
    if woba is None or pd.isna(woba):
        return None
    if woba >= 0.380:
        return "Elite"
    elif woba >= 0.350:
        return "Great"
    elif woba >= 0.320:
        return "Average"
    elif woba >= 0.300:
        return "Below Avg"
    else:
        return "Poor"


# -----------------------
# CLEANING / FLAGGING
# -----------------------

def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + flags for Statcast batter data."""
    df = df.copy()

    # Dates
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Year
    if "game_year" not in df.columns and "game_date" in df.columns:
        df["game_year"] = df["game_date"].dt.year

    # Event flags
    # Event flags
    events = df["events"].fillna("")

    # One row per plate appearance: Statcast only sets `events` on the final pitch.
    df["is_pa"] = events != ""

    # AB / hit / BB / SO flags at the PA level (final pitch only)
    df["is_ab"] = events.isin([
        "single", "double", "triple", "home_run",
        "field_out", "force_out", "grounded_into_double_play"
    ])
    df["is_hit"] = events.isin(["single", "double", "triple", "home_run"])

    # Treat walk + HBP + IBB as "walk" for rate purposes
    df["is_bb"] = events.isin(["walk", "hit_by_pitch", "intent_walk"])

    df["is_hr"] = events.eq("home_run")
    df["is_2b"] = events.eq("double")
    df["is_3b"] = events.eq("triple")
    df["is_so"] = events.eq("strikeout")

    return df


# -----------------------
# OFFENSIVE SUMMARY
# -----------------------

def build_offense_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-level hitter profile:

    Counting: PA, AB, H, 2B, 3B, HR, BB, SO
    Rate: AVG, OBP, SLG, OPS, ISO
    Plate discipline: BB%, K%, K-BB%
    Batted ball: GB%, LD%, FB%, PU%
    Contact quality: HardHit%, avg EV, avg LA, SweetSpot%
    Value: wOBA, xwOBA
    Grading: metric grades + overall_grade
    """
    has_bb_type = "bb_type" in df.columns
    has_launch = "launch_speed" in df.columns and "launch_angle" in df.columns
    has_woba = "woba_value" in df.columns and "woba_denom" in df.columns
    has_xwoba = "estimated_woba_using_speedangle" in df.columns

    rows = []

    for year, g in df.groupby("game_year", dropna=False):
        g = g.copy()

        # Plate appearances: final pitch only (events != "")
        if "is_pa" in g.columns:
            pa = int(g["is_pa"].sum())
        else:
            # Fallback: count non-empty events
            pa = int(g["events"].fillna("").ne("").sum())

        # Counting stats are already PA-level because flags only appear
        # on the final pitch of each PA.
        ab = int(g["is_ab"].sum())
        h = int(g["is_hit"].sum())
        doubles = int(g["is_2b"].sum())
        triples = int(g["is_3b"].sum())
        hr = int(g["is_hr"].sum())
        bb = int(g["is_bb"].sum())
        so = int(g["is_so"].sum())

        # Basic rate stats
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

        # Batted ball mix
        gb_pct = ld_pct = fb_pct = pu_pct = None
        if has_bb_type:
            gb = int((g["bb_type"] == "ground_ball").sum())
            ld = int((g["bb_type"] == "line_drive").sum())
            fb = int((g["bb_type"] == "fly_ball").sum())
            pu = int((g["bb_type"] == "popup").sum())
            bip = gb + ld + fb + pu
            if bip > 0:
                gb_pct = gb / bip * 100
                ld_pct = ld / bip * 100
                fb_pct = fb / bip * 100
                pu_pct = pu / bip * 100

        # Contact quality
        avg_ev = avg_la = hardhit_pct = sweetspot_pct = None
        if has_launch:
            batted = g[g["launch_speed"].notna()].copy()
            n_batted = len(batted)
            if n_batted > 0:
                avg_ev = batted["launch_speed"].mean()
                avg_la = batted["launch_angle"].mean()
                hardhit = (batted["launch_speed"] >= 95).sum()
                hardhit_pct = hardhit / n_batted * 100
                sweet = batted["launch_angle"].between(8, 32, inclusive="both").sum()
                sweetspot_pct = sweet / n_batted * 100

        # Value metrics
        woba = None
        if has_woba:
            woba_value_sum = g["woba_value"].sum()
            woba_denom_sum = g["woba_denom"].sum()
            if woba_denom_sum > 0:
                woba = woba_value_sum / woba_denom_sum

        xwoba = None
        if has_xwoba:
            xwoba = g["estimated_woba_using_speedangle"].mean()

        # Grades
        grade_k = grade_k_pct(k_pct)
        grade_bb = grade_bb_pct(bb_pct)
        grade_iso_val = grade_iso(iso)
        grade_hh = grade_hardhit_pct(hardhit_pct)
        grade_woba_val = grade_woba(woba)

        # Overall grade: lean on wOBA, fall back to ISO if wOBA missing
        overall_grade = grade_woba_val or grade_iso_val

        rows.append({
            "game_year": year,

            # counting
            "PA": pa,
            "AB": ab,
            "H": h,
            "2B": doubles,
            "3B": triples,
            "HR": hr,
            "BB": bb,
            "SO": so,

            # slash / power
            "AVG": avg,
            "OBP": obp,
            "SLG": slg,
            "OPS": ops,
            "ISO": iso,

            # plate discipline
            "BB%": bb_pct,
            "K%": k_pct,
            "K-BB%": k_minus_bb,

            # batted ball mix
            "GB%": gb_pct,
            "LD%": ld_pct,
            "FB%": fb_pct,
            "PU%": pu_pct,

            # contact quality
            "avg_ev": avg_ev,
            "avg_la": avg_la,
            "HardHit%": hardhit_pct,
            "SweetSpot%": sweetspot_pct,

            # value metrics
            "wOBA": woba,
            "xwOBA": xwoba,

            # grades
            "grade_K%": grade_k,
            "grade_BB%": grade_bb,
            "grade_ISO": grade_iso_val,
            "grade_HardHit%": grade_hh,
            "grade_wOBA": grade_woba_val,
            "overall_grade": overall_grade,
        })

    summary = pd.DataFrame(rows)

    # --------- ROUNDING / FORMATTING (still numeric) ---------
    # 3-decimal stats
    three_dec = ["AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "xwOBA"]
    for col in three_dec:
        if col in summary.columns:
            summary[col] = summary[col].round(3)

    # 1-decimal stats (percent-like & quality metrics)
    one_dec = [
        "BB%", "K%", "K-BB%",
        "GB%", "LD%", "FB%", "PU%",
        "avg_ev", "avg_la",
        "HardHit%", "SweetSpot%",
    ]
    for col in one_dec:
        if col in summary.columns:
            summary[col] = summary[col].round(1)

    return summary


# -----------------------
# MAIN / CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Statcast batter data and build hitter summary.")
    parser.add_argument(
        "--raw",
        default="vinny_statcast_2022_2025.csv",
        help="Raw Statcast CSV filename in data/raw/ (default: vinny_statcast_2022_2025.csv)",
    )
    parser.add_argument(
        "--prefix",
        default="vinny",
        help="Prefix for processed output files (default: vinny)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_filename = args.raw
    prefix = args.prefix

    # 1. Load raw Statcast from Savant export
    raw = load_raw(raw_filename)
    print(f"[INFO] Loaded raw Statcast '{raw_filename}': {raw.shape[0]} rows, {raw.shape[1]} columns")

    # 2. Clean & save full pitch-level table
    cleaned = clean_statcast(raw)
    clean_name = f"{prefix}_statcast_clean.csv"
    save_processed(cleaned, clean_name)
    print(f"[INFO] Saved cleaned → data/processed/{clean_name}")

    # 3. Build advanced year-over-year summary
    offense_summary = build_offense_summary(cleaned)
    summary_name = f"{prefix}_offense_summary.csv"
    save_processed(offense_summary, summary_name)
    print(f"[INFO] Saved advanced summary → data/processed/{summary_name}")

    # 4. Print compact view to console
    cols_to_show = [
        "game_year", "PA", "AVG", "OBP", "SLG", "OPS", "ISO",
        "BB%", "K%", "K-BB%",
        "HardHit%", "wOBA", "xwOBA",
        "overall_grade",
    ]
    print("\n--- Advanced Year-Over-Year Summary ---")
    print(offense_summary[cols_to_show])


if __name__ == "__main__":
    main()
