"""
build_hitting_season.py

Builds a league-wide hitters table for a given season and saves:

    data/league/hitting_season_<year>.csv

Primary source:
    pybaseball.batting_stats_bref (Baseball-Reference)

Fallback:
    pybaseball.batting_stats (FanGraphs/MLB source)

Run from project root, e.g.:

    python src/build_hitting_season.py --year 2024
"""

from pathlib import Path
import argparse

import pandas as pd
from pybaseball import batting_stats_bref, batting_stats


ROOT = Path(__file__).resolve().parents[1]
DATA_LEAGUE = ROOT / "data" / "league"


def fetch_bref_batting(year: int) -> pd.DataFrame:
    """Try to fetch hitting stats from Baseball-Reference via pybaseball."""
    print(f"[hitting_season] Trying batting_stats_bref for {year}...")
    raw = batting_stats_bref(year)
    print(f"[hitting_season] batting_stats_bref({year}) shape: {raw.shape}")
    return raw


def fetch_fg_batting(year: int) -> pd.DataFrame:
    """Fallback: try pybaseball.batting_stats (FanGraphs/MLB source)."""
    print(f"[hitting_season] Trying fallback batting_stats for {year}...")
    raw = batting_stats(year, year)
    print(f"[hitting_season] batting_stats({year}, {year}) shape: {raw.shape}")
    return raw


def build_hitting_season(year: int) -> Path | None:
    """
    Pull league-wide hitting stats for a season and save a cleaned CSV.

    Returns:
        Path to the written CSV, or None if no data could be retrieved.
    """
    # ---- 1) Try Baseball-Reference via batting_stats_bref ----
    try:
        raw = fetch_bref_batting(year)
        if raw is None or raw.empty:
            raise ValueError("batting_stats_bref returned empty DataFrame")
        source = "bref"
    except Exception as e:
        print(f"[hitting_season] batting_stats_bref failed for {year}: {e}")
        raw = None
        source = None

    # ---- 2) Fallback to batting_stats if needed ----
    if raw is None or raw.empty:
        try:
            raw = fetch_fg_batting(year)
            if raw is None or raw.empty:
                raise ValueError("batting_stats returned empty DataFrame")
            source = "fg"
        except Exception as e:
            print(f"[hitting_season] batting_stats fallback failed for {year}: {e}")
            print("[hitting_season] No usable league data. Aborting.")
            return None

    df = raw.copy()

    print(f"[hitting_season] Using source: {source}, shape: {df.shape}")

    # ---- 3) Standardize column names depending on source ----
    if source == "bref":
        # BRef style columns
        col_map = {
            "Name": "name",
            "Tm": "team",
            "Age": "age",
            "G": "G",
            "PA": "PA",
            "AB": "AB",
            "R": "R",
            "H": "H",
            "2B": "2B",
            "3B": "3B",
            "HR": "HR",
            "RBI": "RBI",
            "SB": "SB",
            "CS": "CS",
            "BB": "BB",
            "SO": "SO",
            "BA": "AVG",      # BRef uses 'BA'
            "OBP": "OBP",
            "SLG": "SLG",
            "OPS": "OPS",
            "ISO": "ISO",
            "BABIP": "BABIP",
            "BB%": "BB_pct",
            "SO%": "K_pct",
            "wOBA": "wOBA",
            "wRC+": "wRC_plus",
        }
    else:
        # Generic/FanGraphs style columns – this may vary by pybaseball version
        col_map = {
            "Name": "name",
            "Team": "team",
            "G": "G",
            "PA": "PA",
            "AB": "AB",
            "R": "R",
            "H": "H",
            "2B": "2B",
            "3B": "3B",
            "HR": "HR",
            "RBI": "RBI",
            "BB": "BB",
            "SO": "SO",
            "SB": "SB",
            "CS": "CS",
            "AVG": "AVG",
            "OBP": "OBP",
            "SLG": "SLG",
            "OPS": "OPS",
            "ISO": "ISO",
            "BABIP": "BABIP",
            "BB%": "BB_pct",
            "K%": "K_pct",
            "wOBA": "wOBA",
            "wRC+": "wRC_plus",
        }

    # Keep only existing columns
    available = {src: dst for src, dst in col_map.items() if src in df.columns}
    if not available:
        print("[hitting_season] No expected stat columns found in source. Aborting.")
        return None

    df = df[list(available.keys())].rename(columns=available)

    # ---- 4) Add year, sort, save ----
    df.insert(0, "year", year)
    if "PA" in df.columns:
        df = df.sort_values("PA", ascending=False).reset_index(drop=True)

    DATA_LEAGUE.mkdir(parents=True, exist_ok=True)
    out_path = DATA_LEAGUE / f"hitting_season_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"[hitting_season] Saved to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build league-wide hitting season table.")
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year to pull (e.g. 2024). Default: 2025",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_hitting_season(args.year)


if __name__ == "__main__":
    main()
