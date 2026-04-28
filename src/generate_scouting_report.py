"""
generate_scouting_report.py

Phase 3: Turn Statcast + derived summary files into a
front-office style scouting report for ANY MLB hitter.

Inputs (from data/processed/):
    <prefix>_offense_summary.csv
    <prefix>_splits_by_pitch_type.csv
    <prefix>_splits_vs_hand.csv
    <prefix>_zone_discipline.csv
    <prefix>_statcast_clean.csv   (for player_name only)

Output:
    reports/<prefix>_scouting_report.md

Usage:
    python generate_scouting_report.py --prefix vinny
    python generate_scouting_report.py --prefix trout
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd

from utils import ROOT, DATA_PROCESSED


# -----------------------
# GRADING / BENCHMARKS
# -----------------------

def grade_woba(woba: float | None) -> Optional[str]:
    if woba is None or pd.isna(woba):
        return None
    if woba >= 0.370:
        return "Elite"
    elif woba >= 0.340:
        return "Great"
    elif woba >= 0.320:
        return "Average"
    elif woba >= 0.300:
        return "Below Avg"
    else:
        return "Poor"


def grade_iso(iso: float | None) -> Optional[str]:
    if iso is None or pd.isna(iso):
        return None
    if iso >= 0.230:
        return "Elite"
    elif iso >= 0.190:
        return "Great"
    elif iso >= 0.160:
        return "Average"
    elif iso >= 0.120:
        return "Below Avg"
    else:
        return "Poor"


def grade_k_pct(k_pct: float | None) -> Optional[str]:
    if k_pct is None or pd.isna(k_pct):
        return None
    if k_pct <= 15:
        return "Elite"
    elif k_pct <= 20:
        return "Great"
    elif k_pct <= 23:
        return "Average"
    elif k_pct <= 28:
        return "Below Avg"
    else:
        return "Poor"


def grade_bb_pct(bb_pct: float | None) -> Optional[str]:
    if bb_pct is None or pd.isna(bb_pct):
        return None
    if bb_pct >= 12:
        return "Elite"
    elif bb_pct >= 9:
        return "Great"
    elif bb_pct >= 7:
        return "Average"
    elif bb_pct >= 5:
        return "Below Avg"
    else:
        return "Poor"


def grade_hardhit(hh: float | None) -> Optional[str]:
    if hh is None or pd.isna(hh):
        return None
    if hh >= 45:
        return "Elite"
    elif hh >= 40:
        return "Great"
    elif hh >= 35:
        return "Average"
    elif hh >= 30:
        return "Below Avg"
    else:
        return "Poor"


def trend_label(start: float | None,
                end: float | None,
                bigger_is_better: bool = True,
                tol: float = 0.02) -> str:
    """
    Simple trend label: Improved / Declined / Stable.

    tol = relative threshold (2% default) to ignore small noise.
    """
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return "Unknown"

    if start == 0:
        delta = end - start
        if bigger_is_better:
            return "Improved" if delta > 0 else "Declined" if delta < 0 else "Stable"
        else:
            return "Improved" if delta < 0 else "Declined" if delta > 0 else "Stable"

    rel_change = (end - start) / abs(start)

    if abs(rel_change) <= tol:
        return "Stable"

    if bigger_is_better:
        return "Improved" if rel_change > 0 else "Declined"
    else:
        return "Improved" if rel_change < 0 else "Declined"


# -----------------------
# UTIL HELPERS
# -----------------------

def safe_load_processed(name: str) -> Optional[pd.DataFrame]:
    path = DATA_PROCESSED / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_player_name(prefix: str) -> str:
    """
    Pull player_name from the statcast_clean file if possible,
    otherwise just capitalise the prefix.
    """
    clean_name = f"{prefix}_statcast_clean.csv"
    df = safe_load_processed(clean_name)
    if df is None or "player_name" not in df.columns:
        return prefix.capitalize()

    names = df["player_name"].dropna().value_counts()
    if names.empty:
        return prefix.capitalize()
    return names.index[0]


def fmt_3(x: float | None) -> str:
    """Format to .3f and strip leading zero: 0.319 -> .319"""
    if x is None or pd.isna(x):
        return "---"
    return f"{x:.3f}".lstrip("0")


def fmt_1(x: float | None, suffix: str = "") -> str:
    """Format to .1f and optionally add a suffix (e.g. '%')."""
    if x is None or pd.isna(x):
        return "---"
    s = f"{x:.1f}"
    return s + suffix


# -----------------------
# SCOUTING LOGIC
# -----------------------

def summarize_overall(prefix: str) -> Tuple[str, pd.DataFrame, Dict[str, float | str | None]]:
    """
    Pull multi-year offense summary and build:
      - Text block
      - Table of yearly stats
      - Dict of key aggregate metrics for downstream sections
    """
    summary_name = f"{prefix}_offense_summary.csv"
    df = safe_load_processed(summary_name)
    if df is None:
        return "No offense summary file found.\n", pd.DataFrame(), {}

    df = df.sort_values("game_year")
    df["game_year"] = df["game_year"].astype(int)
    years = list(df["game_year"].unique())
    first_year, last_year = years[0], years[-1]

    # Make sure expected columns exist
    cols_needed = [
        "game_year", "PA", "AVG", "OBP", "SLG", "OPS",
        "ISO", "BB%", "K%", "HardHit%", "wOBA"
    ]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = pd.NA

    total_pa = df["PA"].sum()

    def pa_weighted(col: str) -> float | None:
        if total_pa == 0 or df[col].isna().all():
            return None
        return (df[col] * df["PA"]).sum() / total_pa

    avg = pa_weighted("AVG")
    obp = pa_weighted("OBP")
    slg = pa_weighted("SLG")
    ops = pa_weighted("OPS")
    iso = pa_weighted("ISO")
    bb_pct = pa_weighted("BB%")
    k_pct = pa_weighted("K%")
    hh_pct = pa_weighted("HardHit%")
    woba = pa_weighted("wOBA")

    start = df.iloc[0]
    end = df.iloc[-1]

    trend_iso = trend_label(start["ISO"], end["ISO"], bigger_is_better=True)
    trend_bb = trend_label(start["BB%"], end["BB%"], bigger_is_better=True, tol=0.05)
    trend_k = trend_label(start["K%"], end["K%"], bigger_is_better=False, tol=0.05)
    trend_hh = trend_label(start["HardHit%"], end["HardHit%"], bigger_is_better=True, tol=0.05)
    trend_woba = trend_label(start["wOBA"], end["wOBA"], bigger_is_better=True)

    g_woba = grade_woba(woba)
    g_iso = grade_iso(iso)
    g_k = grade_k_pct(k_pct)
    g_bb = grade_bb_pct(bb_pct)
    g_hh = grade_hardhit(hh_pct)

    text_lines: List[str] = []
    text_lines.append(f"**Overall Hitting Profile ({first_year}–{last_year})**")
    text_lines.append("")
    text_lines.append(
        f"- Multi-year slash line: **{fmt_3(avg)}/{fmt_3(obp)}/{fmt_3(slg)}** "
        f"(OPS **{fmt_3(ops)}**)"
    )
    text_lines.append(
        f"- Power: ISO **{fmt_3(iso)}** ({g_iso or 'Unknown'})"
    )
    text_lines.append(
        f"- Run value (wOBA): **{fmt_3(woba)}** "
        f"({g_woba or 'Unknown'})"
    )
    text_lines.append(
        f"- Plate discipline: BB% **{fmt_1(bb_pct)}** ({g_bb or 'Unknown'}), "
        f"K% **{fmt_1(k_pct)}** ({g_k or 'Unknown'})"
    )
    text_lines.append(
        f"- Contact quality (HardHit%): **{fmt_1(hh_pct, '%')}** "
        f"({g_hh or 'Unknown'})"
    )
    text_lines.append("")
    text_lines.append("**Trends over time:**")
    text_lines.append(f"- Power (ISO): **{trend_iso}** from {first_year} to {last_year}")
    text_lines.append(
        f"- Discipline (BB% / K%): BB% trend **{trend_bb}**, "
        f"K% trend **{trend_k}**"
    )
    text_lines.append(
        f"- Hard contact (HardHit%): **{trend_hh}**"
    )
    text_lines.append(
        f"- Overall run value (wOBA): **{trend_woba}**"
    )
    text_lines.append("")

    # Prepare a nicer year-by-year table (preformatted strings)
    mini = df[[
        "game_year", "PA", "AVG", "OBP", "SLG", "OPS",
        "ISO", "BB%", "K%", "HardHit%", "wOBA"
    ]].copy()

    mini["game_year"] = mini["game_year"].astype(int)
    mini["PA"] = mini["PA"].astype(int)

    mini["AVG"] = mini["AVG"].apply(fmt_3)
    mini["OBP"] = mini["OBP"].apply(fmt_3)
    mini["SLG"] = mini["SLG"].apply(fmt_3)
    mini["OPS"] = mini["OPS"].apply(fmt_3)
    mini["ISO"] = mini["ISO"].apply(fmt_3)
    mini["BB%"] = mini["BB%"].apply(lambda v: fmt_1(v))
    mini["K%"] = mini["K%"].apply(lambda v: fmt_1(v))
    mini["HardHit%"] = mini["HardHit%"].apply(lambda v: fmt_1(v, "%"))
    mini["wOBA"] = mini["wOBA"].apply(fmt_3)

    metrics = {
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": ops,
        "iso": iso,
        "bb_pct": bb_pct,
        "k_pct": k_pct,
        "hh_pct": hh_pct,
        "woba": woba,
        "grade_iso": g_iso,
        "grade_woba": g_woba,
        "grade_bb": g_bb,
        "grade_k": g_k,
        "grade_hh": g_hh,
        "trend_iso": trend_iso,
        "trend_woba": trend_woba,
        "trend_bb": trend_bb,
        "trend_k": trend_k,
        "trend_hh": trend_hh,
    }

    return "\n".join(text_lines), mini, metrics


def summarize_pitch_types(prefix: str, min_pa: int = 50) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Pitch-type profile for the MOST RECENT season.

    Returns:
      - text block
      - list of top pitch dicts (best damage)
      - list of bottom pitch dicts (weakest)
    """
    name = f"{prefix}_splits_by_pitch_type.csv"
    df = safe_load_processed(name)
    if df is None:
        return "_No pitch-type split file found._\n", [], []

    if "game_year" not in df.columns:
        return "_Pitch-type splits missing 'game_year' column._\n", [], []

    df["game_year"] = df["game_year"].astype(int)
    last_year = df["game_year"].max()
    df_year = df[df["game_year"] == last_year].copy()

    if df_year.empty:
        return "_No pitch-type data for the most recent season._\n", [], []

    df_year = df_year[df_year["PA"] >= min_pa].copy()
    if df_year.empty:
        return f"_Not enough PA by pitch type (min {min_pa}) in {last_year} to evaluate._\n", [], []

    # Use wOBA primarily; fallback to OPS if missing
    df_year["wOBA_filled"] = df_year["wOBA"].fillna(df_year["OPS"])
    df_year = df_year.sort_values("wOBA_filled", ascending=False)

    best = df_year.head(3).copy()
    worst = df_year.tail(3).copy()

    lines: List[str] = []
    lines.append(f"**Pitch-Type Performance ({last_year}, min PA = {min_pa})**")
    lines.append("")
    lines.append("**Best pitches (he does the most damage against):**")

    def pitch_label(row) -> str:
        if "pitch_name" in row and isinstance(row["pitch_name"], str):
            return row["pitch_name"]
        return str(row.get("pitch_type", "Unknown"))

    best_list: List[Dict] = []
    worst_list: List[Dict] = []

    for _, row in best.iterrows():
        pt = pitch_label(row)
        woba = row.get("wOBA")
        ops = row.get("OPS")
        pa = int(row["PA"])
        lines.append(
            f"- vs **{pt}** — wOBA **{fmt_3(woba)}**, "
            f"OPS **{fmt_3(ops)}** (PA {pa})"
        )
        best_list.append({
            "pitch": pt,
            "wOBA": woba,
            "OPS": ops,
            "PA": pa,
        })

    lines.append("")
    lines.append("**Weak spots (pitches that give him the most trouble):**")

    for _, row in worst.iterrows():
        pt = pitch_label(row)
        woba = row.get("wOBA")
        ops = row.get("OPS")
        pa = int(row["PA"])
        lines.append(
            f"- vs **{pt}** — wOBA **{fmt_3(woba)}**, "
            f"OPS **{fmt_3(ops)}** (PA {pa})"
        )
        worst_list.append({
            "pitch": pt,
            "wOBA": woba,
            "OPS": ops,
            "PA": pa,
        })

    lines.append("")
    return "\n".join(lines), best_list, worst_list


def summarize_vs_hand(prefix: str) -> Tuple[str, Dict[str, Dict]]:
    """
    Multi-year performance vs LHP/RHP.

    Returns:
      - text block
      - dict: { 'L': {...}, 'R': {...} } with wOBA/ISO/PA per side
    """
    name = f"{prefix}_splits_vs_hand.csv"
    df = safe_load_processed(name)
    if df is None:
        return "_No vs-hand split file found._\n", {}

    if "pitcher_hand" not in df.columns:
        return "_Vs-hand splits missing 'pitcher_hand' column._\n", {}

    df = df.sort_values("game_year")

    def agg_side(g: pd.DataFrame) -> Dict[str, float | int | None]:
        pa = g["PA"].sum()
        if pa == 0:
            return {"PA": 0, "wOBA": None, "ISO": None}
        woba = (g["wOBA"] * g["PA"]).sum() / pa if "wOBA" in g.columns else None
        iso = (g["ISO"] * g["PA"]).sum() / pa if "ISO" in g.columns else None
        return {"PA": int(pa), "wOBA": woba, "ISO": iso}

    agg_df = df.groupby("pitcher_hand").apply(agg_side).to_dict()

    lines: List[str] = []
    lines.append("**Vs Lefties vs Righties (multi-year)**")
    lines.append("")

    for hand in sorted(agg_df.keys()):
        side = agg_df[hand]
        woba = side["wOBA"]
        iso = side["ISO"]
        pa = side["PA"]
        label = "L-handed pitchers" if hand.upper().startswith("L") else "R-handed pitchers"
        lines.append(
            f"- vs **{label}** — wOBA **{fmt_3(woba)}**, "
            f"ISO **{fmt_3(iso)}** (PA {pa})"
        )

    lines.append("")

    return "\n".join(lines), agg_df


def summarize_zone_discipline(prefix: str) -> Tuple[str, Dict[str, float | None]]:
    """
    Multi-year zone discipline & contact profile.
    """
    name = f"{prefix}_zone_discipline.csv"
    df = safe_load_processed(name)
    if df is None:
        return "_No zone discipline file found._\n", {}

    def avg(col: str) -> float | None:
        if col not in df.columns or df[col].isna().all():
            return None
        return df[col].mean()

    z_swing = avg("Z_Swing%")
    o_swing = avg("O_Swing%")
    z_contact = avg("Z_Contact%")
    o_contact = avg("O_Contact%")

    info = {
        "Z_Swing%": z_swing,
        "O_Swing%": o_swing,
        "Z_Contact%": z_contact,
        "O_Contact%": o_contact,
    }

    lines: List[str] = []
    lines.append("**Zone Discipline (multi-year)**")
    lines.append("")
    lines.append(f"- In-zone swing rate (Z-Swing%): **{fmt_1(z_swing, '%')}**")
    lines.append(f"- Out-of-zone chase rate (O-Swing%): **{fmt_1(o_swing, '%')}**")
    lines.append(f"- In-zone contact (Z-Contact%): **{fmt_1(z_contact, '%')}**")
    lines.append(f"- Out-of-zone contact (O-Contact%): **{fmt_1(o_contact, '%')}**")

    # Short interpretation
    if o_swing is not None:
        if o_swing <= 25:
            lines.append("- Very selective; rarely chases out of the zone.")
        elif o_swing <= 32:
            lines.append("- Solid eye with a fairly normal chase rate.")
        else:
            lines.append("- Tends to chase more than average out of the zone.")
    if z_contact is not None:
        if z_contact >= 88:
            lines.append("- In the zone, makes excellent contact when he swings.")
        elif z_contact >= 82:
            lines.append("- In the zone, contact rate is solid/normal.")
        else:
            lines.append("- Misses more than average even on pitches in the zone.")

    lines.append("")

    return "\n".join(lines), info

from utils import load_processed  # make sure this is there

def summarize_league_context(prefix: str) -> str:
    """
    Read {prefix}_league_context.csv and build a Development & League Context section.
    Focuses on the most recent season in the data.
    """
    try:
        df = load_processed(f"{prefix}_league_context.csv")
    except FileNotFoundError:
        return ""

    if df.empty:
        return ""

    # Use the most recent year
    latest_year = int(df["year"].max())
    sub = df[df["year"] == latest_year].copy()

    if sub.empty:
        return ""

    player_name = sub["player_name"].iloc[0]
    cohort_size = int(sub["cohort_size"].max())
    min_pa = int(sub["min_pa"].max())

    # Grab metrics as a dict: metric -> (value, percentile)
    metric_map = {}
    for _, row in sub.iterrows():
        metric_map[row["metric"]] = (row["value"], row["percentile"])

    lines = []
    lines.append(f"## 8. Development & League Context ({latest_year})\n")
    lines.append(
        f"In {latest_year}, {player_name} is evaluated against MLB corner bats (1B/DH) "
        f"with at least {min_pa} plate appearances. The comparison cohort size is {cohort_size}."
    )
    lines.append("")

    def fmt_metric(name: str, label: str) -> str:
        if name not in metric_map:
            return ""
        v, pct = metric_map[name]
        return f"- {label}: **{v}** — approximately the **{pct}th percentile** among 1B/DH."

    iso_line = fmt_metric("ISO", "Power (ISO)")
    bb_line = fmt_metric("BB%", "Walk rate (BB%)")
    k_line = fmt_metric("K%", "Strikeout rate (K%)")
    woba_line = fmt_metric("wOBA", "Run value (wOBA)")
    ops_line = fmt_metric("OPS", "Overall production (OPS)")

    for line in [iso_line, bb_line, k_line, woba_line, ops_line]:
        if line:
            lines.append(line)

    lines.append("")
    lines.append(
        "Taken together, these percentiles describe where this bat stands relative to other everyday "
        "corner bats rather than the league as a whole."
    )
    lines.append("")

    return "\n".join(lines)


# -----------------------
# REPORT GENERATION
# -----------------------

def build_report(prefix: str) -> str:
    player_name = get_player_name(prefix)

    overall_text, yearly_table, metrics = summarize_overall(prefix)
    pitch_text, best_pitches, worst_pitches = summarize_pitch_types(prefix)
    hand_text, hand_data = summarize_vs_hand(prefix)
    zone_text, zone_info = summarize_zone_discipline(prefix)

    lines: List[str] = []

    # Header
    lines.append(f"# Hitter Scouting Report – {player_name}")
    lines.append("")
    lines.append(f"_Statcast-based profile for prefix `{prefix}`._")
    lines.append("")

    # 1. Overall
    lines.append("## 1. Overall Performance")
    lines.append("")
    lines.append(overall_text)
    if not yearly_table.empty:
        lines.append("Year-by-year key stats:")
        lines.append("")
        tbl = yearly_table.to_markdown(index=False)
        lines.append(tbl)
        lines.append("")

    # 2. Pitch-type
    lines.append("## 2. Pitch-Type Profile")
    lines.append("")
    lines.append(pitch_text)

    # 3. Platoon
    lines.append("## 3. Vs Lefties / Righties")
    lines.append("")
    lines.append(hand_text)

    # 4. Zone discipline
    lines.append("## 4. Zone Discipline / Approach")
    lines.append("")
    lines.append(zone_text)

    # 5. How This Hitter Wins
    lines.append("## 5. How This Hitter Wins")
    lines.append("")

    g_woba = metrics.get("grade_woba")
    g_iso = metrics.get("grade_iso")
    g_hh = metrics.get("grade_hh")
    woba = metrics.get("woba")
    iso = metrics.get("iso")
    hh_pct = metrics.get("hh_pct")

    # Best pitches in text form
    best_pitch_names = [bp["pitch"] for bp in best_pitches]
    worst_pitch_names = [wp["pitch"] for wp in worst_pitches]

    if g_woba or g_iso:
        lines.append(
            f"- Profiles as a **{(g_woba or g_iso or 'Unknown').lower()} overall bat**, "
            f"with ISO **{fmt_3(iso)}** and wOBA **{fmt_3(woba)}**."
        )
    if g_hh:
        lines.append(
            f"- Contact quality (HardHit%) sits at **{fmt_1(hh_pct, '%')}**, "
            f"graded as **{g_hh}**."
        )
    if best_pitch_names:
        lines.append(
            f"- Does his most damage against: **{', '.join(best_pitch_names)}**."
        )

    # Handedness advantage
    if hand_data:
        # crude: pick better wOBA side
        best_side_label = None
        best_side_val = None
        for hand, data in hand_data.items():
            w = data.get("wOBA")
            if w is None or pd.isna(w):
                continue
            if best_side_val is None or w > best_side_val:
                best_side_val = w
                best_side_label = hand

        if best_side_label is not None:
            side_text = "right-handed pitching" if best_side_label.upper().startswith("R") else "left-handed pitching"
            lines.append(f"- Most dangerous vs **{side_text}**.")
    lines.append("")

    # 6. How to Pitch Him
    lines.append("## 6. How to Pitch Him")
    lines.append("")

    if worst_pitch_names:
        lines.append(
            f"- Game plan should lean on his weakest pitch types: "
            f"**{', '.join(worst_pitch_names)}** where command allows."
        )
    if zone_info:
        o_swing = zone_info.get("O_Swing%")
        z_swing = zone_info.get("Z_Swing%")
        z_contact = zone_info.get("Z_Contact%")

        if o_swing is not None and o_swing > 32:
            lines.append(
                "- Attack just outside the zone with secondary stuff; "
                "he will expand more than league average."
            )
        elif o_swing is not None and o_swing <= 25:
            lines.append(
                "- Will not chase much; pitchers need to finish at-bats in the zone."
            )

        if z_contact is not None and z_contact >= 88:
            lines.append(
                "- Avoid living in the zone; he converts in-zone pitches into contact at an elite clip."
            )
        elif z_contact is not None and z_contact < 82:
            lines.append(
                "- Can be beaten in the zone with velocity and well-located fastballs."
            )

        if z_swing is not None and z_swing >= 65:
            lines.append(
                "- Aggressive when pitches enter the zone; early-count execution matters."
            )

    if not worst_pitch_names and not zone_info:
        lines.append("- Insufficient split/discipline data to prescribe a clear attack plan.")

    lines.append("")
    # 7. TL;DR (kept as quick usage notes, not cutesy)
    lines.append("## 7. TL;DR Usage Notes")
    lines.append("")
    lines.append(
        "- Use overall wOBA/ISO grades to gauge **run production level**.\n"
        "- Use year-to-year trends to see if the bat is **improving, declining, or stable**.\n"
        "- Use pitch-type profile to understand **what he punishes vs what exposes him**.\n"
        "- Use vs-hand splits to decide on **platoon usage or matchups**.\n"
        "- Use zone discipline to understand whether to **challenge in the zone** or "
        "**work edges and expand**."
    )
    lines.append("")

    league_section = summarize_league_context(prefix)
    if league_section:
        lines.append(league_section)
    return "\n".join(lines)


def save_report(prefix: str, content: str) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{prefix}_scouting_report.md"
    path.write_text(content, encoding="utf-8")
    return path


# -----------------------
# MAIN / CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a front-office style scouting report.")
    p.add_argument(
        "--prefix",
        default="vinny",
        help="Prefix used for processed files (default: vinny).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix

    report_md = build_report(prefix)
    path = save_report(prefix, report_md)

    print(f"[INFO] Scouting report written to: {path}")
    print()
    print("----- QUICK CONSOLE SUMMARY -----")
    print()
    # print header + overall section + TL;DR
    lines = report_md.splitlines()
    keep: List[str] = []
    in_overall = False
    in_tldr = False
    for line in lines:
        if line.startswith("## 1. Overall Performance"):
            in_overall = True
        if line.startswith("## 2. Pitch-Type Profile"):
            in_overall = False
        if line.startswith("## 7. TL;DR Usage Notes"):
            in_tldr = True
        if line.startswith("# Hitter Scouting Report"):
            keep.append(line)
        if in_overall or in_tldr:
            keep.append(line)
    print("\n".join(keep))


if __name__ == "__main__":
    main()
