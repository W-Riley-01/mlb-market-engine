import streamlit as st
import numpy as np
from datetime import datetime

# Ensure the large pitch_matrix.parquet (~120MB) is present locally before
# the resolver tries to load it. On Streamlit Cloud's first cold start this
# downloads from GitHub Releases; on subsequent runs it's a no-op. Wrapped
# in try/except so a network failure produces a friendly error rather than
# a Python stack trace.
try:
    from bootstrap_data import ensure_data_files
    ensure_data_files()
except Exception as e:
    st.error(f"Could not bootstrap data files: {e}")
    st.stop()

from resolver import MatchupResolver
from engine_runner import run_slate

# ==========================================
# UI CONFIG
# ==========================================
st.set_page_config(page_title="MLB Engine", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e14;
    color: #c5cdd9;
}

/* Header */
.syndicate-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #4a9eff;
    text-transform: uppercase;
    padding: 6px 0 2px 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 18px;
}
.syndicate-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #e8edf3;
    letter-spacing: -0.01em;
    margin: 0 0 2px 0;
}

/* Game Card */
.game-card {
    background: #0d1520;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 18px 22px 14px 22px;
    margin-bottom: 12px;
}
.game-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #4a9eff;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    text-transform: uppercase;
}

/* Live scoreboard (shown when game is in-progress or final) */
.live-scoreboard {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #0a141e;
    border: 1px solid #1e2a3a;
    border-left: 3px solid #3dd68c;
    border-radius: 4px;
    padding: 8px 14px;
    margin: -8px 0 14px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}
.live-scoreboard.final { border-left-color: #4a6080; }
.live-scoreboard .score-block {
    display: flex;
    flex-direction: column;
    min-width: 60px;
}
.live-scoreboard .team-abbr {
    color: #4a6080;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.live-scoreboard .score-value {
    color: #e8edf3;
    font-size: 22px;
    font-weight: 600;
    line-height: 1.1;
}
.live-scoreboard .score-value.winning { color: #3dd68c; }
.live-scoreboard .separator {
    color: #2a3a50;
    font-size: 16px;
    padding: 0 4px;
}
.live-scoreboard .status-block {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
.live-scoreboard .status-label {
    color: #3dd68c;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}
.live-scoreboard.final .status-label { color: #4a6080; }
.live-scoreboard .inning-info {
    color: #8090a8;
    font-size: 10px;
    margin-top: 2px;
}
.live-indicator {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #3dd68c;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* Section label */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    color: #4a6080;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #141e2b;
    padding-bottom: 5px;
}

/* Metric chips */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 4px; }
.metric-chip {
    background: #111a27;
    border: 1px solid #1e2a3a;
    border-radius: 4px;
    padding: 8px 14px;
    text-align: center;
    min-width: 110px;
}
.metric-chip .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    color: #4a6080;
    text-transform: uppercase;
    display: block;
    margin-bottom: 3px;
}
.metric-chip .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #e8edf3;
}
.metric-chip .value.good { color: #3dd68c; }
.metric-chip .value.warn { color: #f0b429; }
.metric-chip .value.bad  { color: #7a8a9e; }

/* HR-score chip: elevated treatment, matches the score tier */
.hr-chip {
    background: #111a27;
    border: 1px solid #1e2a3a;
    border-left: 3px solid #4a6080;
    border-radius: 4px;
    padding: 8px 16px;
    text-align: left;
    min-width: 180px;
}
.hr-chip.good { border-left-color: #3dd68c; }
.hr-chip.warn { border-left-color: #f0b429; }
.hr-chip.bad  { border-left-color: #4a6080; }
.hr-chip .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    color: #4a6080;
    text-transform: uppercase;
    display: block;
    margin-bottom: 3px;
}
.hr-chip .score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #e8edf3;
    line-height: 1.1;
}
.hr-chip.good .score { color: #3dd68c; }
.hr-chip.warn .score { color: #f0b429; }
.hr-chip.bad  .score { color: #8090a8; }
.hr-chip .tier {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    color: #8090a8;
    margin-top: 2px;
    text-transform: uppercase;
}

/* Pitcher block */
.pitcher-block {
    display: flex;
    gap: 10px;
    margin-bottom: 4px;
    flex-wrap: wrap;
}
.pitcher-card {
    background: #111a27;
    border: 1px solid #1e2a3a;
    border-radius: 4px;
    padding: 10px 16px;
    flex: 1;
    min-width: 200px;
}
.pitcher-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #c5cdd9;
    margin-bottom: 6px;
}
.pitcher-side {
    font-size: 9px;
    letter-spacing: 0.1em;
    color: #4a6080;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.k-row { display: flex; gap: 8px; flex-wrap: wrap; }
.k-chip {
    background: #0a0e14;
    border: 1px solid #1e2a3a;
    border-radius: 3px;
    padding: 4px 10px;
    text-align: center;
}
.k-chip .klabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    color: #4a6080;
    display: block;
    letter-spacing: 0.1em;
}
.k-chip .kval {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #4a9eff;
}

/* Lineup table */
.lineup-wrap { overflow-x: auto; }
table.lineup-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
}
table.lineup-table th {
    background: #0a0e14;
    color: #4a6080;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid #1e2a3a;
    white-space: nowrap;
}
table.lineup-table th.left { text-align: left; }
table.lineup-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #0f1822;
    text-align: right;
    color: #c5cdd9;
    white-space: nowrap;
}
table.lineup-table td.left { text-align: left; color: #e8edf3; font-weight: 600; }
table.lineup-table td.order { color: #4a6080; text-align: center; }
table.lineup-table tr:hover td { background: #111a27; }

/* Color coding for probabilities */
.p-high  { color: #3dd68c !important; font-weight: 600; }
.p-mid   { color: #f0b429 !important; }
.p-low   { color: #607080 !important; }

/* Team label pill */
.team-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 6px;
}
.team-pill.away { background: #1a2535; color: #4a9eff; border: 1px solid #2a3f5f; }
.team-pill.home { background: #1a2a1e; color: #3dd68c; border: 1px solid #2a4a30; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
<div class="syndicate-header">Quantitative Engine &nbsp;·&nbsp; {datetime.today().strftime('%A, %B %d %Y')}</div>
<div class="syndicate-title">MLB Market Engine</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# HELPERS
# ==========================================
def pct(val: float) -> str:
    return f"{val * 100:.1f}%"

# Per-prop color thresholds calibrated to realistic MLB outcomes.
# Each prop has its own scale because a 20% HR probability is elite
# while a 20% "1+ Hits" probability means the hitter is awful.
PROP_THRESHOLDS = {
    "1+ Hits": {"high": 0.70, "mid": 0.60},
    "2+ TB":   {"high": 0.45, "mid": 0.35},
    "1+ HR":   {"high": 0.18, "mid": 0.12},
    "1+ RBI":  {"high": 0.35, "mid": 0.25},
    "1+ SB":   {"high": 0.15, "mid": 0.08},
}

# Pitcher K thresholds: 3+K is a gimme, 7+K is elite.
# Each threshold gets its own calibration.
K_PROP_THRESHOLDS = {
    "3+K": {"high": 0.70, "mid": 0.55},
    "4+K": {"high": 0.60, "mid": 0.45},
    "5+K": {"high": 0.50, "mid": 0.35},
    "6+K": {"high": 0.35, "mid": 0.22},
    "7+K": {"high": 0.22, "mid": 0.12},
}

# Game market thresholds (moneyline, F5, NRFI) centered around 50% coinflip.
GAME_THRESHOLDS = {
    "ML":   {"high": 0.58, "mid": 0.50},
    "F5":   {"high": 0.55, "mid": 0.48},
    "NRFI": {"high": 0.58, "mid": 0.50},
}

def color_class(val: float, thresholds: dict = None) -> str:
    """Returns the CSS class for a probability given a threshold dict.
    Falls back to the generic 60/42 scale if no thresholds provided."""
    if thresholds is None:
        thresholds = {"high": 0.60, "mid": 0.42}
    if val >= thresholds["high"]: return "p-high"
    if val >= thresholds["mid"]:  return "p-mid"
    return "p-low"

def color_for_prop(prop_name: str, val: float) -> str:
    return color_class(val, PROP_THRESHOLDS.get(prop_name))

def color_for_k(threshold_label: str, val: float) -> str:
    return color_class(val, K_PROP_THRESHOLDS.get(threshold_label))

def color_for_game(market: str, val: float) -> str:
    return color_class(val, GAME_THRESHOLDS.get(market))

def color_for_hr_score(score: int) -> str:
    """Maps the HR-conditions score to chip color classes. Each tier
    represents ~10 ft of carry difference from neutral.
      55+  : HR-friendly (green)
      45-54: neutral   (yellow)
      <45  : suppressed (muted)"""
    if score >= 55: return "good"
    if score >= 45: return "warn"
    return "bad"

def k_threshold_probs(k_dist: list, thresholds: list) -> dict:
    """Given a list of K totals across simulations, return over-X probabilities."""
    arr = np.array(k_dist)
    return {f"{t}+K": np.mean(arr >= t) for t in thresholds}

def build_lineup_table(team_name: str, lineup_details: list, player_props: dict, side: str) -> str:
    pill = f'<span class="team-pill {side}">{team_name}</span>'
    rows = ""
    prop_keys = ['1+ Hits', '2+ TB', '1+ HR', '1+ RBI', '1+ SB']

    for i, player in enumerate(lineup_details):
        name  = player['name']
        props = player_props.get(name, {})
        order_cell = f'<td class="order">{i+1}</td>'
        name_cell  = f'<td class="left">{name}</td>'
        prop_cells = ""
        for pk in prop_keys:
            v = props.get(pk, 0.0)
            cc = color_for_prop(pk, v)
            prop_cells += f'<td class="{cc}">{pct(v)}</td>'
        rows += f"<tr>{order_cell}{name_cell}{prop_cells}</tr>"

    header_props = "".join(f'<th>{k}</th>' for k in prop_keys)
    table = f"""
    {pill}
    <div class="lineup-wrap">
    <table class="lineup-table">
      <thead><tr>
        <th style="text-align:center">#</th>
        <th class="left">Player</th>
        {header_props}
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """
    return table

def render_live_scoreboard(game: dict) -> str:
    """
    Renders a small scoreboard strip under the game title for in-progress
    or final games. Returns empty string for scheduled games (nothing to show).
    """
    status = game.get('status', '')
    away_score = game.get('away_score')
    home_score = game.get('home_score')

    # Nothing to show for pre-game states
    if away_score is None or home_score is None:
        return ''

    # Team abbreviations: strip down the full name to first word or short form.
    # Using full names would be too wide; abbr keeps the scoreboard compact.
    def short_name(full: str) -> str:
        # Map of common multi-word city names to their distinguishing suffix
        # For two-word teams like "Red Sox" we want "BOS" style abbreviations,
        # but we don't have that map here — use first 3 uppercase chars of
        # last word, which reads well for most teams ("Yankees" -> "YAN",
        # "Red Sox" -> "SOX", "Blue Jays" -> "JAY").
        parts = full.split()
        return parts[-1][:3].upper() if parts else full[:3].upper()

    away_abbr = short_name(game.get('away_team', ''))
    home_abbr = short_name(game.get('home_team', ''))

    # Highlight the currently-winning score in green. Tie = neither.
    away_class = 'score-value winning' if away_score > home_score else 'score-value'
    home_class = 'score-value winning' if home_score > away_score else 'score-value'

    # Status block — pulses with a dot for live games, muted for finals.
    is_final = status == 'Final'
    is_live  = status in ('In Progress', 'Manager challenge', 'Umpire Review')

    card_class = 'live-scoreboard final' if is_final else 'live-scoreboard'

    if is_final:
        status_html = '<span class="status-label">Final</span>'
    elif is_live:
        inning_ord  = game.get('inning_ordinal') or ''
        half        = game.get('inning_half') or ''
        outs        = game.get('outs')

        # Build the inning line: e.g. "Top 5th · 2 out"
        parts = []
        if half and inning_ord:
            parts.append(f"{half} {inning_ord}")
        elif inning_ord:
            parts.append(inning_ord)
        if outs is not None:
            parts.append(f"{outs} out{'s' if outs != 1 else ''}")
        inning_line = ' · '.join(parts) if parts else ''

        status_html = (
            f'<span class="status-label"><span class="live-indicator"></span>Live</span>'
            f'<span class="inning-info">{inning_line}</span>'
        )
    else:
        # Delayed, postponed, suspended, etc.
        status_html = f'<span class="status-label">{status}</span>'

    return (
        f'<div class="{card_class}">'
        f'<div class="score-block">'
        f'<span class="team-abbr">{away_abbr}</span>'
        f'<span class="{away_class}">{away_score}</span>'
        f'</div>'
        f'<span class="separator">-</span>'
        f'<div class="score-block">'
        f'<span class="team-abbr">{home_abbr}</span>'
        f'<span class="{home_class}">{home_score}</span>'
        f'</div>'
        f'<div class="status-block">{status_html}</div>'
        f'</div>'
    )


# ==========================================
# WEATHER / HR-CONDITIONS SECTION
# ==========================================
# Weather is now fetched inside engine_runner.run_slate() and arrives at
# the UI via the GameRun.weather field on the per-game callback. The
# render function below just consumes that dict — no caching layer needed
# here, and importing get_game_weather directly into app.py is no longer
# required.


def render_weather_section(weather: dict) -> str:
    """Renders the First Pitch Conditions panel — environmental readout
    plus the physics-based HR-conditions score. Input is the dict
    returned by weather.get_game_weather()."""
    park         = weather['park']
    first_pitch  = weather.get('first_pitch_local')
    is_dome      = weather['is_dome']

    # Section header carries park name and (if known) local first-pitch time.
    header_bits = [park]
    if first_pitch:
        header_bits.append(first_pitch)
    header = ' · '.join(header_bits)

    # Domes: skip the environmental chips — the HR score is fixed at 50 and
    # showing fake "72F indoors" values would be misleading.
    if is_dome:
        chips = (
            f'<div class="metric-chip">'
            f'<span class="label">Conditions</span>'
            f'<span class="value">Dome</span>'
            f'</div>'
            + _hr_score_chip(weather['score'], weather['label'],
                             weather.get('carry_delta_ft', 0))
        )
        return (
            f'<div class="section-label">First Pitch Conditions · {header}</div>'
            f'<div class="metric-row">{chips}</div>'
        )

    # Outdoor chips
    temp_str = f"{weather['temp_f']:.0f}°F"
    hum_str  = f"{weather['humidity_pct']}%"
    wind_str = f"{weather['wind_speed_mph']:.0f} mph {weather['wind_from_compass']}"
    press_str = f"{weather['pressure_inhg']:.2f} inHg"

    # CF-axis wind chip color: green if out, yellow if in, neutral if calm
    wind_out = weather['wind_out_mph']
    if   wind_out >= 3:  cf_class = 'good'
    elif wind_out <= -3: cf_class = 'warn'
    else:                cf_class = ''

    # Carry delta — the most physically meaningful single number. Each foot
    # of carry change maps directly to score points.
    carry_ft = weather['carry_delta_ft']
    if   carry_ft >= 5:  carry_class = 'good'
    elif carry_ft <= -5: carry_class = 'warn'
    else:                carry_class = ''
    carry_str = f"{carry_ft:+.0f} ft"

    chips = (
        f'<div class="metric-chip">'
        f'<span class="label">Temp</span>'
        f'<span class="value">{temp_str}</span>'
        f'</div>'
        f'<div class="metric-chip">'
        f'<span class="label">Humidity</span>'
        f'<span class="value">{hum_str}</span>'
        f'</div>'
        f'<div class="metric-chip">'
        f'<span class="label">Pressure</span>'
        f'<span class="value">{press_str}</span>'
        f'</div>'
        f'<div class="metric-chip">'
        f'<span class="label">Wind</span>'
        f'<span class="value">{wind_str}</span>'
        f'</div>'
        f'<div class="metric-chip">'
        f'<span class="label">CF Axis</span>'
        f'<span class="value {cf_class}">{weather["wind_label"]}</span>'
        f'</div>'
        f'<div class="metric-chip">'
        f'<span class="label">Carry Δ</span>'
        f'<span class="value {carry_class}">{carry_str}</span>'
        f'</div>'
        + _hr_score_chip(weather['score'], weather['label'], carry_ft)
    )

    return (
        f'<div class="section-label">First Pitch Conditions · {header}</div>'
        f'<div class="metric-row">{chips}</div>'
    )


def _hr_score_chip(score: int, label: str, carry_ft: float = 0) -> str:
    """The headline HR-conditions chip. The tier color (left bar + score
    color) reflects the score, and the subtitle shows the tier name plus
    the carry delta in feet — which is the physically meaningful number."""
    tier = color_for_hr_score(score)
    # Don't show carry delta for domes (it's always 0 and would look odd)
    if abs(carry_ft) < 0.5:
        subtitle = label
    else:
        subtitle = f'{label} · {carry_ft:+.0f} ft carry'
    return (
        f'<div class="hr-chip {tier}">'
        f'<span class="label">HR Conditions</span>'
        f'<span class="score">{score}<span style="font-size:13px;color:#4a6080;">/100</span></span>'
        f'<div class="tier">{subtitle}</div>'
        f'</div>'
    )


def render_pitcher_section(
    away_name: str, home_name: str,
    away_k_dist: list, home_k_dist: list,
    away_median_k: float, home_median_k: float,
    away_first_inn: dict = None, home_first_inn: dict = None,
):
    thresholds = [3, 4, 5, 6, 7]

    def get_k_probs(k_dist, median_k):
        if k_dist:
            arr = np.array(k_dist)
            return {f"{t}+K": float(np.mean(arr >= t)) for t in thresholds}
        import math
        lam = max(median_k, 0.1)
        result = {}
        for t in thresholds:
            cdf = sum((math.exp(-lam) * lam**k) / math.factorial(k) for k in range(t))
            result[f"{t}+K"] = max(1.0 - cdf, 0.0)
        return result

    def _first_inn_chips_html(fi: dict) -> str:
        """
        Renders a compact 'First Inning' row: ERA | NRFI record | Opp OBP.
        Returns empty string if the pitcher's 1st-inning sample is too thin
        (get_first_inning_stats returns None in that case).
        """
        if not fi:
            return ''

        # ERA chip — muted unless we have a sample worth coloring.
        if fi.get('era_1st') is None:
            era_display, era_cc = '—', ''
        else:
            era = fi['era_1st']
            era_display = f"{era:.2f}"
            # Color calibrated to 1st-inning ERA ranges (roughly: elite <2.5,
            # average ~4.5, blow-up prone >6). These are tighter than season
            # ERA because 1st-inning runs are inherently lumpier.
            if   era <= 2.50: era_cc = 'good'
            elif era <= 4.50: era_cc = ''
            elif era <  6.00: era_cc = 'warn'
            else:             era_cc = 'warn'

        # NRFI record chip — "18-4" format, colored by rate.
        nrfi_starts = fi.get('nrfi_starts', 0)
        yrfi_starts = fi.get('yrfi_starts', 0)
        nrfi_rate   = fi.get('nrfi_rate', 0.0)
        record_str  = f"{nrfi_starts}-{yrfi_starts}"
        if   nrfi_rate >= 0.70: rec_cc = 'good'
        elif nrfi_rate >= 0.55: rec_cc = ''
        else:                   rec_cc = 'warn'

        # Opp OBP in the 1st — smaller secondary signal, just a number.
        obp = fi.get('opp_obp_1st')
        if obp is None:
            obp_display = '—'
        else:
            # .xxx convention (drop leading zero) — baseball-standard display.
            obp_display = f"{obp:.3f}".lstrip('0') if obp < 1 else f"{obp:.3f}"

        n = fi.get('games_started', 0)

        chips = (
            f'<div class="k-chip"><span class="klabel">1st ERA</span>'
            f'<span class="kval {era_cc}">{era_display}</span></div>'
            f'<div class="k-chip"><span class="klabel">Scoreless 1st</span>'
            f'<span class="kval {rec_cc}">{record_str}</span></div>'
            f'<div class="k-chip"><span class="klabel">Opp OBP</span>'
            f'<span class="kval">{obp_display}</span></div>'
        )

        return (
            '<div style="margin-top:10px;padding-top:9px;border-top:1px dashed #1e2a3a;">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:8px;letter-spacing:0.14em;'
            'color:#4a6080;text-transform:uppercase;margin-bottom:6px;">'
            f'First Inning Profile <span style="color:#2a3a50;">· n={n}</span></div>'
            f'<div style="display:flex;gap:7px;flex-wrap:wrap;">{chips}</div>'
            '</div>'
        )

    def pitcher_card_html(name, side_label, k_dist, median_k, first_inn):
        probs = get_k_probs(k_dist, median_k)
        k_chips = ""
        for k_label, v in probs.items():
            cc = color_for_k(k_label, v)
            k_chips += (
                f'<div class="k-chip"><span class="klabel">{k_label}</span>'
                f'<span class="kval {cc}">{pct(v)}</span></div>'
            )
        first_inn_html = _first_inn_chips_html(first_inn)
        return (
            f'<div style="background:#111a27;border:1px solid #1e2a3a;border-radius:4px;padding:12px 16px;">' +
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:0.12em;' +
            f'color:#4a6080;text-transform:uppercase;margin-bottom:5px;">{side_label} Starter</div>' +
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:13px;font-weight:600;' +
            f'color:#c5cdd9;margin-bottom:10px;">{name}</div>' +
            f'<div style="display:flex;gap:7px;flex-wrap:wrap;">{k_chips}</div>' +
            f'<div style="margin-top:9px;font-family:IBM Plex Mono,monospace;font-size:10px;color:#4a6080;">' +
            f'Median Ks: <span style="color:#4a9eff;font-weight:600;">{median_k:.1f}</span></div>' +
            first_inn_html +
            f'</div>'
        )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(pitcher_card_html(away_name, "Away", away_k_dist, away_median_k, away_first_inn), unsafe_allow_html=True)
    with pcol2:
        st.markdown(pitcher_card_html(home_name, "Home", home_k_dist, home_median_k, home_first_inn), unsafe_allow_html=True)


# ==========================================
# ENGINE LOAD
# ==========================================
@st.cache_resource
def load_engine():
    return MatchupResolver(
        pqm_path='./data/pitch_matrix.parquet',
        cqm_path='./data/contact_matrix_env.parquet'
    )

resolver = load_engine()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### Engine Settings")
    iterations = st.slider("Simulations", min_value=1000, max_value=10000, value=5000, step=500)
    st.markdown("---")
    st.markdown("### Prop Color Guide")
    st.markdown("""
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #c5cdd9;">
<div style="margin-bottom:8px; color:#4a6080; font-size:9px; letter-spacing:0.1em;">STRONG / MONITOR THRESHOLDS</div>

<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>1+ Hits</span><span><span style="color:#3dd68c">70%</span> / <span style="color:#f0b429">60%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>2+ TB</span><span><span style="color:#3dd68c">45%</span> / <span style="color:#f0b429">35%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>1+ HR</span><span><span style="color:#3dd68c">18%</span> / <span style="color:#f0b429">12%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>1+ RBI</span><span><span style="color:#3dd68c">35%</span> / <span style="color:#f0b429">25%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>1+ SB</span><span><span style="color:#3dd68c">15%</span> / <span style="color:#f0b429">8%</span></span>
</div>

<div style="margin-top:12px; color:#4a6080; font-size:9px; letter-spacing:0.1em;">PITCHER K THRESHOLDS</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>3+K</span><span><span style="color:#3dd68c">70%</span> / <span style="color:#f0b429">55%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>5+K</span><span><span style="color:#3dd68c">50%</span> / <span style="color:#f0b429">35%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>7+K</span><span><span style="color:#3dd68c">22%</span> / <span style="color:#f0b429">12%</span></span>
</div>

<div style="margin-top:12px; color:#4a6080; font-size:9px; letter-spacing:0.1em;">GAME MARKETS</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>ML / NRFI</span><span><span style="color:#3dd68c">58%</span> / <span style="color:#f0b429">50%</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>F5</span><span><span style="color:#3dd68c">55%</span> / <span style="color:#f0b429">48%</span></span>
</div>

<div style="margin-top:12px; color:#4a6080; font-size:9px; letter-spacing:0.1em;">HR CONDITIONS SCORE</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>Very Friendly</span><span><span style="color:#3dd68c">65+</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>Friendly</span><span><span style="color:#3dd68c">55-64</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>Neutral</span><span><span style="color:#f0b429">45-54</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>Suppressed</span><span><span style="color:#8090a8">35-44</span></span>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 0;">
    <span>Very Suppressed</span><span><span style="color:#8090a8">&lt; 35</span></span>
</div>
<div style="margin-top:6px; color:#4a6080; font-size:8px; letter-spacing:0.06em;">
    1 point ≈ 1 ft of expected carry change. Calibrated to Alan Nathan's fly-ball physics.
</div>
</div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN RUN
# ==========================================
# The slate pipeline (fetch schedule → fetch rosters → simulate → apply
# weather → log to Supabase) lives in engine_runner.run_slate(). This file
# is responsible only for the UI: the button, a per-game render callback,
# and the summary footer. The same run_slate() is called headlessly by
# auto_log_predictions.py from a GitHub Actions cron — keeping that single
# source of truth means UI and cron can never drift.
col_run, col_date = st.columns([2, 3])
with col_run:
    run_button = st.button("Calculate Today's Edges", use_container_width=True)


def _render_game_card(run, i: int, total: int, progress_bar) -> None:
    """
    Callback invoked by run_slate() once per game on the slate.

    Renders one full game card. If the game was skipped (no lineups yet,
    sim error, etc.), renders a short status message in place of the
    markets/props sections so the user still sees the matchup, scoreboard,
    and weather.

    All sim + DB-write work has already been done by run_slate before this
    fires — this function is purely presentation. Errors during DB logging
    (which run_slate intentionally does NOT raise) surface as a soft toast
    so a transient Supabase hiccup never breaks the page render.
    """
    game = run.game

    # ---- Game card shell -------------------------------------------------
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="game-title">{game["matchup"]}</div>',
                unsafe_allow_html=True)

    # Live scoreboard for in-progress / final games (empty string for
    # scheduled, so it's safe to always call).
    scoreboard_html = render_live_scoreboard(game)
    if scoreboard_html:
        st.markdown(scoreboard_html, unsafe_allow_html=True)

    # ---- First Pitch Conditions -----------------------------------------
    # Rendered BEFORE any skip checks below so weather chips show even for
    # games whose lineups haven't dropped yet — useful for pre-game research.
    if run.weather:
        st.markdown(render_weather_section(run.weather), unsafe_allow_html=True)

    def _close_card_and_advance() -> None:
        """Close the .game-card div and tick the progress bar. Called from
        every return path so the DOM stays balanced and progress monotonic."""
        st.markdown('</div>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total)

    # ---- Skip cases ------------------------------------------------------
    # Each branch shows a short status line, closes the card, advances
    # progress, and returns BEFORE touching run.results (which is None
    # for any skipped game and would crash the markets render).
    if run.skipped_reason == "no_lineups":
        st.info("Lineups not yet posted — check back closer to first pitch.")
        _close_card_and_advance()
        return
    if run.skipped_reason == "sim_error":
        st.warning("Simulation failed for this game. Check the server log.")
        _close_card_and_advance()
        return
    if not run.results:
        # Catch-all for any future skip reason or unexpected None results
        # (e.g., 'already_logged' if someone flips the flag back on). Fail
        # gracefully rather than crashing the loop.
        st.info("No prediction available for this game.")
        _close_card_and_advance()
        return

    # ---- Surface DB log errors as a soft toast ---------------------------
    # run_slate keeps these as strings on GameRun.log_error rather than
    # raising — one bad write never aborts the rest of the slate.
    if run.log_error:
        st.toast(f"Prediction logging failed: {run.log_error}", icon="⚠️")

    # ---- Markets, pitcher props, lineups --------------------------------
    away, home, results = run.away, run.home, run.results

    st.markdown('<div class="section-label">Game Markets</div>',
                unsafe_allow_html=True)

    away_win   = results['away_win_prob']
    home_win   = 1.0 - away_win
    f5_away    = results['f5_away_win_prob']
    f5_home    = 1.0 - f5_away - results['f5_tie_prob']
    nrfi       = results['nrfi_prob']
    total_runs = results['median_total']

    def chip(label, value, market_key=None):
        display = pct(value)
        cc = color_for_game(market_key, value) if market_key else ""
        return (f'<div class="metric-chip">'
                f'<span class="label">{label}</span>'
                f'<span class="value {cc}">{display}</span>'
                f'</div>')

    chips_html = (
        '<div class="metric-row">'
        + chip(f"{away['team_name']} ML", away_win, "ML")
        + chip(f"{home['team_name']} ML", home_win, "ML")
        + chip("F5 Away", f5_away, "F5")
        + chip("F5 Home", f5_home, "F5")
        + chip("F5 Tie", results['f5_tie_prob'])
        + chip("NRFI", nrfi, "NRFI")
        + f'<div class="metric-chip"><span class="label">Median Total</span>'
          f'<span class="value">{total_runs}</span></div>'
        + '</div>'
    )
    st.markdown(chips_html, unsafe_allow_html=True)

    # ---- Pitcher Props ---------------------------------------------------
    st.markdown('<div class="section-label">Starting Pitcher Props</div>',
                unsafe_allow_html=True)
    render_pitcher_section(
        away_name=away['starter_name'],
        home_name=home['starter_name'],
        away_k_dist=results.get('away_k_dist', []),
        home_k_dist=results.get('home_k_dist', []),
        away_median_k=results['away_pitcher_median_k'],
        home_median_k=results['home_pitcher_median_k'],
        away_first_inn=results.get('away_starter_first_inn'),
        home_first_inn=results.get('home_starter_first_inn'),
    )

    # ---- Full Lineup Props ----------------------------------------------
    st.markdown('<div class="section-label">Full Lineup Player Props</div>',
                unsafe_allow_html=True)
    lineup_col1, lineup_col2 = st.columns(2)
    with lineup_col1:
        st.markdown(
            build_lineup_table(away['team_name'], away['lineup_details'],
                               results['player_props'], side='away'),
            unsafe_allow_html=True,
        )
    with lineup_col2:
        st.markdown(
            build_lineup_table(home['team_name'], home['lineup_details'],
                               results['player_props'], side='home'),
            unsafe_allow_html=True,
        )

    _close_card_and_advance()


if run_button:
    progress_bar = st.progress(0)

    # skip_already_logged=False so EVERY game re-renders on every click.
    # Per requirement: the user should never see "Already calculated and
    # logged earlier today" — they always get a fresh card. Idempotency
    # lives in prediction_logger.log_prediction (delete-then-insert keyed
    # on game_id + game_date + model_version) so re-running the slate
    # doesn't pile up duplicate rows in Supabase. Last-write-wins
    # semantics: the most recent prediction (closest to first pitch,
    # tightest weather forecast, confirmed lineups) wins.
    #
    # Note: run_slate now processes every game on the slate regardless of
    # status — Scheduled, In Progress, Final, etc. all get cards rendered.
    # DB logging is gated inside run_slate to PRE_FIRST_PITCH_STATUSES
    # only, so In Progress / Final cards show fresh markets next to their
    # live scoreboard but don't overwrite morning predictions in the DB.
    summary = run_slate(
        resolver=resolver,
        iterations=iterations,
        skip_already_logged=False,
        on_game_complete=lambda run, i, total: _render_game_card(
            run, i, total, progress_bar
        ),
    )

    if summary.total_games == 0:
        st.warning("No games found on today's slate.")
    else:
        # Slate-level result line — tells the user what just happened
        # without making them count cards.
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace; font-size:11px; "
            "color:#4a6080; margin-top:14px;'>"
            f"DONE — simulated {summary.simulated} · "
            f"logged {summary.logged} · "
            f"awaiting lineups {summary.skipped_no_lineups} · "
            f"log failures {summary.log_failures}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace; font-size:10px; "
            "color:#2a3a50; margin-top:20px; text-align:center;'>"
            f"SYNDICATE ENGINE · {datetime.today().strftime('%Y-%m-%d %H:%M')} · "
            f"{iterations} ITERATIONS</div>",
            unsafe_allow_html=True,
        )