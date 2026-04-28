"""
prediction_logger.py
--------------------
Persists every Monte Carlo prediction the MLB engine generates to Supabase
Postgres so we can later compare predictions to actual outcomes, measure
calibration, and apply correction models (isotonic regression in Phase 4).

One log_prediction() call writes:
    - 1 row in game_predictions   (game markets + pitcher Ks + weather audit)
    - N rows in player_predictions (one per batter in each lineup)

Designed to never crash the UI — the caller in app.py wraps this in try/except
so a transient DB hiccup degrades to a soft toast warning instead of a stack
trace. A single SQL transaction means partial writes are impossible.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import streamlit as st
from sqlalchemy import text

# Bump this when the simulator or resolver changes in a way that meaningfully
# shifts predictions. Lets calibration analysis segment results by version so
# we don't mix pre- and post-change predictions in the same reliability curve.
MODEL_VERSION = "v1"

# K thresholds we report on. Must match what's rendered in app.py so the
# logged probability == the displayed probability (no drift between UI & DB).
K_THRESHOLDS = (3, 4, 5, 6, 7)


# ---------------------------------------------------------------------------
#  Connection (Streamlit-cached so we don't rebuild on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_conn():
    """
    Returns the Streamlit SQL connection configured under
    [connections.predictions_db] in .streamlit/secrets.toml.
    See secrets.toml.example for the exact format.
    """
    return st.connection("predictions_db", type="sql")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _k_threshold_probs(k_dist: list, median_k: float) -> dict[int, float]:
    """
    Mirror of the rendering logic in app.py: empirical CDF when we have a
    full K distribution, Poisson fallback on the median otherwise. We log
    exactly what the user sees on screen.
    """
    if k_dist:
        arr = np.asarray(k_dist)
        return {t: float(np.mean(arr >= t)) for t in K_THRESHOLDS}

    lam = max(float(median_k or 0.0), 0.1)
    out = {}
    for t in K_THRESHOLDS:
        cdf = sum((math.exp(-lam) * lam**k) / math.factorial(k) for k in range(t))
        out[t] = max(1.0 - cdf, 0.0)
    return out


def _player_id(player: dict) -> int | None:
    """
    Be defensive — different roster scrapers stash the player ID under
    different keys. Try the common ones, return None if absent.
    """
    for k in ("id", "player_id", "mlb_id", "personId"):
        v = player.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _parse_game_date(game_datetime: str | None, fallback: str) -> date:
    """
    MLB Stats API returns ISO 8601 like '2025-04-15T23:05:00Z'. We just want
    the date. Falls back to as_of_date, then today's date, before giving up.
    """
    if game_datetime:
        try:
            cleaned = game_datetime.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).date()
        except (ValueError, AttributeError):
            pass
    try:
        return date.fromisoformat(fallback)
    except (ValueError, AttributeError):
        return date.today()


def _native(v: Any) -> Any:
    """Convert numpy scalars to native Python so SQL parameter binding is clean."""
    if v is None:
        return None
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def log_prediction(
    *,
    game: dict,
    away: dict,
    home: dict,
    results: dict,
    iterations: int,
    as_of_date: str,
    weather: dict | None = None,
) -> int:
    """
    Persist one full game prediction.

    All parameters mirror variables already in scope at the call site in
    app.py — no extra plumbing required. Keyword-only to prevent argument-
    order bugs when the call site evolves.

    Returns
    -------
    int
        The id of the inserted game_predictions row. Useful for testing and
        for chaining outcome ingestion later.
    """
    away_k_probs = _k_threshold_probs(
        results.get("away_k_dist", []),
        results.get("away_pitcher_median_k", 0),
    )
    home_k_probs = _k_threshold_probs(
        results.get("home_k_dist", []),
        results.get("home_pitcher_median_k", 0),
    )

    away_win = float(results.get("away_win_prob", 0.0))
    f5_away  = float(results.get("f5_away_win_prob", 0.0))
    f5_tie   = float(results.get("f5_tie_prob", 0.0))

    game_row = {
        "game_id":           str(game.get("game_id", "")),
        "game_date":         _parse_game_date(game.get("game_datetime"), as_of_date),
        "away_team":         away.get("team_name"),
        "home_team":         home.get("team_name"),
        "away_starter_id":   _native(away.get("starter_id")),
        "away_starter_name": away.get("starter_name"),
        "home_starter_id":   _native(home.get("starter_id")),
        "home_starter_name": home.get("starter_name"),

        "iterations":        int(iterations),
        "as_of_date":        as_of_date,
        "model_version":     MODEL_VERSION,

        # weather snapshot (None for missing or dome where appropriate) ------
        "weather_park":           _native((weather or {}).get("park")),
        "weather_is_dome":        _native((weather or {}).get("is_dome")),
        "weather_temp_f":         _native((weather or {}).get("temp_f")),
        "weather_wind_mph":       _native((weather or {}).get("wind_speed_mph")),
        "weather_carry_delta_ft": _native((weather or {}).get("carry_delta_ft")),
        "weather_hr_score":       _native((weather or {}).get("score")),

        # game markets ------------------------------------------------------
        "away_win_prob":  away_win,
        "home_win_prob":  1.0 - away_win,
        "f5_away_prob":   f5_away,
        "f5_home_prob":   max(1.0 - f5_away - f5_tie, 0.0),
        "f5_tie_prob":    f5_tie,
        "nrfi_prob":      float(results.get("nrfi_prob", 0.0)),
        "median_total":   float(results.get("median_total", 0.0)),

        # pitcher Ks --------------------------------------------------------
        "away_pitcher_median_k": float(results.get("away_pitcher_median_k", 0.0)),
        "away_p_3k": away_k_probs[3],
        "away_p_4k": away_k_probs[4],
        "away_p_5k": away_k_probs[5],
        "away_p_6k": away_k_probs[6],
        "away_p_7k": away_k_probs[7],

        "home_pitcher_median_k": float(results.get("home_pitcher_median_k", 0.0)),
        "home_p_3k": home_k_probs[3],
        "home_p_4k": home_k_probs[4],
        "home_p_5k": home_k_probs[5],
        "home_p_6k": home_k_probs[6],
        "home_p_7k": home_k_probs[7],
    }

    conn = _get_conn()
    with conn.session as s:
        # ---- Insert game-level row, capture the new id --------------------
        result = s.execute(
            text("""
                INSERT INTO game_predictions (
                    game_id, game_date, away_team, home_team,
                    away_starter_id, away_starter_name,
                    home_starter_id, home_starter_name,
                    iterations, as_of_date, model_version,
                    weather_park, weather_is_dome, weather_temp_f,
                    weather_wind_mph, weather_carry_delta_ft, weather_hr_score,
                    away_win_prob, home_win_prob,
                    f5_away_prob, f5_home_prob, f5_tie_prob,
                    nrfi_prob, median_total,
                    away_pitcher_median_k,
                    away_p_3k, away_p_4k, away_p_5k, away_p_6k, away_p_7k,
                    home_pitcher_median_k,
                    home_p_3k, home_p_4k, home_p_5k, home_p_6k, home_p_7k
                ) VALUES (
                    :game_id, :game_date, :away_team, :home_team,
                    :away_starter_id, :away_starter_name,
                    :home_starter_id, :home_starter_name,
                    :iterations, :as_of_date, :model_version,
                    :weather_park, :weather_is_dome, :weather_temp_f,
                    :weather_wind_mph, :weather_carry_delta_ft, :weather_hr_score,
                    :away_win_prob, :home_win_prob,
                    :f5_away_prob, :f5_home_prob, :f5_tie_prob,
                    :nrfi_prob, :median_total,
                    :away_pitcher_median_k,
                    :away_p_3k, :away_p_4k, :away_p_5k, :away_p_6k, :away_p_7k,
                    :home_pitcher_median_k,
                    :home_p_3k, :home_p_4k, :home_p_5k, :home_p_6k, :home_p_7k
                )
                RETURNING id
            """),
            game_row,
        )
        game_pred_id = result.scalar_one()

        # ---- Build per-player rows from both lineups ----------------------
        player_props = results.get("player_props", {}) or {}
        player_rows = []
        for side, roster in (("away", away), ("home", home)):
            for order_idx, player in enumerate(roster.get("lineup_details", []), start=1):
                name = player.get("name")
                if not name:
                    continue
                props = player_props.get(name, {}) or {}
                player_rows.append({
                    "game_prediction_id": game_pred_id,
                    "game_id":            game_row["game_id"],
                    "game_date":          game_row["game_date"],
                    "player_id":          _player_id(player),
                    "player_name":        name,
                    "side":               side,
                    "batting_order":      order_idx,
                    "p_1h":   float(props.get("1+ Hits", 0.0)),
                    "p_2tb":  float(props.get("2+ TB",   0.0)),
                    "p_1hr":  float(props.get("1+ HR",   0.0)),
                    "p_1rbi": float(props.get("1+ RBI",  0.0)),
                    "p_1sb":  float(props.get("1+ SB",   0.0)),
                })

        if player_rows:
            s.execute(
                text("""
                    INSERT INTO player_predictions (
                        game_prediction_id, game_id, game_date,
                        player_id, player_name, side, batting_order,
                        p_1h, p_2tb, p_1hr, p_1rbi, p_1sb
                    ) VALUES (
                        :game_prediction_id, :game_id, :game_date,
                        :player_id, :player_name, :side, :batting_order,
                        :p_1h, :p_2tb, :p_1hr, :p_1rbi, :p_1sb
                    )
                """),
                player_rows,
            )

        s.commit()

    return game_pred_id