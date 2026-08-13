"""
calibration.py
--------------
Foundation for calibration analysis: builds the long-format comparison view
that everything downstream (Brier scores, reliability diagrams, audit-finding
validation, v1↔v2 comparisons) reads from.

One row in the comparison view = one
    (game_prediction_id, market_type, predicted_prob, actual_outcome)
tuple, plus enough context columns to slice by version, date, park, pitcher,
batting order, weather conditions, etc.

Scope of this module
--------------------
Build & return the long-format DataFrame. That's it. Metric computation,
reliability binning, and plotting live in the next module (planned:
calibration_metrics.py). Keeping the data view and the metric layer in
separate files makes both easier to test and re-use.

Design notes
------------
1. Thin SQL + Python melt, not a Supabase view.

   The unpivot is non-trivial: the totals_thresholds JSONB ladder has
   variable-length keys per row, K thresholds are spread across 10 columns
   per game, and player props are 5 columns per batter. Doing that in SQL
   would mean a multi-page query with LATERAL JOINs and a Supabase migration
   to register the view; doing it in pandas takes 30 lines, is unit-testable
   without a DB roundtrip, and lets us add a new market_type by appending a
   chunk to a list. The SQL stays simple — one join per side.

2. We reuse prediction_logger._get_conn() rather than rebuild it.

   The dual-mode (Streamlit context vs. headless SQLAlchemy) connection
   pattern is what makes both app.py and the GitHub Actions cron work off
   the same writer. We use the same reader plumbing here so calibration
   notebooks and ad-hoc scripts both pick up DATABASE_URL the way
   record_outcomes.py does.

3. status='final' is the only resolved-outcome filter.

   Games with status in ('pending', 'postponed', 'suspended', 'cancelled')
   are excluded entirely — their predictions never had a chance to resolve
   against reality and counting them as misses would be wrong.

4. Pitcher K predictions are voided when the listed starter did not pitch.

   A scratched starter's K threshold probabilities were predicted but never
   resolved. The away_starter_pitched / home_starter_pitched booleans in
   game_outcomes are exactly the void flag we need — drop the row, do NOT
   treat as zero.

5. Player prop outcomes are pre-binarized in player_outcomes.

   hit_1h, hit_2tb, hit_1hr, hit_1rbi, hit_1sb are already booleans matching
   the p_1h ... p_1sb prediction columns. We join and rename — no
   "actual_hits >= 1" arithmetic in this module.

6. Don't pool model versions.

   Every public function in this module takes a model_version argument and
   defaults to a single version. Mixing v1 and v2 in one calibration curve
   is the mistake the handoff explicitly calls out.

Markets covered
---------------
Game-level (one row per game per market):
    away_ml, home_ml
    f5_away, f5_home, f5_tie
    nrfi
    run_line_home_-1.5, run_line_away_-1.5
    over_total_X  for each threshold in the JSONB ladder
    pitcher_3k_away ... pitcher_7k_away, same for home

Player-prop (one row per batter per market):
    player_1h, player_2tb, player_1hr, player_1rbi, player_1sb
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
from sqlalchemy import text

# Reuse the dual-mode connection from prediction_logger. _get_conn is
# module-private by convention but is the documented integration point —
# both the Streamlit reader (app.py) and the headless writer go through it,
# so calibration code does too. Don't reimplement.
from prediction_logger import _get_conn


# ---------------------------------------------------------------------------
#  DATABASE_URL bootstrap (headless / notebook / ad-hoc script convenience)
# ---------------------------------------------------------------------------
def _load_db_url_from_streamlit_secrets() -> str | None:
    """
    Read the predictions DB URL from .streamlit/secrets.toml so headless
    callers (CLI, notebooks, ad-hoc scripts) don't have to redundantly set
    DATABASE_URL in their shell when the value already lives in the
    Streamlit config the writer and app use.

    Looks in:
      ./.streamlit/secrets.toml      (project-local — primary)
      ~/.streamlit/secrets.toml      (user-global — fallback)

    Returns the URL string, or None if no usable config is found. Silent on
    I/O or TOML parse errors — falling through to the existing
    "DATABASE_URL not set" error in _get_conn() is the better failure mode
    than crashing on a malformed file before the user sees what's wrong.
    """
    try:
        import tomllib
        from pathlib import Path
    except ImportError:
        return None

    candidates = [
        Path(".streamlit/secrets.toml"),
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                secrets = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        conn = (secrets.get("connections") or {}).get("predictions_db") or {}
        url = conn.get("url")
        if url:
            return str(url)
        # Streamlit also supports assembling a URL from individual fields.
        # Build a SQLAlchemy URL the same way st.connection would internally
        # so we cover both common secrets.toml styles.
        dialect = conn.get("dialect")
        host = conn.get("host")
        database = conn.get("database")
        username = conn.get("username")
        if dialect and host and database and username:
            port = conn.get("port")
            password = conn.get("password") or ""
            port_str = f":{port}" if port else ""
            return f"{dialect}://{username}:{password}@{host}{port_str}/{database}"
    return None


# Bootstrap DATABASE_URL for headless use only. No-op in Streamlit (the
# st.connection path in _get_conn never touches the env var) and no-op in
# cron (DATABASE_URL is already set there as a GitHub Actions secret).
# Idempotent: only fires when DATABASE_URL is unset.
if not os.environ.get("DATABASE_URL"):
    _bootstrapped = _load_db_url_from_streamlit_secrets()
    if _bootstrapped:
        os.environ["DATABASE_URL"] = _bootstrapped


# K thresholds the engine logs. Kept in sync with prediction_logger.K_THRESHOLDS
# manually rather than re-imported — keeps this module's dependency surface
# minimal and makes the dependency direction one-way (calibration → logger,
# not both ways).
K_THRESHOLDS = (3, 4, 5, 6, 7)

# Player-prop column → (market_type, outcome_column) mapping. Used by the
# player-prop melt. Outcome columns are pre-binarized in player_outcomes.
PLAYER_PROP_MAP = {
    "p_1h":   ("player_1h",   "hit_1h"),
    "p_2tb":  ("player_2tb",  "hit_2tb"),
    "p_1hr":  ("player_1hr",  "hit_1hr"),
    "p_1rbi": ("player_1rbi", "hit_1rbi"),
    "p_1sb":  ("player_1sb",  "hit_1sb"),
}


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
def _date_filter_clause(
    start_date: date | str | None,
    end_date: date | str | None,
) -> tuple[str, dict]:
    """Build the WHERE fragment + params dict for an optional date range."""
    clause = ""
    params: dict = {}
    if start_date is not None:
        clause += " AND gp.game_date >= :start_date"
        params["start_date"] = start_date
    if end_date is not None:
        clause += " AND gp.game_date <= :end_date"
        params["end_date"] = end_date
    return clause, params


def _empty_long() -> pd.DataFrame:
    """Shape-stable empty long-format DataFrame so callers can concat safely."""
    return pd.DataFrame(columns=[
        "game_prediction_id", "game_date", "predicted_at",
        "model_version", "iterations", "away_team", "home_team",
        "weather_park", "weather_label", "weather_hr_score",
        "market_type", "predicted_prob", "actual_outcome",
    ])


# ---------------------------------------------------------------------------
#  Data inventory — run this BEFORE any analysis
# ---------------------------------------------------------------------------
def data_inventory(model_version: str | None = None) -> pd.DataFrame:
    """
    Summary of how many predictions we have, broken down by model_version
    and iteration count. Always run this first.

    The v1↔v2 sample split AND the iter=2000 vs iter=5000 split inside each
    version are confounders we want on the record before drawing inferences
    from any Brier number. If iter=2000 rows are concentrated in v1, then
    a v2-better-than-v1 Brier improvement is partially MC-noise-reduction,
    not the Tier 1 model fix.

    Returns one row per (model_version, iterations) cell with:
      n_predictions       — total game_predictions rows
      n_resolved          — joined to a 'final' game_outcomes row
      n_pending_or_void   — joined to a non-'final' outcome (pp/sus/cancel)
      n_no_outcome_row    — no game_outcomes row at all (cron didn't get to it)
      first_date, last_date
      n_player_predictions
      n_player_resolved
      n_player_dnp        — predicted but did_not_play (lineup change)
    """
    where = ""
    params: dict = {}
    if model_version is not None:
        where = " WHERE gp.model_version = :mv"
        params["mv"] = model_version

    sql_games = f"""
        SELECT
            gp.model_version,
            gp.iterations,
            COUNT(*)                                             AS n_predictions,
            COUNT(*) FILTER (WHERE go.status = 'final')          AS n_resolved,
            COUNT(*) FILTER (WHERE go.status IS NOT NULL
                              AND go.status <> 'final')          AS n_pending_or_void,
            COUNT(*) FILTER (WHERE go.id IS NULL)                AS n_no_outcome_row,
            MIN(gp.game_date)                                    AS first_date,
            MAX(gp.game_date)                                    AS last_date
        FROM game_predictions gp
        LEFT JOIN game_outcomes go ON go.game_prediction_id = gp.id
        {where}
        GROUP BY gp.model_version, gp.iterations
        ORDER BY gp.model_version, gp.iterations
    """

    sql_players = f"""
        SELECT
            gp.model_version,
            gp.iterations,
            COUNT(pp.id)                                            AS n_player_predictions,
            COUNT(*) FILTER (WHERE po.status = 'final')             AS n_player_resolved,
            COUNT(*) FILTER (WHERE po.status = 'did_not_play')      AS n_player_dnp
        FROM game_predictions gp
        JOIN player_predictions pp ON pp.game_prediction_id = gp.id
        LEFT JOIN player_outcomes po ON po.player_prediction_id = pp.id
        {where}
        GROUP BY gp.model_version, gp.iterations
        ORDER BY gp.model_version, gp.iterations
    """

    conn = _get_conn()
    with conn.session as s:
        game_rows = [dict(r) for r in s.execute(text(sql_games), params).mappings().all()]
    with conn.session as s:
        player_rows = [dict(r) for r in s.execute(text(sql_players), params).mappings().all()]

    game_df = pd.DataFrame(game_rows)
    player_df = pd.DataFrame(player_rows)
    if game_df.empty:
        return pd.DataFrame()
    return game_df.merge(
        player_df, on=["model_version", "iterations"], how="left"
    )


# ---------------------------------------------------------------------------
#  Game-level markets → long format
# ---------------------------------------------------------------------------
def _fetch_game_wide(
    model_version: str,
    start_date: date | str | None,
    end_date: date | str | None,
) -> pd.DataFrame:
    """
    Pull final game predictions joined to outcomes as a wide DataFrame.
    One row per game, all probability and outcome columns side by side.

    The jsonb_array_length() check on away_k_dist/home_k_dist tells us
    whether the K threshold probabilities came from the empirical 5000-iter
    CDF (preferred) or the Poisson fallback (~22 rows in late April, plus
    all v1 rows logged before K-dist persistence was added). This matters
    for Finding #2 validation — Poisson over-predicts the tail relative to
    the real K distribution, so we'd want to filter those rows out before
    concluding the 3rd-time-through-order bias is the explanation.
    """
    date_clause, date_params = _date_filter_clause(start_date, end_date)
    sql = f"""
        SELECT
            gp.id                AS game_prediction_id,
            gp.game_id,
            gp.game_date,
            gp.predicted_at,
            gp.model_version,
            gp.iterations,
            gp.away_team,
            gp.home_team,
            gp.away_starter_id   AS away_pitcher_id,
            gp.away_starter_name AS away_pitcher_name,
            gp.home_starter_id   AS home_pitcher_id,
            gp.home_starter_name AS home_pitcher_name,
            gp.weather_park,
            gp.weather_label,
            gp.weather_hr_score,
            gp.weather_carry_delta_ft,
            -- Game markets
            gp.away_win_prob, gp.home_win_prob,
            gp.f5_away_prob, gp.f5_home_prob, gp.f5_tie_prob,
            gp.nrfi_prob,
            gp.total_mean, gp.total_std, gp.total_thresholds,
            gp.run_line_home_minus_1_5_prob,
            gp.run_line_away_minus_1_5_prob,
            -- Pitcher K threshold probs
            gp.away_p_3k, gp.away_p_4k, gp.away_p_5k, gp.away_p_6k, gp.away_p_7k,
            gp.home_p_3k, gp.home_p_4k, gp.home_p_5k, gp.home_p_6k, gp.home_p_7k,
            -- K-distribution availability (empirical CDF vs Poisson fallback)
            COALESCE(jsonb_array_length(gp.away_k_dist), 0) > 0 AS away_k_dist_available,
            COALESCE(jsonb_array_length(gp.home_k_dist), 0) > 0 AS home_k_dist_available,
            -- Outcomes
            go.away_score, go.home_score, go.total_runs, go.away_won,
            go.f5_away_score, go.f5_home_score, go.f5_away_won, go.f5_tied,
            go.nrfi_hit,
            go.away_starter_actual_k, go.away_starter_pitched,
            go.home_starter_actual_k, go.home_starter_pitched
        FROM game_predictions gp
        JOIN game_outcomes go ON go.game_prediction_id = gp.id
        WHERE go.status = 'final'
          AND gp.model_version = :mv
          {date_clause}
        ORDER BY gp.game_date, gp.game_id
    """
    params = {"mv": model_version, **date_params}
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(text(sql), params).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def _melt_game_markets(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Unpivot the wide game DataFrame into long format.

    Voiding rules:
      - Pitcher K rows where the starter did not pitch are dropped.
      - Run line -1.5 markets resolve to FALSE when the score diff is 0 or 1
        (i.e. the favorite won by 1, didn't cover). Modeled as "diff >= 2".
    """
    if wide.empty:
        return _empty_long()

    context_cols = [
        "game_prediction_id", "game_date", "predicted_at",
        "model_version", "iterations",
        "away_team", "home_team",
        "weather_park", "weather_label", "weather_hr_score",
    ]
    out_chunks: list[pd.DataFrame] = []

    # Cast booleans to nullable BooleanArray so negation and arithmetic with
    # NA propagate cleanly. After the status='final' filter these should
    # never be NA, but be defensive against the rare half-finalized row.
    away_won = wide["away_won"].astype("boolean")
    f5_away_won = wide["f5_away_won"].astype("boolean")
    f5_tied = wide["f5_tied"].astype("boolean")
    nrfi_hit = wide["nrfi_hit"].astype("boolean")

    # --- moneyline -----------------------------------------------------------
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="away_ml",
        predicted_prob=wide["away_win_prob"],
        actual_outcome=away_won.astype("Int64"),
    ))
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="home_ml",
        predicted_prob=wide["home_win_prob"],
        actual_outcome=(~away_won).astype("Int64"),
    ))

    # --- first-5 -------------------------------------------------------------
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="f5_away",
        predicted_prob=wide["f5_away_prob"],
        actual_outcome=(f5_away_won & ~f5_tied).astype("Int64"),
    ))
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="f5_home",
        predicted_prob=wide["f5_home_prob"],
        actual_outcome=(~f5_away_won & ~f5_tied).astype("Int64"),
    ))
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="f5_tie",
        predicted_prob=wide["f5_tie_prob"],
        actual_outcome=f5_tied.astype("Int64"),
    ))

    # --- NRFI ----------------------------------------------------------------
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="nrfi",
        predicted_prob=wide["nrfi_prob"],
        actual_outcome=nrfi_hit.astype("Int64"),
    ))

    # --- run lines (-1.5 each side) ------------------------------------------
    home_diff = wide["home_score"] - wide["away_score"]
    away_diff = wide["away_score"] - wide["home_score"]
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="run_line_home_-1.5",
        predicted_prob=wide["run_line_home_minus_1_5_prob"],
        actual_outcome=(home_diff >= 2).astype("Int64"),
    ))
    out_chunks.append(_make_chunk(
        wide, context_cols,
        market_type="run_line_away_-1.5",
        predicted_prob=wide["run_line_away_minus_1_5_prob"],
        actual_outcome=(away_diff >= 2).astype("Int64"),
    ))

    # --- totals ladder (one row per threshold per game) ----------------------
    out_chunks.append(_melt_totals_ladder(wide, context_cols))

    # --- pitcher K thresholds (one row per side per threshold) ---------------
    out_chunks.append(_melt_pitcher_k(wide, context_cols))

    return pd.concat(out_chunks, ignore_index=True)


def _make_chunk(
    wide: pd.DataFrame,
    context_cols: list[str],
    *,
    market_type: str,
    predicted_prob: pd.Series,
    actual_outcome: pd.Series,
) -> pd.DataFrame:
    """Build a long-format chunk for a single game-level market."""
    chunk = wide[context_cols].copy()
    chunk["market_type"]    = market_type
    chunk["predicted_prob"] = predicted_prob.astype(float).values
    chunk["actual_outcome"] = actual_outcome.values
    return chunk[~chunk["actual_outcome"].isna()].copy()


def _melt_totals_ladder(wide: pd.DataFrame, context_cols: list[str]) -> pd.DataFrame:
    """
    Explode total_thresholds JSONB into one row per (game, threshold).
    Iterates rows in Python — fine for ~hundreds-of-games scale; would
    vectorize if this ever ran on the full historical Statcast dataset.
    """
    rows = []
    for _, w in wide.iterrows():
        thresholds = w.get("total_thresholds") or {}
        if not isinstance(thresholds, dict):
            continue
        total_runs = w.get("total_runs")
        if total_runs is None:
            continue
        for k_str, prob in thresholds.items():
            try:
                threshold = float(k_str)
            except (TypeError, ValueError):
                continue
            row = {c: w[c] for c in context_cols}
            row.update({
                "market_type":     f"over_total_{k_str}",
                "predicted_prob":  float(prob),
                "actual_outcome":  int(float(total_runs) > threshold),
                "threshold_value": threshold,
            })
            rows.append(row)
    if not rows:
        return _empty_long()
    return pd.DataFrame(rows)


def _melt_pitcher_k(wide: pd.DataFrame, context_cols: list[str]) -> pd.DataFrame:
    """
    Pitcher K threshold markets, one row per (game, side, threshold).
    Voids rows where the starter did not pitch.

    Adds k_dist_available context flag so downstream analysis can filter to
    empirical-CDF-derived probabilities only (Finding #2 validation).
    """
    rows = []
    for side in ("away", "home"):
        actual_k_col   = f"{side}_starter_actual_k"
        pitched_col    = f"{side}_starter_pitched"
        pitcher_id_col = f"{side}_pitcher_id"
        pitcher_nm_col = f"{side}_pitcher_name"
        k_dist_col     = f"{side}_k_dist_available"

        side_df = wide[wide[pitched_col] == True].copy()  # noqa: E712
        if side_df.empty:
            continue

        for _, w in side_df.iterrows():
            k_actual = w.get(actual_k_col)
            if k_actual is None:
                continue
            for t in K_THRESHOLDS:
                prob_col = f"{side}_p_{t}k"
                row = {c: w[c] for c in context_cols}
                row.update({
                    "market_type":      f"pitcher_{t}k_{side}",
                    "predicted_prob":   float(w[prob_col]),
                    "actual_outcome":   int(int(k_actual) >= t),
                    "side":             side,
                    "pitcher_id":       w[pitcher_id_col],
                    "pitcher_name":     w[pitcher_nm_col],
                    "k_dist_available": bool(w[k_dist_col]),
                    "threshold_value":  float(t),
                })
                rows.append(row)
    if not rows:
        return _empty_long()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Player props → long format
# ---------------------------------------------------------------------------
def fetch_player_prop_long(
    model_version: str = "v2",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """
    Player-prop predictions joined to outcomes, returned in long format.
    One row per (player_prediction_id, market_type).

    Filter: po.status='final' only. Drops did_not_play (lineup scratches,
    pinch-hit-only appearances that never hit the predicted slot) and
    pending rows.
    """
    date_clause, date_params = _date_filter_clause(start_date, end_date)
    sql = f"""
        SELECT
            gp.id              AS game_prediction_id,
            pp.id              AS player_prediction_id,
            gp.game_date,
            gp.predicted_at,
            gp.model_version,
            gp.iterations,
            gp.away_team,
            gp.home_team,
            gp.weather_park,
            gp.weather_label,
            gp.weather_hr_score,
            pp.player_id,
            pp.player_name,
            pp.side,
            pp.batting_order,
            pp.p_1h, pp.p_2tb, pp.p_1hr, pp.p_1rbi, pp.p_1sb,
            po.hit_1h, po.hit_2tb, po.hit_1hr, po.hit_1rbi, po.hit_1sb,
            po.plate_appearances, po.at_bats
        FROM game_predictions gp
        JOIN player_predictions pp ON pp.game_prediction_id = gp.id
        JOIN player_outcomes    po ON po.player_prediction_id = pp.id
        WHERE po.status = 'final'
          AND gp.model_version = :mv
          {date_clause}
        ORDER BY gp.game_date, gp.id, pp.batting_order
    """
    params = {"mv": model_version, **date_params}
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(text(sql), params).mappings().all()
    wide = pd.DataFrame([dict(r) for r in rows])
    if wide.empty:
        return _empty_long()

    context_cols = [
        "game_prediction_id", "player_prediction_id",
        "game_date", "predicted_at",
        "model_version", "iterations",
        "away_team", "home_team",
        "weather_park", "weather_label", "weather_hr_score",
        "player_id", "player_name", "side", "batting_order",
        "plate_appearances", "at_bats",
    ]

    chunks: list[pd.DataFrame] = []
    for prob_col, (market_type, outcome_col) in PLAYER_PROP_MAP.items():
        chunk = wide[context_cols].copy()
        chunk["market_type"]    = market_type
        chunk["predicted_prob"] = wide[prob_col].astype(float).values
        # The outcome booleans should never be NA on status='final' rows but
        # guard anyway — drop NaN before casting to plain int.
        outcome = wide[outcome_col].astype("boolean")
        chunk["actual_outcome"] = outcome.astype("Int64").values
        chunks.append(chunk[~chunk["actual_outcome"].isna()].copy())
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
#  Orchestrator: full comparison view
# ---------------------------------------------------------------------------
def build_comparison_view(
    model_version: str = "v2",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """
    Return the unified long-format DataFrame: game-level + player-prop
    markets concatenated, with a uniform schema.

    Common columns (all rows):
      market_type, predicted_prob, actual_outcome,
      game_prediction_id, game_date, predicted_at,
      model_version, iterations,
      away_team, home_team,
      weather_park, weather_label, weather_hr_score

    Player-prop-only columns (NaN on game-level rows):
      player_prediction_id, player_id, player_name, side, batting_order,
      plate_appearances, at_bats

    K-market-only columns:
      side, pitcher_id, pitcher_name, k_dist_available

    Totals-only columns:
      threshold_value (also populated on K-market rows)

    Single-version by design. Call once with model_version='v1' and once
    with 'v2' if you want to compare. The metric layer (next module)
    enforces this discipline by requiring an explicit version on every call.
    """
    game_wide = _fetch_game_wide(model_version, start_date, end_date)
    game_long = _melt_game_markets(game_wide)
    player_long = fetch_player_prop_long(model_version, start_date, end_date)

    out = pd.concat([game_long, player_long], ignore_index=True)
    if out.empty:
        return out

    out["predicted_prob"] = out["predicted_prob"].astype(float)
    out["actual_outcome"] = out["actual_outcome"].astype(int)
    return out


# ---------------------------------------------------------------------------
#  CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with:  python calibration.py
    # Requires DATABASE_URL in env (same env var record_outcomes.py uses).
    print("=== Data inventory ===")
    inv = data_inventory()
    print(inv.to_string(index=False) if not inv.empty else "(no data)")
    print()

    for mv in ("v1", "v2"):
        print(f"=== Comparison view: {mv} ===")
        df = build_comparison_view(model_version=mv)
        if df.empty:
            print(f"(no resolved {mv} data)")
            continue
        print(f"Total rows: {len(df):,}")
        print()
        print("Rows per market_type:")
        counts = df["market_type"].value_counts().sort_index()
        print(counts.to_string())
        print()
        print("Predicted-prob range & actual base rate per market:")
        summary = (df.groupby("market_type")
                     .agg(n=("predicted_prob", "size"),
                          pred_mean=("predicted_prob", "mean"),
                          actual_rate=("actual_outcome", "mean"))
                     .round(4))
        print(summary.to_string())
        print()