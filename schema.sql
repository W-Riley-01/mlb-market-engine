-- ============================================================================
--  MLB Market Engine — Prediction Logger Schema
-- ============================================================================
--  Run this once against your Supabase Postgres (SQL Editor or psql) to set
--  up the tables. Re-running is safe: every statement uses IF NOT EXISTS.
--
--  Two tables:
--    game_predictions   — one row per game/sim run (game markets + pitcher Ks)
--    player_predictions — one row per batter per game (full player props)
--
--  Outcomes will live in their own tables (added in Phase 2).
-- ============================================================================

CREATE TABLE IF NOT EXISTS game_predictions (
    id                   BIGSERIAL    PRIMARY KEY,
    prediction_timestamp TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- identity ---------------------------------------------------------------
    game_id              TEXT         NOT NULL,
    game_date            DATE         NOT NULL,
    away_team            TEXT         NOT NULL,
    home_team            TEXT         NOT NULL,

    -- starting pitchers (IDs are the join key for outcome resolution) -------
    away_starter_id      BIGINT,
    away_starter_name    TEXT,
    home_starter_id      BIGINT,
    home_starter_name    TEXT,

    -- engine inputs (audit trail) -------------------------------------------
    iterations           INTEGER      NOT NULL,
    as_of_date           DATE,
    model_version        TEXT         NOT NULL DEFAULT 'v1',

    -- weather snapshot (nullable: domes / missing data) ---------------------
    weather_park           TEXT,
    weather_is_dome        BOOLEAN,
    weather_temp_f         REAL,
    weather_wind_mph       REAL,
    weather_carry_delta_ft REAL,
    weather_hr_score       INTEGER,

    -- game markets ----------------------------------------------------------
    away_win_prob   REAL,
    home_win_prob   REAL,
    f5_away_prob    REAL,
    f5_home_prob    REAL,
    f5_tie_prob     REAL,
    nrfi_prob       REAL,
    median_total    REAL,

    -- pitcher K props (one set per starter) ---------------------------------
    away_pitcher_median_k REAL,
    away_p_3k REAL,
    away_p_4k REAL,
    away_p_5k REAL,
    away_p_6k REAL,
    away_p_7k REAL,

    home_pitcher_median_k REAL,
    home_p_3k REAL,
    home_p_4k REAL,
    home_p_5k REAL,
    home_p_6k REAL,
    home_p_7k REAL
);

CREATE INDEX IF NOT EXISTS idx_game_predictions_game_id   ON game_predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_game_predictions_game_date ON game_predictions(game_date);


CREATE TABLE IF NOT EXISTS player_predictions (
    id                   BIGSERIAL  PRIMARY KEY,
    game_prediction_id   BIGINT     NOT NULL
        REFERENCES game_predictions(id) ON DELETE CASCADE,

    -- denormalized for easier ad-hoc querying without a join ----------------
    game_id              TEXT       NOT NULL,
    game_date            DATE       NOT NULL,

    player_id            BIGINT,
    player_name          TEXT       NOT NULL,
    side                 TEXT       NOT NULL CHECK (side IN ('away','home')),
    batting_order        INTEGER,

    -- props -----------------------------------------------------------------
    p_1h    REAL,
    p_2tb   REAL,
    p_1hr   REAL,
    p_1rbi  REAL,
    p_1sb   REAL
);

CREATE INDEX IF NOT EXISTS idx_player_predictions_game_id   ON player_predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_player_predictions_player_id ON player_predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_player_predictions_game_date ON player_predictions(game_date);
