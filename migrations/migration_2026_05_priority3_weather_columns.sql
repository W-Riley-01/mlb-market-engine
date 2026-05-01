-- =============================================================================
-- migration_2026_05_priority3_weather_columns.sql
-- =============================================================================
-- Adds the weather and first-pitch fields the read-only viewer needs to
-- render full game cards. Before this migration, app.py reconstructs a
-- partial weather dict from six columns (park, is_dome, temp_f, wind_mph,
-- carry_delta_ft, hr_score) and the hotfix renders '—' for everything
-- else. After this migration the cron writer will start populating the
-- additional fields; once a fresh row exists, the reader can be updated
-- to pull them and the dashes will turn into real values.
--
-- Columns added:
--   weather_humidity_pct       INTEGER       — relative humidity, %
--   weather_pressure_inhg      REAL          — surface pressure, inches Hg
--   weather_wind_from_compass  TEXT          — cardinal direction wind is FROM
--                                              ('NW', 'SSE', etc.)
--   weather_wind_label         TEXT          — human-readable CF-axis
--                                              description ('Out to CF (6 mph)',
--                                              'Calm (CF axis)', 'Indoors')
--   weather_wind_out_mph       REAL          — signed CF-axis component:
--                                              positive = out toward CF,
--                                              negative = in from CF
--   weather_label              TEXT          — HR-conditions tier label
--                                              ('HR Friendly', 'Neutral', etc.)
--                                              — pairs with the existing
--                                              weather_hr_score INTEGER
--   game_datetime_utc          TIMESTAMPTZ   — first pitch in UTC. Stored
--                                              as UTC, converted to
--                                              America/Chicago at render time
--                                              by app.py's _to_central() helper.
--
-- All seven columns are nullable. Existing rows retain NULL until they
-- are next overwritten by the cron's idempotent upsert (which we DON'T
-- expect to happen for past dates — log_prediction's delete-then-insert
-- only fires for today's slate during the active prediction window).
-- The Priority-2 hotfix in render_weather_section already renders '—' for
-- any of these fields when missing, so the table is safe to migrate in
-- isolation: nothing breaks until the writer (Step 3b) starts populating
-- them, and nothing breaks if rollback ever became necessary.
--
-- Deployment sequence (the "writer first, then reader" pattern from
-- past lessons):
--
--   Step 3a — THIS FILE. Run in Supabase SQL editor. Schema-only.
--   Step 3b — Update prediction_logger.log_prediction to write the new
--             columns. Push to main. Wait for the next cron run.
--   Step 3b verify — Run the diagnostic query at the bottom of this file
--             on a freshly-written row to confirm population.
--   Step 3c — Update prediction_logger.fetch_predictions_for_date and
--             app.py's _render_game_card_from_db to consume the new
--             columns. Push to main.
--   Step 3d — Reload the live app, verify the dashes are gone.
--
-- The reader update (3c) is intentionally LAST so the viewer never tries
-- to SELECT a column that doesn't exist yet, even momentarily.
-- =============================================================================

ALTER TABLE game_predictions
    ADD COLUMN IF NOT EXISTS weather_humidity_pct       INTEGER,
    ADD COLUMN IF NOT EXISTS weather_pressure_inhg      REAL,
    ADD COLUMN IF NOT EXISTS weather_wind_from_compass  TEXT,
    ADD COLUMN IF NOT EXISTS weather_wind_label         TEXT,
    ADD COLUMN IF NOT EXISTS weather_wind_out_mph       REAL,
    ADD COLUMN IF NOT EXISTS weather_label              TEXT,
    ADD COLUMN IF NOT EXISTS game_datetime_utc          TIMESTAMPTZ;


-- =============================================================================
-- VERIFICATION (run after the ALTER above)
-- =============================================================================
-- Confirms every new column exists and has the expected type. Should return
-- exactly 7 rows. If the count is < 7, the ALTER didn't complete; if a type
-- doesn't match, drop and re-add the offending column.

SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'game_predictions'
  AND column_name IN (
        'weather_humidity_pct',
        'weather_pressure_inhg',
        'weather_wind_from_compass',
        'weather_wind_label',
        'weather_wind_out_mph',
        'weather_label',
        'game_datetime_utc'
      )
ORDER BY column_name;
-- Expected:
--   game_datetime_utc          | timestamp with time zone | YES
--   weather_humidity_pct       | integer                  | YES
--   weather_label              | text                     | YES
--   weather_pressure_inhg      | real                     | YES
--   weather_wind_from_compass  | text                     | YES
--   weather_wind_label         | text                     | YES
--   weather_wind_out_mph       | real                     | YES


-- =============================================================================
-- POST-WRITER-DEPLOY VERIFICATION  (run AFTER Step 3b cron completes)
-- =============================================================================
-- After Step 3b is deployed and one cron has run, this query confirms the
-- new columns are being populated on fresh rows. Filter to today's date
-- and check that the latest row has non-NULL values for the outdoor-weather
-- columns. Dome games legitimately have NULL pressure/humidity (the writer
-- only fills the outdoor fields when not is_dome), so look for an outdoor
-- park to verify.

SELECT
    id,
    game_id,
    away_team || ' @ ' || home_team    AS matchup,
    weather_park,
    weather_is_dome,
    game_datetime_utc,
    weather_humidity_pct,
    weather_pressure_inhg,
    weather_wind_from_compass,
    weather_wind_out_mph,
    weather_wind_label,
    weather_label,
    predicted_at
FROM game_predictions
WHERE game_date = CURRENT_DATE
  AND weather_is_dome = false
ORDER BY predicted_at DESC NULLS LAST
LIMIT 5;
-- Expected for outdoor games written by the new logger: every column
-- populated. Rows from before Step 3b deploy will still have NULLs in
-- the new columns — that's fine; only the FRESH ones need to populate.

-- =============================================================================
-- End of migration.
-- =============================================================================
