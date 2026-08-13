-- =============================================================================
-- migration_2026_05_cleanup_legacy_duplicates.sql
-- =============================================================================
-- One-time data cleanup. NOT a schema change — every existing column stays as-
-- is. Two issues are repaired here:
--
--   (A) Six legacy duplicate rows in game_predictions from 2026-04-28 evening
--       (game 825017 × 4, game 823310 × 2). Written before the upsert in
--       prediction_logger.log_prediction was deployed; they remain in the DB
--       because the upsert only collapses incoming writes, not historical
--       state. Each duplicate has its own child rows in player_predictions,
--       game_outcomes, and player_outcomes, so a calibration query joining on
--       game_prediction_id will count those games 2-4 times. We keep the
--       latest row per (game_id, game_date, model_version) — matching the
--       last-write-wins semantic the upsert would have produced — and delete
--       the rest.
--
--   (B) `predicted_at` was added to game_predictions in
--       migration_2025_05_phase2c_columns.sql. The migration's NOW() default
--       fired once at apply-time on every existing row, so all 19 rows that
--       predated the migration share an identical `predicted_at` timestamp
--       that has nothing to do with when the prediction was actually made.
--       For those rows, the original `prediction_timestamp` column (set on
--       INSERT) holds the correct value. We copy it over so `predicted_at`
--       means what its name implies on every row in the table.
--
-- Run order (each block is a separate execution in the Supabase SQL editor):
--   1. PRE-FLIGHT diagnostic — confirms expected counts. Run first; if
--      anything looks off, stop and ask.
--   2. CLEANUP TRANSACTION — wrapped in BEGIN/COMMIT. Includes a verification
--      SELECT before commit. Review the output, then COMMIT (or ROLLBACK if
--      it doesn't match).
--   3. PREDICTED_AT BACKFILL — separate transaction, same pattern.
--   4. POST-FLIGHT verification — confirms the table is in the expected
--      state after both cleanups.
--
-- Safety notes:
--   - Steps 2 and 3 are idempotent. Running them a second time finds no work
--     to do (zero rows match the deletion / update predicates).
--   - Supabase Free / Pro tiers retain a 7-day point-in-time recovery window
--     by default; if you want belt-and-suspenders, take a manual backup
--     (Project Settings → Database → Backups) before running.
-- =============================================================================


-- =============================================================================
-- 1. PRE-FLIGHT DIAGNOSTIC — run first, examine, then proceed.
-- =============================================================================
-- Expected output: exactly 2 rows.
--   game_id=823310  group_size=2  oldest_id=18  newest_id=21
--   game_id=825017  group_size=4  oldest_id=13  newest_id=20
-- If the count differs from 2, or the IDs don't match the audit, STOP.

SELECT
    game_id,
    game_date,
    model_version,
    COUNT(*)                                         AS group_size,
    MIN(id)                                          AS oldest_id,
    MAX(id)                                          AS newest_id,
    MIN(prediction_timestamp)                        AS first_written,
    MAX(prediction_timestamp)                        AS last_written
FROM game_predictions
GROUP BY game_id, game_date, model_version
HAVING COUNT(*) > 1
ORDER BY game_id;


-- Same query but listing every row in every duplicate group (for full
-- visibility into what will survive vs. delete). Survivors are flagged
-- by `keep = true`; rest are slated for deletion.

WITH ranked AS (
    SELECT
        id,
        game_id,
        game_date,
        model_version,
        prediction_timestamp,
        away_win_prob,
        iterations,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, game_date, model_version
            ORDER BY prediction_timestamp DESC
        ) AS rn
    FROM game_predictions
)
SELECT
    id,
    game_id,
    game_date,
    prediction_timestamp,
    iterations,
    away_win_prob,
    (rn = 1) AS keep
FROM ranked
WHERE game_id IN (
    SELECT game_id
    FROM game_predictions
    GROUP BY game_id, game_date, model_version
    HAVING COUNT(*) > 1
)
ORDER BY game_id, prediction_timestamp;


-- Cascade preview — counts the child rows that will be removed.
-- Expected: 4 game_predictions, 72 player_predictions, 4 game_outcomes,
--           72 player_outcomes. (4 × 18 batters per game on each side.)

WITH to_delete AS (
    SELECT id
    FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY game_id, game_date, model_version
                ORDER BY prediction_timestamp DESC
            ) AS rn
        FROM game_predictions
    ) t
    WHERE rn > 1
)
SELECT
    (SELECT COUNT(*) FROM to_delete)                                              AS game_predictions_to_delete,
    (SELECT COUNT(*) FROM player_predictions WHERE game_prediction_id IN (SELECT id FROM to_delete)) AS player_predictions_to_delete,
    (SELECT COUNT(*) FROM game_outcomes      WHERE game_prediction_id IN (SELECT id FROM to_delete)) AS game_outcomes_to_delete,
    (SELECT COUNT(*) FROM player_outcomes
        WHERE player_prediction_id IN (
            SELECT pp.id FROM player_predictions pp
            WHERE pp.game_prediction_id IN (SELECT id FROM to_delete)
        )
    )                                                                             AS player_outcomes_to_delete;


-- =============================================================================
-- 2. CLEANUP TRANSACTION — review the SELECT inside, then COMMIT (or ROLLBACK).
-- =============================================================================
-- Order of deletions matters so this works whether or not the outcomes side
-- has ON DELETE CASCADE on its FK to predictions:
--   1. player_outcomes  — by player_prediction_id, derived from the doomed
--                          game_prediction_ids
--   2. game_outcomes    — by game_prediction_id directly
--   3. player_predictions — defensive (cascades anyway via schema.sql FK)
--   4. game_predictions — the parent rows themselves
-- A temp table holds the doomed game_prediction_ids so every step references
-- the same set without re-evaluating the ROW_NUMBER() window.

BEGIN;

CREATE TEMP TABLE _doomed_predictions ON COMMIT DROP AS
SELECT id
FROM (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, game_date, model_version
            ORDER BY prediction_timestamp DESC
        ) AS rn
    FROM game_predictions
) t
WHERE rn > 1;

-- Sanity check — should be 4. If not, ROLLBACK.
SELECT COUNT(*) AS doomed_count FROM _doomed_predictions;

-- (1) player_outcomes
DELETE FROM player_outcomes
WHERE player_prediction_id IN (
    SELECT pp.id
    FROM player_predictions pp
    WHERE pp.game_prediction_id IN (SELECT id FROM _doomed_predictions)
);

-- (2) game_outcomes
DELETE FROM game_outcomes
WHERE game_prediction_id IN (SELECT id FROM _doomed_predictions);

-- (3) player_predictions  (would cascade, but explicit is safer)
DELETE FROM player_predictions
WHERE game_prediction_id IN (SELECT id FROM _doomed_predictions);

-- (4) game_predictions
DELETE FROM game_predictions
WHERE id IN (SELECT id FROM _doomed_predictions);

-- Verification BEFORE commit. Expected:
--   remaining_game_predictions = 38
--   remaining_player_predictions = 684   (38 × 18)
--   remaining_game_outcomes = 38
--   remaining_player_outcomes = 684
--   any_remaining_dupes = 0
SELECT
    (SELECT COUNT(*) FROM game_predictions)        AS remaining_game_predictions,
    (SELECT COUNT(*) FROM player_predictions)      AS remaining_player_predictions,
    (SELECT COUNT(*) FROM game_outcomes)           AS remaining_game_outcomes,
    (SELECT COUNT(*) FROM player_outcomes)         AS remaining_player_outcomes,
    (
        SELECT COUNT(*)
        FROM (
            SELECT 1
            FROM game_predictions
            GROUP BY game_id, game_date, model_version
            HAVING COUNT(*) > 1
        ) d
    )                                              AS any_remaining_dupes;

-- If the numbers above match the expected counts, run:
--      COMMIT;
-- Otherwise:
--      ROLLBACK;

COMMIT;


-- =============================================================================
-- 3. PREDICTED_AT BACKFILL — fix the iter=2000 rows whose `predicted_at` was
--    set by the migration's DEFAULT NOW() instead of by an actual insert.
-- =============================================================================
-- Restoring `predicted_at = prediction_timestamp` makes the column mean what
-- its name says on every row going forward. iter=5000 rows already have the
-- two columns equal (delta = 0 across all 23 audited rows) so this UPDATE is
-- a no-op for them; the predicate filters to only the rows that need fixing.

BEGIN;

-- Preview what will change. Expected: 19 rows on first run, 0 on subsequent.
SELECT COUNT(*) AS rows_to_backfill
FROM game_predictions
WHERE predicted_at <> prediction_timestamp;

UPDATE game_predictions
SET predicted_at = prediction_timestamp
WHERE predicted_at <> prediction_timestamp;

-- Verify post-state. Expected: rows_remaining_skewed = 0.
SELECT COUNT(*) AS rows_remaining_skewed
FROM game_predictions
WHERE predicted_at <> prediction_timestamp;

COMMIT;


-- =============================================================================
-- 4. POST-FLIGHT VERIFICATION — final state checks.
-- =============================================================================

-- (a) Row counts match the expected post-cleanup totals.
SELECT
    (SELECT COUNT(*) FROM game_predictions)                              AS game_predictions,
    (SELECT COUNT(*) FROM player_predictions)                            AS player_predictions,
    (SELECT COUNT(*) FROM game_outcomes)                                 AS game_outcomes,
    (SELECT COUNT(*) FROM player_outcomes)                               AS player_outcomes,
    (SELECT COUNT(DISTINCT (game_id, game_date)) FROM game_predictions)  AS distinct_games,
    (SELECT COUNT(DISTINCT (game_id, game_date)) FROM game_outcomes)     AS distinct_outcome_games;
-- Expected: all four counts consistent (38 / 684 / 38 / 684), and
-- distinct_games == game_predictions == 38 (one row per game, no dupes).


-- (b) No duplicate (game_id, game_date, model_version) tuples remain.
SELECT
    game_id, game_date, model_version, COUNT(*) AS n
FROM game_predictions
GROUP BY game_id, game_date, model_version
HAVING COUNT(*) > 1;
-- Expected: zero rows.


-- (c) Every game_predictions row has exactly 18 player_predictions children.
SELECT
    gp.id, gp.game_id, gp.game_date, COUNT(pp.id) AS player_count
FROM game_predictions gp
LEFT JOIN player_predictions pp ON pp.game_prediction_id = gp.id
GROUP BY gp.id, gp.game_id, gp.game_date
HAVING COUNT(pp.id) <> 18
ORDER BY gp.game_date, gp.game_id;
-- Expected: zero rows.


-- (d) predicted_at is now equal to prediction_timestamp on every row.
SELECT COUNT(*) AS rows_with_skewed_predicted_at
FROM game_predictions
WHERE predicted_at <> prediction_timestamp;
-- Expected: 0.

-- =============================================================================
-- End of cleanup migration.
-- =============================================================================
