-- =============================================================
-- migration_2025_05_phase2c_kdist.sql
-- =============================================================
-- Adds columns to persist the full K-distribution arrays and the
-- pitcher 1st-inning stat dicts. Required so the Streamlit Cloud
-- read-only viewer can render the K histogram and the 1st-inning
-- section in the pitcher card without re-running the simulator.
--
-- Run ONCE in the Supabase SQL editor BEFORE deploying the updated
-- prediction_logger.py + app.py. New columns are nullable; old rows
-- will read as NULL and the renderer falls back gracefully (same
-- pattern the simulator already uses for pitchers without enough
-- starts to compute 1st-inning stats).
--
-- Storage estimate: ~25KB per row for both K dists at 5000 iters,
-- ~1KB for both 1st-inning dicts. Across ~15 games/day × 90 days
-- ≈ 35MB — comfortably within Supabase free tier.
-- =============================================================

ALTER TABLE game_predictions
    ADD COLUMN IF NOT EXISTS away_k_dist            JSONB,
    ADD COLUMN IF NOT EXISTS home_k_dist            JSONB,
    ADD COLUMN IF NOT EXISTS away_starter_first_inn JSONB,
    ADD COLUMN IF NOT EXISTS home_starter_first_inn JSONB;

-- =============================================================
-- After running this, verify with:
--   SELECT column_name, data_type
--   FROM information_schema.columns
--   WHERE table_name = 'game_predictions'
--     AND column_name IN (
--       'away_k_dist', 'home_k_dist',
--       'away_starter_first_inn', 'home_starter_first_inn'
--     );
-- Should return 4 rows, all data_type = 'jsonb'.
-- =============================================================
