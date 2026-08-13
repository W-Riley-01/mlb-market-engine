"""
diagnostic_player_outcomes.py
-----------------------------
One-off diagnostic to verify the player_outcomes "stuck pending" bug
identified during calibration analysis (May 27, 2026).

Hypothesis
----------
fetch_player_predictions_for_game() in record_outcomes.py filters with
`WHERE po.id IS NULL`. That means once any player_outcomes row exists for
a given player_prediction, it is never re-fetched on subsequent record_
outcomes passes. So for evening games in progress at the 23:00 UTC pass:
the player rows get written with status='pending', then the next day's
13:00 UTC --retry-pending pass refreshes the game_outcome but never
revisits the player rows. They stay pending forever.

Three queries
-------------
  Q1: player_outcomes.status distribution by model_version
      Expect: large 'pending' count in v2 if bug is real.

  Q2: joint distribution player_outcomes.status x game_outcomes.status (v2)
      Smoking gun: count of (po.status='pending', go.status='final').
      Should not exist under correct behavior.

  Q3: timing summary for pending v2 player rows
      Expect: go.recorded_at > po.recorded_at (game refreshed later than
      the orphaned player row).

Run
---
    python diagnostic_player_outcomes.py
    python diagnostic_player_outcomes.py > diagnostics/2026-05-27_player_outcomes.txt

Standalone — does not depend on calibration.py beyond borrowing its
DATABASE_URL bootstrap. Reuses prediction_logger._get_conn() for the same
dual-mode connection the writer and reader use.
"""

from __future__ import annotations

import sys

import pandas as pd
from sqlalchemy import text

# Borrow calibration.py's DATABASE_URL bootstrap as a side effect of import.
# The diagnostic itself does not call any calibration.py functions — just
# leverages its secrets.toml fallback so this script works the same way
# from PyCharm, a notebook, or a shell.
import calibration  # noqa: F401
from prediction_logger import _get_conn


def _run(sql: str, params: dict | None = None) -> pd.DataFrame:
    conn = _get_conn()
    with conn.session as s:
        rows = s.execute(text(sql), params or {}).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    _section("Player outcomes diagnostic")
    print("Date: 2026-05-27")
    print("Purpose: verify the stuck-pending player_outcomes bug before")
    print("patching fetch_player_predictions_for_game in record_outcomes.py.")

    # ----------------------------------------------------------------------
    # Q1 — status distribution by model_version
    # ----------------------------------------------------------------------
    _section("Q1. player_outcomes.status distribution by model_version")
    print("Counts every player_prediction grouped by (model_version,")
    print("player_outcomes.status). NULL outcome rows show as '(none)'.")
    print()
    print("Bug signature: large count of v2 rows with status='pending'.")
    print("Correct behavior would be 'final' dominant with a small tail of")
    print("'did_not_play' (~5-10%) and effectively zero 'pending'.")
    print()

    q1 = """
        SELECT
            gp.model_version,
            COALESCE(po.status, '(none)') AS player_outcome_status,
            COUNT(*) AS n
        FROM player_predictions pp
        JOIN game_predictions   gp ON gp.id = pp.game_prediction_id
        LEFT JOIN player_outcomes po ON po.player_prediction_id = pp.id
        GROUP BY gp.model_version, po.status
        ORDER BY gp.model_version, COUNT(*) DESC
    """
    df1 = _run(q1)
    if df1.empty:
        print("(no rows — nothing to diagnose)")
        return 0
    print(df1.to_string(index=False))

    # ----------------------------------------------------------------------
    # Q2 — joint distribution player x game (v2)
    # ----------------------------------------------------------------------
    _section("Q2. Joint: player_outcomes.status x game_outcomes.status (v2)")
    print("For every v2 player_prediction, cross-tabs its player_outcome")
    print("status with its parent game_outcome status.")
    print()
    print("Bug signature: large count in the (player='pending', game='final')")
    print("cell. That combination is the impossible state — the game")
    print("finalized but the player row never caught up.")
    print()

    q2 = """
        SELECT
            COALESCE(po.status, '(none)') AS player_status,
            COALESCE(go.status, '(none)') AS game_status,
            COUNT(*) AS n
        FROM player_predictions pp
        JOIN game_predictions   gp ON gp.id = pp.game_prediction_id
        LEFT JOIN player_outcomes po ON po.player_prediction_id = pp.id
        LEFT JOIN game_outcomes   go ON go.game_prediction_id   = gp.id
        WHERE gp.model_version = 'v2'
        GROUP BY po.status, go.status
        ORDER BY COUNT(*) DESC
    """
    df2 = _run(q2)
    print(df2.to_string(index=False) if not df2.empty else "(no v2 rows)")

    # ----------------------------------------------------------------------
    # Q3 — timing summary for pending v2 player rows
    # ----------------------------------------------------------------------
    _section("Q3. Timing: pending v2 player rows vs their parent games")
    print("For player_outcomes rows still at status='pending' in v2, compares")
    print("when the player row was written (po.recorded_at) to when the")
    print("parent game_outcome was last written (go.recorded_at).")
    print()
    print("Bug signature: most rows show go.recorded_at > po.recorded_at —")
    print("the game outcome was refreshed by a later --retry-pending pass")
    print("but the orphaned player row was never revisited.")
    print()

    q3 = """
        SELECT
            COUNT(*) AS n_pending_v2_players,
            COUNT(*) FILTER (WHERE go.status = 'final')               AS parent_game_is_final,
            COUNT(*) FILTER (WHERE go.recorded_at > po.recorded_at)   AS game_recorded_later,
            MIN(po.recorded_at)::text AS earliest_pending_player_row,
            MAX(po.recorded_at)::text AS latest_pending_player_row,
            MIN(go.recorded_at)::text AS earliest_parent_game_row,
            MAX(go.recorded_at)::text AS latest_parent_game_row
        FROM player_predictions pp
        JOIN game_predictions   gp ON gp.id = pp.game_prediction_id
        JOIN player_outcomes    po ON po.player_prediction_id = pp.id
        LEFT JOIN game_outcomes go ON go.game_prediction_id   = gp.id
        WHERE gp.model_version = 'v2'
          AND po.status = 'pending'
    """
    df3 = _run(q3)
    if df3.empty or (not df3.empty and int(df3.iloc[0]["n_pending_v2_players"]) == 0):
        print("(no pending v2 player_outcomes — Q1 will tell us whether they")
        print("just don't exist as rows, or have a different status)")
    else:
        # Transpose for readability — one column of values is easier to scan
        # than one very wide row.
        print(df3.T.to_string(header=False))

    # ----------------------------------------------------------------------
    # Interpretation
    # ----------------------------------------------------------------------
    _section("Interpretation guide")
    print("Bug confirmed if ALL of the following hold:")
    print()
    print("  [Q1] v2 row count for status='pending' is in the low thousands")
    print("       (~3,500 was the prediction from the inventory math).")
    print()
    print("  [Q2] The (player='pending', game='final') cell is large.")
    print("       Under correct behavior that cell should be zero.")
    print()
    print("  [Q3] game_recorded_later count is close to n_pending_v2_players")
    print("       (the parent game was refreshed AFTER the player row was")
    print("       written, which means a --retry-pending pass ran but did")
    print("       not touch the player row).")
    print()
    print("If confirmed: proceed to Step 2 — patch fetch_player_predictions_")
    print("for_game in record_outcomes.py to relax the po.id IS NULL filter.")
    print()
    print("If Q1 shows v2 has thousands of '(none)' player_outcome_status")
    print("rows instead of 'pending': different bug. The player rows were")
    print("never written at all by record_outcomes. Stop and re-evaluate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())