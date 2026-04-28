"""
OddsWarehouse MLB CSV loader and matchup lookup.

This module is read-only infrastructure for the backtester and analysis.
It loads the CSV once, normalizes team codes and dates to match the
rest of our pipeline, and exposes a fast (date, away, home) lookup.

Usage:
    from odds_ingestion import OddsBook
    book = OddsBook('./data/MLB_Basic.csv')
    row = book.get('2024-04-05', 'ATL', 'PHI')
"""

import os
import pandas as pd


# ==========================================
# TEAM CODE NORMALIZATION
# Maps OddsWarehouse codes -> MLB Stats API codes (what our pipeline uses).
# Most codes match; these are the exceptions.
# ==========================================
TEAM_CODE_MAP = {
    'SLC': 'STL',   # St. Louis Cardinals
    'CHW': 'CWS',   # Chicago White Sox
    'WAS': 'WSH',   # Washington Nationals
    'FLO': 'MIA',   # Florida Marlins -> Miami Marlins (renamed in 2012)
}


def normalize_team(code: str) -> str:
    """Converts an OddsWarehouse team abbreviation to our pipeline's standard."""
    if pd.isna(code):
        return code
    code = str(code).strip().upper()
    return TEAM_CODE_MAP.get(code, code)


# ==========================================
# COLUMN NORMALIZATION
# The CSV header uses spaces and mixed wording. We rename to snake_case
# for stable downstream access regardless of source quirks.
# ==========================================
COLUMN_RENAMES = {
    'game id':          'game_id',
    'date':             'date_raw',
    'away team':        'away_team',
    'away score':       'away_score',
    'away ml open':     'away_ml_open',
    'away ml close':    'away_ml_close',
    'over open':        'over_open',
    'over open odds':   'over_open_odds',
    'over close':       'over_close',
    'over close odds':  'over_close_odds',
    'home team':        'home_team',
    'home score':       'home_score',
    'home ml open':     'home_ml_open',
    'home ml close':    'home_ml_close',
    'under open':       'under_open',
    'under open odds':  'under_open_odds',
    'under close':      'under_close',
    'under close odds': 'under_close_odds',
}


# ==========================================
# THE ODDSBOOK CLASS
# ==========================================
class OddsBook:
    """Loads the OddsWarehouse MLB CSV and serves fast per-game lookups."""

    def __init__(self, csv_path: str = './data/MLB_Basic.csv'):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Odds CSV not found at {csv_path}. "
                f"Place MLB_Basic.csv from OddsWarehouse in ./data/"
            )
        print(f"[OddsBook] Loading {csv_path}...")
        df = pd.read_csv(csv_path)

        # Lowercase + strip the headers so minor formatting differences are ignored
        df.columns = [c.strip().lower() for c in df.columns]

        # Rename to snake_case; only rename columns that exist to be resilient
        rename = {k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}
        df.rename(columns=rename, inplace=True)

        # --- Date normalization: 20090405 -> 2009-04-05 ---
        df['date'] = pd.to_datetime(df['date_raw'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # --- Team code normalization ---
        df['away_team'] = df['away_team'].apply(normalize_team)
        df['home_team'] = df['home_team'].apply(normalize_team)

        # --- Coerce numerics (odds columns may contain stray strings) ---
        numeric_cols = [c for c in df.columns if any(k in c for k in ['score', 'ml_', 'over_', 'under_'])]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # --- Build a fast lookup index on (date, away, home) ---
        # sort_index() removes the "indexing past lexsort depth" warning and
        # switches pandas from linear scan to O(log n) binary search per lookup.
        # On a 40k-game CSV with 4800+ lookups this is a meaningful speedup.
        self.df = df
        self._index = df.set_index(['date', 'away_team', 'home_team']).sort_index()

        print(f"[OddsBook] Loaded {len(df):,} games "
              f"from {df['date'].min()} to {df['date'].max()}")

    def get(self, date: str, away: str, home: str) -> dict | None:
        """
        Returns the odds row for the given matchup as a dict, or None if missing.
        Applies team normalization to the inputs so callers don't have to.
        """
        key = (date, normalize_team(away), normalize_team(home))
        try:
            row = self._index.loc[key]
        except KeyError:
            return None
        # If doubleheaders exist (same matchup twice on same date), take the first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row.to_dict()

    def coverage_summary(self) -> pd.DataFrame:
        """How many games we have per year — useful for sanity checks."""
        summary = self.df.copy()
        summary['year'] = pd.to_datetime(summary['date']).dt.year
        return summary.groupby('year').size().reset_index(name='games')


# ==========================================
# AMERICAN ODDS HELPERS
# These are the only two pieces of math that matter for EV work.
# ==========================================
def american_to_implied_prob(odds: float) -> float:
    """
    Converts American odds to implied probability (including the book's juice).
    -110 -> 0.524
    +120 -> 0.455
    """
    if pd.isna(odds):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def american_to_payout(odds: float) -> float:
    """
    Converts American odds to decimal payout per 1 unit risked.
    -110 -> 0.909  (risk 1 to win 0.909)
    +120 -> 1.200  (risk 1 to win 1.200)
    """
    if pd.isna(odds):
        return None
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


def expected_value(model_prob: float, american_odds: float) -> float:
    """
    EV per 1 unit risked. Positive = +EV bet, negative = -EV bet.
    EV = (p_win * payout) - (1 - p_win)
    """
    payout = american_to_payout(american_odds)
    if payout is None or model_prob is None:
        return None
    return (model_prob * payout) - (1 - model_prob)


# ==========================================
# CLI SANITY CHECK
# Run this module directly to verify the CSV loaded correctly.
# ==========================================
if __name__ == "__main__":
    book = OddsBook('./data/MLB_Basic.csv')

    print("\n--- Coverage by year ---")
    print(book.coverage_summary().to_string(index=False))

    print("\n--- Sample lookup: 2009-04-05 ATL @ PHI ---")
    row = book.get('2009-04-05', 'ATL', 'PHI')
    if row:
        for k, v in row.items():
            print(f"  {k}: {v}")
    else:
        print("  Not found.")

    print("\n--- Odds math sanity check ---")
    print(f"  -110 -> implied {american_to_implied_prob(-110):.3f}, payout {american_to_payout(-110):.3f}")
    print(f"  +120 -> implied {american_to_implied_prob(+120):.3f}, payout {american_to_payout(+120):.3f}")
    print(f"  EV of model_prob=0.55 at +120: {expected_value(0.55, +120):+.3f} units per bet")