"""
Post-hoc analysis of the backtest ledger.

Uses real historical closing odds (from odds_ingestion.OddsBook) to compute:
  - Expected value per bet at real market prices (not flat -110 assumption)
  - Closing line value (CLV): did we get a better price than the market?
  - Calibration curves: does the model mean what it says?
  - Threshold sweeps: where does edge actually exist?

Run this after backtester.py has produced the ledger.
"""

import pandas as pd
import numpy as np
import json
import os

from odds_ingestion import american_to_implied_prob, american_to_payout, expected_value


LEDGER_PATH = './data/backtest_ledger.csv'


# ==========================================
# LOADER — enriches raw ledger with derived columns
# ==========================================
def load_ledger(path: str = LEDGER_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No ledger at {path}. Run backtester.py first.")

    df = pd.read_csv(path)

    # --- Model derived columns ---
    df['model_home_win'] = 1.0 - df['model_away_win']
    df['model_f5_home']  = 1.0 - df['model_f5_away'] - df['model_f5_tie']

    # --- Actual outcomes ---
    df['actual_ml_winner'] = np.where(
        df['actual_away_score'] > df['actual_home_score'], 'away', 'home'
    )
    df['actual_f5_winner'] = np.where(
        df['actual_f5_away'] > df['actual_f5_home'], 'away',
        np.where(df['actual_f5_away'] < df['actual_f5_home'], 'home', 'tie')
    )
    df['actual_nrfi_bool'] = df['actual_nrfi'].astype(bool)

    # --- Model picks ---
    df['ml_pick']       = np.where(df['model_away_win'] > 0.5, 'away', 'home')
    df['ml_confidence'] = df[['model_away_win', 'model_home_win']].max(axis=1)
    df['ml_won']        = (df['ml_pick'] == df['actual_ml_winner']).astype(int)

    df['f5_pick']       = np.where(df['model_f5_away'] > df['model_f5_home'], 'away', 'home')
    df['f5_confidence'] = df[['model_f5_away', 'model_f5_home']].max(axis=1)
    df['f5_won'] = (
        (df['f5_pick'] == df['actual_f5_winner']) &
        (df['actual_f5_winner'] != 'tie')
    ).astype(int)

    df['nrfi_won'] = df['actual_nrfi_bool'].astype(int)

    # --- Market-implied probabilities from closing odds ---
    if 'odds_away_ml_close' in df.columns:
        df['market_away_prob'] = df['odds_away_ml_close'].apply(american_to_implied_prob)
        df['market_home_prob'] = df['odds_home_ml_close'].apply(american_to_implied_prob)

        # Book's vig: implied probs sum to >1.00; the excess is the book's margin
        df['market_total_implied'] = df['market_away_prob'] + df['market_home_prob']

        # De-vigged probabilities (what the market "really" thinks)
        df['market_away_fair'] = df['market_away_prob'] / df['market_total_implied']
        df['market_home_fair'] = df['market_home_prob'] / df['market_total_implied']

        # Model edge over the fair market probability (not the juiced price)
        df['ml_model_edge'] = np.where(
            df['ml_pick'] == 'away',
            df['model_away_win'] - df['market_away_fair'],
            df['model_home_win'] - df['market_home_fair'],
        )

        # Price we'd get at close for our pick
        df['ml_pick_price'] = np.where(
            df['ml_pick'] == 'away',
            df['odds_away_ml_close'],
            df['odds_home_ml_close'],
        )
        df['ml_pick_market_prob'] = np.where(
            df['ml_pick'] == 'away',
            df['market_away_prob'],
            df['market_home_prob'],
        )

    return df


# ==========================================
# EV CALCULATION — bet units won/lost using real prices
# ==========================================
def grade_ml_bets_ev(df: pd.DataFrame, min_edge: float = 0.00) -> pd.DataFrame:
    sub = df.dropna(subset=['ml_pick_price', 'ml_model_edge']).copy()
    sub = sub[sub['ml_model_edge'] >= min_edge]

    if len(sub) == 0:
        return pd.DataFrame()

    sub['payout']     = sub['ml_pick_price'].apply(american_to_payout)
    sub['units']      = np.where(sub['ml_won'] == 1, sub['payout'], -1.0)
    sub['ev_per_bet'] = sub.apply(
        lambda r: expected_value(
            r['model_away_win'] if r['ml_pick'] == 'away' else r['model_home_win'],
            r['ml_pick_price']
        ), axis=1
    )
    return sub


def ml_edge_sweep(df: pd.DataFrame, edges: list = None) -> pd.DataFrame:
    if edges is None:
        edges = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    rows = []
    for e in edges:
        graded = grade_ml_bets_ev(df, min_edge=e)
        n = len(graded)
        if n == 0:
            continue
        rows.append({
            'min_edge':    e,
            'n_bets':      n,
            'win_rate':    round(graded['ml_won'].mean(), 3),
            'mean_ev':     round(graded['ev_per_bet'].mean(), 4),
            'total_units': round(graded['units'].sum(), 1),
            'roi_pct':     round(graded['units'].sum() / n * 100, 2),
        })
    return pd.DataFrame(rows)


# ==========================================
# CLV — Closing Line Value
# The single best indicator of real model skill.
# ==========================================
def closing_line_value(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=['ml_pick_market_prob']).copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub['model_pick_prob'] = np.where(
        sub['ml_pick'] == 'away', sub['model_away_win'], sub['model_home_win']
    )
    sub['clv_prob'] = sub['model_pick_prob'] - sub['ml_pick_market_prob']

    return pd.DataFrame([{
        'mean_clv_pts':     round(sub['clv_prob'].mean() * 100, 2),
        'positive_clv_pct': round((sub['clv_prob'] > 0).mean() * 100, 1),
        'n_bets':           len(sub),
    }])


# ==========================================
# CALIBRATION CURVE
# ==========================================
def calibration_table(df: pd.DataFrame, prob_col: str, outcome_col: str,
                      bucket_size: float = 0.05, min_n: int = 20) -> pd.DataFrame:
    bins = np.arange(0, 1.01, bucket_size)
    c = df.copy()
    c['bucket'] = pd.cut(c[prob_col], bins=bins, include_lowest=True)

    agg = c.groupby('bucket', observed=True).agg(
        n=(outcome_col, 'count'),
        predicted_mean=(prob_col, 'mean'),
        actual_rate=(outcome_col, 'mean'),
    ).reset_index()

    agg['gap'] = agg['actual_rate'] - agg['predicted_mean']
    agg['bucket'] = agg['bucket'].astype(str)
    return agg[agg['n'] >= min_n].round(3)


def pick_side_bias(df: pd.DataFrame, pick_col: str, won_col: str) -> pd.DataFrame:
    rows = []
    for side in ['away', 'home']:
        sub = df[df[pick_col] == side]
        if len(sub) == 0:
            continue
        rows.append({
            'pick': side, 'n': len(sub),
            'win_rate': round(sub[won_col].mean(), 3),
        })
    return pd.DataFrame(rows)


# ==========================================
# MAIN REPORT
# ==========================================
def report(path: str = LEDGER_PATH):
    df = load_ledger(path)
    has_odds = 'odds_away_ml_close' in df.columns and df['odds_away_ml_close'].notna().any()

    print(f"Loaded {len(df):,} games from {path}")
    if has_odds:
        matched = df['odds_away_ml_close'].notna().sum()
        print(f"Odds matched on {matched:,} games ({matched/len(df)*100:.1f}%)\n")
    else:
        print("WARNING: No odds in ledger. Rerun backtester.py with MLB_Basic.csv in ./data/\n")

    print("=" * 72)
    print("1. MONEYLINE — ACCURACY")
    print("=" * 72)
    print("\n-- Calibration --")
    print(calibration_table(df, 'ml_confidence', 'ml_won').to_string(index=False))
    print("\n-- Side bias --")
    print(pick_side_bias(df, 'ml_pick', 'ml_won').to_string(index=False))

    if has_odds:
        print("\n" + "=" * 72)
        print("2. MONEYLINE — REAL MONEY (closing odds)")
        print("=" * 72)
        print("\n-- Edge sweep (min model edge over de-vigged market) --")
        sweep = ml_edge_sweep(df)
        if len(sweep) > 0:
            print(sweep.to_string(index=False))
            print("\n  INTERPRETATION:")
            print("  roi_pct > 0 means profit. An ROI of 5% means $100 bet nets $5.")
            print("  mean_ev is what the model THOUGHT it was getting per bet.")
            print("  If win_rate keeps up with mean_ev, calibration is holding.")
        else:
            print("  No graded bets available.")

        print("\n-- Closing Line Value (CLV) --")
        clv = closing_line_value(df)
        if len(clv) > 0:
            print(clv.to_string(index=False))
            print("\n  mean_clv_pts > 0      -> model systematically beats the close")
            print("  positive_clv_pct > 53 -> evidence of real edge, independent of results")

    print("\n" + "=" * 72)
    print("3. F5")
    print("=" * 72)
    print("\n-- Calibration --")
    print(calibration_table(df, 'f5_confidence', 'f5_won').to_string(index=False))
    print("\n-- Side bias --")
    print(pick_side_bias(df, 'f5_pick', 'f5_won').to_string(index=False))

    print("\n" + "=" * 72)
    print("4. NRFI")
    print("=" * 72)
    print("\n-- Calibration --")
    print(calibration_table(df, 'model_nrfi', 'nrfi_won').to_string(index=False))

    print("\n" + "=" * 72)
    print("5. PITCHER STRIKEOUTS")
    print("=" * 72)
    k_rows = []
    for _, r in df.iterrows():
        try:
            actual_ks_map = json.loads(r['actual_pitcher_ks'])
        except Exception:
            continue
        for pid_col, med_col in [
            ('away_starter_id', 'model_away_k_med'),
            ('home_starter_id', 'model_home_k_med'),
        ]:
            pid = str(int(r[pid_col])) if pd.notna(r[pid_col]) else None
            if pid and pid in actual_ks_map:
                k_rows.append({'actual': actual_ks_map[pid], 'median': r[med_col]})
    if k_rows:
        k_df = pd.DataFrame(k_rows)
        print(f"n pitcher-starts: {len(k_df):,}")
        print(f"Mean absolute error: {(k_df['actual'] - k_df['median']).abs().mean():.2f} Ks")
        print(f"Mean signed bias:    {(k_df['actual'] - k_df['median']).mean():+.2f} Ks")


if __name__ == '__main__':
    report()