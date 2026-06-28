# bet_tracker.py — Replace print-based logging with structured JSONL

import json
import logging
import time
import os
from pathlib import Path

LOG_PATH = Path("bets_log.jsonl")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bet_tracker")

# File handler for structured JSONL
file_handler = logging.FileHandler("bet_tracker_events.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


def log_bet(match: str, market: str, outcome: str,
            model_prob: float, decimal_odds: float,
            no_vig_prob: float, edge: float,
            kelly_fraction: float, stake_units: float,
            result: str = None, closing_odds: float = None):
    """
    Log a bet to bets_log.jsonl with full CLV tracking fields.
    result: 'win' | 'loss' | 'void' | None (pending)
    """
    record = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "match":         match,
        "market":        market,
        "outcome":       outcome,
        "model_prob":    round(model_prob, 4),
        "decimal_odds":  round(decimal_odds, 3),
        "no_vig_prob":   round(no_vig_prob, 4),
        "edge":          round(edge, 4),
        "kelly_fraction":round(kelly_fraction, 4),
        "stake_units":   round(stake_units, 3),
        "result":        result,
        "closing_odds":  closing_odds,
        "clv_pct":       round((closing_odds / decimal_odds) - 1, 4)
                         if closing_odds and decimal_odds else None,
        "pnl_units":     round(stake_units * (decimal_odds - 1), 3)
                         if result == 'win' else
                         round(-stake_units, 3)
                         if result == 'loss' else None
    }

    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f"BET LOGGED | {match} | {outcome} | edge={edge:+.2%} | "
                f"stake={stake_units:.3f}u | result={result}")
    return record


def clv_report(min_bets: int = 20):
    """
    Read bets_log.jsonl and compute CLV summary.
    Requires closing_odds field to be updated after market closes.
    """
    if not LOG_PATH.exists():
        print("No bets logged yet.")
        return

    import pandas as pd
    df = pd.read_json(LOG_PATH, lines=True)

    if len(df) < min_bets:
        print(f"Only {len(df)} bets. Need {min_bets}+ for reliable CLV estimate.")

    settled = df[df['result'].notna()].copy()
    clv_eligible = df[df['clv_pct'].notna()].copy()

    print("\n-- Bet Tracker CLV Report --------------------------------")
    print(f"  Total bets logged:    {len(df)}")
    print(f"  Settled bets:         {len(settled)}")
    print(f"  CLV-eligible bets:    {len(clv_eligible)}")

    if len(settled) > 0:
        wins  = (settled['result'] == 'win').sum()
        pnl   = settled['pnl_units'].sum()
        roi   = pnl / settled['stake_units'].sum() if settled['stake_units'].sum() > 0 else 0
        print(f"\n  Win Rate:             {wins/len(settled):.1%}")
        print(f"  Total P&L:            {pnl:+.2f} units")
        print(f"  ROI:                  {roi:+.1%}")

    if len(clv_eligible) > 0:
        mean_clv = clv_eligible['clv_pct'].mean()
        print(f"\n  Mean CLV:             {mean_clv:+.3%}")
        print(f"  Median CLV:           {clv_eligible['clv_pct'].median():+.3%}")
        print(f"  CLV by Market:")
        print(clv_eligible.groupby('market')['clv_pct'].mean().apply(
            lambda x: f"    {x:+.3%}").to_string())

    print("----------------------------------------------------------\n")
