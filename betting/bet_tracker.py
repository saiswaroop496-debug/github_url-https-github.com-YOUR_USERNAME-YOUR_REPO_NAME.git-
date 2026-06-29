import json, time, logging
from pathlib import Path
import pandas as pd
import numpy as np

LOG_PATH = Path("bets_log.jsonl")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bet_tracker")

def log_bet(match: str, market: str, outcome: str, model_prob: float, decimal_odds: float, no_vig_prob: float, edge: float, kelly_fraction: float, stake_units: float, result: str = None, closing_odds: float = None, btts_prob: float = None, over25_prob: float = None) -> dict:
    clv_pct = None
    if closing_odds and decimal_odds and decimal_odds > 1.0:
        clv_pct = round((closing_odds / decimal_odds) - 1, 4)

    pnl = None
    if result == 'win':
        pnl = round(stake_units * (decimal_odds - 1), 3)
    elif result == 'loss':
        pnl = round(-stake_units, 3)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "match": match, "market": market, "outcome": outcome, "model_prob": round(model_prob, 4),
        "decimal_odds": round(decimal_odds, 3), "no_vig_prob": round(no_vig_prob, 4), "edge": round(edge, 4),
        "kelly_fraction": round(kelly_fraction, 4), "stake_units": round(stake_units, 3), "result": result,
        "closing_odds": closing_odds, "clv_pct": clv_pct, "pnl_units": pnl, "btts_prob": btts_prob, "over25_prob": over25_prob,
    }

    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f"BET | {match} | {outcome} | edge={edge:+.2%} | stake={stake_units:.2f}u | result={result or 'pending'}")
    return record

def clv_report(min_bets: int = 20) -> dict:
    if not LOG_PATH.exists():
        return {"error": "No bets logged yet."}
    df = pd.read_json(LOG_PATH, lines=True)
    if len(df) == 0:
        return {"error": "Empty bet log."}
    report = {"total_bets": len(df), "settled": int(df['result'].notna().sum()), "clv_eligible": int(df['clv_pct'].notna().sum()), "pending": int(df['result'].isna().sum())}
    settled = df[df['result'].notna()].copy()
    if len(settled) > 0:
        wins = (settled['result'] == 'win').sum()
        total_stake = settled['stake_units'].sum()
        total_pnl   = settled['pnl_units'].sum()
        report.update({"win_rate": round(wins / len(settled), 4), "total_pnl": round(float(total_pnl), 3), "roi": round(float(total_pnl / total_stake) if total_stake > 0 else 0, 4)})
    clv_df = df[df['clv_pct'].notna()].copy()
    if len(clv_df) > 0:
        report.update({"mean_clv": round(float(clv_df['clv_pct'].mean()), 4), "median_clv": round(float(clv_df['clv_pct'].median()), 4), "clv_positive_pct": round(float((clv_df['clv_pct'] > 0).mean()), 4), "clv_by_market": clv_df.groupby('market')['clv_pct'].mean().round(4).to_dict()})
        mean_clv = report['mean_clv']
        if mean_clv > 0.02: report['clv_verdict'] = "STRONG EDGE — CLV > 2% sustained"
        elif mean_clv > 0.015: report['clv_verdict'] = "GENUINE EDGE — CLV > 1.5%"
        elif mean_clv > 0.005: report['clv_verdict'] = "MARGINAL EDGE — monitor closely"
        else: report['clv_verdict'] = "NO PROVEN EDGE — do not scale stakes"
    if len(df) < min_bets: report['warning'] = f"Only {len(df)} bets. Need {min_bets}+ for reliable CLV."
    return report
