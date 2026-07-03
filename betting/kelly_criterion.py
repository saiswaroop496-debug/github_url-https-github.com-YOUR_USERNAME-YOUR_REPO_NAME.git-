# betting/kelly_criterion.py — V7.3 complete replacement

import numpy as np
import json
from pathlib import Path
from collections import defaultdict

BET_HISTORY_PATH = Path("data/bet_history.json")


def _load_bet_history() -> list:
    if BET_HISTORY_PATH.exists():
        with open(BET_HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_bet_history(history: list):
    BET_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BET_HISTORY_PATH, 'w') as f:
        json.dump(history[-200:], f)   # keep last 200 bets


# ── 1. Autocorrelation Veto ───────────────────────────────────────────────────
def autocorrelation_veto(market_type: str,
                          history: list = None,
                          consecutive_loss_threshold: int = 5) -> bool:
    """
    Veto bet if the last N bets on this market type all lost.
    Detects when the model is systematically wrong on a market (not just unlucky).
    Returns True = VETO (do not bet), False = proceed.
    """
    if history is None:
        history = _load_bet_history()

    relevant = [b for b in history if b.get('market') == market_type
                and b.get('result') in ['win', 'loss']]

    if len(relevant) < consecutive_loss_threshold:
        return False   # not enough history to veto

    last_n = relevant[-consecutive_loss_threshold:]
    all_losses = all(b['result'] == 'loss' for b in last_n)
    if all_losses:
        print(f"  ⛔ AUTOCORRELATION VETO: {consecutive_loss_threshold} consecutive "
              f"losses on {market_type}. Model may be misspecified for this market.")
    return all_losses


# ── 2. Market Impact Model ────────────────────────────────────────────────────
def market_impact_adjustment(raw_stake_units: float,
                               available_liquidity: float = None,
                               max_liquidity_fraction: float = 0.20) -> float:
    """
    Limit stake to a fraction of available exchange liquidity.
    Prevents moving the line against yourself.
    If liquidity unknown, apply conservative default cap.
    """
    if available_liquidity is None or available_liquidity <= 0:
        # Conservative: assume 500 unit pool when liquidity unknown
        available_liquidity = 500.0

    max_stake = available_liquidity * max_liquidity_fraction
    return min(raw_stake_units, max_stake)


# ── 3. Sharpe-Adjusted Kelly ─────────────────────────────────────────────────
def sharpe_kelly(edge: float,
                  decimal_odds: float,
                  target_sharpe: float = 1.5,
                  ece: float = 0.05,
                  fold_std: float = 0.03,
                  bankroll: float = 1000.0,
                  available_liquidity: float = None) -> dict:
    """
    V7.3 Kelly: Sharpe-adjusted + ECE penalty + stability penalty +
                market impact cap + autocorrelation veto awareness.

    Sharpe-adjusted formula:
        f* = edge / (odds_variance * target_sharpe) / 4

    This maximizes risk-adjusted return (Sharpe) rather than log-wealth (raw Kelly),
    producing smaller but more consistent stakes — critical for sports betting
    where model miscalibration is common.
    """
    if edge <= 0 or decimal_odds <= 1.01:
        return {'stake': 0.0, 'fraction': 0.0, 'reason': 'no_edge'}

    b   = decimal_odds - 1.0
    p   = edge + (1 / decimal_odds)   # implied win prob from edge + novig prob
    p   = min(0.95, max(0.05, p))
    q   = 1.0 - p

    # Full Kelly
    f_kelly = (b*p - q) / b
    if f_kelly <= 0:
        return {'stake': 0.0, 'fraction': 0.0, 'reason': 'negative_kelly'}

    # Odds variance (Bernoulli): σ² = p(1-p)b² + 2pb(1-b)... simplified:
    odds_variance = p * q * b**2

    # Sharpe-adjusted fraction
    f_sharpe = edge / (odds_variance * target_sharpe + 1e-9)

    # Quarter the more conservative of Kelly vs Sharpe
    f_base = min(f_kelly, f_sharpe) / 4.0

    # Calibration penalty
    ece_mult    = max(0.0, 1.0 - ece * 3.0)
    stab_mult   = max(0.3, 1.0 - fold_std)
    f_penalized = f_base * ece_mult * stab_mult

    # Hard caps
    f_capped = min(f_penalized, 0.05)   # max 5% bankroll
    stake    = f_capped * bankroll

    # Market impact cap
    stake = market_impact_adjustment(stake, available_liquidity)

    return {
        'stake':             round(stake, 2),
        'fraction':          round(f_capped, 4),
        'bankroll_pct':      round(f_capped * 100, 2),
        'full_kelly':        round(f_kelly, 4),
        'sharpe_kelly':      round(f_sharpe, 4),
        'ece_multiplier':    round(ece_mult, 4),
        'stab_multiplier':   round(stab_mult, 4),
        'reason':            'bet',
    }


# ── 4. Master Bet Decision Function ──────────────────────────────────────────
def evaluate_bet(market_type: str,
                  model_prob: float,
                  decimal_odds: float,
                  novig_prob: float,
                  ece: float = 0.05,
                  fold_std: float = 0.03,
                  bankroll: float = 1000.0,
                  available_liquidity: float = None,
                  min_edge: float = 0.025) -> dict:
    """
    Full bet evaluation pipeline:
    1. Compute no-vig edge
    2. Check autocorrelation veto
    3. Apply Sharpe-adjusted Kelly sizing
    4. Return bet recommendation
    """
    edge = model_prob - novig_prob

    if edge < min_edge:
        return {'action': 'NO_BET', 'edge': round(edge, 4),
                'reason': f'edge {edge:.3f} < threshold {min_edge}'}

    # Autocorrelation veto check
    if autocorrelation_veto(market_type):
        return {'action': 'VETOED', 'edge': round(edge, 4),
                'reason': 'consecutive_loss_streak_on_market'}

    sizing = sharpe_kelly(edge, decimal_odds, ece=ece,
                           fold_std=fold_std, bankroll=bankroll,
                           available_liquidity=available_liquidity)

    return {
        'action':        'BET',
        'market':        market_type,
        'edge':          round(edge, 4),
        'model_prob':    round(model_prob, 4),
        'novig_prob':    round(novig_prob, 4),
        **sizing,
    }
