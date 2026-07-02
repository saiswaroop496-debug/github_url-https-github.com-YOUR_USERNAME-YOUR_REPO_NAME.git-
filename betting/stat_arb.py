"""
Statistical arbitrage across correlated football markets.
Converts Renaissance/DE Shaw stat-arb to cross-market football mispricing.

Three markets for the same match are cointegrated:
  1X2 implied probability
  Asian Handicap implied probability
  Over/Under implied probability
When they diverge, the cheapest market has positive EV.
"""
import numpy as np

def compute_cross_market_arb(
        # 1X2 market
        odds_h: float, odds_d: float, odds_a: float,
        # Asian Handicap (AH) market — e.g., Home -0.5 at decimal odds
        ah_home_odds: float, ah_handicap: float,
        # Over/Under 2.5 market
        over_odds: float, under_odds: float,
        # Your model's DC scoreline matrix
        score_matrix: np.ndarray) -> dict:
    """
    Find which market is most mispriced relative to your scoreline model.
    Returns the single best bet across all three markets.
    """
    # ── Remove overround from each market ────────────────────────────────
    vig_1x2 = 1/odds_h + 1/odds_d + 1/odds_a
    novig_h  = (1/odds_h) / vig_1x2
    novig_d  = (1/odds_d) / vig_1x2
    novig_a  = (1/odds_a) / vig_1x2

    vig_ah  = 1/ah_home_odds + 1/(2.0 - 1/ah_home_odds)
    novig_ah_home = (1/ah_home_odds) / vig_ah

    vig_ou  = 1/over_odds + 1/under_odds
    novig_over = (1/over_odds) / vig_ou

    # ── Model probabilities from DC scoreline matrix ──────────────────────
    n = score_matrix.shape[0]
    model_h = np.tril(score_matrix, k=-1).sum()
    model_d = np.trace(score_matrix)
    model_a = np.triu(score_matrix, k=1).sum()

    # AH model prob: P(home wins by more than |handicap|)
    # For -0.5 handicap: home must win by 1+
    if ah_handicap <= -0.5:
        model_ah_home = model_h
    elif ah_handicap == 0:
        model_ah_home = model_h + model_d * 0.5
    else:
        model_ah_home = model_h + model_d

    # Over 2.5 from scoreline matrix
    model_over = sum(
        score_matrix[i, j]
        for i in range(n) for j in range(n)
        if i + j > 2
    )

    # ── Compute edges across all markets ─────────────────────────────────
    edges = {
        '1X2_Home':   model_h - novig_h,
        '1X2_Draw':   model_d - novig_d,
        '1X2_Away':   model_a - novig_a,
        'AH_Home':    model_ah_home - novig_ah_home,
        'Over_2.5':   model_over - novig_over,
        'Under_2.5':  (1 - model_over) - (1 - novig_over),
    }

    # Best bet = highest edge above minimum threshold
    MIN_EDGE = 0.025
    valid_bets = {k: v for k, v in edges.items() if v >= MIN_EDGE}

    if not valid_bets:
        return {'best_bet': None, 'edge': 0.0, 'all_edges': edges}

    best_market = max(valid_bets, key=lambda k: valid_bets[k])
    return {
        'best_bet':    best_market,
        'edge':        round(valid_bets[best_market], 4),
        'all_edges':   {k: round(v, 4) for k, v in edges.items()},
        'arb_signal':  _detect_cross_market_inconsistency(edges),
    }


def _detect_cross_market_inconsistency(edges: dict) -> str:
    """
    If 1X2_Home is positive but AH_Home is negative,
    the markets are inconsistent — strong arb signal.
    """
    if edges.get('1X2_Home', 0) > 0.03 and edges.get('AH_Home', 0) < -0.01:
        return "INCONSISTENT: 1X2 cheap but AH expensive — bet 1X2 Home"
    if edges.get('Over_2.5', 0) > 0.03 and edges.get('1X2_Draw', 0) > 0.03:
        return "INCONSISTENT: Both Over and Draw positive — model sees low-score/draw game"
    return "CONSISTENT"
