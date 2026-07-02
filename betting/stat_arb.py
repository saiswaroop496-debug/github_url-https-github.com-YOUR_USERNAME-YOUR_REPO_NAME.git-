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
        odds_h: float, odds_d: float, odds_a: float,
        ah_home_odds: float, ah_away_odds: float,
        ah_handicap: float,
        over_odds: float, under_odds: float,
        score_matrix) -> dict:
    """
    FIXED: ah_away_odds is passed directly — no longer fabricated.
    Previously used (2.0 - 1/ah_home_odds) which is mathematically wrong.
    """
    # ── No-vig normalisation ──────────────────────────────────────────────
    vig_1x2  = 1/odds_h + 1/odds_d + 1/odds_a
    novig_h  = (1/odds_h) / vig_1x2
    novig_d  = (1/odds_d) / vig_1x2
    novig_a  = (1/odds_a) / vig_1x2

    # FIXED: use actual ah_away_odds, not fabricated value
    if ah_away_odds and ah_away_odds > 1.01:
        vig_ah       = 1/ah_home_odds + 1/ah_away_odds
        novig_ah_h   = (1/ah_home_odds) / vig_ah
        novig_ah_a   = (1/ah_away_odds) / vig_ah
    else:
        # Fallback: assume balanced market
        novig_ah_h = 0.50
        novig_ah_a = 0.50

    vig_ou      = 1/over_odds + 1/under_odds
    novig_over  = (1/over_odds) / vig_ou
    novig_under = (1/under_odds) / vig_ou

    # ── Model probabilities from DC matrix ───────────────────────────────
    import numpy as np
    n        = score_matrix.shape[0]
    model_h  = float(np.tril(score_matrix, k=-1).sum())
    model_d  = float(np.trace(score_matrix))
    model_a  = float(np.triu(score_matrix, k=1).sum())

    # AH model probability
    if ah_handicap <= -0.75:
        model_ah_h = model_h
    elif ah_handicap == -0.5:
        model_ah_h = model_h
    elif ah_handicap == -0.25:
        model_ah_h = model_h / (1 - model_d * 0.5)
    elif ah_handicap == 0:
        model_ah_h = model_h / (1 - model_d)
    elif ah_handicap == 0.25:
        model_ah_h = model_h + model_d * 0.5
    else:
        model_ah_h = model_h + model_d

    model_ah_a = 1 - model_ah_h

    # Over 2.5
    model_over  = float(sum(
        score_matrix[i, j] for i in range(n) for j in range(n) if i+j > 2
    ))
    model_under = 1 - model_over

    # ── Edges ──────────────────────────────────────────────────────────────
    edges = {
        '1X2_Home':   round(model_h   - novig_h,    4),
        '1X2_Draw':   round(model_d   - novig_d,    4),
        '1X2_Away':   round(model_a   - novig_a,    4),
        'AH_Home':    round(model_ah_h - novig_ah_h, 4),
        'AH_Away':    round(model_ah_a - novig_ah_a, 4),
        'Over_2.5':   round(model_over  - novig_over, 4),
        'Under_2.5':  round(model_under - novig_under, 4),
    }

    MIN_EDGE = 0.025
    valid    = {k: v for k, v in edges.items() if v >= MIN_EDGE}
    best     = max(valid, key=valid.get) if valid else None

    return {
        'best_bet':    best,
        'edge':        round(valid.get(best, 0.0), 4) if best else 0.0,
        'all_edges':   edges,
        'arb_signal':  _detect_cross_market_inconsistency(edges),
        'no_bet':      best is None,
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
