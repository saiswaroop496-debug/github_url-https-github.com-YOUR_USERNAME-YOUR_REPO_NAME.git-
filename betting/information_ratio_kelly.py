"""
JP Morgan-style Information Ratio maximisation for bet selection.
IR = mean_edge / std_edge
High IR bets (same edge every time) get larger Kelly fractions.
Low IR bets (variable edge) get smaller Kelly fractions even if mean edge is equal.
"""
import numpy as np

def ir_adjusted_kelly(model_prob: float, decimal_odds: float,
                       historical_edges: list,   # past edges on similar bets
                       ece: float = 0.05,
                       fold_std: float = 0.03) -> dict:
    """
    Kelly fraction adjusted by Information Ratio.
    historical_edges: list of past edge observations on this market type.
    """
    b = decimal_odds - 1.0
    p = model_prob
    q = 1 - p

    # Base Kelly
    f_star = (b * p - q) / b
    if f_star <= 0:
        return {'stake_fraction': 0.0, 'ir': 0.0, 'reason': 'negative_edge'}

    # Information Ratio from historical edge distribution
    if len(historical_edges) >= 10:
        ir = np.mean(historical_edges) / max(np.std(historical_edges), 0.001)
        ir_mult = np.clip(ir / 3.0, 0.3, 1.0)  # normalise: IR of 3 = full Kelly
    else:
        ir_mult = 0.5   # conservative until we have enough history
        ir = 0.0

    # Calibration penalty (ECE)
    ece_mult = max(0.0, 1.0 - ece * 3.0)

    # Stability penalty (fold std)
    stab_mult = max(0.3, 1.0 - fold_std)

    # Combined fraction
    f_final = (f_star / 4.0) * ir_mult * ece_mult * stab_mult
    f_final = min(f_final, 0.05)  # hard cap 5%

    return {
        'stake_fraction':     round(f_final, 4),
        'full_kelly':         round(f_star, 4),
        'ir':                 round(ir, 3),
        'ir_multiplier':      round(ir_mult, 3),
        'ece_multiplier':     round(ece_mult, 3),
        'stability_mult':     round(stab_mult, 3),
        'reason':             'bet'
    }
