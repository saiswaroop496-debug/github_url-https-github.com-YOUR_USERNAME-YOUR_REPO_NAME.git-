import numpy as np

def compute_kelly_stake(model_prob: float, decimal_odds: float, bankroll: float = 1000.0, ece: float = 0.05, fold_std: float = 0.03, max_fraction: float = 0.05, max_units: float = 50.0) -> dict:
    if decimal_odds <= 1.0 or model_prob <= 0 or model_prob >= 1:
        return {"stake": 0.0, "fraction": 0.0, "reason": "invalid_inputs"}

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    f_star = (b * p - q) / b
    if f_star <= 0:
        return {"stake": 0.0, "fraction": 0.0, "reason": "negative_edge"}

    f_quarter = f_star / 4.0
    ece_penalty = max(0.0, 1.0 - ece * 3.0)
    stability_penalty = max(0.3, 1.0 - fold_std)
    f_penalized = f_quarter * ece_penalty * stability_penalty
    max_fraction_cap = min(f_penalized, max_fraction)
    stake = min(max_fraction_cap * bankroll, max_units)

    return {
        "stake": round(stake, 2), "fraction": round(f_penalized, 4),
        "full_kelly": round(f_star, 4), "quarter_kelly": round(f_quarter, 4),
        "ece_penalty": round(ece_penalty, 4), "stability_penalty": round(stability_penalty, 4),
        "bankroll_pct": round(f_penalized * 100, 2), "reason": "bet"
    }
