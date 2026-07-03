"""
Two Sigma-style regime detection for football team form.
Detects: Trending (momentum), Mean-Reverting (regression), or Random.
"""
import numpy as np
from scipy.stats import linregress

def detect_team_regime(team: str, df, lookback: int = 8) -> dict:
    """
    Detect whether a team is in momentum or mean-reversion regime.
    Returns regime label and strength coefficient for bet sizing.
    """
    team_matches = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].sort_values('date').tail(lookback)

    if len(team_matches) < 4:
        return {'regime': 'insufficient_data', 'coefficient': 0.0}

    # Build xG-for series
    xg_for = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            val = row.get('home_xg', 1.2)
            xg_for.append(float(val) if not np.isnan(float(val)) else 1.2)
        else:
            val = row.get('away_xg', 1.0)
            xg_for.append(float(val) if not np.isnan(float(val)) else 1.0)

    xg_arr = np.array(xg_for)

    # Linear trend slope
    slope, _, r_value, p_value, _ = linregress(range(len(xg_arr)), xg_arr)

    # Autocorrelation at lag-1 (positive = momentum, negative = mean reversion)
    if len(xg_arr) > 2:
        autocorr = np.corrcoef(xg_arr[:-1], xg_arr[1:])[0, 1]
    else:
        autocorr = 0.0

    # Hurst exponent approximation (H > 0.5 = trending, H < 0.5 = mean reverting)
    lags = [1, 2]
    tau  = [np.std(np.subtract(xg_arr[l:], xg_arr[:-l])) for l in lags]
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] if all(t > 0 for t in tau) else 0.5

    if hurst > 0.55 and slope > 0.05:
        regime = 'momentum_improving'
        coef = 1.2
    elif hurst > 0.55 and slope < -0.05:
        regime = 'momentum_declining'
        coef = -1.2
    elif hurst < 0.45:
        regime = 'mean_reverting'
        coef = 0.8
    else:
        regime = 'random_walk'
        coef = 1.0

    return {'regime': regime, 'coefficient': coef}
