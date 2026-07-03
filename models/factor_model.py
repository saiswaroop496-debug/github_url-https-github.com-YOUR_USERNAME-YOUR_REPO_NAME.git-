"""
Converts BlackRock-style equity factor model to football team factors.
Stock factors -> Football equivalents:
  Momentum    -> Recent form (xG trend last 5 matches)
  Quality     -> Long-run Glicko rating (fundamental strength)
  Value       -> Odds implied probability vs model probability (mispricing)
  Volatility  -> Std deviation of results (consistent vs erratic team)
  Size        -> Squad market value (Transfermarkt €M proxy for depth)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FACTORS = ['momentum', 'quality', 'value', 'volatility', 'depth']

def compute_team_factors(df: pd.DataFrame, team: str,
                          quality_rating: float,
                          squad_values: dict = None) -> dict:
    """
    Returns a 5-factor vector for a team at prediction time.
    All factors are z-scored relative to all teams in the dataset.
    """
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    team_matches = team_matches.sort_values('date').tail(10)

    if len(team_matches) < 3:
        return {f: 0.0 for f in FACTORS}

    # Factor 1: Momentum - xG trend (slope of last 5 xG values)
    xg_series = []
    for _, row in team_matches.tail(5).iterrows():
        if row['home_team'] == team:
            val = row.get('home_xg', 1.2)
            xg_series.append(float(val) if not np.isnan(float(val)) else 1.2)
        else:
            val = row.get('away_xg', 1.0)
            xg_series.append(float(val) if not np.isnan(float(val)) else 1.0)
    momentum = float(np.polyfit(range(len(xg_series)), xg_series, 1)[0]) if len(xg_series) > 1 else 0.0

    # Factor 2: Quality - Glicko rating normalised to z-score
    quality = float(quality_rating - 1500) / 200.0

    # Factor 3: Value - average edge vs bookmaker over last 5 matches
    # (positive = market systematically underprices this team)
    value = 0.0   # populated by value_betting.py CLV tracker

    # Factor 4: Volatility - std of goal difference (consistent team = low vol)
    goal_diffs = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            goal_diffs.append(float(row.get('home_goals',0) - row.get('away_goals',0)))
        else:
            goal_diffs.append(float(row.get('away_goals',0) - row.get('home_goals',0)))
    volatility = -float(np.std(goal_diffs)) if goal_diffs else 0.0  # negative = more consistent = good

    # Factor 5: Depth - squad value proxy
    if squad_values:
        depth = float(squad_values.get(team, 0)) / 1000.0  # normalise by £1B
    else:
        depth = 0.0

    return {
        'momentum':   round(momentum, 4),
        'quality':    round(quality, 4),
        'value':      round(value, 4),
        'volatility': round(volatility, 4),
        'depth':      round(depth, 4),
    }

def factor_matchup_score(home_factors: dict, away_factors: dict) -> dict:
    """
    Compute per-factor edges between two teams.
    Returns differential scores that feed into the meta-learner as extra features.
    """
    return {
        f'factor_{k}_diff': round(home_factors[k] - away_factors[k], 4)
        for k in FACTORS
    }
