import pandas as pd
import numpy as np

def h2h_draw_rate_fast(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    # Create a canonical match key (alphabetically sorted teams)
    df['h2h_key'] = df.apply(
        lambda r: '|'.join(sorted([r['home_team'], r['away_team']])), axis=1
    )
    # Expanding draw count per h2h pair (shift to avoid leakage)
    df['is_draw'] = (df['home_goals'] == df['away_goals']).astype(int)
    
    # Calculate sum and count shifted
    grouped = df.groupby('h2h_key')['is_draw']
    h2h_sum = grouped.transform(lambda x: x.shift(1).expanding().sum())
    h2h_count = grouped.transform(lambda x: x.shift(1).expanding().count())
    
    df['h2h_draw_rate'] = np.where(
        h2h_count >= 4,
        h2h_sum / h2h_count,
        0.29 # global international draw prior
    )
    df.drop(columns=['h2h_key', 'is_draw'], inplace=True)
    return df

def compute_rolling_features(df):
    """
    Computes rolling features with STRICT shift(1) anti-leakage protocol.
    """
    df = df.sort_values('date').copy()
    
    if 'stage' not in df.columns:
        df['stage'] = 'group-stage'
        
    team_histories = []
    
    for _, row in df.iterrows():
        is_draw = int(row['home_goals'] == row['away_goals'])
        team_histories.append({'date': row['date'], 'team': row['home_team'], 'xg': row['home_xg'], 'conceded': row['away_goals'], 'is_draw': is_draw, 'match_id': row['match_id'], 'is_home': True})
        team_histories.append({'date': row['date'], 'team': row['away_team'], 'xg': row['away_xg'], 'conceded': row['home_goals'], 'is_draw': is_draw, 'match_id': row['match_id'], 'is_home': False})
        
    history_df = pd.DataFrame(team_histories).sort_values(['team', 'date'])
    
    # STRICT ANTI-LEAKAGE
    history_df['xg_rolling_3'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).mean())
    history_df['xg_rolling_std'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).std())
    history_df['goals_conceded_roll3'] = history_df.groupby('team')['conceded'].transform(lambda x: x.shift(1).rolling(3).mean())
    history_df['draw_rate'] = history_df.groupby('team')['is_draw'].transform(lambda x: x.shift(1).rolling(10).mean().fillna(0.27))
    
    # Form momentum
    def slope(series):
        if len(series.dropna()) < 3: return np.nan
        return np.polyfit([1, 2, 3], series.values, 1)[0]
        
    history_df['xg_form_momentum'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).apply(slope, raw=False))
    
    # Days since last match
    history_df['days_since_last'] = history_df.groupby('team')['date'].transform(lambda x: (x - x.shift(1)).dt.days)
    
    # Map back to df
    for col, side in [('home', True), ('away', False)]:
        mapping = history_df[history_df['is_home'] == side].drop_duplicates('match_id').set_index('match_id')
        df[f'{col}_xg_rolling_3'] = df['match_id'].map(mapping['xg_rolling_3'])
        df[f'{col}_xg_rolling_std'] = df['match_id'].map(mapping['xg_rolling_std'])
        df[f'{col}_xg_form_momentum'] = df['match_id'].map(mapping['xg_form_momentum'])
        df[f'{col}_days_since_last'] = df['match_id'].map(mapping['days_since_last'])
        df[f'{col}_goals_conceded_roll3'] = df['match_id'].map(mapping['goals_conceded_roll3'])
        df[f'{col}_draw_rate'] = df['match_id'].map(mapping['draw_rate'])
        
    df['home_days_since_last'] = df['home_days_since_last'].fillna(7)
    df['away_days_since_last'] = df['away_days_since_last'].fillna(7)
    df['rest_differential'] = df['home_days_since_last'] - df['away_days_since_last']
    
    df['defensive_balance'] = (df['home_goals_conceded_roll3'].fillna(1.0) + df['away_goals_conceded_roll3'].fillna(1.0)) / 2
    
    stage_weights = {
        'group-stage': 0.0,
        'round-of-32': 0.4,
        'round-of-16': 0.6,
        'quarter-final': 0.8,
        'semi-final': 0.9,
        'third-place': 0.5,
        'final': 1.0
    }
    df['stage_pressure'] = df['stage'].map(stage_weights).fillna(0.3)
    
    df['team1_neutral_xg'] = df['home_xg_rolling_3'] # Since all matches are neutral effectively
    df['team2_neutral_xg'] = df['away_xg_rolling_3']
    
    df['xg_ratio'] = df['team1_neutral_xg'] / (df['team2_neutral_xg'] + 1e-6)
    
    # Add H2H draw rate
    df = h2h_draw_rate_fast(df)
    
    return df

def compute_v6_features(df):
    """
    Adds V6.1 specific features (draw_affinity, xg_supremacy, glicko_signal)
    Must be called AFTER Glicko-2 ratings are computed since it uses those fields.
    """
    df = df.copy()
    
    # 1. Draw Affinity: historical draw rates
    if 'home_draw_rate' in df.columns and 'away_draw_rate' in df.columns:
        df['draw_affinity'] = (df['home_draw_rate'] + df['away_draw_rate']) / 2
    else:
        df['draw_affinity'] = 0.27

    # 2. xG Supremacy Index: relative dominance
    total_xg = df['home_xg_rolling_3'] + df['away_xg_rolling_3'] + 1e-9
    df['xg_supremacy'] = df['home_xg_rolling_3'] / total_xg  # 0.5 = even match

    # 3. Glicko Signal: strength gap scaled by combined uncertainty
    if 'home_rd' in df.columns and 'away_rd' in df.columns:
        rd_combined = np.sqrt(df['home_rd']**2 + df['away_rd']**2) + 1e-9
        df['glicko_signal'] = (df['home_glicko'] - df['away_glicko']) / rd_combined
    else:
        df['glicko_signal'] = 0.0

    return df

import os
import requests
import warnings

def add_sharp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sharp Line Movement: change in implied probability from opening to current odds.
    Positive = money coming in on Home (sharp backing Home).
    """
    if 'opening_odds_home' not in df.columns or 'current_odds_home' not in df.columns:
        df['sharp_line_movement'] = 0.0
        return df

    # Convert to implied probability (no overround correction needed here — direction matters)
    df['sharp_line_movement'] = (
        (1.0 / df['opening_odds_home'].clip(1.01)) -
        (1.0 / df['current_odds_home'].clip(1.01))
    ).fillna(0.0)

    return df


def squad_availability_score(team_id: int, fixture_id: int,
                               api_key: str = None) -> float:
    """
    Fetch lineup data from API-Football and compute Squad Availability Score.
    Returns 0.0 (many key absences) to 1.0 (full squad).
    Gracefully defaults to 1.0 if API unavailable or key missing.

    Key player weight: GK=0.12, CB=0.08, CM=0.07, ST=0.10, others=0.05
    """
    POSITION_WEIGHTS = {
        'G': 0.12,   # Goalkeeper
        'D': 0.08,   # Defender
        'M': 0.07,   # Midfielder
        'F': 0.10,   # Forward
    }
    DEFAULT_WEIGHT = 0.05

    if not api_key:
        api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return 1.0   # graceful degradation — full squad assumed

    try:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers,
                                params={"fixture": fixture_id},
                                timeout=10)
        data = response.json()

        if not data.get('response'):
            return 1.0   # no lineup data yet

        # Find the matching team
        for team_data in data['response']:
            if team_data['team']['id'] != team_id:
                continue

            # Get expected 11 from startXI
            start_xi = team_data.get('startXI', [])
            if len(start_xi) < 10:
                return 0.85   # incomplete lineup, penalise slightly

            # Compute weighted availability
            score = 0.0
            for player in start_xi:
                pos = player['player'].get('pos', 'X')
                weight = POSITION_WEIGHTS.get(pos, DEFAULT_WEIGHT)
                score += weight

            # Normalize to [0, 1]
            return min(1.0, score / 1.0)
            
        return 1.0
    except Exception as e:
        warnings.warn(f"SAS fetch failed: {e}. Defaulting to 1.0")
        return 1.0


def add_squad_features(df: pd.DataFrame, api_key: str = None) -> pd.DataFrame:
    """Add SAS columns to dataframe. Uses cached values if fixture_id repeated."""
    cache = {}
    sas_home, sas_away = [], []

    for _, row in df.iterrows():
        fid = row.get('fixture_id', None)
        htid = row.get('home_team_id', None)
        atid = row.get('away_team_id', None)

        sh = cache.get((htid, fid), squad_availability_score(htid, fid, api_key))
        sa = cache.get((atid, fid), squad_availability_score(atid, fid, api_key))
        cache[(htid, fid)] = sh
        cache[(atid, fid)] = sa
        sas_home.append(sh)
        sas_away.append(sa)

    df['home_sas'] = sas_home
    df['away_sas'] = sas_away
    df['sas_differential'] = df['home_sas'] - df['away_sas']
    return df
