import pandas as pd
import numpy as np

def compute_rolling_features(df):
    """
    Computes rolling features with STRICT shift(1) anti-leakage protocol.
    """
    df = df.sort_values('date').copy()
    
    # 1. xG Rolling and Std
    for side in ['home', 'away']:
        # To compute team's historical xG, we must align by team regardless of venue
        # This requires a more complex grouping. For simplicity and as requested:
        # We will assume df has been shaped or we use a helper. 
        # Actually, the spec implies doing this cleanly. We will create team-based histories.
        pass
        
    # Standard implementation to match spec:
    # Build a long-form history for each team, calculate rolling, then map back to match.
    team_histories = []
    
    for _, row in df.iterrows():
        team_histories.append({'date': row['date'], 'team': row['home_team'], 'xg': row['home_xg'], 'match_id': row['match_id'], 'is_home': True})
        team_histories.append({'date': row['date'], 'team': row['away_team'], 'xg': row['away_xg'], 'match_id': row['match_id'], 'is_home': False})
        
    history_df = pd.DataFrame(team_histories).sort_values(['team', 'date'])
    
    # STRICT ANTI-LEAKAGE
    history_df['xg_rolling_3'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).mean())
    history_df['xg_rolling_std'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).std())
    
    # Form momentum
    def slope(series):
        if len(series.dropna()) < 3: return np.nan
        return np.polyfit([1, 2, 3], series.values, 1)[0]
        
    history_df['xg_form_momentum'] = history_df.groupby('team')['xg'].transform(lambda x: x.shift(1).rolling(3).apply(slope, raw=False))
    
    # Days since last match
    history_df['days_since_last'] = history_df.groupby('team')['date'].transform(lambda x: (x - x.shift(1)).dt.days)
    
    # Map back to df
    for col, side in [('home', True), ('away', False)]:
        mapping = history_df[history_df['is_home'] == side].set_index('match_id')
        df[f'{col}_xg_rolling_3'] = df['match_id'].map(mapping['xg_rolling_3'])
        df[f'{col}_xg_rolling_std'] = df['match_id'].map(mapping['xg_rolling_std'])
        df[f'{col}_xg_form_momentum'] = df['match_id'].map(mapping['xg_form_momentum'])
        df[f'{col}_days_since_last'] = df['match_id'].map(mapping['days_since_last'])
        
    df['home_days_since_last'] = df['home_days_since_last'].fillna(7)
    df['away_days_since_last'] = df['away_days_since_last'].fillna(7)
    df['rest_differential'] = df['home_days_since_last'] - df['away_days_since_last']
    
    df['home_neutral_venue_form'] = np.where(df['is_neutral'], 0.6 * df['away_xg_rolling_3'] + 0.4 * df['home_xg_rolling_3'], df['home_xg_rolling_3'])
    df['away_neutral_venue_form'] = np.where(df['is_neutral'], 0.6 * df['home_xg_rolling_3'] + 0.4 * df['away_xg_rolling_3'], df['away_xg_rolling_3'])
    
    df['xg_ratio'] = df['home_xg_rolling_3'] / (df['away_xg_rolling_3'] + 1e-6)
    
    return df
