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

def add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build injury features from home_injuries/away_injuries columns.
    Uses .shift(1) per team group to prevent target leakage.
    Gracefully defaults to 0 if columns are absent.
    """
    for col in ['home_injuries', 'away_injuries', 'home_key_injury', 'away_key_injury']:
        if col not in df.columns:
            df[col] = 0

    # Shift injury counts per team so match N uses injuries from match N-1
    df = df.sort_values('date').reset_index(drop=True)
    df['home_injuries_lagged']    = df.groupby('home_team')['home_injuries'].shift(1).fillna(0)
    df['away_injuries_lagged']    = df.groupby('away_team')['away_injuries'].shift(1).fillna(0)
    df['home_key_injury_lagged']  = df.groupby('home_team')['home_key_injury'].shift(1).fillna(0)
    df['away_key_injury_lagged']  = df.groupby('away_team')['away_key_injury'].shift(1).fillna(0)

    df['injury_differential'] = df['away_injuries_lagged'] - df['home_injuries_lagged']
    df['key_injury_factor']   = (
        -0.15 * df['home_key_injury_lagged'] +
         0.15 * df['away_key_injury_lagged']
    )
    return df


def add_movement_features(df: pd.DataFrame,
                           stats_path: str = "data/movement_stats.json") -> pd.DataFrame:
    """
    Merge pre-computed player movement stats into df.
    If stats_path doesn't exist -> all movement cols default to 0.0.
    Applies .shift(1) per team group (anti-leakage).
    """
    import json
    from pathlib import Path
    
    movement_cols = ['speed_diff', 'home_total_distance_m',
                     'away_total_distance_m', 'home_total_sprints', 'away_total_sprints']

    for col in movement_cols:
        df[col] = 0.0

    if not Path(stats_path).exists():
        return df   # graceful fallback � all zeros

    with open(stats_path) as f:
        movement = json.load(f)

    lookup = {}
    for key, stats in movement.items():
        parts = key.split("_vs_")
        if len(parts) == 2:
            home = parts[0]
            rest = parts[1].split("_")
            away = rest[0]
            date = rest[1] if len(rest) > 1 else ""
            lookup[(home, away, date)] = stats

    for idx, row in df.iterrows():
        k = (row.get('home_team',""), row.get('away_team',""), str(row.get('date',""))[:10])
        if k in lookup:
            s = lookup[k]
            df.at[idx, 'speed_diff']              = s.get('speed_diff', 0.0)
            df.at[idx, 'home_total_distance_m']   = s.get('home_total_distance_m', 0.0)
            df.at[idx, 'away_total_distance_m']   = s.get('away_total_distance_m', 0.0)
            df.at[idx, 'home_total_sprints']      = s.get('home_total_sprints', 0)
            df.at[idx, 'away_total_sprints']      = s.get('away_total_sprints', 0)

    # Anti-leakage shift: physical data from match N-1 predicts match N
    for col in movement_cols:
        df[col] = df.groupby('home_team')[col].shift(1).fillna(0)

    return df


def add_institutional_signals(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['date_dt'] = pd.to_datetime(df['date'])

    def get_past_matches(team, current_idx, n=None, days=None, current_date=None):
        mask = ((df['home_team'] == team) | (df['away_team'] == team)) & (df.index < current_idx)
        past = df[mask]
        if days is not None and current_date is not None:
            past = past[past['date_dt'] >= (current_date - pd.Timedelta(days=days))]
        if n is not None:
            past = past.tail(n)
        return past

    def get_team_goals(past, team):
        home_mask = past['home_team'] == team
        goals = np.where(home_mask, past['home_goals'], past['away_goals'])
        conc = np.where(home_mask, past['away_goals'], past['home_goals'])
        return goals, conc

    for side in ['home', 'away']:
        team_col = f'{side}_team'
        
        momentum = []
        velocity = []
        conv_rates = []
        def_index = []
        
        for idx, row in df.iterrows():
            team = row[team_col]
            current_date = row['date_dt']
            
            # Momentum (net goals in last 2 matches)
            past2 = get_past_matches(team, idx, n=2)
            if len(past2) < 2:
                momentum.append(0.0)
            else:
                goals, conc = get_team_goals(past2, team)
                momentum.append(float(np.sum(goals) - np.sum(conc)))
                
            # Velocity (30 day Glicko change)
            glicko_col = f'{side}_glicko'
            past_30d = get_past_matches(team, idx, days=30, current_date=current_date)
            if len(past_30d) < 2:
                velocity.append(0.0)
            else:
                # Need to find the glicko of this team in the past match
                past_match = past_30d.iloc[0]
                past_glicko = past_match['home_glicko'] if past_match['home_team'] == team else past_match['away_glicko']
                delta = float(row.get(glicko_col, 1500.0)) - float(past_glicko)
                velocity.append(delta / 30.0)
                
            # Conversion rate (actual goals / xG in last 3)
            past3 = get_past_matches(team, idx, n=3)
            if len(past3) < 2:
                conv_rates.append(1.0)
                def_index.append(1.0)
            else:
                goals, conc = get_team_goals(past3, team)
                
                home_mask = past3['home_team'] == team
                # xG
                xg = np.where(home_mask, past3.get('home_xg', pd.Series([1.2]*len(past3))), 
                                         past3.get('away_xg', pd.Series([1.2]*len(past3))))
                # Conceded shots / xG
                opp_shots = np.where(home_mask, past3.get('away_shots', pd.Series([10.0]*len(past3))), 
                                                past3.get('home_shots', pd.Series([10.0]*len(past3))))
                opp_xg = np.where(home_mask, past3.get('away_xg', pd.Series([1.2]*len(past3))), 
                                             past3.get('home_xg', pd.Series([1.2]*len(past3))))
                
                rate = np.sum(goals) / max(np.sum(xg), 0.5)
                conv_rates.append(float(np.clip(rate, 0.3, 2.5)))
                
                idx_val = np.sum(opp_shots) / max(np.sum(opp_xg) * 10, 1.0)
                def_index.append(float(np.clip(idx_val, 0.3, 3.0)))
                
        df[f'{side}_tournament_momentum'] = momentum
        df[f'{side}_glicko_velocity'] = velocity
        df[f'{side}_xg_conversion'] = conv_rates
        df[f'{side}_defensive_shape'] = def_index
        
        squad_col = f'{side}_sas'
        df[f'{side}_lineup_continuity'] = df[squad_col].fillna(1.0) if squad_col in df.columns else 1.0

    df['tournament_momentum_diff'] = df['home_tournament_momentum'] - df['away_tournament_momentum']
    df['glicko_velocity_diff'] = df['home_glicko_velocity'] - df['away_glicko_velocity']
    df['conversion_diff'] = df['home_xg_conversion'] - df['away_xg_conversion']
    df['defensive_shape_diff'] = df['home_defensive_shape'] - df['away_defensive_shape']
    df['lineup_continuity_diff'] = df['home_lineup_continuity'] - df['away_lineup_continuity']

    df.drop(columns=['date_dt'], inplace=True)
    return df
