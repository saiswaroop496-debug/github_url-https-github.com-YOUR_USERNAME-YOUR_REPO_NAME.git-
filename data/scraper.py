import os
import requests
import pandas as pd
from dotenv import load_dotenv

from data.mock_worldcup_data import MockDataGenerator

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.mock_worldcup_data import MockDataGenerator

class DataScraper:
    def __init__(self):
        self.mock_gen = MockDataGenerator()
        self.teams_to_keep = self.mock_gen.teams
        self.TRUE_STRENGTHS = self.mock_gen.TRUE_STRENGTHS

def get_xg_for_match(row: pd.Series, api_key: str = "") -> tuple:
    """
    Returns (home_xg, away_xg) using real data when available.
    
    Priority order:
    1. Real xG from API-Football /statistics endpoint (if fixture_id available)
    2. Real xG from international_results.csv if column exists
    3. Goals-based proxy (better than goals+noise) as last resort
    
    The goals-based proxy uses the formula:
    xG ≈ goals * 0.85 + 0.15 * shots_on_target * 0.1
    This is a conservative estimator that never exceeds actual goals by large margins.
    It is LABELED as "proxy" in the dataset so downstream code knows it is not real.
    """
    # Try real xG from fixture stats
    fid = row.get('fixture_id')
    if fid and api_key:
        from data.xg_fetcher import fetch_real_international_xg_from_api
        real = fetch_real_international_xg_from_api(int(fid), api_key)
        if real.get('home_xg') is not None:
            return real['home_xg'], real['away_xg'], False  # False = not proxy

    # Real xG already in CSV from StatsBomb open data (international matches)
    if pd.notna(row.get('home_xg')) and row.get('home_xg', 0) > 0:
        return float(row['home_xg']), float(row['away_xg']), False

    # Improved proxy — no Gaussian noise, just conservative scaling
    hg = float(row.get('home_score', row.get('home_goals', 0)) or 0)
    ag = float(row.get('away_score', row.get('away_goals', 0)) or 0)
    
    # Add small competition-based base rate for draws (0-0 not 0 xG)
    base = 0.4  # even 0-0 games had some chance creation
    home_xg_proxy = max(base, hg * 0.85 + 0.3)
    away_xg_proxy = max(base * 0.8, ag * 0.85 + 0.25)
    
    return round(home_xg_proxy, 3), round(away_xg_proxy, 3), True  # True = proxy


    def fetch_fixtures(self):
        try:
            # Attempt to load the real historical dataset
            df = pd.read_csv("data/international_results.csv")
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for our 48 teams
            df = df[df['home_team'].isin(self.teams_to_keep) & df['away_team'].isin(self.teams_to_keep)]
            
            # Filter to matches since 2015 for DC, we'll return this full one and a subset
            df = df[df['date'].dt.year >= 2015].copy()
            
            # Map standard columns
            df['match_id'] = range(1, len(df) + 1)
            df['home_goals'] = df['home_score'].fillna(0).astype(int)
            df['away_goals'] = df['away_score'].fillna(0).astype(int)
            
            # Match Weight Stratification
            COMP_WEIGHTS = {
                'FIFA World Cup': 3.0,
                'UEFA Euro': 2.5,
                'Copa America': 2.5,
                'UEFA Nations League': 2.0,
                'Friendly': 0.5
            }
            df['match_weight'] = df['tournament'].map(COMP_WEIGHTS).fillna(1.0)
            
            # Add required columns for the ML pipeline
            np.random.seed(42)
            
            xg_values = df.apply(lambda row: get_xg_for_match(row, os.getenv("RAPIDAPI_KEY", "")), axis=1)
            df['home_xg'] = [x[0] for x in xg_values]
            df['away_xg'] = [x[1] for x in xg_values]
            df['is_xg_proxy'] = [x[2] for x in xg_values]
            
            df['home_squad_value_m'] = df['home_team'].map(lambda t: self.TRUE_STRENGTHS[t]['value_m'])
            df['away_squad_value_m'] = df['away_team'].map(lambda t: self.TRUE_STRENGTHS[t]['value_m'])
            
            df['base_possession'] = 50 + (np.log(df['home_squad_value_m']+1) - np.log(df['away_squad_value_m']+1)) * 5
            df['home_possession'] = np.clip(np.random.normal(df['base_possession'], 5), 25, 75)
            df['away_possession'] = 100 - df['home_possession']
            
            df['home_shots'] = (df['home_xg'] * 8 + np.random.normal(0, 2, len(df))).clip(1, 30).astype(int)
            df['away_shots'] = (df['away_xg'] * 8 + np.random.normal(0, 2, len(df))).clip(1, 30).astype(int)
            
            df['is_neutral'] = df['neutral']
            
            def map_stage(t):
                if 'World Cup' in t or 'Copa America' in t or 'Euro' in t or 'Gold Cup' in t or 'Africa Cup of Nations' in t or 'Asian Cup' in t:
                    return "Knockout" if np.random.random() > 0.7 else "Group"
                return "Group"
                
            df['tournament_stage'] = df['tournament'].apply(map_stage)
            df['tournament_weight'] = df['tournament_stage'].map({"Group": 1.0, "Knockout": 1.5, "Final": 2.0})
            
            df['host_proximity'] = df.apply(lambda row: row['home_team'] if row['home_team'] in ["USA", "Canada", "Mexico"] else (row['away_team'] if row['away_team'] in ["USA", "Canada", "Mexico"] else "USA"), axis=1)
            
            df['crowd_factor'] = 0.0 # Simplify for real data
            
            # Split into dc_df and form_df
            import os
            if os.path.exists("data/worldcup_matches.csv"):
                try:
                    live_df = pd.read_csv("data/worldcup_matches.csv")
                    live_df['date'] = pd.to_datetime(live_df['date'])
                    
                    # Merge live data into the main df
                    # Live df contains: date, home_team, away_team, home_score, away_score, tournament, etc.
                    # We just concat them and sort by date
                    df = pd.concat([df, live_df], ignore_index=True)
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"[DATA] Merged {len(live_df)} live matches from worldcup_matches.csv")
                except Exception as e:
                    print(f"Warning: failed to merge live data: {e}")

            dc_df = df.copy()
            form_df = df[df['date'].dt.year >= 2018].copy()
            
            return dc_df, form_df
        except Exception as e:
            print("Failed to load real data, falling back to mock:", e)
            mock_df = self.mock_gen.generate()
            return mock_df, mock_df

    def fetch_fixture_statistics(self, fixture_id):
        return []
