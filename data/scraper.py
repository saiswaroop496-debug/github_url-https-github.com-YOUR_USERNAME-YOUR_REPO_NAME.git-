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
            
            def calculate_xg(goals):
                return np.clip(np.random.normal(goals + 0.1, 0.3), 0.05, 5.0)
                
            df['home_xg'] = df['home_goals'].apply(calculate_xg)
            df['away_xg'] = df['away_goals'].apply(calculate_xg)
            
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
            dc_df = df.copy()
            form_df = df[df['date'].dt.year >= 2018].copy()
            
            return dc_df, form_df
        except Exception as e:
            print("Failed to load real data, falling back to mock:", e)
            mock_df = self.mock_gen.generate()
            return mock_df, mock_df

    def fetch_fixture_statistics(self, fixture_id):
        return []
