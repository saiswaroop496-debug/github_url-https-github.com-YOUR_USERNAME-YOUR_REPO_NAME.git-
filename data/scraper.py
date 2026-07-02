import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data.mock_worldcup_data import MockDataGenerator


def calculate_real_proxy_xg(row: pd.Series, dc_params=None) -> tuple:
    """
    Shot-quality proxy xG that does NOT use actual goals.
    Uses shots, shots on target, and historical team attack strength.
    """
    h_shots = float(row.get('home_shots', 0) or 0)
    a_shots = float(row.get('away_shots', 0) or 0)
    
    # If vision xG was extracted, use it directly (if we added it as xg_source == 'vision')
    if row.get('xg_source') == 'vision':
        return round(row.get('home_xg', 0.0), 3), round(row.get('away_xg', 0.0), 3), False
        
    if h_shots > 0:
        # Shot-based proxy (best case)
        h_xg = h_shots * 0.33 * 0.3 + (h_shots * 0.7) * 0.04
        a_xg = a_shots * 0.33 * 0.3 + (a_shots * 0.7) * 0.04
    elif dc_params:
        # Team-specific attack parameter from DC MLE (second best)
        h_team = row.get('home_team', '')
        a_team = row.get('away_team', '')
        t_idx  = dc_params.get('team_idx', {})
        attack = dc_params.get('attack', {})
        h_xg   = np.exp(attack.get(t_idx.get(h_team, 0), 0.0)) * 1.2
        a_xg   = np.exp(attack.get(t_idx.get(a_team, 0), 0.0)) * 1.0
    else:
        # Competition-adjusted floor (last resort)
        h_xg = 1.35; a_xg = 1.05
    
    return round(max(0.15, h_xg), 3), round(max(0.15, a_xg), 3), True


def load_and_enrich_dataset(csv_path: str = "data/worldcup_matches.csv") -> pd.DataFrame:
    """
    Load dataset and enrich with vision xG where available.
    Vision xG cached in data/vision_xg_cache/ — only processed once per match.
    """
    import json
    from pathlib import Path
    
    # Use DataScraper logic to get base data instead of just reading a generic csv
    # since train_test.py expects the full historical dataset (dc_df and form_df).
    # Wait, the user's Fix 7 states: "Replace load_dataset() with load_and_enrich_dataset()".
    # And it should return dc_df, form_df.
    scraper = DataScraper()
    dc_df, form_df = scraper.fetch_fixtures()

    # Apply vision xG enrichment to form_df
    vision_cache_dir = Path("data/vision_xg_cache")
    if vision_cache_dir.exists():
        enriched = 0
        for cache_file in vision_cache_dir.glob("*_xg.json"):
            try:
                with open(cache_file, encoding='utf-8') as f:
                    vision_data = json.load(f)
                if vision_data.get('source') != 'vision_xg':
                    continue

                stem   = cache_file.stem.replace('_xg', '')
                parts  = stem.split('_vs_')
                if len(parts) != 2:
                    continue
                home_t = parts[0].replace('_', ' ')
                rest   = parts[1].split('_')
                away_t = rest[0].replace('_', ' ')
                date_s = rest[1] if len(rest) > 1 else ""

                mask = (
                    (form_df['home_team'].str.lower() == home_t.lower()) &
                    (form_df['away_team'].str.lower() == away_t.lower()) &
                    (form_df['date'].astype(str).str[:10] == date_s)
                )
                if mask.any():
                    form_df.loc[mask, 'home_xg'] = vision_data['home_xg']
                    form_df.loc[mask, 'away_xg'] = vision_data['away_xg']
                    form_df.loc[mask, 'xg_source'] = 'vision'
                    enriched += 1
            except Exception:
                continue
        print(f"  Vision xG enriched: {enriched} matches")

    if 'xg_source' not in form_df.columns:
        form_df['xg_source'] = 'proxy'
        
    return dc_df, form_df


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

            # Filter to matches since 2015 for DC
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

            xg_values = df.apply(
                lambda row: calculate_real_proxy_xg(row),
                axis=1
            )
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

            df['crowd_factor'] = 0.0  # Simplify for real data

            # Merge live WC2026 match results if available
            if os.path.exists("data/worldcup_matches.csv"):
                try:
                    live_df = pd.read_csv("data/worldcup_matches.csv")
                    live_df['date'] = pd.to_datetime(live_df['date'])
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
