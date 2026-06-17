import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataGenerator:
    def __init__(self):
        self.teams = [
            "Brazil", "France", "Argentina", "England", "Spain",
            "Germany", "Portugal", "Netherlands", "Italy", "Belgium",
            "Croatia", "Uruguay", "Colombia", "Senegal", "USA",
            "Mexico", "Japan", "Morocco", "Switzerland", "Denmark"
        ]
        
    def generate(self, n_matches=358):
        np.random.seed(42)
        start_date = datetime(2018, 6, 14)
        
        data = []
        current_date = start_date
        
        for i in range(n_matches):
            home_team, away_team = np.random.choice(self.teams, 2, replace=False)
            
            # Simulate goals via Poisson
            home_goals = np.random.poisson(1.35)
            away_goals = np.random.poisson(1.05)
            
            # Simulate xG
            home_xg = np.clip(np.random.normal(home_goals + 0.1, 0.3), 0.05, 5.0)
            away_xg = np.clip(np.random.normal(away_goals + 0.1, 0.3), 0.05, 5.0)
            
            home_possession = np.random.uniform(35, 65)
            away_possession = 100 - home_possession
            
            home_shots = int(np.clip(np.random.normal(home_xg * 8, 3), 1, 30))
            away_shots = int(np.clip(np.random.normal(away_xg * 8, 3), 1, 30))
            
            is_neutral = np.random.choice([True, False], p=[0.85, 0.15])
            stage = np.random.choice(["Group", "Knockout", "Final"], p=[0.7, 0.25, 0.05])
            
            weight = 1.0
            if stage == "Knockout": weight = 1.5
            elif stage == "Final": weight = 2.0
            
            data.append({
                'match_id': i + 1,
                'date': current_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_possession': home_possession,
                'away_possession': away_possession,
                'home_shots': home_shots,
                'away_shots': away_shots,
                'is_neutral': is_neutral,
                'tournament_stage': stage,
                'tournament_weight': weight
            })
            
            current_date += timedelta(days=np.random.randint(1, 4))
            
        return pd.DataFrame(data)
