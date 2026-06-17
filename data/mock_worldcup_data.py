import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataGenerator:
    def __init__(self):
        # 48 Qualified Teams for 2026 World Cup
        self.teams = [
            "USA", "Canada", "Mexico", # Hosts
            "Germany", "France", "England", "Spain", "Portugal", "Netherlands",
            "Belgium", "Italy", "Croatia", "Switzerland", "Austria", "Denmark", 
            "Serbia", "Poland", "Scotland", "Turkey", # UEFA
            "Brazil", "Argentina", "Colombia", "Uruguay", "Ecuador", "Chile", # CONMEBOL
            "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon", "Ivory Coast",
            "Algeria", "Tunisia", "Ghana", # CAF
            "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
            "Qatar", "Iraq", "Jordan", # AFC
            "Panama", "Costa Rica", "Honduras", # CONCACAF
            "New Zealand" # OFC
        ]
        
    def _get_city_host(self, home_team, away_team):
        hosts = ["USA", "Canada", "Mexico"]
        if home_team in hosts: return home_team
        if away_team in hosts: return away_team
        return np.random.choice(hosts)

    def _get_crowd_factor(self, team1, team2, city_host):
        CONCACAF_TEAMS = ["USA", "Canada", "Mexico", "Panama", "Costa Rica", "Honduras"]
        HOST_CROWD_FACTORS = {"USA": 0.55, "Mexico": 0.60, "Canada": 0.50}
        
        is_1_concacaf = team1 in CONCACAF_TEAMS
        is_2_concacaf = team2 in CONCACAF_TEAMS
        
        # Both CONCACAF — split crowd
        if is_1_concacaf and is_2_concacaf:
            return 0.2
        
        # Team IS the host nation of this city
        if team1 == city_host:
            return HOST_CROWD_FACTORS[city_host]
        
        # CONCACAF team playing in a different host's city
        if is_1_concacaf and team1 != city_host:
            return 0.15
        
        # All other cases — pure neutral
        return 0.0

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
            
            is_neutral = True
            stage = np.random.choice(["Group", "Knockout", "Final"], p=[0.7, 0.25, 0.05])
            
            weight = 1.0
            if stage == "Knockout": weight = 1.5
            elif stage == "Final": weight = 2.0
            
            city_host = self._get_city_host(home_team, away_team)
            
            home_cf = self._get_crowd_factor(home_team, away_team, city_host)
            away_cf = self._get_crowd_factor(away_team, home_team, city_host)
            crowd_factor = home_cf - away_cf
            
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
                'tournament_weight': weight,
                'host_proximity': city_host,
                'crowd_factor': crowd_factor # positive favors home, negative favors away
            })
            
            current_date += timedelta(days=np.random.randint(1, 4))
            
        return pd.DataFrame(data)
