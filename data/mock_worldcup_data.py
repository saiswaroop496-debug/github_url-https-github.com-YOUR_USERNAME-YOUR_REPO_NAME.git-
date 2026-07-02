import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataGenerator:
    def __init__(self):
        # Actual 48 Qualified Teams for 2026 FIFA World Cup (Groups A-L)
        self.teams = [
            "USA", "Canada", "Mexico",
            "South Africa", "South Korea", "Czech Republic",
            "Switzerland", "Qatar", "Bosnia and Herzegovina",
            "Brazil", "Morocco", "Scotland", "Haiti",
            "Paraguay", "Australia", "Turkey",
            "Germany", "Curacao", "Ivory Coast", "Ecuador",
            "Netherlands", "Japan", "Sweden", "Tunisia",
            "Belgium", "Egypt", "Iran", "New Zealand",
            "Spain", "Uruguay", "Cape Verde", "Saudi Arabia",
            "France", "Senegal", "Iraq", "Norway",
            "Argentina", "Algeria", "Austria", "Jordan",
            "Portugal", "DR Congo", "Uzbekistan", "Colombia",
            "England", "Croatia", "Ghana", "Panama"
        ]

        # Realistic Base Strengths for Mocking (Attack, Defense (lower is better), Squad Value €M)
        self.TRUE_STRENGTHS = {
            # TIER 1
            "France": {"att": 2.2, "def": 0.7, "value_m": 1200},
            "England": {"att": 2.1, "def": 0.7, "value_m": 1300},
            "Brazil": {"att": 2.1, "def": 0.8, "value_m": 1100},
            "Argentina": {"att": 2.0, "def": 0.7, "value_m": 900},
            "Spain": {"att": 2.0, "def": 0.8, "value_m": 950},
            "Portugal": {"att": 2.1, "def": 0.8, "value_m": 1050},
            "Germany": {"att": 1.9, "def": 0.9, "value_m": 850},
            "Netherlands": {"att": 1.8, "def": 0.8, "value_m": 700},
            "Belgium": {"att": 1.8, "def": 1.0, "value_m": 500},
            
            # TIER 2
            "Uruguay": {"att": 1.6, "def": 0.9, "value_m": 450},
            "Colombia": {"att": 1.5, "def": 0.9, "value_m": 300},
            "Croatia": {"att": 1.4, "def": 1.0, "value_m": 350},
            "Switzerland": {"att": 1.4, "def": 1.0, "value_m": 300},
            "Senegal": {"att": 1.5, "def": 1.1, "value_m": 250},
            "Morocco": {"att": 1.4, "def": 0.8, "value_m": 320},
            "USA": {"att": 1.5, "def": 1.1, "value_m": 350},
            "Mexico": {"att": 1.4, "def": 1.2, "value_m": 200},
            "Japan": {"att": 1.4, "def": 1.1, "value_m": 280},
            "Ecuador": {"att": 1.3, "def": 1.0, "value_m": 220},
            "Austria": {"att": 1.5, "def": 1.1, "value_m": 250},
            "Turkey": {"att": 1.5, "def": 1.2, "value_m": 200},
            
            # TIER 3
            "South Korea": {"att": 1.3, "def": 1.2, "value_m": 150},
            "Ivory Coast": {"att": 1.3, "def": 1.2, "value_m": 220},
            "Sweden": {"att": 1.3, "def": 1.1, "value_m": 280},
            "Norway": {"att": 1.5, "def": 1.2, "value_m": 350},
            "Czech Republic": {"att": 1.2, "def": 1.2, "value_m": 150},
            "Algeria": {"att": 1.4, "def": 1.3, "value_m": 180},
            "Egypt": {"att": 1.3, "def": 1.2, "value_m": 140},
            "Scotland": {"att": 1.1, "def": 1.2, "value_m": 200},
            "Canada": {"att": 1.3, "def": 1.4, "value_m": 180},
            "Paraguay": {"att": 1.1, "def": 1.1, "value_m": 120},
            "Iran": {"att": 1.2, "def": 1.1, "value_m": 60},
            "Australia": {"att": 1.1, "def": 1.2, "value_m": 50},
            "Saudi Arabia": {"att": 1.0, "def": 1.3, "value_m": 40},
            "Tunisia": {"att": 1.0, "def": 1.2, "value_m": 50},
            
            # TIER 4 / DEBUTANTS
            "Ghana": {"att": 1.2, "def": 1.4, "value_m": 150},
            "Bosnia and Herzegovina": {"att": 1.1, "def": 1.4, "value_m": 80},
            "South Africa": {"att": 1.0, "def": 1.4, "value_m": 40},
            "Panama": {"att": 1.0, "def": 1.5, "value_m": 25},
            "Qatar": {"att": 0.9, "def": 1.6, "value_m": 20},
            "New Zealand": {"att": 0.8, "def": 1.5, "value_m": 30},
            "Uzbekistan": {"att": 1.0, "def": 1.4, "value_m": 35},
            "Iraq": {"att": 0.9, "def": 1.4, "value_m": 15},
            "Jordan": {"att": 0.8, "def": 1.5, "value_m": 10},
            "DR Congo": {"att": 1.1, "def": 1.4, "value_m": 80},
            "Cape Verde": {"att": 0.9, "def": 1.5, "value_m": 25},
            "Haiti": {"att": 0.7, "def": 1.8, "value_m": 15},
            "Curacao": {"att": 0.7, "def": 1.8, "value_m": 15}
        }
        
    def _get_city_host(self, home_team, away_team):
        hosts = ["USA", "Canada", "Mexico"]
        if home_team in hosts: return home_team
        if away_team in hosts: return away_team
        return np.random.choice(hosts)

    def _get_crowd_factor(self, team1, team2, city_host):
        CONCACAF_TEAMS = ["USA", "Canada", "Mexico", "Panama"]
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
            
            # Fetch TRUE strengths
            home_att = self.TRUE_STRENGTHS[home_team]['att']
            home_def = self.TRUE_STRENGTHS[home_team]['def']
            away_att = self.TRUE_STRENGTHS[away_team]['att']
            away_def = self.TRUE_STRENGTHS[away_team]['def']
            
            # Determine Venue
            city_host = self._get_city_host(home_team, away_team)
            home_cf = self._get_crowd_factor(home_team, away_team, city_host)
            away_cf = self._get_crowd_factor(away_team, home_team, city_host)
            crowd_factor = home_cf - away_cf
            
            # Match Lambda = Base * Attack * Defense * Venue Advantage
            h_lambda = 1.1 * home_att * away_def * (1 + (crowd_factor * 0.3))
            a_lambda = 1.1 * away_att * home_def * (1 - (crowd_factor * 0.3))
            
            # Simulate goals via Poisson
            home_goals = np.random.poisson(h_lambda)
            away_goals = np.random.poisson(a_lambda)
            
            # Simulate xG proportionally with noise
            home_xg = np.clip(np.random.normal(h_lambda, 0.3), 0.05, 5.0)
            away_xg = np.clip(np.random.normal(a_lambda, 0.3), 0.05, 5.0)
            
            # Possession skewed toward team with higher squad value and attack
            h_val = self.TRUE_STRENGTHS[home_team]['value_m']
            a_val = self.TRUE_STRENGTHS[away_team]['value_m']
            
            base_possession = 50 + (np.log(h_val+1) - np.log(a_val+1)) * 5
            home_possession = np.clip(np.random.normal(base_possession, 5), 25, 75)
            away_possession = 100 - home_possession
            
            home_shots = int(np.clip(np.random.normal(home_xg * 8, 3), 1, 30))
            away_shots = int(np.clip(np.random.normal(away_xg * 8, 3), 1, 30))
            
            home_yellow = np.random.poisson(1.5)
            away_yellow = np.random.poisson(1.5)
            home_corners = np.random.poisson(4.5 + home_possession / 20.0)
            away_corners = np.random.poisson(4.5 + away_possession / 20.0)
            
            is_neutral = True
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
                'home_yellow': home_yellow,
                'away_yellow': away_yellow,
                'home_corners': home_corners,
                'away_corners': away_corners,
                'is_neutral': is_neutral,
                'tournament_stage': stage,
                'tournament_weight': weight,
                'host_proximity': city_host,
                'crowd_factor': crowd_factor,
                'home_squad_value_m': h_val,
                'away_squad_value_m': a_val
            })
            
            current_date += timedelta(days=np.random.randint(1, 4))
            
        return pd.DataFrame(data)

if __name__ == "__main__":
    generator = MockDataGenerator()
    df = generator.generate(400)
    df.to_csv("worldcup_matches.csv", index=False)
    print(f"Generated {len(df)} matches to worldcup_matches.csv")
