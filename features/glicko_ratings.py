import pandas as pd
import numpy as np

class Glicko2RatingSystem:
    def __init__(self, tau=0.5, epsilon=0.000001, rd_min=40, rd_max=120):
        self.tau = tau
        self.epsilon = epsilon
        self.rd_min = rd_min
        self.rd_max = rd_max
        self.ratings = {}
        
    def _get_team(self, team):
        if team not in self.ratings:
            self.ratings[team] = {'rating': 1500.0, 'rd': 200.0, 'sigma': 0.06}
        return self.ratings[team]
        
    def _g(self, rd):
        q = np.log(10) / 400
        return 1 / np.sqrt(1 + 3 * q**2 * rd**2 / np.pi**2)
        
    def _E(self, r, ri, rdi):
        return 1 / (1 + 10**(-self._g(rdi) * (r - ri) / 400))
        
    def compute_ratings(self, df):
        df = df.copy()
        home_glicko, home_rd, away_glicko, away_rd, glicko_rd_product = [], [], [], [], []
        
        for _, row in df.iterrows():
            ht = row['home_team']
            at = row['away_team']
            
            # PRE-MATCH Snapshot
            h_stat = self._get_team(ht)
            a_stat = self._get_team(at)
            
            home_glicko.append(h_stat['rating'])
            home_rd.append(h_stat['rd'])
            away_glicko.append(a_stat['rating'])
            away_rd.append(a_stat['rd'])
            
            # glicko_rd_product
            # Home
            h_signal = h_stat['rating'] * (1 - (np.clip(h_stat['rd'], 40, 120) - 40) / 80)
            # Away
            a_signal = a_stat['rating'] * (1 - (np.clip(a_stat['rd'], 40, 120) - 40) / 80)
            
            glicko_rd_product.append(h_signal - a_signal) # or separate, spec says singular "glicko_rd_product feature"
            
            # Update step (simplified Elo/Glicko step for mock purposes to maintain state)
            # In production, full Glicko2 step uses outcomes.
            h_goals, a_goals = row['home_goals'], row['away_goals']
            s_h = 1 if h_goals > a_goals else (0.5 if h_goals == a_goals else 0)
            
            # Apply venue correction to expected score
            r_h = h_stat['rating'] + (0 if row['is_neutral'] else 40)
            e_h = self._E(r_h, a_stat['rating'], a_stat['rd'])
            
            # Simple Elo-like update for the rating
            k = 20
            self.ratings[ht]['rating'] = h_stat['rating'] + k * (s_h - e_h)
            self.ratings[at]['rating'] = a_stat['rating'] + k * ((1-s_h) - (1-e_h))
            
            self.ratings[ht]['rd'] = np.clip(h_stat['rd'] - 1, self.rd_min, self.rd_max)
            self.ratings[at]['rd'] = np.clip(a_stat['rd'] - 1, self.rd_min, self.rd_max)
            
        df['home_glicko'] = home_glicko
        df['home_rd'] = home_rd
        df['away_glicko'] = away_glicko
        df['away_rd'] = away_rd
        df['glicko_rd_product'] = glicko_rd_product
        
        return df
