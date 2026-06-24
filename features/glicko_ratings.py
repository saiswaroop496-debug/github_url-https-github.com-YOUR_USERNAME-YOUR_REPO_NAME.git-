import pandas as pd
import numpy as np

import hashlib
import pickle
from pathlib import Path

GLICKO_CACHE_DIR = Path(".glicko_cache")
GLICKO_CACHE_DIR.mkdir(exist_ok=True)

class Glicko2RatingSystem:
    def __init__(self, tau=0.5, epsilon=0.000001, rd_min=45, rd_max=130):
        self.tau = tau
        self.epsilon = epsilon
        self.rd_min = rd_min
        self.rd_max = rd_max
        self.ratings = {}
        
    def _get_team(self, team):
        if team not in self.ratings:
            self.ratings[team] = {'rating': 1500.0, 'rd': 130.0, 'sigma': 0.06}
        return self.ratings[team]
        
    def _g(self, rd):
        q = np.log(10) / 400
        return 1 / np.sqrt(1 + 3 * q**2 * rd**2 / np.pi**2)
        
    def _E(self, r, ri, rdi):
        return 1 / (1 + 10**(-self._g(rdi) * (r - ri) / 400))
        
    def compute_ratings(self, df, force_recompute=False):
        df = df.copy()
        
        # Caching Logic
        key = f"{len(df)}_{df['date'].iloc[-1]}_{df['date'].iloc[0]}"
        cache_key = hashlib.md5(key.encode()).hexdigest()[:12]
        cache_path = GLICKO_CACHE_DIR / f"glicko_{cache_key}.pkl"

        if not force_recompute and cache_path.exists():
            print(f"  [CACHE HIT] Glicko: {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"  [COMPUTING] Glicko-2 from scratch ({len(df)} matches)...")

        home_glicko, home_rd, away_glicko, away_rd, glicko_rd_product, glicko_gap = [], [], [], [], [], []
        
        df['rating_period'] = df['date'].dt.to_period('Q')
        
        for period in sorted(df['rating_period'].unique()):
            period_indices = df[df['rating_period'] == period].index
            period_matches = df.loc[period_indices]
            
            # 1. STORE PRE-PERIOD RATINGS FIRST (CRITICAL FREEZE BUG FIX)
            for _, row in period_matches.iterrows():
                ht = row['home_team']
                at = row['away_team']
                
                h_stat = self._get_team(ht)
                a_stat = self._get_team(at)
                
                home_glicko.append(h_stat['rating'])
                home_rd.append(h_stat['rd'])
                away_glicko.append(a_stat['rating'])
                away_rd.append(a_stat['rd'])
                
                h_signal = h_stat['rating'] * (1 - (np.clip(h_stat['rd'], self.rd_min, self.rd_max) - self.rd_min) / (self.rd_max - self.rd_min))
                a_signal = a_stat['rating'] * (1 - (np.clip(a_stat['rd'], self.rd_min, self.rd_max) - self.rd_min) / (self.rd_max - self.rd_min))
                
                glicko_rd_product.append(h_signal - a_signal)
                glicko_gap.append(abs(h_stat['rd'] - a_stat['rd']))
                
            # 2. THEN UPDATE
            for _, row in period_matches.iterrows():
                ht = row['home_team']
                at = row['away_team']
                
                h_stat = self._get_team(ht)
                a_stat = self._get_team(at)
                
                h_goals, a_goals = row['home_goals'], row['away_goals']
                s_h = 1 if h_goals > a_goals else (0.5 if h_goals == a_goals else 0)
                
                r_h = h_stat['rating'] + (0 if row.get('is_neutral', True) else 40)
                e_h = self._E(r_h, a_stat['rating'], a_stat['rd'])
                
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
        df['glicko_uncertainty_gap'] = glicko_gap
        
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"  [CACHED] Saved to {cache_path}")
        
        return df

def clear_glicko_cache():
    """Call this when you update historical data or change RD parameters."""
    for f in GLICKO_CACHE_DIR.glob("*.pkl"):
        f.unlink()
    print("Glicko cache cleared.")
