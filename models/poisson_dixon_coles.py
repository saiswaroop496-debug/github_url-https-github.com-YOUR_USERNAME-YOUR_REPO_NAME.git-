import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import exp, log
from scipy.stats import poisson

class DixonColesModel:
    def __init__(self):
        self.teams = []
        self.params = {}
        
    def _tau(self, x, y, lambda_, mu, rho):
        if x == 0 and y == 0: return 1 - lambda_ * mu * rho
        elif x == 0 and y == 1: return 1 + lambda_ * rho
        elif x == 1 and y == 0: return 1 + mu * rho
        elif x == 1 and y == 1: return 1 - rho
        else: return 1.0
        
    def _log_likelihood(self, params_array, df, max_date):
        n_teams = len(self.teams)
        attack = params_array[:n_teams]
        defense = params_array[n_teams:2*n_teams]
        home_gamma = params_array[2*n_teams]
        neutral_gamma = params_array[2*n_teams + 1]
        rho = params_array[2*n_teams + 2]
        
        ll = 0.0
        for _, row in df.iterrows():
            i = self.team_to_idx[row['home_team']]
            j = self.team_to_idx[row['away_team']]
            x = row['home_goals']
            y = row['away_goals']
            
            # Use continuous venue_factor (crowd_factor) instead of boolean is_neutral
            venue_factor = row.get('crowd_factor', 0.0)
            gamma_eff = neutral_gamma + venue_factor * (home_gamma - neutral_gamma)
            
            lambda_ = exp(attack[i] + defense[j] + gamma_eff)
            mu = exp(attack[j] + defense[i])
            
            delta_days = (max_date - row['date']).days
            w_t = exp(-0.0065 * delta_days)
            
            tau_val = self._tau(x, y, lambda_, mu, rho)
            if tau_val <= 0:
                return 1e9 # Penalty
                
            ll += w_t * (log(tau_val) + poisson.logpmf(x, lambda_) + poisson.logpmf(y, mu))
            
        return -ll # Minimize negative log-likelihood
        
    def fit(self, df):
        self.teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)
        
        # Initial guess
        x0 = np.zeros(2*n_teams + 3)
        x0[2*n_teams] = 0.3 # home_gamma
        x0[2*n_teams + 1] = 0.0 # neutral_gamma
        x0[2*n_teams + 2] = 0.0 # rho
        
        # Bounds: rho in [-0.2, 0.2]
        bounds = [(None, None)] * (2*n_teams + 2) + [(-0.2, 0.2)]
        
        res = minimize(self._log_likelihood, x0, args=(df, df['date'].max()), method='L-BFGS-B', bounds=bounds)
        
        opt_params = res.x
        self.attack = {t: opt_params[i] for i, t in enumerate(self.teams)}
        self.defense = {t: opt_params[n_teams + i] for i, t in enumerate(self.teams)}
        self.home_gamma = opt_params[2*n_teams]
        self.neutral_gamma = opt_params[2*n_teams + 1]
        self.rho = opt_params[2*n_teams + 2]
        
    def predict_proba(self, home_team, away_team, venue_factor=0.0):
        # venue_factor: float [0.0=pure neutral, 0.6=host, 1.0=true home]
        if home_team not in self.attack or away_team not in self.attack:
            return {'Home': 0.33, 'Draw': 0.34, 'Away': 0.33}
            
        gamma_eff = self.neutral_gamma + venue_factor * (self.home_gamma - self.neutral_gamma)
        lambda_ = exp(self.attack[home_team] + self.defense[away_team] + gamma_eff)
        mu = exp(self.attack[away_team] + self.defense[home_team])
        
        p_home = p_draw = p_away = 0.0
        
        for x in range(11):
            for y in range(11):
                p = self._tau(x, y, lambda_, mu, self.rho) * poisson.pmf(x, lambda_) * poisson.pmf(y, mu)
                if x > y: p_home += p
                elif x == y: p_draw += p
                else: p_away += p
                
        # Normalize just in case grid truncation causes minor loss
        total = p_home + p_draw + p_away
        return {'Home': p_home/total, 'Draw': p_draw/total, 'Away': p_away/total}
