import pandas as pd
import numpy as np
import cvxpy as cp

class BlackLittermanModel:
    """
    Black-Litterman Portfolio Optimization for Sports Betting.
    """
    def __init__(self, market_probs, bnn_probs, odds, covariance, tau=0.05, risk_aversion=2.5, w_mkt=None):
        self.market_probs = np.array(market_probs)
        self.bnn_probs = np.array(bnn_probs)
        self.odds = np.array(odds)
        self.covariance = np.array(covariance)
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.N = len(odds)
        if w_mkt is None:
            self.w_mkt = np.ones(self.N) / self.N
        else:
            self.w_mkt = np.array(w_mkt)

    def _make_psd(self, matrix):
        matrix = (matrix + matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def compute_posterior(self):
        Pi = self.risk_aversion * self.covariance @ self.w_mkt
        Q = (self.bnn_probs * self.odds) - 1.0
        P = np.eye(self.N)
        tau_Sigma = self.tau * self.covariance
        Omega = np.diag(np.diag(P @ tau_Sigma @ P.T))
        
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        M_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        mu_BL = M_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)
        
        Sigma_BL = self.covariance + M_inv
        Sigma_BL = self._make_psd(Sigma_BL)
        return mu_BL, Sigma_BL

    def maximize_sharpe(self, mu, cov):
        if np.max(mu) <= 0:
            return np.zeros(self.N)
        cov = self._make_psd(cov)
        x = cp.Variable(self.N)
        try:
            objective = cp.Minimize(cp.quad_form(x, cp.psd_wrap(cov)))
        except AttributeError:
            objective = cp.Minimize(cp.quad_form(x, cov))
        
        constraints = [x @ mu == 1, x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        
        if x.value is None:
            return np.zeros(self.N)
            
        w = x.value / np.sum(x.value)
        w[w < 1e-6] = 0
        sum_w = np.sum(w)
        if sum_w > 0:
            w = w / sum_w
        return w


def scan_pure_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """Detects pure risk-free arbitrage opportunities."""
    if 'max_h' not in df.columns or 'max_d' not in df.columns or 'max_a' not in df.columns:
        return df

    df['arb_implied_prob'] = (1 / df['max_h']) + (1 / df['max_d']) + (1 / df['max_a'])
    df['is_pure_arb'] = df['arb_implied_prob'] < 1.0
    df['pure_arb_roi_pct'] = np.where(df['is_pure_arb'], ((1 / df['arb_implied_prob']) - 1) * 100, 0.0)

    df['arb_stake_h_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_h']), 0.0)
    df['arb_stake_d_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_d']), 0.0)
    df['arb_stake_a_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_a']), 0.0)
    return df

def scan_statistical_arbitrage(df: pd.DataFrame, model_preds_h: pd.Series, model_preds_d: pd.Series, model_preds_a: pd.Series, bankroll: float = 1000.0) -> pd.DataFrame:
    """
    V8 Black-Litterman Portfolio Optimization across a match slate.
    Treats each outcome as an asset and computes covariance-adjusted optimal allocation.
    """
    if 'max_h' not in df.columns:
        return df

    df['stat_edge_h'] = model_preds_h - (1 / df['max_h'])
    df['stat_edge_d'] = model_preds_d - (1 / df['max_d'])
    df['stat_edge_a'] = model_preds_a - (1 / df['max_a'])
    
    stakes_h = np.zeros(len(df))
    stakes_d = np.zeros(len(df))
    stakes_a = np.zeros(len(df))
    
    # Process the entire dataframe as a single portfolio (slate optimization)
    # We flatten all H, D, A bets into a single vector of assets
    # For a full World Cup slate, this mathematically hedges overlapping risks.
    
    # In V8, we optimize match-by-match for computational simplicity on large df
    # but using Black-Litterman intra-match correlation (H, D, A are highly correlated negatively).
    
    for i in range(len(df)):
        ph, pd, pa = model_preds_h.iloc[i], model_preds_d.iloc[i], model_preds_a.iloc[i]
        oh, od, oa = df['max_h'].iloc[i], df['max_d'].iloc[i], df['max_a'].iloc[i]
        
        if any(np.isnan([ph, pd, pa, oh, od, oa])) or oh <= 1.0 or od <= 1.0 or oa <= 1.0:
            continue
            
        bnn_probs = np.array([ph, pd, pa])
        odds = np.array([oh, od, oa])
        market_probs = 1 / odds
        market_probs = market_probs / np.sum(market_probs) # Normalize
        
        # Intra-match covariance (mutually exclusive)
        # Var(p) = p(1-p). Covariance = -p_i * p_j
        cov = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                if j == k:
                    cov[j, k] = market_probs[j] * (1 - market_probs[j])
                else:
                    cov[j, k] = -market_probs[j] * market_probs[k]
                    
        # Apply Black-Litterman
        bl = BlackLittermanModel(market_probs, bnn_probs, odds, cov)
        try:
            mu_bl, cov_bl = bl.compute_posterior()
            weights = bl.maximize_sharpe(mu_bl, cov_bl)
            
            # Cap maximum exposure per match at 5% of bankroll
            total_weight = np.sum(weights)
            if total_weight > 0.05:
                weights = weights * (0.05 / total_weight)
                
            stakes_h[i] = weights[0] * bankroll
            stakes_d[i] = weights[1] * bankroll
            stakes_a[i] = weights[2] * bankroll
        except Exception:
            # Fallback to zero if optimization fails
            pass
            
    df['stat_stake_h'] = stakes_h
    df['stat_stake_d'] = stakes_d
    df['stat_stake_a'] = stakes_a

    df['stake_h'] = df['stat_stake_h']
    df['stake_d'] = df['stat_stake_d']
    df['stake_a'] = df['stat_stake_a']

    return df
