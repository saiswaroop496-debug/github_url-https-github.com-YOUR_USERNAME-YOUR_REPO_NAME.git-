import numpy as np
from typing import Dict, List, Tuple
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class OrnsteinUhlenbeckModel:
    """
    Fits an Ornstein-Uhlenbeck (OU) process to a time series (e.g., the spread between two markets).
    The OU process models mean reversion: dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    """
    def __init__(self):
        self.theta = 0.0  # Speed of mean reversion
        self.mu = 0.0     # Long-term mean
        self.sigma = 0.0  # Volatility
        self.is_fitted = False

    def fit(self, spread_series: np.ndarray, dt: float = 1.0):
        """
        Calibrate the OU process using linear regression.
        X_{t+1} = a + b * X_t + epsilon
        """
        if len(spread_series) < 10:
            return False
            
        x = spread_series[:-1]
        y = spread_series[1:]
        
        # Fit linear regression y = b*x + a
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        a, b = model.params
        
        # Prevent division by zero or negative log arguments
        if b <= 0 or b >= 1:
            return False
            
        self.theta = -np.log(b) / dt
        self.mu = a / (1 - b)
        
        residuals = y - (a + b * x)
        self.sigma = np.std(residuals) * np.sqrt(2 * self.theta / (1 - b**2))
        self.is_fitted = True
        return True

    def get_z_score(self, current_spread: float) -> float:
        if not self.is_fitted or self.sigma == 0:
            return 0.0
        # Calculate standard deviation of the invariant distribution
        std_eq = self.sigma / np.sqrt(2 * self.theta)
        return (current_spread - self.mu) / std_eq

class StatArbEngine:
    """
    Statistical Arbitrage Engine exploiting cointegration across correlated markets
    (e.g., Match Odds vs Asian Handicaps).
    """
    def __init__(self, z_score_threshold: float = 2.0):
        self.z_score_threshold = z_score_threshold
        self.ou_models: Dict[str, OrnsteinUhlenbeckModel] = {}
        
    def test_cointegration(self, series_y: np.ndarray, series_x: np.ndarray) -> Tuple[bool, float]:
        """
        Tests if two markets (e.g. implied prob of Home Win vs -0.5 Asian Handicap) are cointegrated
        using the Engle-Granger two-step method.
        """
        if len(series_y) < 30:
            return False, 1.0
            
        # Run cointegration test
        # Null hypothesis: no cointegration. 
        # Low p-value (<0.05) implies cointegration.
        score, pvalue, _ = coint(series_y, series_x)
        
        return pvalue < 0.05, pvalue

    def calculate_spread(self, series_y: np.ndarray, series_x: np.ndarray) -> np.ndarray:
        """Calculate the spread using linear regression hedge ratio."""
        X = sm.add_constant(series_x)
        model = sm.OLS(series_y, X).fit()
        hedge_ratio = model.params[1]
        return series_y - hedge_ratio * series_x

    def analyze_pair(self, pair_id: str, series_y: np.ndarray, series_x: np.ndarray, current_y: float, current_x: float) -> Dict:
        """
        Analyze a pair of markets in real-time.
        """
        is_coint, p_val = self.test_cointegration(series_y, series_x)
        
        if not is_coint:
            return {"trade_signal": "NONE", "reason": f"Not cointegrated (p={p_val:.3f})"}
            
        spread_series = self.calculate_spread(series_y, series_x)
        
        if pair_id not in self.ou_models:
            self.ou_models[pair_id] = OrnsteinUhlenbeckModel()
            
        success = self.ou_models[pair_id].fit(spread_series)
        
        if not success:
             return {"trade_signal": "NONE", "reason": "Failed to fit OU process"}
             
        # Calculate current spread
        X_hist = sm.add_constant(series_x)
        hedge_ratio = sm.OLS(series_y, X_hist).fit().params[1]
        
        current_spread = current_y - hedge_ratio * current_x
        z_score = self.ou_models[pair_id].get_z_score(current_spread)
        
        signal = "NONE"
        if z_score > self.z_score_threshold:
            signal = "SELL_SPREAD" # Spread is too high, expect mean reversion down
        elif z_score < -self.z_score_threshold:
            signal = "BUY_SPREAD"  # Spread is too low, expect mean reversion up
            
        return {
            "trade_signal": signal,
            "z_score": round(z_score, 3),
            "half_life": round(np.log(2) / self.ou_models[pair_id].theta, 2) if self.ou_models[pair_id].theta > 0 else float('inf'),
            "hedge_ratio": round(hedge_ratio, 4)
        }
