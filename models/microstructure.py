import numpy as np
import time
from typing import List, Dict

class HawkesProcessEngine:
    """
    Multivariate Hawkes Process for Market Microstructure Analysis.
    Models self-exciting phenomena in betting order flow.
    """
    def __init__(self, background_rate: float = 0.1, decay: float = 1.0, alpha: float = 0.5):
        self.mu = background_rate  # Base intensity
        self.beta = decay          # Exponential decay rate of past events
        self.alpha = alpha         # Jump size per event (must be < decay for stability)
        
        # Event histories: list of timestamps when orders were placed
        self.history_home: List[float] = []
        self.history_away: List[float] = []
        
    def add_event(self, side: str, timestamp: float = None):
        """Register a new aggressive market order (tick)."""
        if timestamp is None:
            timestamp = time.time()
            
        if side == 'home':
            self.history_home.append(timestamp)
        elif side == 'away':
            self.history_away.append(timestamp)
            
    def _calculate_intensity(self, history: List[float], current_time: float) -> float:
        """
        Calculate conditional intensity lambda(t) using exponential kernel.
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
        """
        intensity = self.mu
        if not history:
            return intensity
            
        history_arr = np.array(history)
        time_diffs = current_time - history_arr
        
        # Only consider events in the past
        valid_diffs = time_diffs[time_diffs > 0]
        
        # Vectorized intensity calculation
        excitation = np.sum(self.alpha * np.exp(-self.beta * valid_diffs))
        return intensity + excitation
        
    def get_market_imbalance(self, current_time: float = None) -> float:
        """
        Calculate the normalized Hawkes Imbalance score between Home and Away.
        Positive = Sharp money buying Home. Negative = Sharp money buying Away.
        Range: [-1.0, 1.0]
        """
        if current_time is None:
            current_time = time.time()
            
        lambda_home = self._calculate_intensity(self.history_home, current_time)
        lambda_away = self._calculate_intensity(self.history_away, current_time)
        
        total_intensity = lambda_home + lambda_away
        if total_intensity == 0:
            return 0.0
            
        return (lambda_home - lambda_away) / total_intensity

    def check_veto(self, position_side: str, threshold: float = 0.20) -> bool:
        """
        Hard safety protocol based on the PDF blueprint.
        Veto trade if sharp money is heavily betting AGAINST our position.
        """
        imbalance = self.get_market_imbalance()
        
        # If we want to bet Home, but imbalance is strongly negative (money on Away)
        if position_side == 'home' and imbalance < -threshold:
            return True # VETO
            
        # If we want to bet Away, but imbalance is strongly positive (money on Home)
        if position_side == 'away' and imbalance > threshold:
            return True # VETO
            
        return False
