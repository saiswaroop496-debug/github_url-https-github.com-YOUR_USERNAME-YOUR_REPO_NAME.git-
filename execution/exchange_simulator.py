import numpy as np
import gym
from gym import spaces
import sys
import os

# Add parent dir to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from models.microstructure import HawkesProcessEngine
except ImportError:
    class HawkesProcessEngine:
        def __init__(self): pass
        def get_market_imbalance(self): return np.random.uniform(-0.5, 0.5)
        def add_event(self, side, ts): pass

class BettingExchangeEnv(gym.Env):
    """
    Simulated Limit Order Book (LOB) environment for offline RL training.
    Replays historical or stochastic tick data to allow the SAC agent
    to learn order execution and adverse selection avoidance.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_capital=10000.0, max_steps=100):
        super(BettingExchangeEnv, self).__init__()
        
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.current_step = 0
        self.capital = self.initial_capital
        self.inventory = 0.0 # Current exposure
        
        # State space matching rl_agent.py (10 dimensions)
        # 0: BNN Edge, 1: True Prob, 2: Implied Prob, 3: BNN Uncert,
        # 4: Hawkes Imbal, 5: Best Back, 6: Best Lay, 7: Spread,
        # 8: VWAP Slip, 9: Time to Match
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Action space matching rl_agent.py
        # Action[0] = Urgency/Limit Price Offset (-1.0 to 1.0)
        # Action[1] = Tranche Size (0.0 to 1.0)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.hawkes = HawkesProcessEngine()
        self.state = None

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.inventory = 0.0
        self.hawkes = HawkesProcessEngine()
        return self._get_observation()

    def _get_observation(self):
        """Generate a simulated market state."""
        edge = np.random.normal(0.05, 0.02)
        true_prob = np.clip(np.random.normal(0.5, 0.1), 0.1, 0.9)
        implied_prob = true_prob - edge
        uncert = np.random.uniform(0.01, 0.1)
        
        # Simulate tick events
        if np.random.random() > 0.5:
            self.hawkes.add_event('home')
        else:
            self.hawkes.add_event('away')
            
        hawkes_imbalance = self.hawkes.get_market_imbalance()
        best_back = 1.0 / implied_prob if implied_prob > 0 else 100.0
        best_lay = best_back + np.random.uniform(0.01, 0.05)
        spread = best_lay - best_back
        vwap = np.random.uniform(0.0, 0.05)
        time_to_match = 1.0 - (self.current_step / self.max_steps)
        
        self.state = np.array([
            edge, true_prob, implied_prob, uncert, hawkes_imbalance,
            best_back, best_lay, spread, vwap, time_to_match
        ], dtype=np.float32)
        
        return self.state

    def step(self, action):
        """
        Execute agent's action (price offset and size).
        Calculate slippage, fill probability, and resulting PnL (reward).
        """
        urgency, size_fraction = action
        
        # Extract current state features
        edge = self.state[0]
        true_prob = self.state[1]
        best_back = self.state[5]
        spread = self.state[7]
        hawkes_imbal = self.state[4]
        
        # 1. Size constraint
        trade_size = size_fraction * self.capital * 0.10 # Max 10% of capital per tranche
        
        # 2. Fill Probability & Slippage (Market Impact)
        # Positive urgency = crossing the spread (taking liquidity) -> High fill prob, high slippage
        # Negative urgency = providing liquidity (limit order) -> Low fill prob, negative slippage (earn spread)
        
        base_fill_prob = 0.5 + (urgency * 0.4)
        # Adjust fill prob by hawkes imbalance (adverse selection)
        # If we are buying and hawkes is negative, we get filled instantly (toxic flow)
        if hawkes_imbal < -0.2:
            base_fill_prob += 0.2
            
        fill_prob = np.clip(base_fill_prob, 0.0, 1.0)
        is_filled = np.random.random() < fill_prob
        
        reward = 0.0
        if is_filled:
            # Calculate execution price
            exec_price = best_back + (urgency * spread)
            
            # Simulated match outcome based on true probability
            match_won = np.random.random() < true_prob
            
            if match_won:
                # Profit = Stake * (Odds - 1)
                pnl = trade_size * (exec_price - 1.0)
            else:
                pnl = -trade_size
                
            self.capital += pnl
            
            # Reward is risk-adjusted PnL (Sharpe-like), penalized for toxic fills
            # If we got filled when hawkes < -0.2, penalize heavily
            if hawkes_imbal < -0.2:
                pnl -= trade_size * 0.5 # Penalty for adverse selection
                
            reward = pnl
        else:
            # Small penalty for failing to execute a positive edge
            if edge > 0.02:
                reward = -0.01 * trade_size
                
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, {"capital": self.capital}
        
    def render(self, mode='human'):
        pass
