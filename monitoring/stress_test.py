import numpy as np
from typing import List, Dict

class BlackSwanSimulator:
    """
    Stress testing module for the V9 Omni-Quant Engine.
    Simulates catastrophic scenarios (Black Swans) to calculate Maximum Drawdown (MDD)
    and Value at Risk (VaR) to enforce dynamic circuit breakers.
    """
    def __init__(self, initial_bankroll: float = 10000.0, max_allowable_drawdown: float = 0.25):
        self.initial_bankroll = initial_bankroll
        self.max_allowable_drawdown = max_allowable_drawdown
        
    def simulate_heavy_favorite_collapse(self, portfolio: List[Dict]) -> float:
        """
        Scenario 1: All teams with implied probability > 75% lose their matches.
        """
        simulated_pnl = 0.0
        for bet in portfolio:
            implied_prob = 1 / bet['decimal_odds']
            if implied_prob > 0.75:
                # Bet loses
                simulated_pnl -= bet['stake_units']
            else:
                # Assume average EV realization for the rest
                simulated_pnl += bet['stake_units'] * bet['edge']
                
        return simulated_pnl

    def simulate_extreme_weather(self, portfolio: List[Dict]) -> float:
        """
        Scenario 2: Goals plummet globally. All 'Over 2.5' bets lose, 'Under 2.5' bets win.
        """
        simulated_pnl = 0.0
        for bet in portfolio:
            if bet['market'] == 'Over 2.5':
                simulated_pnl -= bet['stake_units']
            elif bet['market'] == 'Under 2.5':
                simulated_pnl += bet['stake_units'] * (bet['decimal_odds'] - 1)
            else:
                simulated_pnl += bet['stake_units'] * bet['edge']
        return simulated_pnl

    def calculate_cvar(self, portfolio: List[Dict], n_simulations: int = 10000, alpha: float = 0.05) -> float:
        """
        Calculates Conditional Value at Risk (Expected Shortfall) at alpha level.
        Assumes binary outcomes based on model probabilities.
        """
        if not portfolio:
            return 0.0
            
        pnls = np.zeros(n_simulations)
        
        # Vectorized Monte Carlo for the portfolio
        for bet in portfolio:
            # Simulate win/loss based on the model's true probability (edge + implied)
            true_prob = (1 / bet['decimal_odds']) + bet['edge']
            wins = np.random.rand(n_simulations) < true_prob
            
            pnl_if_win = bet['stake_units'] * (bet['decimal_odds'] - 1)
            pnl_if_loss = -bet['stake_units']
            
            pnls += np.where(wins, pnl_if_win, pnl_if_loss)
            
        # CVaR is the expected loss in the worst alpha% of cases
        var_threshold = np.percentile(pnls, alpha * 100)
        tail_losses = pnls[pnls <= var_threshold]
        
        return tail_losses.mean() if len(tail_losses) > 0 else 0.0

    def check_circuit_breaker(self, current_bankroll: float, portfolio: List[Dict]) -> Dict:
        """
        Evaluates current risk and triggers circuit breakers if limits are exceeded.
        """
        # 1. Hard Drawdown Check
        current_drawdown = 1.0 - (current_bankroll / self.initial_bankroll)
        if current_drawdown > self.max_allowable_drawdown:
            return {"status": "HALT", "reason": f"Max Drawdown Exceeded ({current_drawdown:.1%} > {self.max_allowable_drawdown:.1%})"}
            
        # 2. Stress Test Check
        pnl_favorite_collapse = self.simulate_heavy_favorite_collapse(portfolio)
        projected_drawdown = 1.0 - ((current_bankroll + pnl_favorite_collapse) / self.initial_bankroll)
        
        if projected_drawdown > self.max_allowable_drawdown:
             return {"status": "HALT", "reason": "Stress Test Failed: Heavy Favorite Collapse violates DD limit."}
             
        # 3. CVaR Check
        cvar = self.calculate_cvar(portfolio)
        projected_cvar_drawdown = 1.0 - ((current_bankroll + cvar) / self.initial_bankroll)
        
        if projected_cvar_drawdown > self.max_allowable_drawdown:
            return {"status": "WARNING", "reason": f"High Tail Risk: CVaR indicates potential {projected_cvar_drawdown:.1%} DD."}
            
        return {"status": "OK", "reason": "All risk limits satisfied."}
