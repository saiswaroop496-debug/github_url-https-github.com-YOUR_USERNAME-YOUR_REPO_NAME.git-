import numpy as np

class LinUCBAgent:
    def __init__(self, n_actions=3, n_features=5):
        self.n_actions = n_actions
        self.n_features = n_features
        
        # Initialize A_a (identity matrix) and b_a (zero vector) for each action
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        self.n_rounds = 0
        
    def select_action(self, context_vector):
        x = np.array(context_vector)
        if len(x) != self.n_features:
            # Fallback to zero-padding if less features provided
            pad = np.zeros(self.n_features)
            pad[:len(x)] = x
            x = pad
            
        # Alpha decay starting from 0.25
        alpha = max(0.05, 0.25 / (1 + 0.05 * self.n_rounds))
        
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            
            # UCB formula
            p[a] = theta_a.T @ x + alpha * np.sqrt(x.T @ A_inv @ x)
            
        action = np.argmax(p)
        return action
        
    def update(self, action, context_vector, reward):
        x = np.array(context_vector)
        if len(x) != self.n_features:
            pad = np.zeros(self.n_features)
            pad[:len(x)] = x
            x = pad
            
        # Update A_a and b_a
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        
        self.n_rounds += 1

    def train_on_historical(self, contexts, actions, rewards):
        for c, a, r in zip(contexts, actions, rewards):
            self.update(a, c, r)

    def backtest_sweep(self, df, novig_col='novig_edge', glicko_col='glicko_uncertainty_gap'):
        """
        Pre-trains LinUCB arms over historical dataset before live prediction.
        Builds 5D context: [epv_edge, hawkes_imbalance, seq_volatility, 
                             novig_edge, glicko_uncertainty_gap]
        Addresses O~(d*sqrt(T)) regret bound — needs T>=50 to converge.
        """
        import pandas as pd
        rewards_log = []
        
        # We need something analogous to tranche_sizes. The original code has 3 actions:
        # 0: Aggressive Limit (20%) -> 1.0 (relative sizing)
        # 1: Passive Peg (10%) -> 0.5
        # 2: TWAP Slice (5%) -> 0.25
        tranche_sizes = {0: 1.0, 1: 0.5, 2: 0.25}
        
        for idx, row in df.iterrows():
            # Build 5D context vector
            context = np.array([
                row.get('epv_edge', 0.0),
                row.get('hawkes_imbalance', 0.0),
                row.get('seq_volatility', 0.1),
                row.get(novig_col, 0.0),
                row.get(glicko_col, 75.0) / 130.0  # normalise to [0,1] using RD_MAX cap
            ], dtype=np.float64)

            arm = self.select_action(context)

            # Simulate reward: novig_edge * tranche_executed minus slippage proxy
            base_reward = row.get(novig_col, 0.0) * tranche_sizes.get(arm, 0.5)
            slippage = np.random.normal(0.0025, 0.001)  # mean 0.25% slippage
            reward = base_reward - slippage

            self.update(arm, context, reward)
            rewards_log.append({'idx': idx, 'arm': arm, 'reward': reward,
                                 'cumulative': sum(r['reward'] for r in rewards_log) + reward})

        print(f"[LinUCB Sweep] {len(df)} contexts processed. "
              f"Mean reward: {np.mean([r['reward'] for r in rewards_log]):.5f}")
        return pd.DataFrame(rewards_log)
