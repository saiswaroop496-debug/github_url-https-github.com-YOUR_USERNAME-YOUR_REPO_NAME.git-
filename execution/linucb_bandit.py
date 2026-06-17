import numpy as np

class LinUCBAgent:
    def __init__(self, n_actions=3, n_features=3):
        self.n_actions = n_actions
        self.n_features = n_features
        
        # Initialize A_a (identity matrix) and b_a (zero vector) for each action
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        self.n_rounds = 0
        
    def select_action(self, context_vector):
        x = np.array(context_vector)
        
        # Alpha decay
        alpha = max(0.1, 1.0 / (1 + 0.05 * self.n_rounds))
        
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
        
        # Update A_a and b_a
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        
        self.n_rounds += 1
