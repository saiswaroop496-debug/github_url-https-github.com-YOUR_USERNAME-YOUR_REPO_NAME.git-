import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """
    Continuous Actor for Soft Actor-Critic (SAC).
    Outputs the mean and log standard deviation for a Gaussian policy.
    Action Space: [Tranche_Size (0 to 1), Price_Aggressiveness (-1 to 1)]
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t) # Squash to [-1, 1]
        
        # Action space mapping:
        # Action 0: Tranche Size (0 to 1) -> (y_t[0] + 1) / 2
        # Action 1: Price Aggressiveness (-1 to 1) -> y_t[1]
        action = y_t.clone()
        action[:, 0] = (y_t[:, 0] + 1.0) / 2.0
        
        # Enforce action bound log_prob correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

class CriticNetwork(nn.Module):
    """
    Twin Q-Networks for SAC to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        # Q1 architecture
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        
        return q1, q2

class SACExecutionAgent:
    """
    Soft Actor-Critic agent for continuous execution in betting limit order books.
    
    State Vector (dim=5):
        [BNN Edge, Model Uncertainty, Hawkes Imbalance, Time to Kickoff, Current Drawdown]
    
    Action Vector (dim=2):
        [Tranche Size %, Limit Price Offset]
    """
    def __init__(self, state_dim=5, action_dim=2, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def get_action(self, state, evaluate=False):
        """
        Takes a raw state array and returns an action array.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
                
        return action.detach().cpu().numpy()[0]
        
    def get_veto_signal(self, hawkes_imbalance: float, threshold: float = -0.20) -> bool:
        """
        Hard safety protocol based on Market Microstructure (Hawkes Process).
        If informed money is betting against us heavily, veto the trade.
        """
        if hawkes_imbalance < threshold:
            return True # VETO trade
        return False
