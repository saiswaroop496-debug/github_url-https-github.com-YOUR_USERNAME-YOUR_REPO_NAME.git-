"""
Kalman Filter for dynamic team strength estimation.
Converts Jane Street's real-time signal tracking to football team quality.

State vector: [team_strength, strength_velocity]
Observation:  match result xG differential (noisy measurement of true strength)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class KalmanTeamState:
    strength:    float = 1500.0   # estimated true strength (like Glicko)
    velocity:    float = 0.0      # rate of change (improving/declining)
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 100)
    # P = error covariance matrix

# System matrices
F = np.array([[1, 1],   # State transition: strength evolves, velocity persists
              [0, 1]])
H = np.array([[1, 0]])  # Observation: we observe strength, not velocity
Q = np.array([[1, 0],   # Process noise: strength changes slowly
              [0, 0.1]])
R = np.array([[50.0]])  # Observation noise: match results are noisy

def kalman_update(state: KalmanTeamState,
                   observed_strength: float) -> KalmanTeamState:
    """
    Update team strength estimate after one observation (match result).
    observed_strength: derived from xG differential in the match.
    """
    x = np.array([[state.strength], [state.velocity]])
    P = state.P

    # Predict
    x_pred = F @ x
    P_pred  = F @ P @ F.T + Q

    # Innovation (difference between observation and prediction)
    z = np.array([[observed_strength]])
    y = z - H @ x_pred

    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Update
    x_new = x_pred + K @ y
    P_new  = (np.eye(2) - K @ H) @ P_pred

    return KalmanTeamState(
        strength = float(x_new[0, 0]),
        velocity = float(x_new[1, 0]),
        P        = P_new
    )

def xg_to_strength_observation(home_xg: float, away_xg: float,
                                 team: str, home_team: str,
                                 base_strength: float = 1500) -> float:
    """
    Convert match xG to a strength observation.
    If team was home: observation = base + (home_xg - away_xg) * 100
    """
    xg_diff = home_xg - away_xg
    if team == home_team:
        return base_strength + xg_diff * 100
    else:
        return base_strength - xg_diff * 100

class KalmanRatingSystem:
    """Drop-in replacement for Glicko-2 with continuous Kalman updating."""

    def __init__(self):
        self.states: Dict[str, KalmanTeamState] = {}

    def get_or_init(self, team: str) -> KalmanTeamState:
        if team not in self.states:
            self.states[team] = KalmanTeamState()
        return self.states[team]

    def update_match(self, home_team: str, away_team: str,
                      home_xg: float, away_xg: float):
        """Update both teams after a match using xG as observation."""
        # Observation for each team
        h_obs = xg_to_strength_observation(home_xg, away_xg, home_team, home_team)
        a_obs = xg_to_strength_observation(home_xg, away_xg, away_team, home_team)

        self.states[home_team] = kalman_update(self.get_or_init(home_team), h_obs)
        self.states[away_team] = kalman_update(self.get_or_init(away_team), a_obs)

    def get_strength(self, team: str) -> float:
        return self.get_or_init(team).strength

    def get_velocity(self, team: str) -> float:
        """Positive velocity = improving team. Powerful feature for upsets."""
        return self.get_or_init(team).velocity
        
    def get_uncertainty(self, team: str) -> float:
        return float(self.get_or_init(team).P[0, 0])
