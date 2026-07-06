"""
Particle Filter for dynamic team strength estimation.
Replaces Kalman Filter to handle non-Gaussian regime shifts (e.g. injuries).

State vector: [team_strength] (velocity is implicitly tracked via finite differences)
Observation:  match result xG differential (noisy measurement of true strength)
"""
import numpy as np
import scipy.stats as stats
from filterpy.monte_carlo import systematic_resample
from typing import Dict, Tuple

class TrueStrengthParticleFilter:
    def __init__(self, num_particles: int = 2000, initial_strength: float = 1500.0,
                 initial_std: float = 50.0, shock_scale: float = 5.0, obs_std: float = 25.0):
        self.num_particles = num_particles
        self.particles = np.random.normal(initial_strength, initial_std, num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.shock_scale = shock_scale
        self.obs_std = obs_std
        self.last_mean = initial_strength

    def predict(self) -> None:
        shocks = stats.cauchy.rvs(loc=0, scale=self.shock_scale, size=self.num_particles)
        self.particles += shocks

    def update(self, z: float) -> None:
        likelihood = stats.norm.pdf(z, loc=self.particles, scale=self.obs_std)
        self.weights *= likelihood
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self) -> None:
        ess = 1.0 / np.sum(np.square(self.weights))
        if ess < self.num_particles / 2.0:
            indexes = systematic_resample(self.weights)
            self.particles = self.particles[indexes]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self) -> Tuple[float, float]:
        mean = np.average(self.particles, weights=self.weights)
        var = np.average((self.particles - mean) ** 2, weights=self.weights)
        return mean, var

    def step(self, z: float) -> Tuple[float, float]:
        self.last_mean, _ = self.estimate()
        self.predict()
        self.update(z)
        self.resample()
        return self.estimate()


def xg_to_strength_observation(home_xg: float, away_xg: float,
                               team: str, home_team: str,
                               opponent_strength: float) -> float:
    """
    Convert match xG to a strength observation.
    If team was home: observation = opp_strength + (home_xg - away_xg) * 100
    """
    xg_diff = home_xg - away_xg
    if team == home_team:
        return opponent_strength + xg_diff * 100
    else:
        return opponent_strength - xg_diff * 100


class KalmanRatingSystem:
    """
    Drop-in replacement for the Kalman filter, now using Particle Filters internally.
    Keeps the same class name for backward compatibility with rolling_features.py.
    """
    def __init__(self):
        self.states: Dict[str, TrueStrengthParticleFilter] = {}

    def get_or_init(self, team: str) -> TrueStrengthParticleFilter:
        if team not in self.states:
            self.states[team] = TrueStrengthParticleFilter(
                num_particles=2000, 
                initial_strength=1500.0, 
                initial_std=50.0, 
                shock_scale=5.0, 
                obs_std=25.0
            )
        return self.states[team]

    def update_match(self, home_team: str, away_team: str, home_xg: float, away_xg: float):
        """Update both teams after a match using xG as observation."""
        h_pf = self.get_or_init(home_team)
        a_pf = self.get_or_init(away_team)

        h_str, _ = h_pf.estimate()
        a_str, _ = a_pf.estimate()

        h_obs = xg_to_strength_observation(home_xg, away_xg, home_team, home_team, opponent_strength=a_str)
        a_obs = xg_to_strength_observation(home_xg, away_xg, away_team, home_team, opponent_strength=h_str)

        h_pf.step(h_obs)
        a_pf.step(a_obs)

    def get_strength(self, team: str) -> float:
        mean, _ = self.get_or_init(team).estimate()
        return float(mean)

    def get_velocity(self, team: str) -> float:
        """Estimate velocity as the finite difference of the PF mean."""
        pf = self.get_or_init(team)
        mean, _ = pf.estimate()
        return float(mean - pf.last_mean)
        
    def get_uncertainty(self, team: str) -> float:
        """Returns the variance of the particle distribution."""
        _, var = self.get_or_init(team).estimate()
        return float(var)
