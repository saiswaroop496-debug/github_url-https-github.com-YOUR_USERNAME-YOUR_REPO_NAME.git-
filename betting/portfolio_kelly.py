import numpy as np
from scipy.optimize import minimize

def portfolio_kelly(edges: np.ndarray,
                    odds:  np.ndarray,
                    corr_matrix: np.ndarray,
                    bankroll: float = 1000.0,
                    max_exposure: float = 0.20) -> np.ndarray:
    """
    Solve for optimal stake fractions across N simultaneous bets.

    Parameters:
    - edges: (N,) array of no-vig edges (model_prob - novig_prob)
    - odds:  (N,) decimal odds
    - corr_matrix: (N,N) correlation between bet outcomes
    - max_exposure: maximum total bankroll fraction across all bets

    Returns: (N,) array of optimal fractions summing to <= max_exposure
    """
    n = len(edges)
    if n == 0:
        return np.array([])

    # Only bet where edge exists
    valid = edges > 0.01
    if not valid.any():
        return np.zeros(n)

    def neg_log_growth(fracs):
        """Maximize expected log growth = Kelly objective."""
        # Approximate log growth with Taylor expansion
        total = sum(
            (edges[i] * odds[i] * fracs[i]) - (0.5 * fracs[i]**2 * (odds[i] - 1))
            for i in range(n) if valid[i]
        )
        # Covariance penalty (Markowitz)
        cov_penalty = 0.5 * fracs @ corr_matrix @ fracs
        return -(total - cov_penalty)

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda f: max_exposure - f.sum()},
    ]
    bounds = [(0, 0.05) for _ in range(n)]   # max 5% per bet

    result = minimize(neg_log_growth,
                       x0=np.ones(n) * 0.01,
                       method='SLSQP',
                       bounds=bounds,
                       constraints=constraints)

    fracs = np.maximum(result.x, 0)
    # Quarter Kelly adjustment
    fracs = fracs * 0.25
    return fracs


def build_bet_correlation_matrix(markets: list) -> np.ndarray:
    """
    Estimate correlation between simultaneous bet outcomes.
    Same match bets (e.g., Home Win + Over 2.5) are correlated.
    Different match bets are near-independent.
    """
    n = len(markets)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            m_i = markets[i]
            m_j = markets[j]
            # Same fixture: moderate positive correlation
            if m_i.get('fixture_id') == m_j.get('fixture_id'):
                corr[i,j] = corr[j,i] = 0.40
            # Same tournament, different match: very low
            else:
                corr[i,j] = corr[j,i] = 0.05
    return corr
