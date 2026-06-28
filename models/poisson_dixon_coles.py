import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# --- STAGE 1: Fit attack/defense with identifiability constraint --------------
def fit_attack_defense(home_teams, away_teams, home_goals, away_goals,
                       all_teams, time_weights=None):
    """
    Stage 1: MLE for attack/defense parameters only.
    Identifiability penalty: 1000 * sum(attack)^2 anchors mean(attack)=0.
    """
    n_teams = len(all_teams)
    team_idx = {t: i for i, t in enumerate(all_teams)}
    W = time_weights if time_weights is not None else np.ones(len(home_goals))

    def neg_log_likelihood(params):
        attack  = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]

        ll = 0.0
        for i, (ht, at, gh, ga) in enumerate(zip(home_teams, away_teams,
                                                   home_goals, away_goals)):
            hi, ai = team_idx[ht], team_idx[at]
            lam_h = np.exp(attack[hi] - defense[ai] + home_adv)
            lam_a = np.exp(attack[ai] - defense[hi])
            ll += W[i] * (poisson.logpmf(gh, lam_h) + poisson.logpmf(ga, lam_a))

        # Identifiability constraint: reduced penalty 1000 (was 10000)
        penalty = 1000.0 * (np.sum(attack) ** 2)
        return -ll + penalty

    x0 = np.zeros(2 * n_teams + 1)
    result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-8})
    params = result.x
    attack  = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    return attack, defense, home_adv, team_idx


# --- STAGE 2: Estimate ? with fixed attack/defense ----------------------------
def dc_tau(gh, ga, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction for 4 outcome cells."""
    if gh == 0 and ga == 0: return 1.0 - lam_h * lam_a * rho
    if gh == 0 and ga == 1: return 1.0 + lam_h * rho
    if gh == 1 and ga == 0: return 1.0 + lam_a * rho
    if gh == 1 and ga == 1: return 1.0 - rho
    return 1.0


def fit_rho_concentrated(home_teams, away_teams, home_goals, away_goals,
                          attack, defense, home_adv, team_idx,
                          time_weights=None):
    """
    Stage 2: Estimate rho via concentrated log-likelihood.
    attack/defense are FIXED from Stage 1 — only rho varies here.
    Bounds widened to (-0.35, 0.35) per V6.2 blueprint.
    """
    W = time_weights if time_weights is not None else np.ones(len(home_goals))

    def neg_ll_rho(rho):
        ll = 0.0
        for i, (ht, at, gh, ga) in enumerate(zip(home_teams, away_teams,
                                                   home_goals, away_goals)):
            hi, ai = team_idx[ht], team_idx[at]
            lam_h = np.exp(attack[hi] - defense[ai] + home_adv)
            lam_a = np.exp(attack[ai] - defense[hi])
            tau = dc_tau(gh, ga, lam_h, lam_a, rho)
            if tau <= 0:
                return 1e9
            ll += W[i] * np.log(max(tau, 1e-10))
        return -ll

    result = minimize_scalar(neg_ll_rho, bounds=(-0.35, 0.35), method='bounded')
    rho = result.x
    print(f"  ? (rho) converged to: {rho:.4f}")
    return rho


# --- Vectorized Score Matrix -------------------------------------------------
def score_probability_matrix(lam_h: float, lam_a: float,
                              rho: float, max_goals: int = 8) -> np.ndarray:
    """Vectorized scoreline probability matrix using np.outer."""
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lam_h)
    pa = poisson.pmf(g, lam_a)
    joint = np.outer(ph, pa)

    # Apply DC tau correction to 2x2 block
    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = 1.0 - lam_h * lam_a * rho
    tau[0, 1] = 1.0 + lam_h * rho
    tau[1, 0] = 1.0 + lam_a * rho
    tau[1, 1] = 1.0 - rho

    matrix = joint * tau
    return matrix / matrix.sum()   # normalize


def outcome_probs(matrix: np.ndarray):
    """Extract 1X2 from scoreline matrix."""
    home_win = np.tril(matrix, k=-1).sum()
    draw     = np.trace(matrix)
    away_win = np.triu(matrix, k=1).sum()
    total = home_win + draw + away_win
    return home_win/total, draw/total, away_win/total


# --- BTTS + Over/Under --------------------------------------------------------
def extract_btts_ou(matrix: np.ndarray, ou_threshold: float = 2.5):
    """
    Extract BTTS (Both Teams to Score) and Over/Under probabilities
    from the scoreline matrix. These markets have thinner bookmaker margins.
    """
    n = matrix.shape[0]

    # BTTS Yes: both teams score at least 1
    btts_yes = matrix[1:, 1:].sum()
    btts_no  = 1.0 - btts_yes

    # Over X.5: total goals > threshold
    total_probs = {}
    for threshold in [1.5, 2.5, 3.5]:
        over = 0.0
        for h in range(n):
            for a in range(n):
                if h + a > threshold:
                    over += matrix[h, a]
        total_probs[f'over_{threshold}'] = over
        total_probs[f'under_{threshold}'] = 1.0 - over

    return {
        'btts_yes': round(btts_yes, 4),
        'btts_no':  round(btts_no, 4),
        **{k: round(v, 4) for k, v in total_probs.items()}
    }
