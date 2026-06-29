import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


def compute_time_decay_weights(dates, xi=0.0065):
    T = pd.to_datetime(dates).max()
    days_ago = (T - pd.to_datetime(dates)).dt.days.values
    weights  = np.exp(-xi * days_ago)
    return weights / weights.sum() * len(weights)


# --- STAGE 1: Fit attack/defense with identifiability constraint --------------
def fit_attack_defense(home_teams, away_teams, home_goals, away_goals,
                       all_teams, time_weights=None, sample_weight=None):
    """
    Stage 1: MLE for attack/defense parameters only.
    Identifiability penalty: 1000 * sum(attack)^2 anchors mean(attack)=0.
    """
    n_teams = len(all_teams)
    team_idx = {t: i for i, t in enumerate(all_teams)}
    if sample_weight is not None:
        W = sample_weight
    else:
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
                      options={'maxiter': 100, 'ftol': 1e-8})
    params = result.x
    attack  = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    return attack, defense, home_adv, team_idx


# --- STAGE 2: Estimate rho with fixed attack/defense ----------------------------
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
    print(f"  rho converged to: {rho:.4f}")
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

def in_play_adjust_lambdas(lam_h: float, lam_a: float,
                            elapsed_minutes: int,
                            full_time: int = 90,
                            red_cards: dict = None) -> tuple:
    """
    Adjust pre-match Poisson intensities for remaining time.

    The fraction of the match remaining scales the remaining expected goals.
    Red card adjustment: each red card removes ~0.35 expected goals/90min.

    Formula: lam_remaining = lam_full * (remaining / full_time)
    """
    remaining = max(0, full_time - elapsed_minutes)
    time_fraction = remaining / full_time

    # Scale by time remaining
    lam_h_rem = lam_h * time_fraction
    lam_a_rem = lam_a * time_fraction

    # Red card penalty: each red card reduces attacking threat
    if red_cards:
        home_reds = red_cards.get("home", 0)
        away_reds = red_cards.get("away", 0)
        RED_CARD_PENALTY = 0.35   # goals per full match
        lam_h_rem = max(0.02, lam_h_rem - home_reds * RED_CARD_PENALTY * time_fraction)
        lam_a_rem = max(0.02, lam_a_rem - away_reds * RED_CARD_PENALTY * time_fraction)

    return lam_h_rem, lam_a_rem


def in_play_outcome_probs(lam_h_rem: float, lam_a_rem: float,
                           current_home: int, current_away: int,
                           rho: float = -0.13,
                           max_additional: int = 6) -> dict:
    """
    Compute live outcome probabilities given:
    - Remaining expected goals (already time-adjusted)
    - Current scoreline

    Method: sum over all possible remaining goal combinations,
    conditioned on the current score to compute final outcome probs.

    P(final outcome | current score) = sum over (delta_h, delta_a) of:
        P(Home scores delta_h more) * P(Away scores delta_a more)
        weighted by DC tau correction
    """
    additional_goals = np.arange(max_additional + 1)

    ph = poisson.pmf(additional_goals, lam_h_rem)
    pa = poisson.pmf(additional_goals, lam_a_rem)
    joint = np.outer(ph, pa)

    # Apply tau correction for 0-0 remaining (i.e., no more goals = 0-0 fragment)
    tau = np.ones_like(joint)
    tau[0, 0] = max(0.01, 1.0 - lam_h_rem * lam_a_rem * rho)
    tau[0, 1] = 1.0 + lam_h_rem * rho
    tau[1, 0] = 1.0 + lam_a_rem * rho
    if max_additional >= 1:
        tau[1, 1] = 1.0 - rho
    joint = joint * tau
    joint = joint / joint.sum()

    p_home_win = 0.0
    p_draw     = 0.0
    p_away_win = 0.0

    for dh in range(max_additional + 1):
        for da in range(max_additional + 1):
            final_h = current_home + dh
            final_a = current_away + da
            p       = joint[dh, da]
            if final_h > final_a:
                p_home_win += p
            elif final_h == final_a:
                p_draw     += p
            else:
                p_away_win += p

    total = p_home_win + p_draw + p_away_win
    if total < 0.001:
        return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}

    return {
        "home_win": round(p_home_win / total, 4),
        "draw":     round(p_draw     / total, 4),
        "away_win": round(p_away_win / total, 4),
    }


def live_in_play_predict(lam_h_prematch: float, lam_a_prematch: float,
                          elapsed: int, home_goals: int, away_goals: int,
                          rho: float = -0.13,
                          red_cards: dict = None,
                          live_state: dict = None) -> dict:
    """
    Master in-play prediction function.
    Bypasses ML meta-learner entirely — pure Dixon-Coles time-decay.

    Also computes live-adjusted BTTS and O/U from remaining goals.
    """
    # Step 1: Adjust lambdas for time remaining + red cards
    lam_h_rem, lam_a_rem = in_play_adjust_lambdas(
        lam_h_prematch, lam_a_prematch, elapsed, red_cards=red_cards
    )

    # Step 2: Condition on current scoreline
    probs = in_play_outcome_probs(
        lam_h_rem, lam_a_rem, home_goals, away_goals, rho
    )

    # Step 3: Live BTTS (already 1 or more goals may have been scored)
    home_already_scored = home_goals > 0
    away_already_scored = away_goals > 0

    if home_already_scored and away_already_scored:
        btts_yes = 1.0
    elif home_already_scored:
        p_away_scores = 1.0 - poisson.pmf(0, lam_a_rem)
        btts_yes = p_away_scores
    elif away_already_scored:
        p_home_scores = 1.0 - poisson.pmf(0, lam_h_rem)
        btts_yes = p_home_scores
    else:
        btts_yes = (1 - poisson.pmf(0, lam_h_rem)) * (1 - poisson.pmf(0, lam_a_rem))

    # Step 4: O/U 2.5 live (goals already scored + remaining expected)
    total_goals_so_far = home_goals + away_goals
    if total_goals_so_far > 2:
        over_25 = 1.0
    elif total_goals_so_far == 2:
        p_at_least_one_more = 1 - poisson.pmf(0, lam_h_rem + lam_a_rem)
        over_25 = p_at_least_one_more
    else:
        remaining_needed = 3 - total_goals_so_far
        p_under = sum(
            poisson.pmf(k, lam_h_rem + lam_a_rem)
            for k in range(remaining_needed)
        )
        over_25 = 1 - p_under

    # Step 5: Momentum signal from live stats
    momentum_signal = _compute_live_momentum(live_state) if live_state else 0.0

    remaining = max(0, 90 - elapsed)
    return {
        "mode":            "in_play",
        "elapsed":         elapsed,
        "remaining":       remaining,
        "current_score":   f"{home_goals}-{away_goals}",
        "home_win_prob":   probs["home_win"],
        "draw_prob":       probs["draw"],
        "away_win_prob":   probs["away_win"],
        "btts_yes":        round(float(btts_yes), 4),
        "over_25":         round(float(over_25), 4),
        "lam_h_remaining": round(lam_h_rem, 4),
        "lam_a_remaining": round(lam_a_rem, 4),
        "momentum_signal": round(momentum_signal, 4),
        "red_cards":       red_cards or {},
    }


def _compute_live_momentum(state: dict) -> float:
    """
    Compute a momentum signal from live stats.
    Positive = home team dominating; negative = away team dominating.
    Based on: shots on target, xG, corners, possession.
    """
    if not state:
        return 0.0

    signals = []

    # Shots on target differential (normalised)
    h_sot = state.get("home_shots_ot", 0) or 0
    a_sot = state.get("away_shots_ot", 0) or 0
    if h_sot + a_sot > 0:
        signals.append((h_sot - a_sot) / (h_sot + a_sot))

    # xG differential
    h_xg = state.get("home_xg_live") or state.get("home_xg_live_ss") or 0
    a_xg = state.get("away_xg_live") or state.get("away_xg_live_ss") or 0
    try:
        h_xg, a_xg = float(h_xg), float(a_xg)
        if h_xg + a_xg > 0:
            signals.append((h_xg - a_xg) / (h_xg + a_xg))
    except (ValueError, TypeError):
        pass

    # Corner differential
    h_cor = state.get("home_corners", 0) or 0
    a_cor = state.get("away_corners", 0) or 0
    if h_cor + a_cor > 0:
        signals.append((h_cor - a_cor) / (h_cor + a_cor) * 0.5)

    # Possession (as fraction deviation from 50/50)
    h_pos = state.get("home_possession", 50) or 50
    try:
        signals.append((float(h_pos) - 50) / 50 * 0.3)
    except (ValueError, TypeError):
        pass

    return float(np.mean(signals)) if signals else 0.0
