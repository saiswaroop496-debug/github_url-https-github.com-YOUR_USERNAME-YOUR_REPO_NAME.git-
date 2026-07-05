import pandas as pd
import numpy as np
from scipy.stats import nbinom, poisson
from scipy.optimize import minimize, minimize_scalar
from models.copula_goals import bivariate_score_matrix


# ── Overdispersion parameter (estimated from WC data) ────────────────────────
# r=5 is the published estimate for international football goals.
# Lower r = more overdispersion (more 0-0 and 5-0 matches).
# Estimate from your data: np.var(goals)/np.mean(goals) = 1 + mean/r → r = mean²/(var-mean)
NB_R = 4.42


def nb_pmf(k: int, mu: float, r: float = NB_R) -> float:
    """
    Negative Binomial PMF. Replaces poisson.pmf() everywhere.
    Captures overdispersion: variance = mu + mu²/r > mu (unlike Poisson where var=mu).
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    p = r / (r + mu)
    return float(nbinom.pmf(k, r, p))


def nb_pmf_array(goals: np.ndarray, mu: float, r: float = NB_R) -> np.ndarray:
    """Vectorized NB PMF for an array of goal counts."""
    if mu <= 0:
        result = np.zeros(len(goals))
        result[0] = 1.0
        return result
    p = r / (r + mu)
    return nbinom.pmf(goals, r, p)


def estimate_overdispersion(goals_series: np.ndarray) -> float:
    """
    Estimate NB overdispersion r from historical goal data.
    r = mean² / (var - mean). Use r=5 if variance <= mean (no overdispersion).
    """
    mu  = goals_series.mean()
    var = goals_series.var()
    if var <= mu or mu <= 0:
        return 5.0   # default
    r = mu**2 / (var - mu)
    return float(np.clip(r, 1.0, 50.0))


# ── Bayesian Shrinkage for Attack/Defense Parameters ─────────────────────────
SHRINKAGE_ALPHA = 3.0   # pseudo-count: teams with <3 matches shrink heavily to prior

def bayesian_shrink_lambdas(lam_estimated: float,
                             lam_global_mean: float,
                             n_matches: int,
                             alpha: float = SHRINKAGE_ALPHA) -> float:
    """
    Empirical Bayes shrinkage toward league mean.
    Formula: λ_shrunk = (n * λ_obs + α * λ_global) / (n + α)
    Teams with n < α get pulled strongly toward the global average.
    """
    return (n_matches * lam_estimated + alpha * lam_global_mean) / (n_matches + alpha)


def fit_dixon_coles_nb(df, dates=None, xi: float = 0.0065,
                        r: float = None) -> dict:
    """
    V7.3 Dixon-Coles MLE using Negative Binomial distributions.
    Two-stage:
      Stage 1: Fit attack/defense with NB likelihood + identifiability constraint
      Stage 2: Estimate rho with concentrated NB likelihood
    Also applies Bayesian shrinkage to stabilize small-sample teams.
    """
    home_teams  = df['home_team'].values
    away_teams  = df['away_team'].values
    home_goals  = df['home_goals'].astype(int).values
    away_goals  = df['away_goals'].astype(int).values
    all_teams   = sorted(set(home_teams) | set(away_teams))
    team_idx    = {t: i for i, t in enumerate(all_teams)}
    n_teams     = len(all_teams)

    # Time decay weights
    if dates is not None:
        from models.base_learners import compute_match_weights
        tournaments = df['tournament'] if 'tournament' in df.columns else None
        
        # We need pandas Series for the new compute_match_weights, dates in fit_dixon_coles_nb 
        # is currently passed as a numpy array in train_test.py (dates.values). We should convert it to Series.
        import pandas as pd
        dates_series = pd.Series(dates)
        weights = compute_match_weights(dates_series, tournaments, xi=xi)
    else:
        weights = np.ones(len(home_goals))

    # Estimate overdispersion from data if not provided
    if r is None:
        all_goals = np.concatenate([home_goals, away_goals])
        r = estimate_overdispersion(all_goals)
        print(f"  NB overdispersion r={r:.2f} (Poisson would assume r→∞)")

    # Pre-map indices for vectorization
    h_idx = np.array([team_idx[t] for t in home_teams])
    a_idx = np.array([team_idx[t] for t in away_teams])

    # Stage 1: Fit attack/defense
    def neg_ll_stage1(params):
        attack   = params[:n_teams]
        defense  = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        
        lh = np.exp(attack[h_idx] - defense[a_idx] + home_adv)
        la = np.exp(attack[a_idx] - defense[h_idx])
        
        p = r / (r + lh)
        ll_h = nbinom.logpmf(home_goals, r, p)
        
        p_a = r / (r + la)
        ll_a = nbinom.logpmf(away_goals, r, p_a)
        
        ll = np.sum(weights * (np.nan_to_num(ll_h, nan=-10.0) + np.nan_to_num(ll_a, nan=-10.0)))
        penalty = 1000.0 * (np.sum(attack)**2)
        return -ll + penalty

    x0 = np.zeros(2*n_teams + 1)
    res = minimize(neg_ll_stage1, x0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-8})
    params      = res.x
    attack_raw  = params[:n_teams]
    defense_raw = params[n_teams:2*n_teams]
    home_adv    = params[2*n_teams]

    # Bayesian shrinkage: count matches per team, shrink toward global mean
    match_counts = {}
    for t in all_teams:
        match_counts[t] = (
            (df['home_team'] == t).sum() + (df['away_team'] == t).sum()
        )
    global_attack_mean  = np.exp(attack_raw).mean()
    global_defense_mean = np.exp(defense_raw).mean()

    attack_shrunk  = np.zeros(n_teams)
    defense_shrunk = np.zeros(n_teams)
    for i, team in enumerate(all_teams):
        n = match_counts[team]
        raw_lam_att = np.exp(attack_raw[i])
        raw_lam_def = np.exp(defense_raw[i])
        attack_shrunk[i]  = np.log(bayesian_shrink_lambdas(raw_lam_att, global_attack_mean, n))
        defense_shrunk[i] = np.log(bayesian_shrink_lambdas(raw_lam_def, global_defense_mean, n))

    # Stage 2: Estimate rho with NB likelihood, fixed attack/defense
    def neg_ll_rho(rho):
        lh = np.exp(attack_shrunk[h_idx] - defense_shrunk[a_idx] + home_adv)
        la = np.exp(attack_shrunk[a_idx] - defense_shrunk[h_idx])
        
        # Vectorized dc_tau computation
        tau = np.ones(len(home_goals))
        
        # 0-0
        mask_00 = (home_goals == 0) & (away_goals == 0)
        tau[mask_00] = np.maximum(0.001, 1.0 - lh[mask_00] * la[mask_00] * rho)
        
        # 0-1
        mask_01 = (home_goals == 0) & (away_goals == 1)
        tau[mask_01] = np.maximum(0.001, 1.0 + lh[mask_01] * rho)
        
        # 1-0
        mask_10 = (home_goals == 1) & (away_goals == 0)
        tau[mask_10] = np.maximum(0.001, 1.0 + la[mask_10] * rho)
        
        # 1-1
        mask_11 = (home_goals == 1) & (away_goals == 1)
        tau[mask_11] = np.maximum(0.001, 1.0 - rho)
        
        ll = np.sum(weights * np.log(tau))
        if np.any(tau <= 0):
            return 1e9
        return -ll

    rho_result = minimize_scalar(neg_ll_rho, bounds=(-0.35, 0.35), method='bounded')
    rho_fitted = float(rho_result.x)
    print(f"  rho={rho_fitted:.4f} (NB-corrected)")

    return {
        'attack':      {t: float(attack_shrunk[i])  for t, i in team_idx.items()},
        'defense':     {t: float(defense_shrunk[i]) for t, i in team_idx.items()},
        'home_adv':    float(home_adv),
        'rho':         rho_fitted,
        'r_nb':        float(r),
        'team_idx':    {t: i for t, i in team_idx.items()},
        'match_counts': match_counts,
    }


def dc_tau_scalar(gh, ga, lh, la, rho) -> float:
    if gh == 0 and ga == 0: return max(0.001, 1.0 - lh*la*rho)
    if gh == 0 and ga == 1: return max(0.001, 1.0 + lh*rho)
    if gh == 1 and ga == 0: return max(0.001, 1.0 + la*rho)
    if gh == 1 and ga == 1: return max(0.001, 1.0 - rho)
    return 1.0


def score_probability_matrix(lam_h: float, lam_a: float,
                               rho: float,
                               r: float = NB_R,
                               max_goals: int = 8) -> np.ndarray:
    """
    V7.3: Score matrix using Negative Binomial + DC tau correction.
    """
    g   = np.arange(max_goals + 1)
    ph  = nb_pmf_array(g, lam_h, r)
    pa  = nb_pmf_array(g, lam_a, r)
    joint = np.outer(ph, pa)

    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = max(0.001, 1.0 - lam_h * lam_a * rho)
    tau[0, 1] = max(0.001, 1.0 + lam_h * rho)
    tau[1, 0] = max(0.001, 1.0 + lam_a * rho)
    if max_goals >= 1:
        tau[1, 1] = max(0.001, 1.0 - rho)

    matrix = joint * tau
    total  = matrix.sum()
    return matrix / total if total > 1e-9 else joint / joint.sum()


def outcome_probs(matrix: np.ndarray) -> tuple:
    h = float(np.tril(matrix, k=-1).sum())
    d = float(np.trace(matrix))
    a = float(np.triu(matrix, k=1).sum())
    total = h + d + a
    return h/total, d/total, a/total


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

    # Red card penalty: each red card reduces attacking threat AND boosts opponent threat
    if red_cards:
        home_reds = red_cards.get("home", 0)
        away_reds = red_cards.get("away", 0)
        RED_CARD_PENALTY = 0.35   # goals per full match (attacking drop)
        RED_CARD_BOOST   = 0.20   # goals per full match (opponent boost)
        
        lam_h_rem = max(0.02, lam_h_rem - home_reds * RED_CARD_PENALTY * time_fraction + away_reds * RED_CARD_BOOST * time_fraction)
        lam_a_rem = max(0.02, lam_a_rem - away_reds * RED_CARD_PENALTY * time_fraction + home_reds * RED_CARD_BOOST * time_fraction)

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
    # Apply Frank Copula for bivariate dependency
    # The copula already handles the exact 0-0 interaction and dependency
    joint = bivariate_score_matrix(lam_h_rem, lam_a_rem, max_g=max_additional)
    # Ensure it sums strictly to 1.0
    joint = joint / np.sum(joint)

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
        "home_win": float(p_home_win / total),
        "draw":     float(p_draw     / total),
        "away_win": float(p_away_win / total),
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
