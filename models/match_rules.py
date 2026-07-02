# models/match_rules.py
"""
FIFA World Cup 2026 match rules engine.
Handles group stage (draw allowed) vs knockout (ET + penalties).
"""
import numpy as np
from models.poisson_dixon_coles import (
    score_probability_matrix, outcome_probs,
    in_play_outcome_probs, in_play_adjust_lambdas
)


KNOCKOUT_STAGES = {'round_of_32', 'round_of_16', 'quarter_final',
                   'semi_final', '3rd_place', 'final'}

GROUP_STAGES    = {'group'}


# ── PENALTY SHOOTOUT MODEL ────────────────────────────────────────────────────
def penalty_win_prob(team_strength_ratio: float = 1.0) -> dict:
    """
    Penalty shootout is approximately 50/50 but slightly favours the stronger
    team. Academic research puts base probability at 50% ± 3-5% based on
    team quality differential.

    team_strength_ratio: home_glicko / away_glicko (normalised)
    Returns: {'home': float, 'away': float}
    """
    base_home = 0.50
    # Glicko-based adjustment: 100-point gap ≈ 2.5% advantage in shootout
    glicko_adjustment = (team_strength_ratio - 1.0) * 0.025
    p_home = np.clip(base_home + glicko_adjustment, 0.35, 0.65)
    return {'home': round(p_home, 4), 'away': round(1 - p_home, 4)}


# ── EXTRA TIME GOAL MODEL ─────────────────────────────────────────────────────
def extra_time_probs(lam_h: float, lam_a: float, rho: float,
                      current_home: int, current_away: int,
                      glicko_ratio: float = 1.0) -> dict:
    """
    Model 30 minutes of Extra Time (ET) as a new Poisson period.
    Goals in ET are rarer — scaling factor ~0.35 of 90-min rate.
    Returns: probabilities for final result after ET.
    """
    ET_SCALE = 0.35   # goals per 30 ET minutes relative to 90-min expectation
    lam_h_et = lam_h * ET_SCALE
    lam_a_et = lam_a * ET_SCALE

    # ET scoreline matrix (max 3 additional goals in 30 min is realistic)
    et_probs = in_play_outcome_probs(
        lam_h_et, lam_a_et,
        current_home, current_away,
        rho=rho,
        max_additional=4
    )

    # P(still tied after ET) — goes to penalties
    p_tied_after_et = _compute_still_tied_prob(
        lam_h_et, lam_a_et, current_home, current_away
    )

    pen = penalty_win_prob(glicko_ratio)

    # Final knockout outcome probabilities
    p_home = et_probs['home_win'] + p_tied_after_et * pen['home']
    p_away = et_probs['away_win'] + p_tied_after_et * pen['away']

    return {
        'home_win':            round(p_home, 4),
        'away_win':            round(p_away, 4),
        'draw':                0.0,   # draws are impossible in knockouts
        'p_extra_time':        round(et_probs['draw'], 4),
        'p_penalties':         round(p_tied_after_et, 4),
        'penalty_home_win':    round(p_tied_after_et * pen['home'], 4),
        'penalty_away_win':    round(p_tied_after_et * pen['away'], 4),
    }


def _compute_still_tied_prob(lam_h_et, lam_a_et,
                               current_home, current_away) -> float:
    """P(both teams score equal additional goals in ET)."""
    from scipy.stats import poisson
    max_g = 3
    p_tied = 0.0
    for dg in range(max_g + 1):
        ph = poisson.pmf(dg, lam_h_et)
        pa = poisson.pmf(dg, lam_a_et)
        p_tied += ph * pa
    return float(p_tied)


# ── MASTER PREDICTION ROUTER ──────────────────────────────────────────────────
def predict_with_stage_rules(stage: str,
                               lam_h: float, lam_a: float,
                               rho: float,
                               glicko_ratio: float = 1.0,
                               # Live params (optional)
                               elapsed: int = None,
                               home_goals: int = 0,
                               away_goals: int = 0,
                               match_period: str = 'regular',
                               red_cards: dict = None) -> dict:
    """
    Master routing function. Applies correct math based on match stage
    and current period (regular / extra_time / penalties).

    match_period: 'regular' | 'extra_time' | 'penalties'
    stage: 'group' | 'round_of_32' | 'round_of_16' | 'quarter_final' |
           'semi_final' | '3rd_place' | 'final'
    """
    is_knockout = stage in KNOCKOUT_STAGES
    is_live     = elapsed is not None

    if not is_knockout:
        # ── GROUP STAGE: draws allowed ──────────────────────────────────────
        if is_live:
            lam_h_rem, lam_a_rem = in_play_adjust_lambdas(
                lam_h, lam_a, elapsed, red_cards=red_cards
            )
            probs = in_play_outcome_probs(
                lam_h_rem, lam_a_rem, home_goals, away_goals, rho
            )
        else:
            matrix = score_probability_matrix(lam_h, lam_a, rho)
            h, d, a = outcome_probs(matrix)
            probs   = {'home_win': h, 'draw': d, 'away_win': a}

        return {**probs, 'stage': stage, 'allows_draw': True,
                'extra_time': False, 'penalties': False}

    # ── KNOCKOUT STAGE ───────────────────────────────────────────────────────
    if match_period == 'penalties':
        # Game is at penalties — 50/50 adjusted for Glicko
        pen = penalty_win_prob(glicko_ratio)
        return {
            'home_win':   pen['home'],
            'away_win':   pen['away'],
            'draw':       0.0,
            'stage':      stage,
            'allows_draw': False,
            'extra_time': False,
            'penalties':  True,
            'message':    'Game at penalties — Glicko-adjusted 50/50'
        }

    if match_period == 'extra_time':
        # In Extra Time — model remaining ET + penalties if still tied
        et_elapsed  = max(0, elapsed - 90) if elapsed else 0
        et_remaining = max(0, 30 - et_elapsed)
        lam_h_et    = lam_h * (et_remaining / 90)
        lam_a_et    = lam_a * (et_remaining / 90)
        probs = extra_time_probs(lam_h_et, lam_a_et, rho,
                                  home_goals, away_goals, glicko_ratio)
        return {**probs, 'stage': stage, 'allows_draw': False,
                'extra_time': True, 'penalties': False}

    # Regular time in a knockout match
    if is_live:
        lam_h_rem, lam_a_rem = in_play_adjust_lambdas(
            lam_h, lam_a, elapsed, red_cards=red_cards
        )
        raw = in_play_outcome_probs(
            lam_h_rem, lam_a_rem, home_goals, away_goals, rho
        )
        # If game is tied after 90, route to ET+penalties model
        p_tied_at_90 = raw['draw']
        et = extra_time_probs(lam_h * 0.35, lam_a * 0.35, rho,
                               home_goals, away_goals, glicko_ratio)
        # Combine: P(win in regular) + P(tied at 90) * P(win in ET/pens)
        p_home = raw['home_win'] + p_tied_at_90 * et['home_win']
        p_away = raw['away_win'] + p_tied_at_90 * et['away_win']
        total  = p_home + p_away
        return {
            'home_win':         round(p_home / total, 4),
            'away_win':         round(p_away / total, 4),
            'draw':             0.0,
            'p_tied_at_90':     round(p_tied_at_90, 4),
            'p_extra_time':     round(p_tied_at_90 * et['p_extra_time'], 4),
            'p_penalties':      round(p_tied_at_90 * et['p_penalties'], 4),
            'stage':            stage,
            'allows_draw':      False,
            'extra_time':       False,
            'penalties':        False,
        }
    else:
        # Pre-match knockout prediction
        matrix = score_probability_matrix(lam_h, lam_a, rho)
        h, d, a = outcome_probs(matrix)
        et = extra_time_probs(lam_h, lam_a, rho, 0, 0, glicko_ratio)
        # P(win in 90) + P(draw at 90) * P(win in ET+pens)
        p_home = h + d * et['home_win']
        p_away = a + d * et['away_win']
        total  = p_home + p_away
        return {
            'home_win':         round(p_home / total, 4),
            'away_win':         round(p_away / total, 4),
            'draw':             0.0,
            'p_draw_at_90':     round(d, 4),
            'p_extra_time':     round(d * et['p_extra_time'], 4),
            'p_penalties':      round(d * et['p_penalties'], 4),
            'penalty_home_win': round(d * et['penalty_home_win'], 4),
            'penalty_away_win': round(d * et['penalty_away_win'], 4),
            'stage':            stage,
            'allows_draw':      False,
            'extra_time':       False,
            'penalties':        False,
        }
