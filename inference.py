# inference.py — Live inference engine for V6.2
# Bridges trained model artifacts -> API server -> Telegram bot

import json
import os
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("model_versions/latest")
STATES_PATH = MODEL_DIR / "team_states.json"

# ─── Cached globals (loaded once at startup) ──────────────────────────────────
_base_learners  = None
_meta_learner   = None
_scaler         = None
_team_states    = None
_feature_cols   = None
_dc_params      = None   # {attack, defense, home_adv, rho, team_idx}


def _load_artifacts():
    """Load all model artifacts into memory. Called once at module import."""
    global _base_learners, _meta_learner, _scaler, _team_states, _feature_cols, _dc_params

    if _meta_learner is not None:
        return   # already loaded

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Run train_test.py first to generate model artifacts."
        )

    print("  🔄 Loading model artifacts...")

    _base_learners = joblib.load(MODEL_DIR / "base_learners.joblib")
    _meta_learner  = joblib.load(MODEL_DIR / "meta_learner.joblib")

    scaler_path = MODEL_DIR / "scaler.joblib"
    _scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(STATES_PATH) as f:
        _team_states = json.load(f)

    with open(MODEL_DIR / "manifest.json") as f:
        manifest = json.load(f)
    _feature_cols = manifest["feature_cols"]

    dc_path = MODEL_DIR / "dc_params.joblib"
    _dc_params = joblib.load(dc_path) if dc_path.exists() else None

    print(f"  ✅ Loaded V{manifest['version']} ({manifest.get('build', 'unknown')}) | "
          f"Features: {_feature_cols} | Teams: {len(_team_states)}")


# Load on import
_load_artifacts()


# ─── Team State Retrieval ─────────────────────────────────────────────────────
def _get_team_state(team_name: str) -> dict:
    """
    Retrieve latest feature state for a team.
    Tries exact match first, then case-insensitive fuzzy match.
    Returns safe defaults if team not found (logs warning).
    """
    # Exact match
    if team_name in _team_states:
        return _team_states[team_name]

    # Case-insensitive match
    lower = {k.lower(): v for k, v in _team_states.items()}
    if team_name.lower() in lower:
        return lower[team_name.lower()]

    # Partial match (e.g. "USA" matches "United States")
    for key in _team_states:
        if team_name.lower() in key.lower() or key.lower() in team_name.lower():
            warnings.warn(f"Team '{team_name}' matched to '{key}' via partial match.")
            return _team_states[key]

    warnings.warn(f"Team '{team_name}' not found in team_states.json. Using defaults.")
    return {
        "glicko": 1500.0, "rd": 200.0,
        "xg_rolling_3": 1.2, "neutral_venue_form": 1.2,
        "last_match_date": ""
    }


# ─── Feature Construction ─────────────────────────────────────────────────────
def _build_feature_vector(home: dict, away: dict,
                           venue_factor: float, stage: str) -> np.ndarray:
    """
    Build the exact FEATURE_COLS vector for the base learners.
    Derived from team states + match context.
    Column order MUST match manifest["feature_cols"].
    """
    stage_pressure_map = {
        "group": 0.5, "round_of_16": 0.65,
        "quarter": 0.80, "semi": 0.90, "final": 1.0
    }
    stage_pressure = stage_pressure_map.get(stage, 0.5)

    # xG gap and draw affinity
    xg_gap = abs(home["xg_rolling_3"] - away["xg_rolling_3"])
    draw_affinity = max(0.0, 1.0 - xg_gap / 3.0)

    # Neutral venue form: 60% away form, 40% home form (V4.1 standard)
    home_neutral = 0.4 * home["xg_rolling_3"] + 0.6 * home.get("neutral_venue_form", home["xg_rolling_3"])
    away_neutral = 0.4 * away["xg_rolling_3"] + 0.6 * away.get("neutral_venue_form", away["xg_rolling_3"])

    # xG supremacy (relative)
    total_xg = home["xg_rolling_3"] + away["xg_rolling_3"] + 1e-9
    xg_supremacy = home["xg_rolling_3"] / total_xg

    # Glicko signal
    rd_combined = np.sqrt(home["rd"]**2 + away["rd"]**2) + 1e-9
    glicko_signal = (home["glicko"] - away["glicko"]) / rd_combined

    # Rest differential — estimate from last_match_date if available
    def days_since(date_str):
        if not date_str:
            return 4   # assume normal rest
        try:
            d = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            return max(1, (datetime.now(timezone.utc) - d).days)
        except Exception:
            return 4

    home_rest = days_since(home.get("last_match_date", ""))
    away_rest = days_since(away.get("last_match_date", ""))
    rest_differential = float(home_rest - away_rest)

    # Regime filter: if Glicko gap > 400, return None to trigger NO_BET
    glicko_gap_raw = abs(home["glicko"] - away["glicko"])
    if glicko_gap_raw > 400:
        warnings.warn(f"Glicko gap {glicko_gap_raw:.0f} > 400. Regime filter active.")
        return None

    # Build vector in FEATURE_COLS order (from manifest)
    feature_map = {
        "home_glicko":             home["glicko"],
        "home_rd":                 home["rd"],
        "away_glicko":             away["glicko"],
        "away_rd":                 away["rd"],
        "xg_supremacy":            xg_supremacy,
        "glicko_signal":           glicko_signal,
        "draw_affinity":           draw_affinity,
        "home_neutral_venue_form": home_neutral,
        "away_neutral_venue_form": away_neutral,
        "rest_differential":       rest_differential,
        "defensive_balance":       home.get("defensive_balance", 0.0) - away.get("defensive_balance", 0.0),
        "stage_pressure":          stage_pressure,
        "h2h_draw_rate":           0.0, # usually from historical df
        # SAS features — default to 1.0 if not in states
        "home_sas":                home.get("sas", 1.0),
        "away_sas":                away.get("sas", 1.0),
        "sas_differential":        home.get("sas", 1.0) - away.get("sas", 1.0),
        "sharp_line_movement":     0.0,   # provided by caller if available
    }

    return np.array([feature_map.get(col, 0.0) for col in _feature_cols], dtype=float)


# ─── Poisson Dixon-Coles Blending ────────────────────────────────────────────
def _dc_predict(home_team: str, away_team: str, venue_factor: float):
    """
    Run Dixon-Coles prediction. Returns (ph, pd, pa) or None if params unavailable.
    """
    if _dc_params is None:
        return None

    try:
        from models.poisson_dixon_coles import score_probability_matrix, outcome_probs, extract_btts_ou

        attack   = _dc_params["attack"]
        defense  = _dc_params["defense"]
        home_adv = _dc_params["home_adv"]
        rho      = _dc_params["rho"]

        if home_team not in attack or away_team not in attack:
            return None

        effective_home_adv = home_adv * venue_factor

        lam_h = np.exp(attack[home_team] - defense[away_team] + effective_home_adv)
        lam_a = np.exp(attack[away_team] - defense[home_team])

        matrix = score_probability_matrix(lam_h, lam_a, rho)
        ph, pd, pa = outcome_probs(matrix)
        btts_ou = extract_btts_ou(matrix)

        return {"home_win": ph, "draw": pd, "away_win": pa, **btts_ou}

    except Exception as e:
        warnings.warn(f"DC prediction failed: {e}")
        return None


# ─── No-Vig Edge Calculation ──────────────────────────────────────────────────
def _compute_betting_math(model_probs: dict,
                           home_odds: float = None,
                           draw_odds: float = None,
                           away_odds: float = None) -> dict:
    """Compute no-vig edge and Kelly fractions for each market."""
    if not all([home_odds, draw_odds, away_odds]):
        return {"no_vig_edge": None, "best_bet": None, "kelly_fraction": None}

    raw = [1/home_odds, 1/draw_odds, 1/away_odds]
    overround = sum(raw)
    novig = [r / overround for r in raw]

    outcomes = ["Home Win", "Draw", "Away Win"]
    model_p  = [model_probs["home_win"], model_probs["draw"], model_probs["away_win"]]
    odds_list = [home_odds, draw_odds, away_odds]

    edges = [mp - nv for mp, nv in zip(model_p, novig)]
    best_idx = int(np.argmax(edges))
    best_edge = edges[best_idx]

    MIN_EDGE = 0.025
    if best_edge < MIN_EDGE:
        return {"no_vig_edge": round(best_edge, 4), "best_bet": "NO BET",
                "kelly_fraction": 0.0, "novig_probs": novig}

    p = model_p[best_idx]
    b = odds_list[best_idx] - 1
    kelly_full = (b * p - (1 - p)) / b
    kelly_quarter = max(0.0, kelly_full * 0.25)
    kelly_capped  = min(kelly_quarter, 0.05)   # never > 5% per bet

    return {
        "no_vig_edge":    round(best_edge, 4),
        "best_bet":       outcomes[best_idx],
        "kelly_fraction": round(kelly_capped, 4),
        "novig_probs":    [round(n, 4) for n in novig],
        "all_edges":      {o: round(e, 4) for o, e in zip(outcomes, edges)}
    }


# ─── Main Inference Function ─────────────────────────────────────────────────
def run_inference(home_team: str, away_team: str,
                  venue_factor: float = 0.3, stage: str = "group",
                  home_odds: float = None, draw_odds: float = None,
                  away_odds: float = None) -> dict:
    """
    Full inference pipeline. Returns prediction dict ready for API/Telegram.
    """
    # 1. Retrieve team states
    home_state = _get_team_state(home_team)
    away_state = _get_team_state(away_team)

    # 2. Build feature vector
    X = _build_feature_vector(home_state, away_state, venue_factor, stage)

    if X is None:
        return {
            "home_team": home_team, "away_team": away_team,
            "regime_filtered": True,
            "message": "Regime filter: Glicko gap > 400. No prediction issued.",
            "model_version": "6.2"
        }

    # 3. Scale if scaler available
    X_input = _scaler.transform(X.reshape(1, -1)) if _scaler else X.reshape(1, -1)

    # 4. Base learner OOF-style predictions
    cat_h, cat_a, xgb_h, xgb_a, ridge_h, ridge_a = _base_learners
    
    pred_h = (cat_h.predict(X_input) + xgb_h.predict(X_input) + ridge_h.predict(X_input)) / 3.0
    pred_a = (cat_a.predict(X_input) + xgb_a.predict(X_input) + ridge_a.predict(X_input)) / 3.0
    
    from train_test import xg_to_probs
    base_blend = xg_to_probs(pred_h, pred_a)[0]

    # 5. Dixon-Coles prediction (50/50 blend)
    dc_result = _dc_predict(home_team, away_team, venue_factor)

    if dc_result:
        dc_probs = np.array([dc_result["home_win"], dc_result["draw"], dc_result["away_win"]])
        ensemble_probs = 0.5 * base_blend + 0.5 * dc_probs
    else:
        ensemble_probs = base_blend
        dc_result = {}

    # 6. Meta-learner gate
    meta_input = ensemble_probs.reshape(1, -1)
    final_proba = _meta_learner.predict_proba(meta_input)[0]

    # Ensure sums to 1 and is clipped
    final_proba = np.clip(final_proba, 0.05, 0.95)
    final_proba = final_proba / final_proba.sum()

    classes = list(_meta_learner.classes_)
    ph_idx = classes.index("Home Win") if "Home Win" in classes else classes.index(0)
    pd_idx = classes.index("Draw") if "Draw" in classes else classes.index(1)
    pa_idx = classes.index("Away Win") if "Away Win" in classes else classes.index(2)

    ph = float(final_proba[ph_idx])
    pd = float(final_proba[pd_idx])
    pa = float(final_proba[pa_idx])

    # Draw threshold gate (validated at 0.34 in V6.1, so keeping 0.34 here as user explicitly told us to matching V6.1 validation or Option C validation. Wait, the user said "Keep 0.34 in inference.py - match what train_test.py validated, or your live predictions will contradict". Wait! Option C validation used 0.38, but the user's snippet explicitly said "Draw threshold gate (validated at 0.34 in V6.1)". Wait! The user's prompt said "Precision Fix 1 — Draw Threshold Mismatch: Your plan says draw_thresh=0.38 with draw_affinity >= 0.45. But your V6.1 validation was tuned at draw_thresh=0.34. Do not change the threshold without re-validating. Keep 0.34 in inference.py — match what train_test.py validated." 
    # Okay, I will use 0.34 in inference.py.
    draw_affinity_val = float(X[_feature_cols.index("draw_affinity")])
    DRAW_THRESH = 0.34
    if pd >= DRAW_THRESH and draw_affinity_val >= 0.45:
        final_pick = "Draw"
    else:
        final_pick = ["Home Win", "Draw", "Away Win"][int(np.argmax(final_proba))]

    # 7. Confidence from ECE + fold_std
    ECE_CURRENT = 0.0813
    fold_std = 0.0283
    if ECE_CURRENT < 0.05 and fold_std < 0.04:
        confidence = "HIGH CONFIDENCE"
    elif ECE_CURRENT < 0.08 and fold_std < 0.06:
        confidence = "MODERATE CONFIDENCE"
    else:
        confidence = "LOW CONFIDENCE — reduced stake recommended"

    # 8. Betting math
    betting = _compute_betting_math(
        {"home_win": ph, "draw": pd, "away_win": pa},
        home_odds, draw_odds, away_odds
    )

    return {
        "home_team":      home_team,
        "away_team":      away_team,
        "home_win_prob":  round(ph, 4),
        "draw_prob":      round(pd, 4),
        "away_win_prob":  round(pa, 4),
        "final_pick":     final_pick,
        "confidence":     confidence,
        "no_vig_edge":    betting.get("no_vig_edge"),
        "best_bet":       betting.get("best_bet"),
        "kelly_fraction": betting.get("kelly_fraction"),
        "all_edges":      betting.get("all_edges"),
        "btts_yes":       dc_result.get("btts_yes"),
        "over_25":        dc_result.get("over_25"),
        "regime_filtered":False,
        "model_version":  "6.2",
        "timestamp":      datetime.now(timezone.utc).isoformat()
    }
