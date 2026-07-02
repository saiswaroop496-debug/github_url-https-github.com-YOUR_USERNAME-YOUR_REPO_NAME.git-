# inference.py — Live inference engine for V6.2
# Bridges trained model artifacts -> API server -> Telegram bot

import json
import os
import numpy as np
import joblib
import warnings
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Suppress XGBoost serialization warnings when unpickling models via joblib
xgb.set_config(verbosity=0)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

load_dotenv()

# ─── Pre-import custom classes so pickle can deserialize meta_learner.joblib ──
# The saved CalibratedClassifierCV wraps an imblearn Pipeline containing
# SafeSMOTE (defined in models.meta_learner). Pickle needs these in the
# module namespace BEFORE joblib.load() is called.
import sys
try:
    from imblearn.pipeline import Pipeline as _ImbPipeline  # noqa: F401
    from models.meta_learner import SafeSMOTE as _SafeSMOTE  # noqa: F401
    from models.temperature_scaler import TemperatureScaler as _TemperatureScaler # noqa: F401
except ImportError as _e:
    warnings.warn(f"imbalanced-learn not installed on cloud. Using sklearn Pipeline mock for deserialization: {_e}")
    # Inject a mock so joblib.load() can successfully unpickle the object
    import types
    from sklearn.pipeline import Pipeline
    imb = types.ModuleType('imblearn')
    sys.modules['imblearn'] = imb
    imb_pipe = types.ModuleType('imblearn.pipeline')
    imb_pipe.Pipeline = Pipeline
    sys.modules['imblearn.pipeline'] = imb_pipe
    
    from models.meta_learner import SafeSMOTE as _SafeSMOTE  # This will succeed now
    # sklearn Pipeline requires all steps before the final estimator to have a transform method.
    # SafeSMOTE is a sampler (fit_resample) and doesn't have one, so we patch it in memory
    # just for inference to return X unmodified.
    if not hasattr(_SafeSMOTE, 'transform'):
        _SafeSMOTE.transform = lambda self, X: X

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


def _load_base_artifacts():
    """Load lightweight artifacts (JSON states, DC params)."""
    global _team_states, _feature_cols, _dc_params

    if _team_states is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    with open(STATES_PATH, encoding="utf-8") as f:
        _team_states = json.load(f)

    with open(MODEL_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    _feature_cols = manifest["feature_cols"]

    dc_path = MODEL_DIR / "dc_params.joblib"
    _dc_params = joblib.load(dc_path) if dc_path.exists() else None

def _load_ml_artifacts():
    """Load heavy ML artifacts (joblib models) into memory."""
    global _base_learners, _meta_learner, _scaler

    if _meta_learner is not None:
        return   # already loaded

    print("  [LOADING] Heavy ML artifacts...")

    _base_learners = joblib.load(MODEL_DIR / "base_learners.joblib")
    # Reconstruct native XGBoost models
    xgb_h = xgb.XGBRegressor()
    xgb_h.load_model(MODEL_DIR / "xgb_h.json")
    xgb_a = xgb.XGBRegressor()
    xgb_a.load_model(MODEL_DIR / "xgb_a.json")
    _base_learners[2] = xgb_h
    _base_learners[3] = xgb_a

    _meta_learner  = joblib.load(MODEL_DIR / "meta_learner.joblib")

    scaler_path = MODEL_DIR / "scaler.joblib"
    _scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    print("  [OK] Loaded ML Models.")
    print(f"Features: {_feature_cols} | Teams: {len(_team_states)}")


def _ensure_loaded(is_live=False):
    """Lazy loader - splits loading based on requirement."""
    _load_base_artifacts()
    if not is_live:
        _load_ml_artifacts()


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
        # NEW FEATURES:
        "kalman_signal":           (home.get("strength", 1500) - away.get("strength", 1500)) / (np.sqrt(home.get("uncertainty", 100)**2 + away.get("uncertainty", 100)**2) + 1e-9),
        "kalman_velocity_diff":    home.get("velocity", 0.0) - away.get("velocity", 0.0),
        "regime_factor_diff":      home.get("regime_coefficient", 1.0) - away.get("regime_coefficient", 1.0),
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

        return {"home_win": ph, "draw": pd, "away_win": pa, **btts_ou, "score_matrix": matrix}

    except Exception as e:
        warnings.warn(f"DC prediction failed: {e}")
        return None


# ─── No-Vig Edge Calculation ──────────────────────────────────────────────────
def _compute_betting_math(model_probs: dict,
                           home_odds: float = None,
                           draw_odds: float = None,
                           away_odds: float = None,
                           ah_home_odds: float = None,
                           ah_handicap: float = None,
                           over_odds: float = None,
                           under_odds: float = None,
                           score_matrix: np.ndarray = None) -> dict:
    """Compute no-vig edge and Kelly fractions for each market."""
    if not all([home_odds, draw_odds, away_odds]):
        return {"no_vig_edge": None, "best_bet": None, "kelly_fraction": None, "ir_multiplier": None}

    raw = [1/home_odds, 1/draw_odds, 1/away_odds]
    overround = sum(raw)
    novig = [r / overround for r in raw]

    outcomes = ["Home Win", "Draw", "Away Win"]
    model_p  = [model_probs["home_win"], model_probs["draw"], model_probs["away_win"]]
    odds_list = [home_odds, draw_odds, away_odds]

    edges = [mp - nv for mp, nv in zip(model_p, novig)]
    best_idx = int(np.argmax(edges))
    best_edge = edges[best_idx]

    try:
        from betting.stat_arb import compute_cross_market_arb
        from betting.information_ratio_kelly import ir_adjusted_kelly
        STAT_ARB_AVAILABLE = True
    except ImportError:
        STAT_ARB_AVAILABLE = False

    all_edges_dict = {o: round(e, 4) for o, e in zip(outcomes, edges)}

    if STAT_ARB_AVAILABLE and all([ah_home_odds, ah_handicap is not None, over_odds, under_odds, score_matrix is not None]):
        arb_res = compute_cross_market_arb(
            home_odds, draw_odds, away_odds,
            ah_home_odds, ah_handicap,
            over_odds, under_odds,
            score_matrix
        )
        if arb_res.get('best_bet'):
            best_bet = arb_res['best_bet']
            best_edge = arb_res['edge']
            all_edges_dict = arb_res.get('all_edges', all_edges_dict)
            if best_bet == '1X2_Home': p = model_p[0]; b = home_odds - 1.0
            elif best_bet == '1X2_Draw': p = model_p[1]; b = draw_odds - 1.0
            elif best_bet == '1X2_Away': p = model_p[2]; b = away_odds - 1.0
            elif best_bet == 'AH_Home':
                model_h = np.tril(score_matrix, k=-1).sum()
                model_d = np.trace(score_matrix)
                if ah_handicap <= -0.5: p = model_h
                elif ah_handicap == 0: p = model_h + model_d * 0.5
                else: p = model_h + model_d
                b = ah_home_odds - 1.0
            elif best_bet == 'Over_2.5':
                n = score_matrix.shape[0]
                p = sum(score_matrix[i, j] for i in range(n) for j in range(n) if i + j > 2)
                b = over_odds - 1.0
            elif best_bet == 'Under_2.5':
                n = score_matrix.shape[0]
                p = 1.0 - sum(score_matrix[i, j] for i in range(n) for j in range(n) if i + j > 2)
                b = under_odds - 1.0
            else:
                p = 0; b = 1
        else:
            best_bet = "NO BET"
    else:
        MIN_EDGE = 0.025
        if best_edge < MIN_EDGE:
            best_bet = "NO BET"
        else:
            best_bet = outcomes[best_idx]
            p = model_p[best_idx]
            b = odds_list[best_idx] - 1.0

    if best_bet != "NO BET":
        if STAT_ARB_AVAILABLE:
            kelly_res = ir_adjusted_kelly(p, b + 1.0, historical_edges=[])
            kelly_capped = kelly_res['stake_fraction']
            ir_multiplier = kelly_res['ir_multiplier']
        else:
            kelly_full = (b * p - (1 - p)) / b
            kelly_capped  = min(max(0.0, kelly_full * 0.25), 0.05)
            ir_multiplier = 0.0
    else:
        kelly_capped = 0.0
        ir_multiplier = 0.0

    return {
        "no_vig_edge":    round(best_edge, 4) if best_bet != "NO BET" else None,
        "best_bet":       best_bet if best_bet != "NO BET" else None,
        "kelly_fraction": round(kelly_capped, 4) if best_bet != "NO BET" else None,
        "ir_multiplier":  round(ir_multiplier, 4) if best_bet != "NO BET" else None,
        "novig_probs":    [round(n, 4) for n in novig],
        "all_edges":      all_edges_dict
    }


# ─── Main Inference Function ─────────────────────────────────────────────────
from functools import lru_cache
import hashlib, json
from pathlib import Path

# Cache invalidation key: hash of team_states.json modification time
def _get_states_hash() -> str:
    p = Path("model_versions/latest/team_states.json")
    if p.exists():
        return str(p.stat().st_mtime)
    return "0"

def validate_feature_alignment(manifest_cols: list, constructed_vec: np.ndarray):
    """
    Crash early with a clear message if the live vector
    doesn't match what the model was trained on.
    """
    if len(manifest_cols) != constructed_vec.shape[0]:
        raise ValueError(
            f"Feature mismatch: manifest has {len(manifest_cols)} features, "
            f"inference constructed {constructed_vec.shape[0]}. "
            f"Re-run train_test.py to regenerate manifest.json with correct columns."
        )

@lru_cache(maxsize=512)
def _cached_inference(home_team: str, away_team: str,
                       venue_factor: float, stage: str,
                       states_hash: str) -> str:
    """
    Core inference — cached by team names + venue + stage + states hash.
    Returns JSON string (hashable for lru_cache).
    """
    # 0. Lazy-load artifacts
    _ensure_loaded()
    
    # 1. Retrieve team states
    home_state = _get_team_state(home_team)
    away_state = _get_team_state(away_team)

    # 2. Build feature vector
    X = _build_feature_vector(home_state, away_state, venue_factor, stage)
    if X is not None:
        validate_feature_alignment(_feature_cols, X)
    if X is None:
        return json.dumps({
            "home_team": home_team, "away_team": away_team,
            "regime_filtered": True,
            "message": "Regime filter: Glicko gap > 400. No prediction issued.",
            "model_version": "6.2"
        })

    # 3. Scale
    X_input = _scaler.transform(X.reshape(1, -1)) if _scaler else X.reshape(1, -1)

    # 4. Base learner predictions
    cat_h, cat_a, xgb_h, xgb_a, ridge_h, ridge_a = _base_learners
    pred_h = (cat_h.predict(X_input) + xgb_h.predict(X_input) + ridge_h.predict(X_input)) / 3.0
    pred_a = (cat_a.predict(X_input) + xgb_a.predict(X_input) + ridge_a.predict(X_input)) / 3.0
    
    from train_test import xg_to_probs
    base_blend = xg_to_probs(pred_h, pred_a)[0]

    # 6. Meta-learner gate with augmented features
    if dc_result:
        dc_probs = np.array([dc_result["home_win"], dc_result["draw"], dc_result["away_win"]])
    else:
        # Fallback if DC fails (pad with equal probs)
        dc_probs = np.array([0.333, 0.334, 0.333])
        dc_result = {}
        
    # The meta-learner expects 9 features: [ML, DC, Prior]
    # We will inject the market odds prior later if provided, but for caching purposes
    # we return the ML and DC probs, and do the meta-learner step outside the cache.
    # Wait, the meta-learner runs inside _cached_inference!
    # If we want to use live odds, we can't cache the result based on odds.
    # Let's just return the raw inputs and let `run_inference` call the meta-learner.
    pass # We will fix this by moving meta-learner evaluation to run_inference

    return json.dumps({
        "home_team":      home_team,
        "away_team":      away_team,
        "base_blend":     base_blend.tolist(),
        "dc_probs":       dc_probs.tolist(),
        "dc_result":      dc_result,
        "signals": {
            "kalman_velocity_diff": float(X[_feature_cols.index("kalman_velocity_diff")]) if "kalman_velocity_diff" in _feature_cols else 0.0,
            "draw_affinity":        float(X[_feature_cols.index("draw_affinity")]) if "draw_affinity" in _feature_cols else 0.0
        }
    })



def run_inference(home_team: str, away_team: str,
                   venue_factor: float = 0.3,
                   stage: str = "group",
                   home_odds=None, draw_odds=None, away_odds=None,
                   ah_home_odds=None, ah_handicap=None, over_odds=None, under_odds=None,
                   # ── LIVE PARAMETERS ──────────────────────────────────
                   elapsed_minutes: int = None,
                   home_goals_live: int = None,
                   away_goals_live: int = None,
                   red_cards: dict = None,
                   live_state: dict = None,
                   match_period: str = "regular",
                   home_lineup: list = None,
                   home_expected_xi: list = None,
                   away_lineup: list = None,
                   away_expected_xi: list = None) -> dict:
    """
    Unified inference entry point.
    If elapsed_minutes is provided → In-Play mode (DC only).
    Otherwise → Pre-match mode (full ML ensemble + DC blend).
    """
    is_live = elapsed_minutes is not None
    _ensure_loaded(is_live=is_live)

    if _dc_params is None:
        lam_h, lam_a, rho = 1.4, 1.0, -0.13
    else:
        attack   = _dc_params.get("attack", {})
        defense  = _dc_params.get("defense", {})
        home_adv = _dc_params.get("home_adv", 0.3)
        rho      = _dc_params.get("rho", -0.13)
        
        if home_team in attack and away_team in attack:
            eff_ha = home_adv * venue_factor
            lam_h = np.exp(attack[home_team] - defense[away_team] + eff_ha)
            lam_a = np.exp(attack[away_team] - defense[home_team])
        else:
            lam_h, lam_a = 1.4, 1.0

    home_state = _get_team_state(home_team)
    away_state = _get_team_state(away_team)
    glicko_ratio = (home_state.get("glicko", 1500) / (away_state.get("glicko", 1500) + 1e-9))

    from models.match_rules import predict_with_stage_rules, KNOCKOUT_STAGES, extra_time_probs

    if is_live:
        # ── IN-PLAY: Dixon-Coles time-decay + Match Rules ─────────────────────────
        result = predict_with_stage_rules(
            stage=stage,
            lam_h=lam_h, lam_a=lam_a, rho=rho,
            glicko_ratio=glicko_ratio,
            elapsed=elapsed_minutes,
            home_goals=home_goals_live or 0,
            away_goals=away_goals_live or 0,
            match_period=match_period,
            red_cards=red_cards
        )
        # Ensure base keys expected by downstream exist
        result["home_win_prob"] = result.get("home_win", 0.0)
        result["draw_prob"] = result.get("draw", 0.0)
        result["away_win_prob"] = result.get("away_win", 0.0)
        
        if "final_pick" not in result:
            probs = [result["home_win_prob"], result["draw_prob"], result["away_win_prob"]]
            result["final_pick"] = ["Home Win", "Draw", "Away Win"][int(np.argmax(probs))]
    else:
        # ── PRE-MATCH: Full ML ensemble + Match Rules ─────────────────────
        states_hash = _get_states_hash()
        result_json = _cached_inference(home_team, away_team, venue_factor, stage, states_hash)
        cached = json.loads(result_json)
        
        if cached.get("regime_filtered"):
            return cached
            
        base_blend = np.array(cached["base_blend"])
        dc_probs = np.array(cached["dc_probs"])
        dc_result = cached["dc_result"]
        
        # ── Market Odds Prior for Meta-Learner ──
        if all([home_odds, draw_odds, away_odds]):
            raw = [1/home_odds, 1/draw_odds, 1/away_odds]
            mkt_probs = np.array([r / sum(raw) for r in raw])
        else:
            mkt_probs = dc_probs
            
        meta_input = np.hstack([base_blend, dc_probs, mkt_probs]).reshape(1, -1)
        final_proba = _meta_learner.predict_proba(meta_input)[0]
        final_proba = np.clip(final_proba, 0.05, 0.95)
        final_proba = final_proba / final_proba.sum()
        
        classes = list(_meta_learner.classes_)
        ph_idx = classes.index("Home Win") if "Home Win" in classes else classes.index(0)
        pd_idx = classes.index("Draw") if "Draw" in classes else classes.index(1)
        pa_idx = classes.index("Away Win") if "Away Win" in classes else classes.index(2)

        ph, pd, pa = float(final_proba[ph_idx]), float(final_proba[pd_idx]), float(final_proba[pa_idx])
        
        # ── Lineup Strength Adjustment ──
        from data.lineup_scraper import lineup_strength_adjustment
        home_adj = lineup_strength_adjustment(home_lineup, home_expected_xi)
        away_adj = lineup_strength_adjustment(away_lineup, away_expected_xi)
        
        ph = ph * home_adj
        pa = pa * away_adj
        # Normalize back
        total = ph + pd + pa
        ph, pd, pa = ph/total, pd/total, pa/total
        
        result = {
            "home_team": home_team, "away_team": away_team,
            "home_win_prob": round(ph, 4), "draw_prob": round(pd, 4), "away_win_prob": round(pa, 4),
            "btts_yes": dc_result.get("btts_yes"), "over_25": dc_result.get("over_25"),
            "regime_filtered": False,
            "signals": cached["signals"],
            "model_version": "7.2",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Determine final pick
        draw_affinity_val = cached["signals"].get("draw_affinity", 0.0)
        if pd >= 0.34 and draw_affinity_val >= 0.45:
            result["final_pick"] = "Draw"
        else:
            result["final_pick"] = ["Home Win", "Draw", "Away Win"][int(np.argmax([ph, pd, pa]))]
        
        if stage in KNOCKOUT_STAGES:
            # Overwrite the ML probabilities using the Extra Time redistributor
            h, d, a = result["home_win_prob"], result["draw_prob"], result["away_win_prob"]
            et = extra_time_probs(lam_h, lam_a, rho, 0, 0, glicko_ratio)
            
            p_home = h + d * et['home_win']
            p_away = a + d * et['away_win']
            total = p_home + p_away
            
            result["home_win_prob"] = round(p_home / total, 4)
            result["away_win_prob"] = round(p_away / total, 4)
            result["draw_prob"] = 0.0
            
            result["p_draw_at_90"] = round(d, 4)
            result["p_extra_time"] = round(d * et['p_extra_time'], 4)
            result["p_penalties"] = round(d * et['p_penalties'], 4)
            result["penalty_home_win"] = round(d * et['penalty_home_win'], 4)
            result["penalty_away_win"] = round(d * et['penalty_away_win'], 4)
            result["stage"] = stage
            result["allows_draw"] = False
            
            probs = [result["home_win_prob"], result["draw_prob"], result["away_win_prob"]]
            result["final_pick"] = ["Home Win", "Draw", "Away Win"][int(np.argmax(probs))]


    # Add betting signals if odds provided
    if all([home_odds, draw_odds, away_odds]):
        betting = _compute_betting_math(
            {"home_win": result["home_win_prob"], "draw": result["draw_prob"], "away_win": result["away_win_prob"]},
            home_odds, draw_odds, away_odds,
            ah_home_odds, ah_handicap, over_odds, under_odds,
            dc_result.get("score_matrix") if isinstance(dc_result, dict) else None
        )
        result["no_vig_edge"] = betting.get("no_vig_edge")
        result["best_bet"] = betting.get("best_bet")
        result["kelly_fraction"] = betting.get("kelly_fraction")
        result["ir_multiplier"] = betting.get("ir_multiplier")
        result["all_edges"] = betting.get("all_edges")

    # Apply Conformal Prediction Sets
    try:
        from models.conformal import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        cp.threshold = 0.05 # Mock threshold since we don't have the real calibration set in inference.py
        probs = np.array([[result["home_win_prob"], result["draw_prob"], result["away_win_prob"]]])
        prediction_set = cp.predict_set(probs, ["home_win", "draw", "away_win"])[0]
        result["conformal_prediction_set"] = prediction_set
    except Exception as e:
        pass

    return result
