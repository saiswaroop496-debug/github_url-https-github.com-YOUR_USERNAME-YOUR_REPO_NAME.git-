<USER_REQUEST>
The plan approved. The three warnings in your plan each have a clean solution. Here is the complete V6.2 implementation with every module at production standard, plus the best-method resolution for all warnings.

***

## Warning Resolutions First

### Warning 1 — API Keys & Secrets

**Best method: GitHub Actions Secrets + `python-dotenv` local + Streamlit Secrets for cloud.**

Never put secrets in code or `.env` committed to git. Use a three-environment pattern:

```
Local dev   → .env file (gitignored) loaded by python-dotenv
CI/CD       → GitHub Actions Secrets (Settings → Secrets → Actions)
Streamlit   → Streamlit Cloud Secrets (app settings → Secrets TOML)
```

Create `.env.template` (committed, safe):
```
RAPIDAPI_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHANNEL_ID=your_channel_id_here
API_SERVER_KEY=your_api_key_here
```

In every Python file, load with:
```python
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("RAPIDAPI_KEY", "MISSING_KEY")
if API_KEY == "MISSING_KEY":
    raise EnvironmentError("RAPIDAPI_KEY not set. Copy .env.template to .env and fill values.")
```

This pattern works identically in all three environments — no code changes needed between local, CI, and cloud. 

### Warning 2 — imbalanced-learn / SMOTE

**Best method: Use `SMOTE(k_neighbors=3)` NOT `SMOTETomek` for your dataset size.**

`SMOTETomek` is Tomek-link cleaning + SMOTE combined. With only ~144 training samples per fold, Tomek removal will delete real majority-class samples you cannot afford to lose. Use pure SMOTE with small k:

```
# requirements.txt additions
imbalanced-learn>=0.12.0
```

```python
# Correct import
from imblearn.over_sampling import SMOTE
# NOT SMOTETomek for small datasets
smote = SMOTE(k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_oof_train, y_oof_train)
```

Apply SMOTE **only** inside the meta-learner training split, never on the calibration slice. Never on test data. 

### Warning 3 — Squad Availability Score

**Best method: Implement as a graceful degradation feature with a `SQUAD_AVAILABLE` flag.**

If the API key is missing, the feature returns 1.0 (full squad assumed) and logs a warning. The pipeline never breaks:

```python
def squad_availability_score(team_id: int, fixture_id: int) -> float:
    """Returns 0.0 (many absences) to 1.0 (full squad). Defaults to 1.0 if API unavailable."""
    if not os.getenv("RAPIDAPI_KEY"):
        return 1.0   # graceful default — no crash
    # ... API call logic below
```

***

## Complete Production Code

### `models/meta_learner.py` — SMOTE + Draw Recalibration + Optuna Cap

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import warnings

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. SMOTE disabled. Run: pip install imbalanced-learn")


# ─── SMOTE Balancing (training split only) ────────────────────────────────────
def balance_oof_for_meta(X_oof: np.ndarray, y_oof: np.ndarray):
    """
    Apply SMOTE to OOF training data to address draw class imbalance.
    k_neighbors=3 is safe for small datasets (~120-150 samples).
    Returns original data unchanged if SMOTE unavailable.
    """
    if not SMOTE_AVAILABLE:
        return X_oof, y_oof

    class_counts = {c: (y_oof == c).sum() for c in np.unique(y_oof)}
    min_class_count = min(class_counts.values())

    if min_class_count < 4:
        warnings.warn(f"Min class count {min_class_count} too small for SMOTE. Skipping.")
        return X_oof, y_oof

    k = min(3, min_class_count - 1)
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = smote.fit_resample(X_oof, y_oof)
    print(f"  SMOTE: {len(y_oof)} → {len(y_res)} samples "
          f"(Draw: {(y_oof=='Draw').sum()} → {(y_res=='Draw').sum()})")
    return X_res, y_res


# ─── Draw-Isolated Platt Recalibration ────────────────────────────────────────
def recalibrate_draw_column(proba: np.ndarray, y_true: np.ndarray,
                             draw_idx: int, ece_threshold: float = 0.08):
    """
    If ECE for the Draw column exceeds threshold, apply isotonic recalibration
    strictly to the Draw probability column. Does not touch Home/Away columns.
    Target range: draw probabilities scaled to [0.28, 0.38].
    """
    draw_probs = proba[:, draw_idx]
    draw_true  = (y_true == 'Draw').astype(int)

    # Compute draw-column ECE
    fraction_pos, mean_pred = calibration_curve(draw_true, draw_probs, n_bins=5)
    draw_ece = np.mean(np.abs(fraction_pos - mean_pred))

    if draw_ece <= ece_threshold:
        print(f"  Draw ECE={draw_ece:.4f} within threshold. No recalibration needed.")
        return proba

    print(f"  Draw ECE={draw_ece:.4f} > {ece_threshold}. Applying isotonic recalibration.")
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.28, y_max=0.38)
    iso.fit(draw_probs, draw_true.astype(float))
    recal_draw = iso.predict(draw_probs)

    # Re-normalize so rows sum to 1
    proba_new = proba.copy()
    proba_new[:, draw_idx] = recal_draw
    row_sums = proba_new.sum(axis=1, keepdims=True)
    return proba_new / row_sums


# ─── Meta-Learner Fit ─────────────────────────────────────────────────────────
def fit_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray, classes: list):
    """
    Full meta-learner training pipeline:
    1. SMOTE balance on training split
    2. Balanced LR fit
    3. Platt calibration on held-out chronological slice
    4. Draw column recalibration if ECE > 0.08
    """
    n = len(y_oof)
    cal_split = int(n * 0.85)

    X_train, y_train = oof_preds[:cal_split], y_oof[:cal_split]
    X_cal, y_cal     = oof_preds[cal_split:], y_oof[cal_split:]

    # Apply SMOTE only to training split
    X_train_bal, y_train_bal = balance_oof_for_meta(X_train, y_train)

    lr = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_bal, y_train_bal)

    calibrated = CalibratedClassifierCV(lr, cv='prefit', method='sigmoid')
    calibrated.fit(X_cal, y_cal)

    return calibrated


def predict_with_draw_threshold(model, X: np.ndarray, classes: list,
                                  draw_thresh: float = 0.34):
    """
    Predict with draw override gate at calibrated threshold.
    draw_thresh=0.34 tuned for class_weight='balanced' output distribution.
    """
    proba = model.predict_proba(X)
    class_list = list(model.classes_)
    draw_idx = class_list.index('Draw')

    preds = []
    for p in proba:
        if p[draw_idx] >= draw_thresh:
            preds.append('Draw')
        else:
            preds.append(class_list[np.argmax(p)])
    return np.array(preds), proba


# ─── Optuna Cap (if used for LR hyperparameter tuning) ────────────────────────
def tune_lr_optuna(X: np.ndarray, y: np.ndarray):
    """Optuna-tuned C parameter. Capped at 20 trials, 30s timeout."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            C = trial.suggest_float('C', 0.01, 2.0, log=True)
            from sklearn.model_selection import cross_val_score
            lr = LogisticRegression(C=C, class_weight='balanced',
                                    solver='lbfgs', max_iter=500)
            scores = cross_val_score(lr, X, y, cv=3, scoring='neg_log_loss')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=30)
        return study.best_params['C']

    except ImportError:
        return 0.5   # safe default if Optuna not installed
```

### `models/poisson_dixon_coles.py` — Two-Stage ρ Estimation

```python
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# ─── STAGE 1: Fit attack/defense with identifiability constraint ──────────────
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


# ─── STAGE 2: Estimate ρ with fixed attack/defense ────────────────────────────
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
    print(f"  ρ (rho) converged to: {rho:.4f}")
    return rho


# ─── Vectorized Score Matrix ─────────────────────────────────────────────────
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


# ─── BTTS + Over/Under ────────────────────────────────────────────────────────
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
```

### `features/rolling_features.py` — Sharp Line Movement + SAS

```python
import numpy as np
import pandas as pd
import os
import requests
import warnings


def add_sharp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sharp Line Movement: change in implied probability from opening to current odds.
    Positive = money coming in on Home (sharp backing Home).
    """
    if 'opening_odds_home' not in df.columns or 'current_odds_home' not in df.columns:
        df['sharp_line_movement'] = 0.0
        return df

    # Convert to implied probability (no overround correction needed here — direction matters)
    df['sharp_line_movement'] = (
        (1.0 / df['opening_odds_home'].clip(1.01)) -
        (1.0 / df['current_odds_home'].clip(1.01))
    ).fillna(0.0)

    return df


def squad_availability_score(team_id: int, fixture_id: int,
                               api_key: str = None) -> float:
    """
    Fetch lineup data from API-Football and compute Squad Availability Score.
    Returns 0.0 (many key absences) to 1.0 (full squad).
    Gracefully defaults to 1.0 if API unavailable or key missing.

    Key player weight: GK=0.12, CB=0.08, CM=0.07, ST=0.10, others=0.05
    """
    POSITION_WEIGHTS = {
        'G': 0.12,   # Goalkeeper
        'D': 0.08,   # Defender
        'M': 0.07,   # Midfielder
        'F': 0.10,   # Forward
    }
    DEFAULT_WEIGHT = 0.05

    if not api_key:
        api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return 1.0   # graceful degradation — full squad assumed

    try:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers,
                                params={"fixture": fixture_id},
                                timeout=10)
        data = response.json()

        if not data.get('response'):
            return 1.0   # no lineup data yet

        # Find the matching team
        for team_data in data['response']:
            if team_data['team']['id'] != team_id:
                continue

            # Get expected 11 from startXI
            start_xi = team_data.get('startXI', [])
            if len(start_xi) < 10:
                return 0.85   # incomplete lineup, penalise slightly

            # Compute weighted availability
            score = 0.0
            for player in start_xi:
                pos = player['player'].get('pos', 'X')
                weight = POSITION_WEIGHTS.get(pos, DEFAULT_WEIGHT)
                score += weight

            # Normalize to [0, 1]
            return min(1.0, score / 1.0)

    except Exception as e:
        warnings.warn(f"SAS fetch failed: {e}. Defaulting to 1.0")
        return 1.0


def add_squad_features(df: pd.DataFrame, api_key: str = None) -> pd.DataFrame:
    """Add SAS columns to dataframe. Uses cached values if fixture_id repeated."""
    cache = {}
    sas_home, sas_away = [], []

    for _, row in df.iterrows():
        fid = row.get('fixture_id', None)
        htid = row.get('home_team_id', None)
        atid = row.get('away_team_id', None)

        sh = cache.get((htid, fid), squad_availability_score(htid, fid, api_key))
        sa = cache.get((atid, fid), squad_availability_score(atid, fid, api_key))
        cache[(htid, fid)] = sh
        cache[(atid, fid)] = sa
        sas_home.append(sh)
        sas_away.append(sa)

    df['home_sas'] = sas_home
    df['away_sas'] = sas_away
    df['sas_differential'] = df['home_sas'] - df['away_sas']
    return df
```

### `api_server.py` — FastAPI with Key Auth + Request Counting

```python
# api_server.py
# Run: uvicorn api_server:app --reload --port 8000

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os, time, json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="WorldCup Quant API",
    version="6.2",
    description="Institutional-grade FIFA World Cup prediction engine"
)
security = HTTPBearer()

# ─── Simple in-memory API key store (replace with DB for production) ──────────
VALID_API_KEYS = {
    os.getenv("API_SERVER_KEY", "dev-key-change-this"): {"tier": "pro", "calls": 0},
}
REQUEST_COUNTS = defaultdict(int)
RATE_LIMIT = {"free": 10, "pro": 500, "api": 10000}  # calls/day


def validate_api_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    key = creds.credentials
    if key not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key")
    tier = VALID_API_KEYS[key]["tier"]
    REQUEST_COUNTS[key] += 1
    if REQUEST_COUNTS[key] > RATE_LIMIT[tier]:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"{tier} tier limit ({RATE_LIMIT[tier]}/day) reached")
    return key


# ─── Request / Response Models ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    home_team: str = Field(..., example="Brazil")
    away_team: str = Field(..., example="Germany")
    venue_factor: float = Field(0.3, ge=0.0, le=1.0,
                                description="0=pure neutral, 1=true home")
    stage: str = Field("group", pattern="^(group|round_of_16|quarter|semi|final)$")
    home_decimal_odds: Optional[float] = Field(None, gt=1.0)
    draw_decimal_odds:  Optional[float] = Field(None, gt=1.0)
    away_decimal_odds:  Optional[float] = Field(None, gt=1.0)

    @field_validator('home_team', 'away_team')
    @classmethod
    def teams_must_differ(cls, v, info):
        if 'home_team' in info.data and v == info.data['home_team']:
            raise ValueError('Home and away teams must be different')
        return v


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: str
    lower_bounds: dict
    upper_bounds: dict
    btts_yes: Optional[float]
    over_25: Optional[float]
    kelly_fraction: Optional[float]
    no_vig_edge: Optional[float]
    best_bet: Optional[str]
    model_version: str
    timestamp: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "6.2"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, api_key: str = Depends(validate_api_key)):
    try:
        # Import your pipeline — adjust path as needed
        from main import run_prediction_pipeline
        result = run_prediction_pipeline(
            home_team=req.home_team,
            away_team=req.away_team,
            venue_factor=req.venue_factor,
            stage=req.stage,
            home_odds=req.home_decimal_odds,
            draw_odds=req.draw_decimal_odds,
            away_odds=req.away_decimal_odds
        )

        # Compute no-vig edge if odds provided
        no_vig_edge = None
        best_bet = None
        kelly_fraction = None

        if all([req.home_decimal_odds, req.draw_decimal_odds, req.away_decimal_odds]):
            raw = [1/req.home_decimal_odds, 1/req.draw_decimal_odds,
                   1/req.away_decimal_odds]
            overround = sum(raw)
            novig = [r / overround for r in raw]
            edges = [result['home_win_prob'] - novig[0],
                     result['draw_prob']     - novig [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html),
                     result['away_win_prob'] - novig [campus.datacamp](https://campus.datacamp.com/courses/deploying-ai-into-production-with-fastapi/securing-and-optimizing-the-api?ex=4)]
            best_idx = int(np.argmax(edges))
            best_bet_edge = edges[best_idx]
            if best_bet_edge >= 0.025:
                best_bet = ["Home Win", "Draw", "Away Win"][best_idx]
                no_vig_edge = round(best_bet_edge, 4)
                # Quarter Kelly
                b = [req.home_decimal_odds, req.draw_decimal_odds,
                     req.away_decimal_odds][best_idx] - 1
                p = [result['home_win_prob'], result['draw_prob'],
                     result['away_win_prob']][best_idx]
                q = 1 - p
                kelly_fraction = round(max(0, (b*p - q) / b) * 0.25, 4)

        return PredictResponse(
            home_team=req.home_team,
            away_team=req.away_team,
            home_win_prob=round(result['home_win_prob'], 4),
            draw_prob=round(result['draw_prob'], 4),
            away_win_prob=round(result['away_win_prob'], 4),
            confidence=result.get('confidence', 'MODERATE'),
            lower_bounds=result.get('lower_bounds', {}),
            upper_bounds=result.get('upper_bounds', {}),
            btts_yes=result.get('btts_yes'),
            over_25=result.get('over_25'),
            kelly_fraction=kelly_fraction,
            no_vig_edge=no_vig_edge,
            best_bet=best_bet,
            model_version="6.2",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test curl:
# curl -X POST http://localhost:8000/predict \
#      -H "Authorization: Bearer dev-key-change-this" \
#      -H "Content-Type: application/json" \
#      -d '{"home_team":"Brazil","away_team":"Germany","venue_factor":0.3}'
```

### `telegram_bot.py` — Async Signal Bot with Scheduling

```python
# telegram_bot.py
# Install: pip install python-telegram-bot apscheduler
# Run: python telegram_bot.py

import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")  # e.g. "@yourchannelname" or "-100xxxx"

if not BOT_TOKEN:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN not set in .env")

from telegram import Bot
from telegram.constants import ParseMode

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logging.warning("apscheduler not installed. Run: pip install apscheduler")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_prediction_message(match: dict) -> str:
    """Format a prediction dict into a Telegram-ready message."""
    home  = match.get('home_team', 'N/A')
    away  = match.get('away_team', 'N/A')
    ph    = match.get('home_win_prob', 0)
    pd_   = match.get('draw_prob', 0)
    pa    = match.get('away_win_prob', 0)
    conf  = match.get('confidence', 'MODERATE')
    edge  = match.get('no_vig_edge')
    best  = match.get('best_bet')
    kelly = match.get('kelly_fraction')
    btts  = match.get('btts_yes')
    o25   = match.get('over_25')

    conf_emoji = {"HIGH CONFIDENCE": "🟢", "MODERATE CONFIDENCE": "🟡",
                  "LOW CONFIDENCE": "🔴"}.get(conf, "⚪")

    msg = (
        f"⚽ *World Cup Signal* — {datetime.utcnow().strftime('%d %b %Y')}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🏠 *{home}* vs ✈️ *{away}*\n\n"
        f"📊 *1X2 Probabilities (No-Vig)*\n"
        f"  Home Win: `{ph:.1%}`\n"
        f"  Draw:     `{pd_:.1%}`\n"
        f"  Away Win: `{pa:.1%}`\n\n"
    )

    if btts is not None:
        msg += f"⚡ BTTS Yes: `{btts:.1%}` | Over 2.5: `{o25:.1%}`\n\n"

    msg += f"{conf_emoji} *Confidence:* {conf}\n"

    if best and edge and kelly:
        msg += (
            f"\n💰 *Best Bet: {best}*\n"
            f"  Edge vs Market: `+{edge:.1%}`\n"
            f"  Kelly Stake: `{kelly:.1%}` of bankroll\n"
        )
    else:
        msg += "\n🚫 *No Value Bet* — edge below 2.5% threshold\n"

    msg += f"\n`Model: V6.2 | #FIFA #WorldCup2026`"
    return msg


async def send_prediction(match: dict):
    """Send a single prediction to the Telegram channel."""
    bot = Bot(token=BOT_TOKEN)
    message = format_prediction_message(match)
    try:
        await bot.send_message(
            chat_id=CHANNEL_ID,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Signal sent: {match.get('home_team')} vs {match.get('away_team')}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


async def send_daily_signals():
    """
    Fetch today's matches, run predictions, send signals.
    Called by scheduler or manually.
    """
    try:
        from main import run_prediction_pipeline, get_todays_fixtures
        fixtures = get_todays_fixtures()

        if not fixtures:
            logger.info("No fixtures today.")
            return

        for fixture in fixtures:
            result = run_prediction_pipeline(
                home_team=fixture['home_team'],
                away_team=fixture['away_team'],
                venue_factor=fixture.get('venue_factor', 0.3),
                stage=fixture.get('stage', 'group')
            )
            result.update(fixture)
            await send_prediction(result)
            await asyncio.sleep(2)   # rate limit: 2s between messages

    except Exception as e:
        logger.error(f"Daily signal job failed: {e}")


def run_bot_with_scheduler():
    """Start async scheduler to send signals daily at 10:00 AM IST."""
    if not SCHEDULER_AVAILABLE:
        logger.warning("Running without scheduler. Call send_daily_signals() manually.")
        asyncio.run(send_daily_signals())
        return

    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(send_daily_signals, 'cron', hour=10, minute=0)
    scheduler.start()
    logger.info("Telegram bot started. Signals will fire at 10:00 AM IST daily.")

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    run_bot_with_scheduler()
```

### `bet_tracker.py` — Structured JSON Logging

```python
# bet_tracker.py — Replace print-based logging with structured JSONL

import json
import logging
import time
import os
from pathlib import Path

LOG_PATH = Path("bets_log.jsonl")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bet_tracker")

# File handler for structured JSONL
file_handler = logging.FileHandler("bet_tracker_events.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


def log_bet(match: str, market: str, outcome: str,
            model_prob: float, decimal_odds: float,
            no_vig_prob: float, edge: float,
            kelly_fraction: float, stake_units: float,
            result: str = None, closing_odds: float = None):
    """
    Log a bet to bets_log.jsonl with full CLV tracking fields.
    result: 'win' | 'loss' | 'void' | None (pending)
    """
    record = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "match":         match,
        "market":        market,
        "outcome":       outcome,
        "model_prob":    round(model_prob, 4),
        "decimal_odds":  round(decimal_odds, 3),
        "no_vig_prob":   round(no_vig_prob, 4),
        "edge":          round(edge, 4),
        "kelly_fraction":round(kelly_fraction, 4),
        "stake_units":   round(stake_units, 3),
        "result":        result,
        "closing_odds":  closing_odds,
        "clv_pct":       round((closing_odds / decimal_odds) - 1, 4)
                         if closing_odds and decimal_odds else None,
        "pnl_units":     round(stake_units * (decimal_odds - 1), 3)
                         if result == 'win' else
                         round(-stake_units, 3)
                         if result == 'loss' else None
    }

    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f"BET LOGGED | {match} | {outcome} | edge={edge:+.2%} | "
                f"stake={stake_units:.3f}u | result={result}")
    return record


def clv_report(min_bets: int = 20):
    """
    Read bets_log.jsonl and compute CLV summary.
    Requires closing_odds field to be updated after market closes.
    """
    if not LOG_PATH.exists():
        print("No bets logged yet.")
        return

    import pandas as pd
    df = pd.read_json(LOG_PATH, lines=True)

    if len(df) < min_bets:
        print(f"Only {len(df)} bets. Need {min_bets}+ for reliable CLV estimate.")

    settled = df[df['result'].notna()].copy()
    clv_eligible = df[df['clv_pct'].notna()].copy()

    print("\n── Bet Tracker CLV Report ────────────────────────────────")
    print(f"  Total bets logged:    {len(df)}")
    print(f"  Settled bets:         {len(settled)}")
    print(f"  CLV-eligible bets:    {len(clv_eligible)}")

    if len(settled) > 0:
        wins  = (settled['result'] == 'win').sum()
        pnl   = settled['pnl_units'].sum()
        roi   = pnl / settled['stake_units'].sum() if settled['stake_units'].sum() > 0 else 0
        print(f"\n  Win Rate:             {wins/len(settled):.1%}")
        print(f"  Total P&L:            {pnl:+.2f} units")
        print(f"  ROI:                  {roi:+.1%}")

    if len(clv_eligible) > 0:
        mean_clv = clv_eligible['clv_pct'].mean()
        print(f"\n  Mean CLV:             {mean_clv:+.3%}")
        print(f"  Median CLV:           {clv_eligible['clv_pct'].median():+.3%}")
        print(f"  CLV by Market:")
        print(clv_eligible.groupby('market')['clv_pct'].mean().apply(
            lambda x: f"    {x:+.3%}").to_string())

    print("──────────────────────────────────────────────────────────\n")
```

### `.github/workflows/deploy.yml` — CI/CD with Gates

```yaml
name: V6.2 Validate & Deploy

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 21 * * 0'   # Every Sunday 9 PM UTC = Monday 2:30 AM IST

jobs:
  validate:
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.gate.outputs.accuracy }}
      passed:   ${{ steps.gate.outputs.passed }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run validation pipeline
        env:
          RAPIDAPI_KEY: ${{ secrets.RAPIDAPI_KEY }}
        run: python train_test.py 2>&1 | tee validation_output.txt

      - name: Check accuracy + log-loss gates
        id: gate
        run: |
          python - <<'EOF'
          import re, sys, os

          with open('validation_output.txt') as f:
              txt = f.read()

          # Parse accuracy
          acc_match = re.search(r'Accuracy[:\s]+(\d+\.?\d*)%', txt)
          ll_match  = re.search(r'Log.Loss[:\s]+([\d.]+)', txt)

          acc = float(acc_match.group(1)) if acc_match else 0.0
          ll  = float(ll_match.group(1))  if ll_match  else 99.0

          print(f"Accuracy: {acc:.1f}%")
          print(f"Log-Loss: {ll:.4f}")

          passed = (acc >= 44.0 and ll <= 1.10)

          # Write GitHub outputs
          with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
              f.write(f"accuracy={acc:.1f}\n")
              f.write(f"passed={'true' if passed else 'false'}\n")

          if not passed:
              print(f"❌ GATE FAILED: Accuracy={acc:.1f}% (need ≥44%), Log-Loss={ll:.4f} (need ≤1.10)")
              sys.exit(1)
          else:
              print(f"✅ Gates passed: Accuracy={acc:.1f}%, Log-Loss={ll:.4f}")
          EOF

      - name: Upload validation report
        uses: actions/upload-artifact@v4
        with:
          name: validation-report-${{ github.run_number }}
          path: validation_output.txt
          retention-days: 30

  deploy:
    needs: validate
    if: needs.validate.outputs.passed == 'true'
    runs-on: ubuntu-latest

    steps:
      - name: Deploy confirmation
        run: |
          echo "✅ Accuracy: ${{ needs.validate.outputs.accuracy }}%"
          echo "✅ Streamlit Cloud redeploys automatically on push to main."
          echo "✅ No manual deploy step needed — Streamlit watches main branch."

  notify-failure:
    needs: validate
    if: failure()
    runs-on: ubuntu-latest

    steps:
      - name: Send failure notification
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHANNEL_ID: ${{ secrets.TELEGRAM_CHANNEL_ID }}
        run: |
          curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHANNEL_ID}" \
            -d text="⚠️ V6.2 CI FAILED — accuracy gate not met. Check GitHub Actions for details." \
            -d parse_mode="Markdown" || echo "Telegram notification skipped (no token)"
```

### `model_versions/` — Versioned Model Export

Add to the end of `train_test.py`:

```python
# model_versions/export.py  — or add inline to train_test.py

import joblib
import json
import hashlib
import time
from pathlib import Path

MODEL_VERSIONS_DIR = Path("model_versions")
MODEL_VERSIONS_DIR.mkdir(exist_ok=True)


def export_model(model, scaler, meta_learner, feature_cols: list,
                 metrics: dict, version: str = "6.2"):
    """
    Export model artifacts with version manifest.
    Automatically increments build number via hash.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    build_hash = hashlib.md5(ts.encode()).hexdigest()[:8]
    build_name = f"v{version}_{ts}_{build_hash}"
    build_dir  = MODEL_VERSIONS_DIR / build_name
    build_dir.mkdir()

    # Save artifacts
    joblib.dump(model,        build_dir / "base_learners.joblib")
    joblib.dump(scaler,       build_dir / "scaler.joblib")
    joblib.dump(meta_learner, build_dir / "meta_learner.joblib")

    # Save manifest
    manifest = {
        "version":       version,
        "build":         build_name,
        "timestamp":     ts,
        "feature_cols":  feature_cols,
        "metrics": {
            "accuracy":    metrics.get("accuracy"),
            "log_loss":    metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "ece":         metrics.get("ece"),
            "draw_recall": metrics.get("draw_recall"),
            "fold_std":    metrics.get("fold_std"),
        },
        "promoted":     metrics.get("accuracy", 0) >= 44.0 and
                        metrics.get("log_loss", 99) <= 1.10
    }

    with open(build_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Symlink latest promoted model
    latest_link = MODEL_VERSIONS_DIR / "latest"
    if manifest["promoted"]:
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(build_dir.name)
        print(f"  ✅ Promoted → model_versions/latest → {build_name}")
    else:
        print(f"  ⚠️  Model archived but not promoted (gate failed): {build_name}")

    return build_dir


# Call at end of train_test.py after validation:
# export_model(base_learners, scaler, meta, FEATURE_COLS, metrics_dict)
```

***

## Updated `requirements.txt` Additions

```
# Add to existing requirements.txt
imbalanced-learn>=0.12.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-telegram-bot>=21.0
apscheduler>=3.10.0
pydantic>=2.0.0
python-dotenv>=1.0.0
statsmodels>=0.14.0
httpx>=0.27.0   # used internally by FastAPI test client
```

***

## Verification Commands

Run in this exact order after implementation:

```bash
# 1. Unit-test new DC two-stage rho
python -c "from models.poisson_dixon_coles import fit_rho_concentrated; print('DC import OK')"

# 2. Full pipeline validation
python train_test.py
# Expected: rho ≈ -0.12 to -0.25, Draw recall > 0, accuracy ≥ 44%

# 3. Start API server
uvicorn api_server:app --reload --port 8000

# 4. Test API endpoint
curl -X POST http://localhost:8000/predict \
     -H "Authorization: Bearer dev-key-change-this" \
     -H "Content-Type: application/json" \
     -d '{"home_team":"Brazil","away_team":"Germany","venue_factor":0.3,"stage":"group"}'

# 5. Verify JSONL logging
python -c "from bet_tracker import log_bet; log_bet('Test vs Test','1X2','Home Win',0.48,2.10,0.46,0.02,0.015,1.0,'win',2.05)"
cat bets_log.jsonl

# 6. Test Telegram (requires real token)
python telegram_bot.py   # sends one signal to channel on startup

# 7. Push to GitHub → Actions auto-runs validation + deploy gate
git add . && git commit -m "V6.2 full implementation" && git push origin main
```
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-06-28T07:33:04+05:30.
</ADDITIONAL_METADATA>







