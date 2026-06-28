# api_server.py
# Run: uvicorn api_server:app --reload --port 8000

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os, time, json
import numpy as np
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

# --- Simple in-memory API key store (replace with DB for production) ----------
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


# --- Request / Response Models ------------------------------------------------
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


# --- Endpoints ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "6.2"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, api_key: str = Depends(validate_api_key)):
    try:
        # Import your pipeline — adjust path as needed
        from app import run_prediction_pipeline
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
                     result['draw_prob']     - novig[1],
                     result['away_win_prob'] - novig[2]]
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
