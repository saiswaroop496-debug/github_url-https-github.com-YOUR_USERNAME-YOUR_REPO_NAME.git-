import json
from inference import run_inference

# Create a mock live state
live_state = {
    "home_team": "United States",
    "away_team": "Canada",
    "match_period": "regular",
    "elapsed": 45,
    "home_goals": 1,
    "away_goals": 0,
    "home_red": 0,
    "away_red": 0,
    "home_xg_live": 1.2,
    "away_xg_live": 0.5,
    "home_shots_ot": 4,
    "away_shots_ot": 2,
    "home_corners": 3,
    "away_corners": 1,
    "home_possession": 55.0,
    "away_possession": 45.0,
}

try:
    result = run_inference(
        home_team="United States",
        away_team="Canada",
        venue_factor=0.3,
        stage="group",
        elapsed_minutes=45,
        home_goals_live=1,
        away_goals_live=0,
        red_cards={"home": 0, "away": 0},
        live_state=live_state,
        match_period="regular"
    )
    print("SUCCESS")
    print(json.dumps(result, indent=2))
except Exception as e:
    import traceback
    print("ERROR:")
    traceback.print_exc()
