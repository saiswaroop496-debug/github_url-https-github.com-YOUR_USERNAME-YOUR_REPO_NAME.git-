# data/espn_scraper.py
import requests
import json
import warnings
from datetime import datetime
from pathlib import Path

LIVE_STATE_PATH = Path("data/live_state.json")

def fetch_espn_live_data(game_id: str) -> dict:
    url = f"http://site.api.espn.com/apis/site/v2/sports/soccer/all/summary?event={game_id}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        warnings.warn(f"Failed to fetch ESPN data: {e}")
        return {}

    state = {
        "fixture_id":       game_id,
        "fetched_at":       datetime.utcnow().isoformat(),
        "status":           "unknown",
        "match_period":     "regular",
        "elapsed":          0,
        "home_team":        "",
        "away_team":        "",
        "home_goals":       0,
        "away_goals":       0,
        "home_yellow":      0,
        "away_yellow":      0,
        "home_red":         0,
        "away_red":         0,
        "home_subs_used":   0,
        "away_subs_used":   0,
        "home_subs_left":   5,
        "away_subs_left":   5,
        "home_shots_ot":    0,
        "away_shots_ot":    0,
        "home_shots_total": 0,
        "away_shots_total": 0,
        "home_corners":     0,
        "away_corners":     0,
        "home_possession":  50.0,
        "away_possession":  50.0,
        "home_passes":      0,
        "away_passes":      0,
        "home_pass_acc":    0.0,
        "away_pass_acc":    0.0,
        "home_fouls":       0,
        "away_fouls":       0,
        "home_offsides":    0,
        "away_offsides":    0,
        "home_xg_live":     None,
        "away_xg_live":     None,
        "goals_timeline":   [],
        "cards_timeline":   [],
        "subs_timeline":    [],
        "home_formation":   "",
        "away_formation":   "",
        "home_xi":          [],
        "away_xi":          [],
    }

    header = data.get("header", {})
    comps = header.get("competitions", [{}])[0]
    competitors = comps.get("competitors", [])

    home_id, away_id = None, None

    for comp in competitors:
        team = comp.get("team", {})
        side = comp.get("homeAway")
        
        name = team.get("displayName", "")
        score = int(comp.get("score", 0))
        formation = comp.get("formation", "")

        if side == "home":
            state["home_team"] = name
            state["home_goals"] = score
            state["home_formation"] = formation
            home_id = team.get("id")
        else:
            state["away_team"] = name
            state["away_goals"] = score
            state["away_formation"] = formation
            away_id = team.get("id")

    # Match Status and Clock
    status = comps.get("status", {})
    state["elapsed"] = int(status.get("clock", 0)) / 60.0
    display_clock = status.get("displayClock", "0'")
    if "'" in display_clock:
        try:
            state["elapsed"] = int(display_clock.replace("'", "").split('+')[0])
        except ValueError:
            pass
            
    match_state = status.get("type", {}).get("state", "")
    period = status.get("period", 1)
    
    if match_state == "pre":
        state["status"] = "NS"
        state["match_period"] = "regular"
    elif match_state == "post":
        state["status"] = "FT"
        state["match_period"] = "finished"
    else:
        state["status"] = "LIVE"
        if period <= 2:
            state["match_period"] = "regular"
        elif period <= 4:
            state["match_period"] = "extra_time"
        else:
            state["match_period"] = "penalties"

    # Boxscore Stats
    boxscore = data.get("boxscore", {})
    for t_stat in boxscore.get("teams", []):
        t_id = t_stat.get("team", {}).get("id")
        side = "home" if t_id == home_id else "away"
        
        for stat in t_stat.get("statistics", []):
            name = stat.get("name")
            val = stat.get("displayValue", "0")
            
            try:
                if "%" in str(val):
                    fval = float(val.replace("%", ""))
                else:
                    fval = float(val)
            except ValueError:
                fval = 0.0

            if name == "possessionPct": state[f"{side}_possession"] = fval
            elif name == "totalShots": state[f"{side}_shots_total"] = int(fval)
            elif name == "shotsOnTarget": state[f"{side}_shots_ot"] = int(fval)
            elif name == "wonCorners": state[f"{side}_corners"] = int(fval)
            elif name == "yellowCards": state[f"{side}_yellow"] = int(fval)
            elif name == "redCards": state[f"{side}_red"] = int(fval)
            elif name == "totalPasses": state[f"{side}_passes"] = int(fval)
            elif name == "passPct": state[f"{side}_pass_acc"] = fval * 100 if fval <= 1.0 else fval
            elif name == "foulsCommitted": state[f"{side}_fouls"] = int(fval)
            elif name == "offsides": state[f"{side}_offsides"] = int(fval)

    # Key Events
    for event in data.get("keyEvents", []):
        text = event.get("text", "")
        clock = event.get("clock", {}).get("displayValue", "0'")
        
        # ESPN typically identifies the team by their name or in the text
        side = "home" 
        if state["home_team"] in text:
            side = "home"
        elif state["away_team"] in text:
            side = "away"

        if "Goal" in text:
            state["goals_timeline"].append({
                "minute": clock, "team": side,
                "player": text.split(" - ")[0] if " - " in text else text, 
                "assist": "", "detail": text
            })
        elif "Yellow Card" in text:
            state["cards_timeline"].append({
                "minute": clock, "team": side,
                "player": text.split(" - ")[0] if " - " in text else text, 
                "card_type": "Yellow Card"
            })
        elif "Red Card" in text:
            state["cards_timeline"].append({
                "minute": clock, "team": side,
                "player": text.split(" - ")[0] if " - " in text else text, 
                "card_type": "Red Card"
            })
        elif "Substitution" in text:
            state[f"{side}_subs_used"] += 1
            state[f"{side}_subs_left"] = max(0, 5 - state[f"{side}_subs_used"])
            state["subs_timeline"].append({
                "minute": clock, "team": side,
                "player_out": text, "player_in": ""
            })

    # Lineups
    rosters = data.get("rosters", [])
    for roster in rosters:
        t_id = roster.get("team", {}).get("id")
        side = "home" if t_id == home_id else "away"
        
        for p in roster.get("lineups", []):
            state[f"{side}_xi"].append({
                "name": p.get("athlete", {}).get("displayName", ""),
                "number": p.get("jersey", ""),
                "position": p.get("position", {}).get("abbreviation", ""),
                "grid": ""
            })

    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return state
