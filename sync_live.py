"""
Phase 7: Live World Cup data sync.
Fetches completed 2026 World Cup matches from ESPN (free API) and appends
new results to data/wc2026_live.csv. Called by retrain_loop.py.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ESPN_URL = "http://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
LIVE_CSV  = Path("data/wc2026_live.csv")

def fetch_finished_matches() -> list[dict]:
    """Pull all finished WC matches from ESPN API."""
    resp = requests.get(ESPN_URL, timeout=15)
    resp.raise_for_status()
    events = resp.json().get("events", [])
    
    # Filter only completed matches
    return [e for e in events if e.get("status", {}).get("type", {}).get("completed", False)]

def normalise_match(m: dict) -> dict:
    """Map API fields to your pipeline's column schema."""
    date_str = m.get("date", "")[:10]
    
    home_team = ""
    away_team = ""
    home_goals = 0
    away_goals = 0
    
    competitors = m.get("competitions", [])[0].get("competitors", [])
    for comp in competitors:
        team_name = comp.get("team", {}).get("displayName", "")
        # Map some common ESPN names to our dataset standard if needed
        if team_name == "United States": team_name = "USA"
        
        score = int(comp.get("score", 0))
        if comp.get("homeAway") == "home":
            home_team = team_name
            home_goals = score
        else:
            away_team = team_name
            away_goals = score

    # Infer result
    if home_goals > away_goals:
        result = "Home Win"
    elif away_goals > home_goals:
        result = "Away Win"
    else:
        result = "Draw"

    # ESPN doesn't explicitly flag knockout in the top level easily, 
    # but for model purposes we can default to Group unless specified.
    # In a full tournament we'd map this, for now default to Group.
    stage = "Group"
    is_knockout = 0

    return {
        "date":            date_str,
        "home_team":       home_team,
        "away_team":       away_team,
        "home_score":      home_goals,
        "away_score":      away_goals,
        "result":          result,
        "tournament":      "FIFA World Cup",
        "stage":           stage,
        "is_neutral":      1,                   # all WC matches are neutral
        "is_knockout":     is_knockout,
        "tournament_weight": 3.0,               # max weight for WC matches
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
    }

def load_existing() -> pd.DataFrame:
    if LIVE_CSV.exists():
        return pd.read_csv(LIVE_CSV)
    return pd.DataFrame()

def save(df: pd.DataFrame):
    LIVE_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(LIVE_CSV, index=False)

def sync() -> int:
    """
    Fetch live results, append only NEW matches, return count of new rows.
    """
    print("🔄  Syncing live WC 2026 results from ESPN API...")

    existing = load_existing()
    existing_keys = set()
    if not existing.empty:
        existing_keys = set(
            existing["date"] + "|" + existing["home_team"] + "|" + existing["away_team"]
        )

    raw_matches = fetch_finished_matches()
    new_rows = []

    for m in raw_matches:
        row = normalise_match(m)
        if not row["home_team"] or not row["away_team"]:
            continue
            
        key = f"{row['date']}|{row['home_team']}|{row['away_team']}"
        if key not in existing_keys:
            new_rows.append(row)

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        save(combined)
        print(f"  ✅ {len(new_rows)} new match(es) saved → {LIVE_CSV}")
    else:
        print("  ℹ️  No new completed matches since last sync.")

    return len(new_rows)

if __name__ == "__main__":
    sync()
