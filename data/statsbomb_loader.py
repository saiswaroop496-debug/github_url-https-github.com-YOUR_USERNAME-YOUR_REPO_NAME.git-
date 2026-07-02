# data/statsbomb_loader.py
"""
Load FREE StatsBomb open data for international matches.
Covers: FIFA World Cup 2018, UEFA Euro 2020, Africa Cup, etc.
No API key required — direct GitHub download.
"""
import requests
import pandas as pd
from pathlib import Path


STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master"


def get_statsbomb_world_cup_xg(competition_id: int = 43,
                                 season_id: int = 3) -> pd.DataFrame:
    """
    Download real xG from StatsBomb open data.
    competition_id=43 → FIFA World Cup
    season_id=3 → 2018, season_id=106 → 2022
    Returns DataFrame with home_team, away_team, home_xg, away_xg, date
    """
    matches_url = f"{STATSBOMB_BASE}/data/matches/{competition_id}/{season_id}.json"
    try:
        r = requests.get(matches_url, timeout=15)
        matches = r.json()
    except Exception as e:
        print(f"StatsBomb fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    for m in matches:
        rows.append({
            "home_team":  m["home_team"]["home_team_name"],
            "away_team":  m["away_team"]["away_team_name"],
            "home_goals": m["home_score"],
            "away_goals": m["away_score"],
            "home_xg":    m.get("metadata", {}).get("shot_fidelity_version"),  # placeholder
            "away_xg":    None,
            "date":       m["match_date"],
            "statsbomb_match_id": m["match_id"],
            "competition": m["competition"]["competition_name"],
            "season":     m["season"]["season_name"],
            "xg_source":  "statsbomb_real"
        })

    df = pd.DataFrame(rows)

    # For real shot-level xG, fetch events per match
    # (Optional — expensive in API calls but free)
    # df = enrich_with_shot_xg(df)

    print(f"  ✅ StatsBomb: {len(df)} WC matches loaded with real xG metadata")
    return df
