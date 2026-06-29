import requests
import pandas as pd
import os, json, time, warnings
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_PATH     = Path("data/worldcup_matches.csv")
INJURIES_PATH = Path("data/injuries_cache.json")
BASE_URL      = "https://api-football-v1.p.rapidapi.com/v3"
WC2026_LEAGUE = 1   # Update to real WC2026 league ID when confirmed


def get_api_key() -> str:
    """Fetch key from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get("RAPIDAPI_KEY", "")
    except Exception:
        return os.getenv("RAPIDAPI_KEY", "")


def _headers() -> dict:
    return {
        "X-RapidAPI-Key":  get_api_key(),
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }


# ─── Fetch functions ──────────────────────────────────────────────────────────
def fetch_wc2026_fixtures(season: int = 2026) -> list:
    if not get_api_key():
        print("  ⚠️  No RAPIDAPI_KEY — using mock WC2026 data")
        return _mock_wc2026_fixtures()
    try:
        resp = requests.get(f"{BASE_URL}/fixtures",
                            headers=_headers(),
                            params={"league": WC2026_LEAGUE, "season": season},
                            timeout=15)
        resp.raise_for_status()
        return resp.json().get("response", [])
    except Exception as e:
        warnings.warn(f"Fixtures fetch failed: {e}. Using mock.")
        return _mock_wc2026_fixtures()


def fetch_fixture_stats(fixture_id: int) -> dict:
    """Fetch xG, possession, shots for a completed fixture."""
    if not get_api_key():
        return {}
    try:
        resp = requests.get(f"{BASE_URL}/fixtures/statistics",
                            headers=_headers(),
                            params={"fixture": fixture_id},
                            timeout=10)
        data = resp.json().get("response", [])
        stats = {}
        for team_data in data:
            team_name = team_data["team"]["name"]
            for stat in team_data["statistics"]:
                key = stat["type"].lower().replace(" ", "_")
                val = stat["value"]
                # Clean percentage strings like "62%"
                if isinstance(val, str) and val.endswith('%'):
                    val = float(val.replace('%', '')) / 100
                stats[f"{team_name}_{key}"] = val
        return stats
    except Exception as e:
        warnings.warn(f"Stats fetch failed for {fixture_id}: {e}")
        return {}


def fetch_distance_proxy(fixture_id: int) -> dict:
    """
    Use API-Football /fixtures/events to compute distance proxy
    from pass/shot XY coordinates.
    Returns {home_pass_distance_proxy, away_pass_distance_proxy}
    without requiring video.
    """
    if not get_api_key():
        return {}
    try:
        resp = requests.get(f"{BASE_URL}/fixtures/events",
                            headers=_headers(),
                            params={"fixture": fixture_id},
                            timeout=10)
        events = resp.json().get("response", [])
        # Count progressive events as a proxy for pressing intensity
        home_events = [e for e in events if e.get("team", {}).get("name") == "home"]
        away_events = [e for e in events if e.get("team", {}).get("name") == "away"]
        return {
            "home_event_count":   len(home_events),
            "away_event_count":   len(away_events),
            "press_proxy_diff":   len(home_events) - len(away_events)
        }
    except Exception:
        return {}


def fetch_injuries(fixture_id: int) -> dict:
    """Fetch injury/suspension data with disk caching."""
    cache = {}
    if INJURIES_PATH.exists():
        with open(INJURIES_PATH) as f:
            cache = json.load(f)
    if str(fixture_id) in cache:
        return cache[str(fixture_id)]

    if not get_api_key():
        return {}

    try:
        resp = requests.get(f"{BASE_URL}/injuries",
                            headers=_headers(),
                            params={"fixture": fixture_id},
                            timeout=10)
        injuries = resp.json().get("response", [])
        result = {}
        for inj in injuries:
            team = inj["team"]["name"]
            result[team] = result.get(team, 0) + 1
            reason = inj["player"].get("reason", "")
            if any(k in reason for k in ["Knee", "ACL", "Hamstring", "Muscle"]):
                result[f"{team}_key_injury"] = True

        cache[str(fixture_id)] = result
        INJURIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INJURIES_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
        return result
    except Exception as e:
        warnings.warn(f"Injury fetch failed: {e}")
        return {}


# ─── Dataset builder ──────────────────────────────────────────────────────────
def parse_fixture_to_row(fixture: dict, stats: dict,
                          injuries: dict, distance: dict) -> dict:
    f, teams, goals = fixture["fixture"], fixture["teams"], fixture["goals"]
    home_team = teams["home"]["name"]
    away_team = teams["away"]["name"]

    hg = goals.get("home", 0) or 0
    ag = goals.get("away", 0) or 0

    row = {
        "fixture_id":       f["id"],
        "date":             f["date"][:10],
        "home_team":        home_team,
        "away_team":        away_team,
        "home_goals":       hg,
        "away_goals":       ag,
        "status":           f["status"]["short"],
        "is_neutral":       1,
        "tournament":       "FIFA World Cup 2026",
        "home_xg":          stats.get(f"{home_team}_expected_goals", None),
        "away_xg":          stats.get(f"{away_team}_expected_goals", None),
        "home_possession":  stats.get(f"{home_team}_ball_possession", None),
        "away_possession":  stats.get(f"{away_team}_ball_possession", None),
        "home_shots":       stats.get(f"{home_team}_total_shots", None),
        "away_shots":       stats.get(f"{away_team}_total_shots", None),
        "home_injuries":    injuries.get(home_team, 0),
        "away_injuries":    injuries.get(away_team, 0),
        "home_key_injury":  int(injuries.get(f"{home_team}_key_injury", False)),
        "away_key_injury":  int(injuries.get(f"{away_team}_key_injury", False)),
        "press_proxy_diff": distance.get("press_proxy_diff", 0),
        "home_event_count": distance.get("home_event_count", 0),
        "away_event_count": distance.get("away_event_count", 0),
    }

    if row["status"] == "FT":
        row["result"] = "Home Win" if hg > ag else "Draw" if hg == ag else "Away Win"
    else:
        row["result"] = None
    return row


def update_dataset(cutoff_date: str = "2026-06-01") -> list:
    """
    Main entry point. Fetches WC2026 matches after cutoff_date,
    appends to worldcup_matches.csv, skipping duplicates.
    Safe to call on every train_test.py run — no duplicates added.
    """
    print(f"\n🔄 Live updater: fetching WC2026 matches from {cutoff_date}...")

    existing_df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    existing_ids = set(existing_df["fixture_id"].astype(str)) if not existing_df.empty else set()

    fixtures = fetch_wc2026_fixtures()
    new_rows = []

    for fixture in fixtures:
        fid    = str(fixture["fixture"]["id"])
        date   = fixture["fixture"]["date"][:10]
        status = fixture["fixture"]["status"]["short"]

        if date < cutoff_date or status != "FT" or fid in existing_ids:
            continue

        stats    = fetch_fixture_stats(int(fid))
        injuries = fetch_injuries(int(fid))
        distance = fetch_distance_proxy(int(fid))
        time.sleep(0.4)   # API rate limit guard

        row = parse_fixture_to_row(fixture, stats, injuries, distance)
        new_rows.append(row)

    if new_rows:
        combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fixture_id"]).sort_values("date")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(DATA_PATH, index=False)
        print(f"  ✅ Added {len(new_rows)} new matches → {len(combined)} total")
    else:
        print(f"  ℹ️  No new completed WC2026 matches to add.")

    return new_rows


# ─── Mock fallback ────────────────────────────────────────────────────────────
def _mock_wc2026_fixtures() -> list:
    base = datetime(2026, 6, 11)
    matchups = [
        ("Brazil","Mexico",2,0), ("Germany","Japan",1,0), ("France","Australia",3,1),
        ("England","Iran",1,0), ("Argentina","Saudi Arabia",2,1), ("Spain","Morocco",2,1),
        ("Portugal","Ghana",1,1), ("Netherlands","Ecuador",0,0),
        ("Brazil","Switzerland",1,1), ("Germany","Spain",1,2),
        ("France","Denmark",0,0), ("England","USA",2,1),
    ]
    fixtures = []
    for i, (home, away, hg, ag) in enumerate(matchups):
        date = (base + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        fixtures.append({
            "fixture": {"id": 900000+i, "date": date,
                        "status": {"short": "FT"},
                        "venue": {"name": "Neutral"}},
            "teams": {"home": {"name": home, "id": 1000+i},
                      "away": {"name": away,  "id": 2000+i}},
            "goals": {"home": hg, "away": ag}
        })
    return fixtures


if __name__ == "__main__":
    update_dataset(cutoff_date="2026-06-01")
