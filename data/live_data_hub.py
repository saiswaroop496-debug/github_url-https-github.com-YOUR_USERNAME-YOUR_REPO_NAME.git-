import requests, json, time, warnings
import pandas as pd
from bs4 import BeautifulSoup

# ─── Priority 1: API-Football (structured, your existing key) ────────────────
def fetch_api_football_live(fixture_id: int, api_key: str) -> dict:
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    endpoints = {
        "stats":   f"https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics?fixture={fixture_id}",
        "events":  f"https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture={fixture_id}",
        "lineups": f"https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture={fixture_id}",
    }
    result = {}
    for key, url in endpoints.items():
        try:
            r = requests.get(url, headers=headers, timeout=8)
            result[key] = r.json().get("response", [])
        except Exception:
            result[key] = []
    return result

# ─── Priority 2: SofaScore unofficial API (live xG, pressures, duels) ────────
def fetch_sofascore_advanced(event_id: int) -> dict:
    """
    Unofficial SofaScore JSON — gives live xG, expected assists,
    ball recoveries, possession won final third, aerial duels.
    Free, no key required. May get rate-limited at high frequency.
    """
    base = f"https://api.sofascore.com/api/v1/event/{event_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Referer": "https://www.sofascore.com/"
    }
    endpoints = {
        "statistics":    f"{base}/statistics",
        "momentum":      f"{base}/activity",   # attack momentum timeline
        "heatmap":       f"{base}/heatmaps",
    }
    result = {}
    for k, url in endpoints.items():
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                result[k] = r.json()
        except Exception:
            pass
        time.sleep(0.3)
    return result

# ─── Priority 3: FBref scraping (post-match, updates within minutes of FT) ───
def fetch_fbref_match(match_url: str) -> dict:
    """
    FBref match report: xG, progressive passes, pressures, PPDA,
    field tilt, and shot map. Free, no key required.
    """
    try:
        dfs = pd.read_html(match_url)
        summary = {}
        for df in dfs:
            if 'xG' in df.columns:
                summary['xg_table'] = df.to_dict()
            if 'Poss' in df.columns:
                summary['possession_table'] = df.to_dict()
        return summary
    except Exception:
        return {}

# ─── Priority 4: BBC Sport / ESPN live text events (free scraping) ────────────
def fetch_bbc_live_commentary(match_url: str) -> list:
    """
    Scrape BBC Sport live text commentary for event parsing.
    Returns list of {minute, event_type, description}.
    """
    events = []
    try:
        r = requests.get(match_url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        # BBC Sport uses data-reactroot with JSON state
        scripts = soup.find_all('script', type='application/json')
        for s in scripts:
            try:
                data = json.loads(s.string)
                # Parse commentary events from JSON structure
                if 'commentary' in str(data):
                    events.append(data)
            except Exception:
                pass
    except Exception:
        pass
    return events

# ─── Priority 5: Odds movement scraping (Oddschecker / Betfair API) ──────────
def fetch_odds_movement(home_team: str, away_team: str) -> dict:
    """
    Track opening vs current odds to detect sharp money movement.
    Uses Oddschecker public JSON (no key, rate-limited).
    """
    try:
        query = f"{home_team.lower().replace(' ', '-')}-v-{away_team.lower().replace(' ', '-')}"
        url   = f"https://www.oddschecker.com/football/world-cup/{query}"
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        # Extract odds table
        rows = soup.select('tr[data-bk]')
        odds = {}
        for row in rows:
            bk   = row.get('data-bk', '')
            vals = [td.text.strip() for td in row.find_all('td', class_='bc')]
            if vals and bk:
                odds[bk] = vals[:3]   # H/D/A
        return odds
    except Exception:
        return {}

# ─── Master aggregator ────────────────────────────────────────────────────────
def get_complete_live_state(fixture_id: int, sofascore_id: int = None,
                             fbref_url: str = None, api_key: str = "") -> dict:
    """
    Pull all sources, merge into unified state dict.
    Fields available for live DC prediction:
    - elapsed, home_goals, away_goals, red_cards
    - home_xg_live, away_xg_live (SofaScore)
    - home_shots_ot, away_shots_ot, corners, passes
    - sharp_movement (odds drift)
    - momentum_index (SofaScore activity)
    """
    state = {}

    # API-Football (primary)
    if api_key:
        af = fetch_api_football_live(fixture_id, api_key)
        state['api_football'] = af

    # SofaScore (supplement)
    if sofascore_id:
        ss = fetch_sofascore_advanced(sofascore_id)
        state['sofascore'] = ss

    # Consolidate into flat structure
    consolidated = _merge_sources(state)
    return consolidated

def _merge_sources(state: dict) -> dict:
    """Flatten multi-source state into single prediction-ready dict."""
    out = {
        "elapsed": 0, "home_goals": 0, "away_goals": 0,
        "home_xg_live": None, "away_xg_live": None,
        "home_shots_ot": 0, "away_shots_ot": 0,
        "home_corners": 0, "away_corners": 0,
        "home_passes": 0, "away_passes": 0,
        "home_possession": 50, "away_possession": 50,
        "red_cards": {"home": 0, "away": 0},
        "momentum_index": 0.0,
        "sharp_movement": 0.0,
    }

    # Unpack API-Football stats
    af = state.get('api_football', {})
    for team_data in af.get('stats', []):
        side = 'home' if af['stats'].index(team_data) == 0 else 'away'
        for stat in team_data.get('statistics', []):
            t   = stat['type'].lower().replace(' ', '_')
            val = stat['value']
            if isinstance(val, str) and val.endswith('%'):
                val = float(val.replace('%','')) / 100
            out[f'{side}_{t}'] = val

    # SofaScore xG override (more accurate)
    ss_stats = state.get('sofascore', {}).get('statistics', {})
    if ss_stats:
        for group in ss_stats.get('statistics', []):
            for item in group.get('statisticsItems', []):
                if 'xG' in item.get('name', ''):
                    out['home_xg_live'] = item.get('home')
                    out['away_xg_live'] = item.get('away')

    return out
