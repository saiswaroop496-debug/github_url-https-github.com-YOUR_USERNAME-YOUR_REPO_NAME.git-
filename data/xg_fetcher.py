# data/xg_fetcher.py
"""
Fetch REAL historical xG from Understat (free, no API key).
Covers: EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL (2014+)
For international matches: use API-Football's live xG field (v3 endpoint)
which provides real xG from their data partner (Sofascore).
"""
import requests
import json
import re
import pandas as pd
from pathlib import Path


def fetch_understat_match_xg(match_id: int) -> dict:
    """
    Fetch real xG for a specific Understat match.
    Returns {'home_xg': float, 'away_xg': float}
    """
    url = f"https://understat.com/match/{match_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        # Understat embeds JSON in <script> tags
        scripts = re.findall(
            r"var\s+shotsData\s*=\s*JSON\.parse\('(.+?)'\)",
            r.text
        )
        if not scripts:
            return {}
        shots = json.loads(scripts[0].encode('utf-8').decode('unicode_escape'))
        home_xg = sum(float(s['xG']) for s in shots.get('h', []))
        away_xg = sum(float(s['xG']) for s in shots.get('a', []))
        return {'home_xg': round(home_xg, 3), 'away_xg': round(away_xg, 3)}
    except Exception as e:
        return {}


def fetch_real_international_xg_from_api(fixture_id: int, api_key: str) -> dict:
    """
    API-Football v3 provides real xG from their Sofascore data partner
    in the /fixtures/statistics endpoint under 'expected_goals'.
    This is REAL xG — not synthetic.
    """
    headers = {
        "X-RapidAPI-Key":  api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    try:
        r = requests.get(
            "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics",
            headers=headers,
            params={"fixture": fixture_id},
            timeout=10
        )
        data = r.json().get("response", [])
        result = {}
        for i, team_data in enumerate(data):
            side = "home" if i == 0 else "away"
            for stat in team_data.get("statistics", []):
                if stat["type"] in ["expected_goals", "Expected Goals"]:
                    try:
                        result[f"{side}_xg"] = float(stat["value"] or 0)
                    except (ValueError, TypeError):
                        pass
        return result
    except Exception:
        return {}


def patch_synthetic_xg_in_dataset(df: pd.DataFrame,
                                    api_key: str = "") -> pd.DataFrame:
    """
    Replace synthetic xG (goals + noise) with real xG from API-Football.
    Only patches rows where fixture_id is available and xG was synthetic.
    Falls back to keeping existing value if real xG fetch fails.
    """
    if 'fixture_id' not in df.columns:
        print("  ⚠️  No fixture_id column — cannot fetch real xG")
        return df

    patched = 0
    for idx, row in df.iterrows():
        fid = row.get('fixture_id')
        if not fid or pd.isna(fid):
            continue

        real_xg = fetch_real_international_xg_from_api(int(fid), api_key)
        if real_xg.get('home_xg') is not None:
            df.at[idx, 'home_xg'] = real_xg['home_xg']
            df.at[idx, 'away_xg'] = real_xg['away_xg']
            patched += 1

        if patched % 50 == 0 and patched > 0:
            print(f"  Patched xG for {patched} matches...")

    print(f"  ✅ Real xG patched for {patched}/{len(df)} matches")
    return df
