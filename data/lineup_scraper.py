import requests

def get_confirmed_lineup(fixture_id: int, api_key: str) -> dict:
    """
    API-Football provides confirmed lineups ~1hr before kick-off.
    This is the highest-value free data upgrade available.
    """
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    r = requests.get(
        "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups",
        headers=headers,
        params={"fixture": fixture_id},
        timeout=10
    )
    lineups = r.json().get("response", [])
    result = {}
    for lineup in lineups:
        team = lineup["team"]["name"]
        result[team] = {
            "formation": lineup.get("formation", ""),
            "start_xi":  [p["player"]["name"] for p in lineup.get("startXI", [])],
            "coach":     lineup.get("coach", {}).get("name", ""),
        }
    return result

def lineup_strength_adjustment(confirmed_xi: list,
                                 expected_xi: list) -> float:
    """
    Compare confirmed lineup to expected starting 11.
    Missing key players = negative adjustment to team strength.
    Returns adjustment multiplier: 1.0 = full strength, 0.8 = major absences.
    """
    if not confirmed_xi or not expected_xi:
        return 1.0
    overlap = len(set(confirmed_xi) & set(expected_xi)) / len(expected_xi)
    return max(0.75, overlap)   # minimum 75% strength even if completely rotated
