# data/live_auto_poller.py
"""
Fully automatic live match state fetcher.
Pulls ALL match data from API-Football every 60 seconds.
No manual input required. Feeds directly into inference.py.
"""
import requests, json, time, os, threading, warnings
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LIVE_STATE_PATH = Path("data/live_state.json")
POLL_INTERVAL   = 60   # seconds between API calls
_poller_thread  = None
_latest_state   = {}


def _headers() -> dict:
    key = os.getenv("RAPIDAPI_KEY", "")
    return {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

BASE = "https://api-football-v1.p.rapidapi.com/v3"


def fetch_all_live_data(fixture_id: int) -> dict:
    """
    Single function — fetches EVERYTHING for a live fixture:
    score, goals, cards, substitutions, stats, lineups, events.
    Returns a completely flat state dict ready for inference.
    """
    state = {
        "fixture_id":       fixture_id,
        "fetched_at":       datetime.utcnow().isoformat(),
        "status":           "unknown",
        "match_period":     "regular",
        "elapsed":          0,
        "home_team":        "",
        "away_team":        "",
        "home_goals":       0,
        "away_goals":       0,
        # Cards
        "home_yellow":      0,
        "away_yellow":      0,
        "home_red":         0,
        "away_red":         0,
        # Substitutions
        "home_subs_used":   0,
        "away_subs_used":   0,
        "home_subs_left":   3,
        "away_subs_left":   3,
        # Stats
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
        # Events timeline
        "goals_timeline":   [],
        "cards_timeline":   [],
        "subs_timeline":    [],
        # Lineups
        "home_formation":   "",
        "away_formation":   "",
        "home_xi":          [],
        "away_xi":          [],
    }

    # ── 1. Fixture overview (score + period) ──────────────────────────────
    try:
        r = requests.get(f"{BASE}/fixtures",
                         headers=_headers(),
                         params={"id": fixture_id},
                         timeout=10)
        fixtures = r.json().get("response", [])
        if fixtures:
            f      = fixtures[0]
            fix    = f["fixture"]
            teams  = f["teams"]
            goals  = f["goals"]
            status = fix.get("status", {})

            state["home_team"]    = teams["home"]["name"]
            state["away_team"]    = teams["away"]["name"]
            state["home_goals"]   = goals.get("home", 0) or 0
            state["away_goals"]   = goals.get("away", 0) or 0
            state["elapsed"]      = int(status.get("elapsed", 0) or 0)
            state["status"]       = status.get("short", "NS")

            # Determine match period
            short = state["status"]
            if short in ["1H", "HT"]:
                state["match_period"] = "regular"
            elif short in ["2H"]:
                state["match_period"] = "regular"
            elif short in ["ET", "BT"]:
                state["match_period"] = "extra_time"
                state["elapsed"]      = state["elapsed"] or 90
            elif short in ["P"]:
                state["match_period"] = "penalties"
            elif short == "FT":
                state["match_period"] = "finished"
    except Exception as e:
        warnings.warn(f"Fixture fetch failed: {e}")

    # ── 2. Statistics (shots, possession, corners, passes, xG) ────────────
    try:
        r = requests.get(f"{BASE}/fixtures/statistics",
                         headers=_headers(),
                         params={"fixture": fixture_id},
                         timeout=10)
        stats_data = r.json().get("response", [])

        for team_data in stats_data:
            team_name = team_data["team"]["name"]
            side = "home" if team_name == state["home_team"] else "away"

            for stat in team_data.get("statistics", []):
                t   = stat["type"]
                val = stat["value"]
                if val is None:
                    continue
                if isinstance(val, str) and val.endswith("%"):
                    val = float(val.replace("%", ""))

                mapping = {
                    "Shots on Goal":       f"{side}_shots_ot",
                    "Total Shots":         f"{side}_shots_total",
                    "Ball Possession":     f"{side}_possession",
                    "Corner Kicks":        f"{side}_corners",
                    "Total passes":        f"{side}_passes",
                    "Passes %":            f"{side}_pass_acc",
                    "Fouls":               f"{side}_fouls",
                    "Offsides":            f"{side}_offsides",
                    "expected_goals":      f"{side}_xg_live",
                    "Expected Goals":      f"{side}_xg_live",
                }
                if t in mapping:
                    try:
                        state[mapping[t]] = float(val)
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        warnings.warn(f"Stats fetch failed: {e}")

    # ── 3. Events (goals, cards, substitutions — with timestamps) ─────────
    try:
        r = requests.get(f"{BASE}/fixtures/events",
                         headers=_headers(),
                         params={"fixture": fixture_id},
                         timeout=10)
        events = r.json().get("response", [])

        for event in events:
            team_name  = event.get("team", {}).get("name", "")
            side       = "home" if team_name == state["home_team"] else "away"
            etype      = event.get("type", "")
            detail     = event.get("detail", "")
            minute     = event.get("time", {}).get("elapsed", 0) or 0
            player     = event.get("player", {}).get("name", "")
            assist     = event.get("assist", {}).get("name", "")

            if etype == "Goal":
                state["goals_timeline"].append({
                    "minute": minute, "team": side,
                    "player": player, "assist": assist,
                    "detail": detail
                })

            elif etype == "Card":
                if detail == "Yellow Card":
                    state[f"{side}_yellow"] += 1
                elif detail in ["Red Card", "Second Yellow"]:
                    state[f"{side}_red"]    += 1
                state["cards_timeline"].append({
                    "minute": minute, "team": side,
                    "player": player, "card_type": detail
                })

            elif etype == "subst":
                state[f"{side}_subs_used"] += 1
                state[f"{side}_subs_left"]  = max(
                    0, 5 - state[f"{side}_subs_used"]  # WC2026 allows 5 subs
                )
                state["subs_timeline"].append({
                    "minute": minute, "team": side,
                    "player_out": player, "player_in": assist
                })
    except Exception as e:
        warnings.warn(f"Events fetch failed: {e}")

    # ── 4. Lineups (starting XI + formation) ──────────────────────────────
    try:
        r = requests.get(f"{BASE}/fixtures/lineups",
                         headers=_headers(),
                         params={"fixture": fixture_id},
                         timeout=10)
        lineups = r.json().get("response", [])

        for lineup in lineups:
            team_name = lineup["team"]["name"]
            side      = "home" if team_name == state["home_team"] else "away"
            formation = lineup.get("formation", "")
            start_xi  = [
                {
                    "name":     p["player"]["name"],
                    "number":   p["player"]["number"],
                    "position": p["player"]["pos"],
                    "grid":     p["player"].get("grid", "")
                }
                for p in lineup.get("startXI", [])
            ]
            state[f"{side}_formation"] = formation
            state[f"{side}_xi"]        = start_xi
    except Exception as e:
        warnings.warn(f"Lineups fetch failed: {e}")

    # ── Save to disk ───────────────────────────────────────────────────────
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return state


def start_auto_poller(fixture_id: int, interval: int = 60):
    """
    Start background thread that auto-fetches live data every `interval` seconds.
    Stops automatically when match finishes.
    """
    global _poller_thread, _latest_state

    def _poll_loop():
        global _latest_state
        print(f"📡 Auto-poller started for fixture {fixture_id} "
              f"(every {interval}s)")
        while True:
            state = fetch_all_live_data(fixture_id)
            _latest_state = state

            elapsed = state.get("elapsed", 0)
            period  = state.get("match_period", "regular")
            status  = state.get("status", "")

            print(f"  [{datetime.utcnow().strftime('%H:%M:%S')}] "
                  f"{state['home_team']} "
                  f"{state['home_goals']}-{state['away_goals']} "
                  f"{state['away_team']} | "
                  f"{elapsed}' | {period} | "
                  f"xG: {state.get('home_xg_live','?')}/{state.get('away_xg_live','?')}")

            if status in ["FT", "AET", "PEN", "ABD"]:
                print(f"  ✅ Match finished ({status}). Poller stopped.")
                break

            time.sleep(interval)

    _poller_thread = threading.Thread(target=_poll_loop, daemon=True)
    _poller_thread.start()
    return _poller_thread


def get_latest_state() -> dict:
    """Return the most recently fetched live state (thread-safe read)."""
    global _latest_state
    if _latest_state:
        return _latest_state
    if LIVE_STATE_PATH.exists():
        with open(LIVE_STATE_PATH) as f:
            return json.load(f)
    return {}
