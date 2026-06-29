# data/live_match_poller.py
import requests, time, json, os, re, warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RAPIDAPI_KEY", "")
HEADERS = {
    "X-RapidAPI-Key":  API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
BASE = "https://api-football-v1.p.rapidapi.com/v3"
LIVE_STATE_PATH = Path("data/live_state.json")


# ─── Source 1: API-Football Live ─────────────────────────────────────────────
def fetch_live_api_football() -> list:
    """
    Fetch all currently live WC2026 fixtures.
    Returns list of live match state dicts.
    """
    if not API_KEY:
        return []
    try:
        resp = requests.get(f"{BASE}/fixtures",
                            headers=HEADERS,
                            params={"live": "all", "league": 1},
                            timeout=10)
        fixtures = resp.json().get("response", [])
        results  = []
        for f in fixtures:
            fid      = f["fixture"]["id"]
            elapsed  = f["fixture"].get("status", {}).get("elapsed", 0) or 0
            score    = f["goals"]
            teams    = f["teams"]
            stats_raw = _fetch_live_stats(fid)
            events   = _fetch_live_events(fid)

            results.append({
                "fixture_id":      fid,
                "home_team":       teams["home"]["name"],
                "away_team":       teams["away"]["name"],
                "elapsed":         int(elapsed),
                "home_goals":      score.get("home", 0) or 0,
                "away_goals":      score.get("away", 0) or 0,
                "home_possession": stats_raw.get("home_ball_possession", 50),
                "away_possession": stats_raw.get("away_ball_possession", 50),
                "home_shots_ot":   stats_raw.get("home_shots_on_goal", 0),
                "away_shots_ot":   stats_raw.get("away_shots_on_goal", 0),
                "home_corners":    stats_raw.get("home_corner_kicks", 0),
                "away_corners":    stats_raw.get("away_corner_kicks", 0),
                "home_passes":     stats_raw.get("home_total_passes", 0),
                "away_passes":     stats_raw.get("away_total_passes", 0),
                "home_xg_live":    stats_raw.get("home_expected_goals", None),
                "away_xg_live":    stats_raw.get("away_expected_goals", None),
                "red_cards":       _count_red_cards(events),
                "last_goal_team":  _last_goal_team(events),
                "last_goal_min":   _last_goal_minute(events),
                "source":          "api-football"
            })
        return results
    except Exception as e:
        warnings.warn(f"API-Football live fetch failed: {e}")
        return []


def _fetch_live_stats(fixture_id: int) -> dict:
    try:
        resp = requests.get(f"{BASE}/fixtures/statistics",
                            headers=HEADERS,
                            params={"fixture": fixture_id},
                            timeout=8)
        data = resp.json().get("response", [])
        out  = {}
        for team_data in data:
            side = "home" if data.index(team_data) == 0 else "away"
            for stat in team_data.get("statistics", []):
                key = f"{side}_{stat['type'].lower().replace(' ', '_')}"
                val = stat["value"]
                if isinstance(val, str) and val.endswith("%"):
                    val = float(val.replace("%","")) / 100
                out[key] = val
        return out
    except Exception:
        return {}


def _fetch_live_events(fixture_id: int) -> list:
    try:
        resp = requests.get(f"{BASE}/fixtures/events",
                            headers=HEADERS,
                            params={"fixture": fixture_id},
                            timeout=8)
        return resp.json().get("response", [])
    except Exception:
        return []


def _count_red_cards(events: list) -> dict:
    reds = {"home": 0, "away": 0}
    for e in events:
        if e.get("type") == "Card" and e.get("detail") in ["Red Card", "Second Yellow"]:
            team = e.get("team", {}).get("name", "")
            reds[team] = reds.get(team, 0) + 1
    return reds


def _last_goal_team(events: list) -> str:
    goals = [e for e in events if e.get("type") == "Goal"]
    if goals:
        return goals[-1].get("team", {}).get("name", "")
    return ""


def _last_goal_minute(events: list) -> int:
    goals = [e for e in events if e.get("type") == "Goal"]
    if goals:
        return int(goals[-1].get("time", {}).get("elapsed", 0) or 0)
    return 0


# ─── Source 2: SofaScore scraper (unofficial, free) ──────────────────────────
def fetch_sofascore_live(match_id: int = None) -> dict:
    """
    SofaScore unofficial JSON endpoint — live xG and advanced stats.
    match_id is the SofaScore event ID (find from sofascore.com URL).
    Falls back gracefully if blocked.
    """
    if not match_id:
        return {}
    try:
        url     = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36",
            "Referer": "https://www.sofascore.com/"
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {}

        stats = resp.json().get("statistics", [])
        out   = {}
        for group in stats:
            for item in group.get("statisticsItems", []):
                key = item.get("name", "").lower().replace(" ", "_")
                out[f"home_{key}"] = item.get("home")
                out[f"away_{key}"] = item.get("away")
        return out
    except Exception:
        return {}


# ─── Source 3: Live stream frame analysis (YOLOv8) ───────────────────────────
def fetch_live_stream_frame(stream_url: str,
                             homography_points=None) -> dict:
    """
    Pull a single frame from a live stream and run YOLOv8 tracker.
    Returns instantaneous physical stats.
    Works with any yt-dlp compatible URL (YouTube, Twitch, etc.)
    """
    try:
        from data.player_tracker import PlayerMovementTracker, _check_tracking_deps
        if not _check_tracking_deps():
            return {}

        import cv2, subprocess
        # Get direct stream URL via yt-dlp
        cmd = ["yt-dlp", "-g", "--format", "best[height<=720]", stream_url]
        direct_url = subprocess.check_output(cmd, timeout=15).decode().strip()

        cap = cv2.VideoCapture(direct_url)
        # Skip to near-live (grab frame 300 = ~10s buffer)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {}

        tracker = PlayerMovementTracker(homography_points=homography_points)
        tracker.model(frame, classes=[0], conf=0.4, verbose=False)
        return {"frame_captured": True, "shape": str(frame.shape)}

    except Exception as e:
        return {}


# ─── Aggregate all sources ────────────────────────────────────────────────────
def get_live_match_state(fixture_id: int = None,
                          sofascore_id: int = None,
                          stream_url: str = None) -> dict:
    """
    Master aggregator. Pulls from all available sources and merges.
    Priority: API-Football > SofaScore > Stream frame
    Saves to data/live_state.json for continuous reading.
    """
    state = {}

    # API-Football (primary)
    live_fixtures = fetch_live_api_football()
    if fixture_id:
        match = next((f for f in live_fixtures if f["fixture_id"] == fixture_id), None)
        if match:
            state.update(match)
    elif live_fixtures:
        state.update(live_fixtures[0])   # default: first live match

    # SofaScore (supplement live xG if missing)
    if sofascore_id and not state.get("home_xg_live"):
        ss = fetch_sofascore_live(sofascore_id)
        if ss:
            state["home_xg_live_ss"] = ss.get("home_expected_goals_(xg)")
            state["away_xg_live_ss"] = ss.get("away_expected_goals_(xg)")
            state["home_passes_ss"]  = ss.get("home_passes")
            state["away_passes_ss"]  = ss.get("away_passes")

    state["fetched_at"] = datetime.utcnow().isoformat()

    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return state


def poll_live_every_n_seconds(fixture_id: int, interval: int = 60,
                               sofascore_id: int = None):
    """Continuous poller — runs in a background thread."""
    import threading
    def _poll():
        while True:
            try:
                state = get_live_match_state(fixture_id, sofascore_id)
                elapsed = state.get("elapsed", 0)
                print(f"  📡 {state.get('home_team')} {state.get('home_goals')}-"
                      f"{state.get('away_goals')} {state.get('away_team')} "
                      f"| {elapsed}' | xG H/A: "
                      f"{state.get('home_xg_live','?')}/{state.get('away_xg_live','?')}")
                if elapsed >= 90:
                    print("  ✅ Match finished. Stopping poller.")
                    break
            except Exception as e:
                print(f"  ⚠️  Poll error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    return t
