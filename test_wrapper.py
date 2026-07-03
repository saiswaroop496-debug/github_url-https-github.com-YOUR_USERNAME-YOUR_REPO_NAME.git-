from data.live_auto_poller import fetch_all_live_data
import json

state = fetch_all_live_data("https://www.espn.co.uk/football/match/_/gameId/760498")
print(json.dumps(state, indent=2))
