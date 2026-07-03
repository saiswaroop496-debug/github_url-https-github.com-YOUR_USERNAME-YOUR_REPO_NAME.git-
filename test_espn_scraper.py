from data.espn_scraper import fetch_espn_live_data
import json

state = fetch_espn_live_data("760498")
print(json.dumps(state, indent=2))
