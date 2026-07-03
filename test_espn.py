import requests
import json

game_id = "760498"
url = f"http://site.api.espn.com/apis/site/v2/sports/soccer/all/summary?event={game_id}"
r = requests.get(url)
data = r.json()

with open("espn_test.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done. Saved to espn_test.json")
