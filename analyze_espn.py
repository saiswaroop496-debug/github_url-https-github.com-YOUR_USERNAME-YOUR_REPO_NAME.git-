import json

with open("espn_test.json") as f:
    data = json.load(f)

print("Top-level keys:", data.keys())

if "boxscore" in data:
    print("Boxscore teams:")
    for team in data["boxscore"]["teams"]:
        print(f"  Team: {team['team']['displayName']}")
        for stat in team.get("statistics", []):
            print(f"    {stat['name']}: {stat['displayValue']}")
else:
    print("No boxscore found")

if "header" in data:
    print("Header info:")
    header = data["header"]
    print("  Status:", header.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("description"))

if "rosters" in data:
    print("Rosters:", len(data["rosters"]))

if "plays" in data:
    print(f"Plays found: {len(data['plays'])}")
    for play in data["plays"][-3:]:
        print(f"  - {play.get('clock', {}).get('displayValue', '')}: {play.get('text', '')}")

if "keyEvents" in data:
    print(f"Key Events found: {len(data['keyEvents'])}")
    for event in data["keyEvents"][-3:]:
        print(f"  - {event.get('clock', {}).get('displayValue', '')}: {event.get('text', '')}")
