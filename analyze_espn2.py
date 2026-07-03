import json

with open("espn_test.json") as f:
    data = json.load(f)

header = data.get("header", {})
comps = header.get("competitions", [{}])[0]
competitors = comps.get("competitors", [])

print(f"Status: {comps.get('status', {})}")
print(f"Clock: {comps.get('status', {}).get('displayClock')}")
print(f"Period: {comps.get('status', {}).get('period')}")
print(f"State: {comps.get('status', {}).get('type', {}).get('state')}")
print(f"Detail: {comps.get('status', {}).get('type', {}).get('detail')}")

for comp in competitors:
    team = comp.get("team", {})
    print(f"\nTeam: {team.get('displayName')} (Home/Away: {comp.get('homeAway')})")
    print(f"Score: {comp.get('score')}")
    print(f"Formation: {comp.get('formation')}")
    
    if "lineups" in data.get("rosters", [{}])[0]:
        # just print first player
        pass
