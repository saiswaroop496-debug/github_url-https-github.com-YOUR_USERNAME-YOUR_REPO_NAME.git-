import sys
import json
import re

html_path = r"C:\Users\MMS Mandapeta\.gemini\antigravity\brain\3b87df0b-690d-4948-9190-44b802ca1365\.system_generated\steps\15293\content.md"

with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

# Look for JSON hydration state window.__INITIAL_STATE__
match = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});", html_content, re.DOTALL)
if match:
    state_json = match.group(1)
    try:
        state = json.loads(state_json)
        print(f"Extracted state with keys: {list(state.keys())}")
        with open("scratch_state.json", "w", encoding="utf-8") as f2:
            json.dump(state, f2, indent=2)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
else:
    print("Could not find __INITIAL_STATE__.")
    
# Or NEXT_DATA
match2 = re.search(r'id="__NEXT_DATA__".*?>(.*?)</script>', html_content, re.DOTALL)
if match2:
    state_json = match2.group(1)
    try:
        state = json.loads(state_json)
        print(f"Extracted NEXT_DATA with keys: {list(state.keys())}")
        with open("scratch_state2.json", "w", encoding="utf-8") as f2:
            json.dump(state, f2, indent=2)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
else:
    print("Could not find __NEXT_DATA__.")

# Find matches around Brazil
brazil_matches = re.findall(r'.{0,100}Brazil.{0,100}', html_content, re.IGNORECASE)
print(f"Found {len(brazil_matches)} mentions of Brazil:")
for m in brazil_matches[:10]:
    print(m)
