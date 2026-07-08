import re

html_path = r"C:\Users\MMS Mandapeta\.gemini\antigravity\brain\3b87df0b-690d-4948-9190-44b802ca1365\.system_generated\steps\15293\content.md"

with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

# find {"ocBetId" and parse out the chunk
matches = re.findall(r'(\{"ocBetId".*?\})', html_content)
print(f"Found {len(matches)} ocBetId blocks.")

parsed = []
for m in matches:
    if "betName" in m:
        try:
            # crude extraction since it might not be perfect json
            name_match = re.search(r'"betName":"([^"]+)"', m)
            if name_match:
                name = name_match.group(1)
                parsed.append(name)
        except:
            pass

from collections import Counter
print("Teams found:", Counter(parsed).most_common(20))
