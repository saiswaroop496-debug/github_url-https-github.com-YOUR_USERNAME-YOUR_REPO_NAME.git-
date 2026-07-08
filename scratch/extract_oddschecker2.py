import re

html_path = r"C:\Users\MMS Mandapeta\.gemini\antigravity\brain\3b87df0b-690d-4948-9190-44b802ca1365\.system_generated\steps\15293\content.md"

with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

matches = re.findall(r'.{0,100}27245319694.{0,100}', html_content)
for m in matches:
    print(m)
