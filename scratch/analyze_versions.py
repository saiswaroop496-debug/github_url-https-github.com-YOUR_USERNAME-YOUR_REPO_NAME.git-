import json
from pathlib import Path

transcript_path = Path(r"C:\Users\MMS Mandapeta\.gemini\antigravity\brain\3b87df0b-690d-4948-9190-44b802ca1365\.system_generated\logs\transcript.jsonl")

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
        
# Search strictly for V7.6 or V7.7 in the transcript
found = []
for line in lines:
    try:
        data = json.loads(line)
        content = data.get('content', '')
        if content and ('v7.6' in content.lower() or 'v7.7' in content.lower() or 'v7.8' in content.lower() or 'v8' in content.lower()):
            found.append((data.get('step_index'), data.get('source'), content))
    except:
        pass
        
print(f"Found {len(found)} mentions of V7.6, V7.7, V8, etc.")
for idx, source, text in found:
    # Print a snippet around the match
    idx_v = max(0, text.lower().find('v7.'))
    print(f"Step {idx} [{source}]: {text[idx_v-100:idx_v+200]}")
