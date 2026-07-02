import json
import re
import os

with open(r'C:\Users\MMS Mandapeta\.gemini\antigravity\scratch\exact_code_utf8.jsonl', 'r', encoding='utf-8-sig') as f:
    for line in f:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = data.get('content', '')
        blocks = re.findall(r'```python\s*#\s*(.*?)\s+(.*?)```', content, re.DOTALL)
        for path, code in blocks:
            path = path.strip()
            print(f'Found: {path}')
            if path in ('betting/stat_arb.py', 'betting/information_ratio_kelly.py', 'models/kalman_strength.py', 'models/factor_model.py', 'features/regime_detector_v2.py'):
                full_path = os.path.join(r'C:\Users\MMS Mandapeta\Downloads\worldcup_v5', path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as out:
                    out.write(code)
                print(f'Wrote {path}')
