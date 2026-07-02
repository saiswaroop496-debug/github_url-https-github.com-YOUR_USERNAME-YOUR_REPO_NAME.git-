with open('v7_1_validation.txt') as f:
    txt = f.read()
checks = ['kalman', 'regime', 'factor', 'velocity', 'temperature', 'ECE', 'Log-Loss', 'Accuracy']
for c in checks:
    found = c.lower() in txt.lower()
    print(f'  {"✅" if found else "❌"} {c} mentioned in output')

import json
s = json.load(open('model_versions/latest/team_states.json'))
sample_team = list(s.keys())[0]
v = s[sample_team]
print(f'\nSample team: {sample_team}')
required_keys = ['glicko', 'rd', 'kalman_strength', 'kalman_velocity', 'regime', 'regime_coef']
for k in required_keys:
    present = k in v
    val = v.get(k, 'MISSING')
    print(f'  {"✅" if present else "❌"} {k}: {val}')

kalman_defaults = [t for t,vs in s.items() if vs.get('kalman_strength', 1500) == 1500.0]
print(f'Teams with default kalman_strength (1500): {len(kalman_defaults)} (should be 0 after 20+ matches)')

m = json.load(open('model_versions/latest/manifest.json'))
cols = m.get('feature_cols', [])
print(f'\nFeatures in manifest: {len(cols)}')
for c in cols:
    print(f'  {c}')
kalman_features = [c for c in cols if 'kalman' in c]
regime_features = [c for c in cols if 'regime' in c]
factor_features = [c for c in cols if 'factor' in c]
print(f'Kalman features: {len(kalman_features)} {kalman_features}')
print(f'Regime features: {len(regime_features)}')
print(f'Factor features: {len(factor_features)}')
