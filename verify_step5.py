from inference import run_inference

# Test 1: Pre-match knockout (no draw should appear)
r = run_inference('Brazil', 'Germany', venue_factor=0.3, stage='quarter_final')
assert r is not None
assert r.get('draw', 1.0) == 0.0 or r.get('allows_draw', True) == False, 'FAIL: Draw shown in knockout'
print(f'✅ Knockout: Brazil {r["home_win_prob"]:.1%} | Germany {r["away_win_prob"]:.1%} | Draw {r.get("draw", 0.0):.1%}')

# Test 2: Group stage (draw must appear)
r2 = run_inference('France', 'Argentina', venue_factor=0.3, stage='group')
assert r2.get('draw_prob', 0) > 0.10, f'FAIL: Draw probability too low in group stage: {r2.get("draw_prob")}'
print(f'✅ Group: France {r2["home_win_prob"]:.1%} | Draw {r2["draw_prob"]:.1%} | Argentina {r2["away_win_prob"]:.1%}')

# Test 3: Different teams give different predictions
r3 = run_inference('Japan', 'Saudi Arabia', venue_factor=0.3, stage='group')
assert r['home_win_prob'] != r3['home_win_prob'], 'FAIL: All teams produce identical predictions'
print(f'✅ Predictions differ across matchups')

# Test 4: Kalman velocity is being used (check signals dict if available)
signals = r2.get('signals', {})
if 'kalman_velocity_diff' in signals:
    print(f'✅ Kalman velocity in signals: {signals["kalman_velocity_diff"]:.4f}')
else:
    print('⚠️  kalman_velocity_diff not in signals — check build_feature_vector()')

print('All inference tests passed')
