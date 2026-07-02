import os
required_files = [
    # New Wall Street modules
    'models/factor_model.py',
    'models/kalman_strength.py',
    'betting/stat_arb.py',
    'features/regime_detector_v2.py',
    'betting/information_ratio_kelly.py',
    # Existing V7 modules (must still exist)
    'models/meta_learner.py',
    'models/temperature_scaler.py',
    'models/poisson_dixon_coles.py',
    'models/match_rules.py',
    'models/conformal.py',
    'models/base_learners.py',
    'data/live_auto_poller.py',
    'data/live_data_hub.py',
    'monitoring/regime_detector.py',
    'betting/portfolio_kelly.py',
    'betting/information_ratio_kelly.py',
    'betting/bet_tracker.py',
    'model_registry.py',
    'inference.py',
    'train_test.py',
    'app.py',
    '.github/workflows/deploy.yml',
]
missing = [f for f in required_files if not os.path.exists(f)]
present = [f for f in required_files if os.path.exists(f)]
print(f'Present: {len(present)}/{len(required_files)}')
if missing:
    print('MISSING FILES:')
    for f in missing:
        print(f'  ❌ {f}')
else:
    print('✅ All required files present')
