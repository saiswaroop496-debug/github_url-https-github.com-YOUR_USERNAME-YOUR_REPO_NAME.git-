import sys, re
with open('train_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace Data loading in main
new_main_top = '''def main():
    print("=" * 70)
    print("  V6.2 QUANTITATIVE ENGINE — TRAIN & TEST REPORT")
    print("=" * 70)

    t_start = time.time()

    # 1. LIVE DATA UPDATE
    from data.live_updater import update_dataset
    update_dataset(cutoff_date="2026-06-01")

    # 1. DATA
    scraper = DataScraper()
    dc_df, form_df = scraper.fetch_fixtures()
    print(f"\\n[DATA]  dc_df (2015+): {len(dc_df)} matches")
    print(f"[DATA]  form_df (2018+): {len(form_df)} matches")

    # 2. FEATURES
    print("\\n[FEATURES]  Computing rolling features …")
    df = compute_rolling_features(form_df)
    print("[FEATURES]  Computing Glicko-2 ratings …")
    glicko = Glicko2RatingSystem()
    df = glicko.compute_ratings(df)

    print("[FEATURES]  Computing V6 specific features …")
    from features.rolling_features import compute_v6_features, add_injury_features, add_movement_features
    df = compute_v6_features(df)
    df = add_injury_features(df)
    df = add_movement_features(df)'''

code = re.sub(r'def main\(\):.*?(?=# 3\. FEATURE COLUMNS & TARGET)', new_main_top + '\n\n    ', code, flags=re.DOTALL)

new_feature_cols = '''# 3. FEATURE COLUMNS & TARGET
    FEATURE_COLS_FULL = [
        # Core Glicko signals (4)
        'home_glicko', 'home_rd', 'away_glicko', 'away_rd',
        # Derived strength signals (3)
        'glicko_signal', 'xg_supremacy', 'draw_affinity',
        # Form signals (2)
        'home_neutral_venue_form', 'away_neutral_venue_form',
        # Context signals (2)
        'rest_differential', 'stage_pressure',
        # NEW — Injury signals (2)
        'injury_differential', 'key_injury_factor',
        # NEW — Movement signals (2, default 0 when no video)
        'speed_diff', 'home_total_sprints',
        # NEW — API distance proxy (1)
        'press_proxy_diff',
    ]

    # Filter to only columns actually present in df (prevents manifest mismatch bug):
    FEATURE_COLS = [c for c in FEATURE_COLS_FULL if c in df.columns]
    print(f"  Features available: {len(FEATURE_COLS)}/{len(FEATURE_COLS_FULL)}")
    print(f"  Missing (will be added as 0): {[c for c in FEATURE_COLS_FULL if c not in df.columns]}")

    # Ensure missing columns exist in df as zeros
    for c in FEATURE_COLS_FULL:
        if c not in df.columns:
            df[c] = 0.0'''

code = re.sub(r'# 3\. FEATURE COLUMNS & TARGET.*?df = df\[df\[\'date\'\] > \'(.*?)\'\]', new_feature_cols + '''\n\n    print(f"[FEATURES]  Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    df = df.dropna(subset=FEATURE_COLS)

    df = df[df['date'] > '\\1']''', code, flags=re.DOTALL)

with open('train_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
