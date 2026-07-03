"""
V7.0 FIFA World Cup Quantitative Engine — Full Train & Test Pipeline
=====================================================================
V7 UPGRADE: Temperature-scaled logit calibration, augmented meta-learner.
"""

import sys, os, time, warnings
import joblib
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from math import exp, log
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from data.scraper import DataScraper
from features.rolling_features import compute_rolling_features
from features.glicko_ratings import Glicko2RatingSystem
from models.kalman_strength import KalmanRatingSystem
from features.regime_detector_v2 import detect_team_regime
from models.factor_model import compute_team_factors, factor_matchup_score
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece

# =====================================================================
# VECTORIZED DIXON-COLES (replaces row-by-row for speed)
# =====================================================================
from models.poisson_dixon_coles import (
    fit_dixon_coles_nb, 
    score_probability_matrix, outcome_probs
)
from math import exp

class FastDixonColes:
    """Wrapper using the new two-stage V7.3 Dixon-Coles NB optimization."""
    
    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, df):
        t0 = time.time()
        dates = df['date'].view('int64') / 10**9 / 86400
        
        res = fit_dixon_coles_nb(df, dates.values)
        
        self.attack = res['attack']
        self.defense = res['defense']
        self.home_adv = res['home_adv']
        self.rho = res['rho']
        print(f"[DC MODEL]  Fitted in {time.time()-t0:.1f}s  (rho={self.rho:.4f}, "
              f"home_gamma={self.home_adv:.4f})")

    def predict_proba(self, team1, team2, venue_factor=0.0):
        if team1 not in self.attack or team2 not in self.attack:
            return np.array([0.33, 0.34, 0.33])
            
        lam_h = exp(self.attack[team1] - self.defense[team2] + self.home_adv * max(0.0, venue_factor))
        lam_a = exp(self.attack[team2] - self.defense[team1])
        lam_h = min(max(lam_h, 0.01), 15.0)
        lam_a = min(max(lam_a, 0.01), 15.0)
        
        matrix = score_probability_matrix(lam_h, lam_a, self.rho)
        ph, pd, pa = outcome_probs(matrix)
        return np.array([ph, pd, pa])

    def predict_proba_batch(self, teams1, teams2, venue_factors=None):
        """Batch prediction for speed."""
        n = len(teams1)
        if venue_factors is None:
            venue_factors = np.zeros(n)
        probs = np.zeros((n, 3))
        for i in range(n):
            probs[i] = self.predict_proba(teams1[i], teams2[i], venue_factors[i])
        return probs


from models.base_learners import xg_to_probs



def compute_dc_probs(row: pd.Series, dc_model_dict: dict) -> np.ndarray:
    from models.poisson_dixon_coles import score_probability_matrix, outcome_probs
    import math
    
    t1 = row['home_team']
    t2 = row['away_team']
    
    attack = dc_model_dict.get('attack', {})
    defense = dc_model_dict.get('defense', {})
    home_adv = dc_model_dict.get('home_adv', 0.0)
    rho = dc_model_dict.get('rho', 0.0)
    
    is_neutral = row.get('is_neutral', 0)
    h_adv = home_adv if not is_neutral else 0.0
    
    lam_h = attack.get(t1, 0.0) - defense.get(t2, 0.0) + h_adv
    lam_a = attack.get(t2, 0.0) - defense.get(t1, 0.0)
    
    lam_h = min(max(math.exp(lam_h), 0.01), 15.0)
    lam_a = min(max(math.exp(lam_a), 0.01), 15.0)
    
    matrix = score_probability_matrix(lam_h, lam_a, rho)
    h, d, a = outcome_probs(matrix)
    return np.array([h, d, a])

def compute_alpha_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create secondary alpha target: deviation from market implied probability.
    Requires historical odds columns (novig_home, novig_draw, novig_away).
    If odds not available, fills with NaN — ready for when you add the data.
    """
    if 'novig_home' in df.columns:
        # True alpha: binary outcome minus market probability
        df['alpha_home'] = (df['result'] == 'Home Win').astype(float) - df['novig_home']
        df['alpha_draw'] = (df['result'] == 'Draw').astype(float)     - df['novig_draw']
        df['alpha_away'] = (df['result'] == 'Away Win').astype(float) - df['novig_away']
        print("  ✅ Alpha targets computed from historical odds")
    else:
        df['alpha_home'] = np.nan
        df['alpha_draw'] = np.nan
        df['alpha_away'] = np.nan
        print("  ℹ️  Alpha targets: NaN (no historical odds — drop CSVs into data/ to activate)")
    return df


def compute_h2h_draw_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head specific draw rate for each match.
    Captures tactical matchup effects (e.g., Italy vs Spain always draws).
    Uses .shift(1).expanding() per pairing to prevent leakage.
    """
    df = df.sort_values('date').reset_index(drop=True)
    draw_rates = []

    for idx, row in df.iterrows():
        ht = row['home_team']
        at = row['away_team']

        # All previous matches between these two teams (either direction)
        past = df[
            (df.index < idx) & (
                ((df['home_team'] == ht) & (df['away_team'] == at)) |
                ((df['home_team'] == at) & (df['away_team'] == ht))
            )
        ]

        if len(past) >= 2:
            rate = float((past['home_goals'] == past['away_goals']).mean())
        else:
            rate = 0.29   # international football prior

        draw_rates.append(rate)

    df['h2h_draw_rate'] = draw_rates
    return df


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    print("=" * 70)
    print("  V7.2 QUANTITATIVE ENGINE — TRAIN & TEST REPORT")
    print("=" * 70)

    t_start = time.time()

    # 1. LIVE DATA UPDATE
    # from data.live_updater import update_dataset
    # update_dataset(cutoff_date="2026-06-01")

    # 1. DATA
    print("\n[DATA] Loading and enriching dataset with vision xG...")
    from data.scraper import load_and_enrich_dataset
    dc_df, form_df = load_and_enrich_dataset("data/worldcup_matches.csv")
    print(f"[DATA] form_df: {len(form_df)} matches")


    # 2. FEATURES
    print("\n[FEATURES]  Computing rolling features â€¦")
    df = compute_rolling_features(form_df)
    print("[FEATURES]  Computing Glicko-2 ratings â€¦")
    glicko = Glicko2RatingSystem()
    df = glicko.compute_ratings(df)

    print("[FEATURES]  Computing V6 & Institutional features ...")
    from features.rolling_features import compute_v6_features, add_injury_features, add_movement_features, add_institutional_signals
    df = compute_v6_features(df)
    df = add_injury_features(df)
    df = add_movement_features(df)
    df = add_institutional_signals(df)

    df = compute_alpha_target(df)
    df = compute_h2h_draw_rate(df)

    print("[FEATURES]  Computing Kalman, Regime & Factor features â€¦")
    kalman_system = KalmanRatingSystem()
    df = df.sort_values('date').reset_index(drop=True)
    
    for i in range(len(df)):
        row = df.iloc[i]
        idx = row.name
        
        # Kalman
        df.at[idx, 'home_kalman_strength'] = kalman_system.get_strength(row['home_team'])
        df.at[idx, 'home_kalman_velocity']  = kalman_system.get_velocity(row['home_team'])
        df.at[idx, 'home_kalman_uncertainty'] = kalman_system.get_uncertainty(row['home_team'])
        df.at[idx, 'away_kalman_strength']  = kalman_system.get_strength(row['away_team'])
        df.at[idx, 'away_kalman_velocity']  = kalman_system.get_velocity(row['away_team'])
        df.at[idx, 'away_kalman_uncertainty'] = kalman_system.get_uncertainty(row['away_team'])
        
        # Signal
        df.at[idx, 'kalman_velocity_diff'] = df.at[idx, 'home_kalman_velocity'] - df.at[idx, 'away_kalman_velocity']
        df.at[idx, 'kalman_signal'] = (df.at[idx, 'home_kalman_strength'] - df.at[idx, 'away_kalman_strength']) / np.sqrt(df.at[idx, 'home_kalman_uncertainty'] + df.at[idx, 'away_kalman_uncertainty'] + 1e-9)

        # Regime and Factors (anti-leakage: pass only past data)
        df_past = df.iloc[:i]
        h_reg = detect_team_regime(row['home_team'], df_past)
        a_reg = detect_team_regime(row['away_team'], df_past)
        df.at[idx, 'home_regime_coef'] = h_reg['coefficient']
        df.at[idx, 'away_regime_coef'] = a_reg['coefficient']
        df.at[idx, 'regime_factor_diff'] = h_reg['coefficient'] - a_reg['coefficient']

        h_fac = compute_team_factors(df_past, row['home_team'], row['home_glicko'])
        a_fac = compute_team_factors(df_past, row['away_team'], row['away_glicko'])
        fac_diffs = factor_matchup_score(h_fac, a_fac)
        for k, v in fac_diffs.items():
            df.at[idx, k] = v

        # NOW update Kalman
        home_xg_val = row.get('home_xg', 1.2)
        home_xg = float(home_xg_val) if pd.notna(home_xg_val) else 1.2
        away_xg_val = row.get('away_xg', 1.0)
        away_xg = float(away_xg_val) if pd.notna(away_xg_val) else 1.0
        kalman_system.update_match(row['home_team'], row['away_team'], home_xg, away_xg)

    # 3. FEATURE COLUMNS & TARGET
    # KEEP Glicko features, ADD Kalman features:
    FEATURE_COLS_FULL = [
        # Existing Glicko (KEEP)
        'home_glicko', 'home_rd', 'away_glicko', 'away_rd', 'glicko_signal',
        # NEW: Kalman additions
        'home_kalman_velocity', 'away_kalman_velocity',   # key: improving/declining
        'kalman_velocity_diff',                            # = home - away velocity
        'kalman_signal',                                   # = (home_str - away_str) / sqrt(h_unc + a_unc)
        # Regime
        'home_regime_coef', 'away_regime_coef', 'regime_factor_diff',
        # Factor model diffs
        'factor_momentum_diff', 'factor_quality_diff', 'factor_volatility_diff',
        # Institutional signals
        'tournament_momentum_diff', 'glicko_velocity_diff', 'conversion_diff',
        'defensive_shape_diff', 'lineup_continuity_diff',
        # Existing features
        'xg_supremacy', 'draw_affinity', 'home_neutral_venue_form',
        'away_neutral_venue_form', 'rest_differential', 'stage_pressure',
        'injury_differential', 'key_injury_factor', 'press_proxy_diff',
        'h2h_draw_rate',
    ]

    available_cols = [c for c in FEATURE_COLS_FULL if c in df.columns]
    print(f"  Features available: {len(available_cols)}/{len(FEATURE_COLS_FULL)}")
    print(f"  Missing (will be added as 0): {[c for c in FEATURE_COLS_FULL if c not in df.columns]}")

    for c in FEATURE_COLS_FULL:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)
            
    available_cols = FEATURE_COLS_FULL
    print(f"[FEATURES]  Using {len(available_cols)} features: {available_cols}")

    df = df.dropna(subset=available_cols).reset_index(drop=True)
    
    # Optional filter for walk-forward efficiency (keep 2018 onwards)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 2018].reset_index(drop=True)
        
    print(f"[FEATURES]  {len(df)} matches after preprocessing")


    X_full_temp = df[available_cols].copy()
    y_full_temp = df['home_goals']
    
    X = df[available_cols].copy()
    y_home_goals = df['home_goals']
    y_away_goals = df['away_goals']
    y_outcome = np.where(df['home_goals'] > df['away_goals'], 0,
                np.where(df['home_goals'] == df['away_goals'], 1, 2))

    print(f"\n[TARGET]  Class distribution:")
    for cls, name in [(0, 'Home Win'), (1, 'Draw'), (2, 'Away Win')]:
        print(f"          {name}:  {(y_outcome == cls).sum()} ({(y_outcome == cls).mean()*100:.1f}%)")

    # DC probs will be computed per-fold

    # 5. WALK-FORWARD VALIDATION
    N_FOLDS = 5
    EMBARGO = 4
    n = len(X)
    fold_size = n // (N_FOLDS + 1)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION  ({N_FOLDS} folds, embargo={EMBARGO})")
    print(f"  Total samples: {n}, fold_size: {fold_size}")
    print(f"{'='*70}")

    fold_results = []
    all_test_preds = np.zeros((n, 3))
    all_test_discrete_preds = np.zeros(n, dtype=int)
    all_test_mask = np.zeros(n, dtype=bool)

    def rank_probability_score(y_true_onehot, y_proba):
        """
        RPS for ordered outcomes: Away Win < Draw < Home Win.
        Lower is better. Bookmaker baseline ~ 0.195.
        """
        cum_true  = np.cumsum(y_true_onehot,  axis=1)
        cum_proba = np.cumsum(y_proba, axis=1)
        return np.mean(np.sum((cum_proba - cum_true) ** 2, axis=1) / (y_proba.shape[1] - 1))

    from models.base_learners import compute_match_weights

    for fold_idx in range(N_FOLDS):
        train_end = (fold_idx + 1) * fold_size
        test_start = train_end + EMBARGO
        test_end = min((fold_idx + 2) * fold_size, n)
        if test_start >= n:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) == 0:
            break
        
        if len(train_idx) < 200:
            print(f"\n  --- Fold {fold_idx+1} ---  Skipping (train_size={len(train_idx)} < 200)")
            continue

        X_train_df = X.iloc[train_idx]
        X_test_df  = X.iloc[test_idx]
        y_tr_h = y_home_goals.iloc[train_idx].values
        y_tr_a = y_away_goals.iloc[train_idx].values
        y_test_out = y_outcome[test_idx]
        y_train_out = y_outcome[train_idx]
    
        dates_train = df['date'].iloc[train_idx]
        w_train = np.asarray(compute_match_weights(dates_train))

        print(f"\n  --- Fold {fold_idx+1} ---  train={len(train_idx)}, test={len(test_idx)}")
        
        # FIX 1: DC Model fitted on fold train
        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()
        print(f"  [Fold {fold_idx+1}] Fitting DC on {len(df_train)} train matches...")
        dc_model_fold = FastDixonColes()
        dc_model_fold.fit(df_train)
        
        dc_model_dict = {
            'attack': dc_model_fold.attack,
            'defense': dc_model_fold.defense,
            'home_adv': dc_model_fold.home_adv,
            'rho': dc_model_fold.rho
        }

        dc_probs_train = np.array([compute_dc_probs(row, dc_model_dict) for _, row in df_train.iterrows()])
        dc_probs_test  = np.array([compute_dc_probs(row, dc_model_dict) for _, row in df_test.iterrows()])

        # FIX 3: Feature selection inside loop
        from sklearn.inspection import permutation_importance
        X_train_full = X_train_df.values
        if len(available_cols) > 15:
            lr_fast = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            lr_fast.fit(X_train_full, y_train_out)
            result_imp = permutation_importance(
                lr_fast, X_train_full, y_train_out,
                n_repeats=3, random_state=42, scoring='neg_log_loss'
            )
            top_idx = np.argsort(result_imp.importances_mean)[::-1][:15]
            selected_cols_fold = [available_cols[i] for i in sorted(top_idx)]
            print(f"  [Fold {fold_idx+1}] Selected 15 features")
        else:
            selected_cols_fold = available_cols

        X_train = X_train_df[selected_cols_fold].values
        X_test  = X_test_df[selected_cols_fold].values

        # FIX 2: Meta-Learner Inner CV OOF
        from sklearn.model_selection import TimeSeriesSplit
        inner_cv = TimeSeriesSplit(n_splits=3)
        oof_ml = np.zeros((len(X_train), 3))
        
        def build_bls():
            cat_h = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10,
                                       iterations=250, learning_rate=0.04, subsample=0.8,
                                       bootstrap_type='Bernoulli',
                                       random_seed=42, verbose=0, thread_count=1)
            xgb_h = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0,
                                  n_estimators=200, learning_rate=0.04, subsample=0.8,
                                  random_state=42, verbosity=0, n_jobs=1)
            ridge_h = Ridge(alpha=15.0)

            cat_a = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10,
                                       iterations=250, learning_rate=0.04, subsample=0.8,
                                       bootstrap_type='Bernoulli',
                                       random_seed=42, verbose=0, thread_count=1)
            xgb_a = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0,
                                  n_estimators=200, learning_rate=0.04, subsample=0.8,
                                  random_state=42, verbosity=0, n_jobs=1)
            ridge_a = Ridge(alpha=15.0)
            return cat_h, xgb_h, ridge_h, cat_a, xgb_a, ridge_a
            
        for inner_tr_idx, inner_val_idx in inner_cv.split(X_train):
            ch, xh, rh, ca, xa, ra = build_bls()
            ch.fit(X_train[inner_tr_idx], y_tr_h[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            xh.fit(X_train[inner_tr_idx], y_tr_h[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            rh.fit(X_train[inner_tr_idx], y_tr_h[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            
            ca.fit(X_train[inner_tr_idx], y_tr_a[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            xa.fit(X_train[inner_tr_idx], y_tr_a[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            ra.fit(X_train[inner_tr_idx], y_tr_a[inner_tr_idx], sample_weight=w_train[inner_tr_idx])
            
            ph = np.clip((ch.predict(X_train[inner_val_idx]) + xh.predict(X_train[inner_val_idx]) + rh.predict(X_train[inner_val_idx])) / 3.0, 0.3, 4.0)
            pa = np.clip((ca.predict(X_train[inner_val_idx]) + xa.predict(X_train[inner_val_idx]) + ra.predict(X_train[inner_val_idx])) / 3.0, 0.3, 4.0)
            oof_ml[inner_val_idx] = xg_to_probs(ph, pa)
            
        # Refit on all train
        cat_h, xgb_h, ridge_h, cat_a, xgb_a, ridge_a = build_bls()
        cat_h.fit(X_train, y_tr_h, sample_weight=w_train)
        xgb_h.fit(X_train, y_tr_h, sample_weight=w_train)
        ridge_h.fit(X_train, y_tr_h, sample_weight=w_train)
        cat_a.fit(X_train, y_tr_a, sample_weight=w_train)
        xgb_a.fit(X_train, y_tr_a, sample_weight=w_train)
        ridge_a.fit(X_train, y_tr_a, sample_weight=w_train)
        
        pred_h_test = np.clip((cat_h.predict(X_test) + xgb_h.predict(X_test) + ridge_h.predict(X_test)) / 3.0, 0.3, 4.0)
        pred_a_test = np.clip((cat_a.predict(X_test) + xgb_a.predict(X_test) + ridge_a.predict(X_test)) / 3.0, 0.3, 4.0)
        ml_probs_test = xg_to_probs(pred_h_test, pred_a_test)
        ml_probs_train = oof_ml
        
        augmented_test = np.hstack([ml_probs_test, dc_probs_test])
        augmented_train = np.hstack([ml_probs_train, dc_probs_train])

        # Meta-Learner
        from models.meta_learner import fit_meta_learner
        label_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        y_train_str = np.array([label_map[y] for y in y_train_out])
        
        meta_lr = fit_meta_learner(augmented_train, y_train_str)
        final_probs = meta_lr.predict_proba(augmented_test)
        preds = np.argmax(final_probs, axis=1)
        
        acc = accuracy_score(y_test_out, preds)
        ll = log_loss(y_test_out, final_probs, labels=[0, 1, 2])
    
        brier = 0.0
        for cls in range(3):
            y_bin = (y_test_out == cls).astype(int)
            brier += brier_score_loss(y_bin, final_probs[:, cls])
        brier /= 3.0

        ece_vals = []
        for cls in range(3):
            y_bin = (y_test_out == cls).astype(float)
            ece_vals.append(expected_calibration_error(y_bin, final_probs[:, cls], n_bins=10))
        ece = np.mean(ece_vals)

        y_test_onehot = np.eye(3)[y_test_out]
        rps = rank_probability_score(y_test_onehot, final_probs)
    
        all_test_preds[test_idx] = final_probs
        all_test_discrete_preds[test_idx] = preds
        all_test_mask[test_idx] = True

        fold_results.append({
            'fold': fold_idx + 1, 'train_size': len(train_idx), 'test_size': len(test_idx),
            'accuracy': acc, 'log_loss': ll, 'brier': brier, 'ece': ece, 'rps': rps
        })

        print(f"    Accuracy:   {acc*100:.2f}%")
        print(f"    Log-Loss:   {ll:.4f}")
        print(f"    Brier:      {brier:.4f}")
        print(f"    ECE:        {ece:.4f}")
        print(f"    RPS:        {rps:.4f}")

    # =====================================================================
    # 6. AGGREGATE REPORT
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS  (Walk-Forward, {len(fold_results)} folds)")
    print(f"{'='*70}")

    accs = [f['accuracy'] for f in fold_results]
    lls = [f['log_loss'] for f in fold_results]
    briers = [f['brier'] for f in fold_results]
    eces = [f['ece'] for f in fold_results]
    rpss = [f['rps'] for f in fold_results]

    print(f"\n  Accuracy:   {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"  Log-Loss:   {np.mean(lls):.4f} +/- {np.std(lls):.4f}")
    print(f"  Brier:      {np.mean(briers):.4f} +/- {np.std(briers):.4f}")
    print(f"  ECE:        {np.mean(eces):.4f} +/- {np.std(eces):.4f}")
    print(f"  RPS:        {np.mean(rpss):.4f} +/- {np.std(rpss):.4f}")

    print(f"\n  [STABILITY]")
    print(f"    Fold accuracy range:  {min(accs)*100:.1f}% - {max(accs)*100:.1f}%")
    print(f"    Fold spread (max-min): {(max(accs)-min(accs))*100:.1f}pp")
    if np.std(accs) > 0.08:
        print(f"    WARNING: Fold std > 8% - model may be unstable")
    else:
        print(f"    OK: Fold std within acceptable range")
    if np.mean(eces) > 0.08:
        print(f"    WARNING: Mean ECE > 0.08 - Kelly stakes may be overconfident")
    elif np.mean(eces) > 0.05:
        print(f"    CAUTION: Mean ECE > 0.05 - monitor calibration drift")
    else:
        print(f"    OK: Calibration within Kelly-safe threshold (ECE < 0.05)")

    # Pooled classification report
    pooled_y = y_outcome[all_test_mask]
    pooled_preds = all_test_discrete_preds[all_test_mask]

    print(f"\n{'='*70}")
    print(f"  POOLED CLASSIFICATION REPORT  (all test folds combined)")
    print(f"{'='*70}")
    label_names = ['Home Win', 'Draw', 'Away Win']
    print(classification_report(pooled_y, pooled_preds, target_names=label_names, digits=3))

    print("Confusion Matrix:")
    cm = confusion_matrix(pooled_y, pooled_preds, labels=[0, 1, 2])
    print(f"                  Predicted")
    print(f"                  HWin  Draw  AWin")
    for i, name in enumerate(label_names):
        print(f"  Actual {name:>8s}  {cm[i][0]:>4d}  {cm[i][1]:>4d}  {cm[i][2]:>4d}")



    # Per-fold table
    print(f"\n{'='*70}")
    print(f"  PER-FOLD SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Fold':>4s}  {'Train':>5s}  {'Test':>4s}  {'Acc%':>6s}  {'LogLoss':>7s}  {'Brier':>6s}  {'ECE':>6s}  {'RPS':>6s}")
    print(f"  {'---':>4s}  {'---':>5s}  {'---':>4s}  {'---':>6s}  {'---':>7s}  {'---':>6s}  {'---':>6s}  {'---':>6s}")
    for f in fold_results:
        print(f"  {f['fold']:>4d}  {f['train_size']:>5d}  {f['test_size']:>4d}  "
              f"{f['accuracy']*100:>5.1f}%  {f['log_loss']:>7.4f}  {f['brier']:>6.4f}  "
              f"{f['ece']:>6.4f}  {f['rps']:>6.4f}")

    # Baselines
    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISONS")
    print(f"{'='*70}")
    prior = np.bincount(y_outcome, minlength=3) / len(y_outcome)
    random_ll = log_loss(y_outcome, np.tile(prior, (len(y_outcome), 1)), labels=[0, 1, 2])
    random_acc = prior.max()
    print(f"  Class-Prior Baseline:  Acc = {random_acc*100:.1f}%,  Log-Loss = {random_ll:.4f}")

    print(f"  V7.2 Ensemble (WF):    Acc = {np.mean(accs)*100:.1f}%,  Log-Loss = {np.mean(lls):.4f}")

    improvement = (np.mean(accs) - random_acc) / random_acc * 100
    print(f"\n  V7.2 vs Baseline:  {improvement:+.1f}% relative accuracy improvement")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  COMPLETED IN {elapsed:.1f}s")
    print(f"{'='*70}")

    # Populate this from your actual walk-forward output
    real_accuracy = np.mean(accs)    # 43.4%
    real_log_loss = np.mean(lls)     # 1.0836
    
    deployment_metrics = {
        "accuracy": real_accuracy,
        "log_loss": real_log_loss,
        "brier_score": float(np.mean(briers)),
        "ece": float(np.mean(eces)),
        "fold_std": float(np.std(accs)),
        "draw_recall": float(cm[1][1] / max(cm[1].sum(), 1)),
        "n_matches": int(len(form_df)),
    }

    # Evaluate against deployment gate
    ACCURACY_GATE = 0.43
    LOG_LOSS_GATE  = 1.11
    is_promoted = bool(
        deployment_metrics.get("accuracy", 0) >= ACCURACY_GATE and 
        deployment_metrics.get("log_loss", 99) <= LOG_LOSS_GATE
    )

    import joblib
    import json

    # --- Export logic ---
    try:
        available_cols = selected_cols_fold
        dc_model = dc_model_fold

        base_learners = [cat_h, cat_a, xgb_h, xgb_a, ridge_h, ridge_a]
        build_dir, is_promoted = export_model(base_learners, None, meta_lr, available_cols, deployment_metrics)
    
        # Save dc_params to build dir
        dc_params = {
            "attack": dc_model.attack,
            "defense": dc_model.defense,
            "home_adv": dc_model.home_adv,
            "rho": dc_model.rho
        }
        joblib.dump(dc_params, build_dir / "dc_params.joblib")
    
        # Symlink latest promoted model
        latest_dir = MODEL_VERSIONS_DIR / "latest"
        if is_promoted:
            import shutil
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(build_dir, latest_dir)
            print(f"  [OK] Promoted -> model_versions/latest -> {build_dir.name}")
        
            export_team_states(
                df=form_df,                      # df with rolling features already computed
                glicko_ratings=glicko.ratings,   # the raw dict returned by compute_glicko_ratings()
                kalman_system=kalman_system,
                feature_cols=available_cols,     # the ACTUAL available cols, not FEATURE_COLS
                output_path=latest_dir / "team_states.json"
            )

            import os
            import sys
            from auto_deploy import deploy
            if "--auto-deploy=false" not in sys.argv:
                deploy(
                    metrics=deployment_metrics,
                    github_url="https://github.com/saiswaroop496-debug/github_url-https-github.com-YOUR_USERNAME-YOUR_REPO_NAME.git-",
                    api_key=os.getenv("RAPIDAPI_KEY", ""),
                version_tag="V7.2",
                )
            else:
                print("Skipping deployment due to --auto-deploy=false flag.")
        else:
            print(f"  [FAIL]  Model archived but not promoted (gate failed): {build_dir.name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Model export failed: {e}")

# --- Function Definitions ---
import hashlib
from pathlib import Path

MODEL_VERSIONS_DIR = Path("model_versions")
MODEL_VERSIONS_DIR.mkdir(exist_ok=True)

def export_model(model, scaler, meta_learner, feature_cols: list,
                 metrics: dict, version: str = "7.2"):
    """
    Export model artifacts with version manifest.
    Automatically increments build number via hash.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    build_hash = hashlib.md5(ts.encode()).hexdigest()[:8]
    build_name = f"v{version}_{ts}_{build_hash}"
    build_dir  = MODEL_VERSIONS_DIR / build_name
    build_dir.mkdir()

    cat_h, cat_a, xgb_h, xgb_a, ridge_h, ridge_a = model
    xgb_h.save_model(build_dir / "xgb_h.json")
    xgb_a.save_model(build_dir / "xgb_a.json")
    joblib.dump([cat_h, cat_a, None, None, ridge_h, ridge_a], build_dir / "base_learners.joblib")
    if scaler is not None:
        joblib.dump(scaler,       build_dir / "scaler.joblib")
    joblib.dump(meta_learner, build_dir / "meta_learner.joblib")

    # Save manifest
    manifest = {
        "version":       version,
        "build":         build_name,
        "timestamp":     ts,
        "feature_cols":  feature_cols,
        "metrics": {
            "accuracy":    metrics.get("accuracy"),
            "log_loss":    metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "ece":         metrics.get("ece"),
            "draw_recall": metrics.get("draw_recall"),
            "fold_std":    metrics.get("fold_std"),
        },
        "promoted":     bool(metrics.get("accuracy", 0) >= 0.43 and
                             metrics.get("log_loss", 99) <= 1.11)
    }

    with open(build_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    return build_dir, manifest["promoted"]

import json

def export_team_states(df: pd.DataFrame, glicko_ratings: dict, kalman_system,
                       feature_cols: list, output_path: str = "model_versions/latest/team_states.json"):
    import json
    from pathlib import Path

    team_states = {}
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    for team in all_teams:
        g = glicko_ratings.get(team, {})
        k_state = kalman_system.get_or_init(team)
        regime  = detect_team_regime(team, df)

        # Latest xG rolling from last match
        last_matches = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].tail(1)
        if len(last_matches) > 0:
            row = last_matches.iloc[0]
            if row['home_team'] == team:
                xg_roll3 = float(row.get('home_xg_rolling_3', 1.2) or 1.2)
                nv_form  = float(row.get('home_neutral_venue_form', 0.5) or 0.5)
                tm = float(row.get('home_tournament_momentum', 0.0) or 0.0)
                gv = float(row.get('home_glicko_velocity', 0.0) or 0.0)
                conv = float(row.get('home_xg_conversion', 1.0) or 1.0)
                df_shape = float(row.get('home_defensive_shape', 1.0) or 1.0)
                lc = float(row.get('home_lineup_continuity', 1.0) or 1.0)
            else:
                xg_roll3 = float(row.get('away_xg_rolling_3', 1.0) or 1.0)
                nv_form  = float(row.get('away_neutral_venue_form', 0.5) or 0.5)
                tm = float(row.get('away_tournament_momentum', 0.0) or 0.0)
                gv = float(row.get('away_glicko_velocity', 0.0) or 0.0)
                conv = float(row.get('away_xg_conversion', 1.0) or 1.0)
                df_shape = float(row.get('away_defensive_shape', 1.0) or 1.0)
                lc = float(row.get('away_lineup_continuity', 1.0) or 1.0)
        else:
            xg_roll3 = 1.2; nv_form = 0.5
            tm = 0.0; gv = 0.0; conv = 1.0; df_shape = 1.0; lc = 1.0

        team_states[team] = {
            # Glicko (kept for stability)
            "glicko":             round(float(g.get('rating', 1500)), 2),
            "rd":                 round(float(g.get('rd', 200)), 2),
            # Kalman (new)
            "kalman_strength":    round(k_state.strength, 2),
            "kalman_velocity":    round(k_state.velocity, 4),
            "kalman_uncertainty": round(float(k_state.P[0, 0]), 2),
            # Regime
            "regime":             regime['regime'],
            "regime_coef":        regime['coefficient'],
            # Rolling features
            "xg_rolling_3":       round(xg_roll3, 3),
            "neutral_venue_form": round(nv_form, 3),
            # Alpha Signals
            "tournament_momentum": round(tm, 3),
            "glicko_velocity":     round(gv, 4),
            "xg_conversion":       round(conv, 3),
            "defensive_shape":     round(df_shape, 3),
            "lineup_continuity":   round(lc, 3),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(team_states, f, indent=2, sort_keys=True)
    print(f"  ✅ team_states.json: {len(team_states)} teams (Glicko + Kalman + Regime)")
    
    return team_states

# Export the trained model and meta learner

if __name__ == '__main__':
    main()
