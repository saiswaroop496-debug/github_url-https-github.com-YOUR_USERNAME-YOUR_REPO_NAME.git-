"""
V5.1 FIFA World Cup Quantitative Engine — Full Train & Test Pipeline
=====================================================================
OPTIMIZED VERSION: Vectorized Dixon-Coles, fast walk-forward.
"""

import sys, os, time, warnings
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
from models.meta_learner import expected_calibration_error

# =====================================================================
# VECTORIZED DIXON-COLES (replaces row-by-row for speed)
# =====================================================================
class FastDixonColes:
    """Dixon-Coles with fully vectorized log-likelihood for speed."""
    
    def __init__(self):
        self.teams = []
        self.attack = {}
        self.defense = {}
        self.home_gamma = 0.3
        self.neutral_gamma = 0.0
        self.rho = 0.0

    def _tau_vec(self, x, y, lam, mu, rho):
        """Vectorized tau correction."""
        tau = np.ones(len(x))
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        return tau

    def _neg_ll(self, params, home_idx, away_idx, x, y, weights, time_weights):
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        hg = params[2*n_teams]
        ng = params[2*n_teams + 1]
        rho = params[2*n_teams + 2]

        # All matches treated as neutral (crowd_factor=0) for speed
        lam = np.exp(attack[home_idx] + defense[away_idx] + ng)
        mu = np.exp(attack[away_idx] + defense[home_idx])

        # Clamp to prevent overflow
        lam = np.clip(lam, 0.01, 15.0)
        mu = np.clip(mu, 0.01, 15.0)

        tau = self._tau_vec(x, y, lam, mu, rho)
        tau = np.clip(tau, 1e-10, None)

        ll = time_weights * weights * (
            np.log(tau) + 
            poisson.logpmf(x, lam) + 
            poisson.logpmf(y, mu)
        )
        return -np.sum(ll)

    def fit(self, df):
        t0 = time.time()
        self.teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        home_idx = df['home_team'].map(team_to_idx).values
        away_idx = df['away_team'].map(team_to_idx).values
        x = df['home_goals'].values.astype(float)
        y = df['away_goals'].values.astype(float)
        weights = df['match_weight'].values if 'match_weight' in df.columns else np.ones(len(df))
        
        max_date = df['date'].max()
        delta_days = (max_date - df['date']).dt.days.values.astype(float)
        time_weights = np.exp(-0.0065 * delta_days)

        x0 = np.zeros(2*n_teams + 3)
        x0[2*n_teams] = 0.3
        x0[2*n_teams + 1] = 0.0
        x0[2*n_teams + 2] = 0.0

        bounds = [(None, None)] * (2*n_teams + 2) + [(-0.35, 0.35)]

        res = minimize(self._neg_ll, x0,
                       args=(home_idx, away_idx, x, y, weights, time_weights),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-6})

        opt = res.x
        self.attack = {t: opt[i] for i, t in enumerate(self.teams)}
        self.defense = {t: opt[n_teams + i] for i, t in enumerate(self.teams)}
        self.home_gamma = opt[2*n_teams]
        self.neutral_gamma = opt[2*n_teams + 1]
        self.rho = opt[2*n_teams + 2]
        print(f"[DC MODEL]  Fitted in {time.time()-t0:.1f}s  (rho={self.rho:.4f}, "
              f"home_gamma={self.home_gamma:.4f}, neutral_gamma={self.neutral_gamma:.4f})")

    def predict_proba(self, team1, team2, venue_factor=0.0):
        if team1 not in self.attack or team2 not in self.attack:
            return np.array([0.33, 0.34, 0.33])
        
        g_n = max(self.neutral_gamma, 1e-6)
        g_f = max(self.home_gamma, 1e-6)
        v = np.clip(venue_factor, 0.0, 1.0)
        if abs(g_n - g_f) < 1e-8:
            gamma_eff = g_n
        else:
            gamma_eff = g_n * (g_f / g_n) ** v
        gamma_eff = np.clip(gamma_eff, -2.0, 2.5)
        
        lam = exp(self.attack[team1] + self.defense[team2] + gamma_eff)
        mu = exp(self.attack[team2] + self.defense[team1])
        lam = min(max(lam, 0.01), 15.0)
        mu = min(max(mu, 0.01), 15.0)

        p_h = p_d = p_a = 0.0
        for gx in range(8):
            for gy in range(8):
                tau = 1.0
                if gx == 0 and gy == 0: tau = 1 - lam * mu * self.rho
                elif gx == 0 and gy == 1: tau = 1 + lam * self.rho
                elif gx == 1 and gy == 0: tau = 1 + mu * self.rho
                elif gx == 1 and gy == 1: tau = 1 - self.rho
                p = max(tau, 0) * poisson.pmf(gx, lam) * poisson.pmf(gy, mu)
                if gx > gy: p_h += p
                elif gx == gy: p_d += p
                else: p_a += p
        total = p_h + p_d + p_a
        if total < 1e-10:
            return np.array([0.33, 0.34, 0.33])
        return np.array([p_h/total, p_d/total, p_a/total])

    def predict_proba_batch(self, teams1, teams2, venue_factors=None):
        """Batch prediction for speed."""
        n = len(teams1)
        if venue_factors is None:
            venue_factors = np.zeros(n)
        probs = np.zeros((n, 3))
        for i in range(n):
            probs[i] = self.predict_proba(teams1[i], teams2[i], venue_factors[i])
        return probs


def xg_to_probs(pred_h, pred_a):
    """Convert xG predictions to match outcome probabilities via Poisson grid."""
    n = len(pred_h)
    probs = np.zeros((n, 3))
    for i in range(n):
        lam, mu = max(pred_h[i], 0.05), max(pred_a[i], 0.05)
        p_h = p_d = p_a = 0.0
        for gx in range(8):
            for gy in range(8):
                p = poisson.pmf(gx, lam) * poisson.pmf(gy, mu)
                if gx > gy: p_h += p
                elif gx == gy: p_d += p
                else: p_a += p
        total = p_h + p_d + p_a
        probs[i] = [p_h/total, p_d/total, p_a/total] if total > 0 else [0.33, 0.34, 0.33]
    return probs


# =====================================================================
# MAIN PIPELINE
# =====================================================================
print("=" * 70)
print("  V5.1 QUANTITATIVE ENGINE — TRAIN & TEST REPORT")
print("=" * 70)

t_start = time.time()

# 1. DATA
scraper = DataScraper()
dc_df, form_df = scraper.fetch_fixtures()
print(f"\n[DATA]  dc_df (2015+): {len(dc_df)} matches")
print(f"[DATA]  form_df (2018+): {len(form_df)} matches")

# 2. FEATURES
print("\n[FEATURES]  Computing rolling features …")
df = compute_rolling_features(form_df)
print("[FEATURES]  Computing Glicko-2 ratings …")
glicko = Glicko2RatingSystem()
df = glicko.compute_ratings(df)

print("[FEATURES]  Computing V6 specific features …")
from features.rolling_features import compute_v6_features
df = compute_v6_features(df)

# 3. FEATURE COLUMNS & TARGET
FEATURE_COLS = [
    'home_glicko', 'home_rd',
    'away_glicko', 'away_rd',
    'xg_supremacy',          # replaces home/away_xg_rolling_3
    'glicko_signal',         # replaces raw glicko + rd pair
    'draw_affinity',         # NEW — draw signal
    'team1_neutral_xg',      # Keeping neutral xg references if needed
    'team2_neutral_xg',
    'rest_differential',
]
available_cols = [c for c in FEATURE_COLS if c in df.columns]
print(f"[FEATURES]  Using {len(available_cols)} features: {available_cols}")

df = df.dropna(subset=available_cols).reset_index(drop=True)
print(f"[FEATURES]  {len(df)} matches after NaN drop")

X = df[available_cols].copy()
y_home_goals = df['home_goals']
y_away_goals = df['away_goals']
y_outcome = np.where(df['home_goals'] > df['away_goals'], 0,
            np.where(df['home_goals'] == df['away_goals'], 1, 2))

print(f"\n[TARGET]  Class distribution:")
for cls, name in [(0, 'Home Win'), (1, 'Draw'), (2, 'Away Win')]:
    print(f"          {name}:  {(y_outcome == cls).sum()} ({(y_outcome == cls).mean()*100:.1f}%)")

# 4. DIXON-COLES (vectorized)
print(f"\n[DC MODEL]  Fitting vectorized Dixon-Coles on {len(dc_df)} matches …")
dc_model = FastDixonColes()
dc_model.fit(dc_df)

# DC probs for the form_df matches
dc_probs = dc_model.predict_proba_batch(
    df['home_team'].values, df['away_team'].values,
    df['crowd_factor'].values if 'crowd_factor' in df.columns else None
)

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
all_test_mask = np.zeros(n, dtype=bool)

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

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_tr_h = y_home_goals.iloc[train_idx]
    y_tr_a = y_away_goals.iloc[train_idx]
    y_test_out = y_outcome[test_idx]
    y_train_out = y_outcome[train_idx]

    print(f"\n  --- Fold {fold_idx+1} ---  train={len(train_idx)}, test={len(test_idx)}")

    # Base Learners
    cat_h = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10,
                               iterations=250, learning_rate=0.04, subsample=0.8,
                               random_seed=42, verbose=0)
    xgb_h = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0,
                          n_estimators=200, learning_rate=0.04, subsample=0.8,
                          random_state=42, verbosity=0)
    ridge_h = Ridge(alpha=15.0)

    cat_a = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10,
                               iterations=250, learning_rate=0.04, subsample=0.8,
                               random_seed=42, verbose=0)
    xgb_a = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0,
                          n_estimators=200, learning_rate=0.04, subsample=0.8,
                          random_state=42, verbosity=0)
    ridge_a = Ridge(alpha=15.0)

    cat_h.fit(X_train, y_tr_h); xgb_h.fit(X_train, y_tr_h); ridge_h.fit(X_train, y_tr_h)
    cat_a.fit(X_train, y_tr_a); xgb_a.fit(X_train, y_tr_a); ridge_a.fit(X_train, y_tr_a)

    # Test predictions
    pred_h_test = np.clip((cat_h.predict(X_test) + xgb_h.predict(X_test) + ridge_h.predict(X_test)) / 3.0, 0.3, 4.0)
    pred_a_test = np.clip((cat_a.predict(X_test) + xgb_a.predict(X_test) + ridge_a.predict(X_test)) / 3.0, 0.3, 4.0)
    ml_probs_test = xg_to_probs(pred_h_test, pred_a_test)

    # Train predictions (for meta-learner)
    pred_h_tr = np.clip((cat_h.predict(X_train) + xgb_h.predict(X_train) + ridge_h.predict(X_train)) / 3.0, 0.3, 4.0)
    pred_a_tr = np.clip((cat_a.predict(X_train) + xgb_a.predict(X_train) + ridge_a.predict(X_train)) / 3.0, 0.3, 4.0)
    ml_probs_train = xg_to_probs(pred_h_tr, pred_a_tr)

    # Blend with DC (50/50)
    blended_test = 0.5 * ml_probs_test + 0.5 * dc_probs[test_idx]
    blended_test /= blended_test.sum(axis=1, keepdims=True)

    blended_train = 0.5 * ml_probs_train + 0.5 * dc_probs[train_idx]
    blended_train /= blended_train.sum(axis=1, keepdims=True)

    # Meta-Learner with chronological calibration split
    cal_split = int(len(train_idx) * 0.75)
    meta_lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, solver='lbfgs')
    meta_lr.fit(blended_train[:cal_split], y_train_out[:cal_split])

    final_probs = meta_lr.predict_proba(blended_test)
    final_probs = np.clip(final_probs, 0.05, 0.95)
    final_probs /= final_probs.sum(axis=1, keepdims=True)

    # Metrics
    preds = final_probs.argmax(axis=1)
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

    rps_vals = []
    for i in range(len(y_test_out)):
        actual_cum = np.cumsum(np.eye(3)[y_test_out[i]])
        pred_cum = np.cumsum(final_probs[i])
        rps_vals.append(np.mean((actual_cum - pred_cum) ** 2))
    rps = np.mean(rps_vals)

    all_test_preds[test_idx] = final_probs
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
pooled_preds = all_test_preds[all_test_mask].argmax(axis=1)

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

# DC standalone
print(f"\n{'='*70}")
print(f"  DIXON-COLES STANDALONE")
print(f"{'='*70}")
dc_preds_class = dc_probs.argmax(axis=1)
dc_acc = accuracy_score(y_outcome, dc_preds_class)
dc_ll = log_loss(y_outcome, dc_probs, labels=[0, 1, 2])
print(f"  Accuracy:   {dc_acc*100:.2f}%")
print(f"  Log-Loss:   {dc_ll:.4f}")

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
print(f"  Dixon-Coles:           Acc = {dc_acc*100:.1f}%,  Log-Loss = {dc_ll:.4f}")
print(f"  V5.1 Ensemble (WF):    Acc = {np.mean(accs)*100:.1f}%,  Log-Loss = {np.mean(lls):.4f}")

improvement = (np.mean(accs) - random_acc) / random_acc * 100
print(f"\n  V5.1 vs Baseline:  {improvement:+.1f}% relative accuracy improvement")

elapsed = time.time() - t_start
print(f"\n{'='*70}")
print(f"  COMPLETED IN {elapsed:.1f}s")
print(f"{'='*70}")
