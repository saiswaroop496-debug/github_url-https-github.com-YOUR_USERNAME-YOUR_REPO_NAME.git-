# tune_optuna.py
import json
import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_OUT  = Path("model_versions/optuna_best_params.json")
FEATURES_IN = Path("data/feature_vectors.joblib")
RANDOM_SEED = 42

def load_feature_vectors():
    if FEATURES_IN.exists():
        bundle = joblib.load(FEATURES_IN)
        return bundle['X'], bundle['y'], bundle['feature_cols']
    print("  feature_vectors.joblib not found — rebuilding from train_test.py pipeline...")
    import train_test
    bundle = joblib.load(FEATURES_IN)
    return bundle['X'], bundle['y'], bundle['feature_cols']

def cv_log_loss(model, X, y, n_splits=5):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    losses = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        losses.append(log_loss(y[val_idx], proba))
    return float(np.mean(losses))

def xgb_objective(trial, X, y):
    params = {
        'max_depth':          trial.suggest_int('max_depth', 2, 7),
        'learning_rate':      trial.suggest_float('learning_rate', 0.005, 0.20, log=True),
        'n_estimators':       trial.suggest_int('n_estimators', 100, 800),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight':   trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'gamma':              trial.suggest_float('gamma', 0.0, 5.0),
        'max_delta_step':     trial.suggest_int('max_delta_step', 0, 5),
    }
    model = XGBClassifier(
        **params,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    return cv_log_loss(model, X, y)

def catboost_objective(trial, X, y):
    params = {
        'depth':              trial.suggest_int('depth', 3, 8),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'iterations':         trial.suggest_int('iterations', 100, 600),
        'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count':       trial.suggest_int('border_count', 32, 255),
        'random_strength':    trial.suggest_float('random_strength', 1e-3, 5.0, log=True),
    }
    model = CatBoostClassifier(
        **params,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=RANDOM_SEED,
        verbose=0,
    )
    return cv_log_loss(model, X, y)

def draw_gate_objective(trial, X, y_binary):
    params = {
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 50, 400),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
    }
    model = XGBClassifier(
        **params,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    losses = []
    for train_idx, val_idx in skf.split(X, y_binary):
        model.fit(X[train_idx], y_binary[train_idx])
        proba = model.predict_proba(X[val_idx])[:, 1]
        losses.append(log_loss(y_binary[val_idx], proba))
    return float(np.mean(losses))

def run_sweep(n_trials: int = 100):
    print("Loading feature vectors...")
    X, y, feature_cols = load_feature_vectors()
    y_binary_draw = (y == 1).astype(int)

    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (0=Home, 1=Draw, 2=Away)")

    results = {}

    print(f"\n[1/3] XGBoost main classifier — {n_trials} trials...")
    study_xgb = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study_xgb.optimize(lambda t: xgb_objective(t, X, y), n_trials=n_trials)
    results['xgboost_main'] = {
        'best_log_loss': round(study_xgb.best_value, 5),
        'params': study_xgb.best_params,
    }
    print(f"  Best log-loss: {study_xgb.best_value:.5f}")
    print(f"  Best params: {study_xgb.best_params}")

    print(f"\n[2/3] CatBoost base learner — {n_trials} trials...")
    study_cat = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study_cat.optimize(lambda t: catboost_objective(t, X, y), n_trials=n_trials)
    results['catboost_base'] = {
        'best_log_loss': round(study_cat.best_value, 5),
        'params': study_cat.best_params,
    }
    print(f"  Best log-loss: {study_cat.best_value:.5f}")

    print(f"\n[3/3] XGBoost draw gate (binary) — {n_trials} trials...")
    study_draw = optuna.create_study(direction='minimize',
                                      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study_draw.optimize(lambda t: draw_gate_objective(t, X, y_binary_draw),
                         n_trials=n_trials)
    results['draw_gate'] = {
        'best_log_loss': round(study_draw.best_value, 5),
        'params': study_draw.best_params,
    }
    print(f"  Best log-loss: {study_draw.best_value:.5f}")

    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Optuna results saved to {PARAMS_OUT}")
    print_summary(results)

def print_summary(results: dict):
    print("\n" + "="*55)
    print("  OPTUNA SWEEP SUMMARY")
    print("="*55)
    for model_name, r in results.items():
        print(f"  {model_name:20}: log-loss = {r['best_log_loss']:.5f}")
    print("\n  Next step: run  python train_test.py --use-optuna-params")
    print("="*55)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    run_sweep(n_trials=args.trials)
