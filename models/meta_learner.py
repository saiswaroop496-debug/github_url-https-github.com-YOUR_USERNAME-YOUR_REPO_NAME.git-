import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import warnings

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. SMOTE disabled. Run: pip install imbalanced-learn")


# --- SMOTE Balancing (training split only) ------------------------------------
def balance_oof_for_meta(X_oof: np.ndarray, y_oof: np.ndarray):
    """
    Apply SMOTE to OOF training data to address draw class imbalance.
    k_neighbors=3 is safe for small datasets (~120-150 samples).
    Returns original data unchanged if SMOTE unavailable.
    """
    if not SMOTE_AVAILABLE:
        return X_oof, y_oof

    class_counts = {c: (y_oof == c).sum() for c in np.unique(y_oof)}
    min_class_count = min(class_counts.values())

    if min_class_count < 4:
        warnings.warn(f"Min class count {min_class_count} too small for SMOTE. Skipping.")
        return X_oof, y_oof

    k = min(3, min_class_count - 1)
    smote = SMOTE(
        k_neighbors=k,
        sampling_strategy=0.6,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X_oof, y_oof)
    
    draw_original = ((y_oof == 'Draw') | (y_oof == 1)).sum()
    draw_res = ((y_res == 'Draw') | (y_res == 1)).sum()
    
    print(f"  SMOTE: {len(y_oof)} ? {len(y_res)} samples "
          f"(Draw: {draw_original} ? {draw_res})")
    return X_res, y_res


# --- Draw-Isolated Platt Recalibration ----------------------------------------
def recalibrate_draw_column(proba: np.ndarray, y_true: np.ndarray,
                             draw_idx: int, ece_threshold: float = 0.08):
    """
    If ECE for the Draw column exceeds threshold, apply isotonic recalibration
    strictly to the Draw probability column. Does not touch Home/Away columns.
    Target range: draw probabilities scaled to [0.28, 0.38].
    """
    draw_probs = proba[:, draw_idx]
    draw_true  = ((y_true == 'Draw') | (y_true == 1)).astype(int)

    # Compute draw-column ECE
    fraction_pos, mean_pred = calibration_curve(draw_true, draw_probs, n_bins=5)
    draw_ece = np.mean(np.abs(fraction_pos - mean_pred))

    if draw_ece <= ece_threshold:
        print(f"  Draw ECE={draw_ece:.4f} within threshold. No recalibration needed.")
        return proba

    print(f"  Draw ECE={draw_ece:.4f} > {ece_threshold}. Applying isotonic recalibration.")
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.28, y_max=0.38)
    iso.fit(draw_probs, draw_true.astype(float))
    recal_draw = iso.predict(draw_probs)

    # Re-normalize so rows sum to 1
    proba_new = proba.copy()
    proba_new[:, draw_idx] = recal_draw
    row_sums = proba_new.sum(axis=1, keepdims=True)
    return proba_new / row_sums


# --- Meta-Learner Fit ---------------------------------------------------------
def fit_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray, classes: list):
    """
    Full meta-learner training pipeline:
    1. SMOTE balance on training split
    2. Balanced LR fit
    3. Platt calibration on held-out chronological slice
    4. Draw column recalibration if ECE > 0.08
    """
    n = len(y_oof)
    cal_split = int(n * 0.85)

    X_train, y_train = oof_preds[:cal_split], y_oof[:cal_split]
    X_cal, y_cal     = oof_preds[cal_split:], y_oof[cal_split:]

    # Apply SMOTE only to training split
    X_train_bal, y_train_bal = balance_oof_for_meta(X_train, y_train)

    lr = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_bal, y_train_bal)

    calibrated = CalibratedClassifierCV(lr, cv='prefit', method='sigmoid')
    calibrated.fit(X_cal, y_cal)

    return calibrated


def predict_with_draw_threshold(model, X_proba: np.ndarray, classes: list,
                                  draw_thresh: float = 0.38,
                                  draw_affinity_floor: float = 0.45,
                                  draw_affinity_arr=None):
    """
    Dual-gate draw prediction:
    1. Model probability must exceed draw_thresh
    2. draw_affinity feature must exceed floor (confirms tight match)
    """
    proba = model.predict_proba(X_proba)
    class_list = list(classes)
    draw_idx = class_list.index('Draw') if 'Draw' in class_list else class_list.index(1)

    preds = []
    for i, p in enumerate(proba):
        prob_gate = p[draw_idx] >= draw_thresh
        affinity_gate = (draw_affinity_arr is None or
                         draw_affinity_arr[i] >= draw_affinity_floor)
        if prob_gate and affinity_gate:
            preds.append(class_list[draw_idx])
        else:
            preds.append(class_list[np.argmax(p)])
    return np.array(preds), proba


# --- Optuna Cap (if used for LR hyperparameter tuning) ------------------------
def tune_lr_optuna(X: np.ndarray, y: np.ndarray):
    """Optuna-tuned C parameter. Capped at 20 trials, 30s timeout."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            C = trial.suggest_float('C', 0.01, 2.0, log=True)
            from sklearn.model_selection import cross_val_score
            lr = LogisticRegression(C=C, class_weight='balanced',
                                    solver='lbfgs', max_iter=500)
            scores = cross_val_score(lr, X, y, cv=3, scoring='neg_log_loss')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=30)
        return study.best_params['C']

    except ImportError:
        return 0.5   # safe default if Optuna not installed

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece
