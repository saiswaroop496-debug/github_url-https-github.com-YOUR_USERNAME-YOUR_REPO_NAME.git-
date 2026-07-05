# models/meta_learner.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from models.temperature_scaler import TemperatureScaler
import warnings

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    import os, sys, site, importlib
    print("imbalanced-learn not found in meta_learner. Attempting dynamic install...")
    os.system("pip install imbalanced-learn==0.11.0")
    importlib.invalidate_caches()
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    try:
        from imblearn.over_sampling import SMOTE
        SMOTE_AVAILABLE = True
    except ImportError:
        SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ─── SafeSMOTE (module scope for pickling) ────────────────────────────────────
class SafeSMOTE:
    def fit_resample(self, X, y):
        if not SMOTE_AVAILABLE:
            return X, y
        counts = {c: (y == c).sum() for c in np.unique(y)}
        min_count = min(counts.values())
        if min_count < 4:
            return X, y
        k = min(3, min_count - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        try:
            return smote.fit_resample(X, y)
        except Exception:
            return X, y


# ─── TWO-HEAD META-LEARNER ────────────────────────────────────────────────────
class TwoHeadMetaLearner:
    """
    Head 1 (draw_gate):      P(Draw) — XGBoost (nonlinear draw signals)
    Head 2 (direction_gate): P(Home Win | not Draw) — Logistic Regression

    Calibration pipeline:
        raw outputs → Isotonic calibration per market → Temperature Scaling
    """
    def __init__(self, C_direction=0.5, draw_gate_params=None):
        if XGB_AVAILABLE:
            if draw_gate_params:
                self.draw_gate = XGBClassifier(
                    **draw_gate_params,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                self.draw_gate = XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    scale_pos_weight=3.5,
                    eval_metric='logloss',
                    verbosity=0,
                    random_state=42
                )
        else:
            self.draw_gate = LogisticRegression(
                C=0.3, class_weight={0: 1.0, 1: 3.5},
                solver='lbfgs', max_iter=1000
            )

        self.direction_gate = LogisticRegression(
            C=C_direction,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.calibrators      = {}       # per-market isotonic regressors
        self.temp_scaler      = TemperatureScaler()
        self.classes_         = ['Home Win', 'Draw', 'Away Win']
        self.is_fitted        = False
        self.temp_fitted      = False

    # ── TRAINING ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit both heads on SMOTE-balanced OOF training split."""
        smote = SafeSMOTE()
        X_bal, y_bal = smote.fit_resample(X, y)

        # Head 1: draw gate
        y_draw = (y_bal == 'Draw').astype(int)
        self.draw_gate.fit(X_bal, y_draw)

        # Head 2: direction gate (non-draw samples only)
        non_draw = y_bal != 'Draw'
        if non_draw.sum() > 10:
            y_dir = (y_bal[non_draw] == 'Home Win').astype(int)
            self.direction_gate.fit(X_bal[non_draw], y_dir)

        self.is_fitted = True
        return self

    def fit_calibration(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Step 1: Fit per-market Isotonic Regression calibrators.
        Step 2: Fit TemperatureScaler on logits from uncalibrated probabilities.
        Uses the same X_cal slice — chronologically AFTER training data.
        """
        # Get raw (uncalibrated) probabilities
        raw_proba = self._raw_predict_proba(X_cal)

        # Isotonic calibration per market
        for i, market in enumerate(self.classes_):
            iso = IsotonicRegression(out_of_bounds='clip',
                                      y_min=0.03, y_max=0.94)
            y_bin = (y_cal == market).astype(float)
            iso.fit(raw_proba[:, i], y_bin)
            self.calibrators[market] = iso

        # Temperature scaling — convert calibrated proba to pseudo-logits
        iso_proba = self._apply_isotonic(raw_proba)
        # Convert probabilities back to logits for temperature fitting
        # log(p) with small epsilon guard
        logits = np.log(np.clip(iso_proba, 1e-6, 1 - 1e-6))
        # Normalize to zero-mean (proper logit form)
        logits = logits - logits.mean(axis=1, keepdims=True)

        self.temp_scaler.fit(logits, y_cal, self.classes_)
        self.temp_fitted = True
        return self

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Raw probabilities from both gates — no calibration applied."""
        # Backward compatibility patch for loading sklearn 1.8.0 pickles in sklearn 1.5.2
        if not hasattr(self.draw_gate, "multi_class"):
            self.draw_gate.multi_class = "auto"
        if not hasattr(self.direction_gate, "multi_class"):
            self.direction_gate.multi_class = "auto"

        p_draw      = self.draw_gate.predict_proba(X)[:, 1]
        p_hw_cond   = self.direction_gate.predict_proba(X)[:, 1]
        p_home      = (1 - p_draw) * p_hw_cond
        p_away      = (1 - p_draw) * (1 - p_hw_cond)
        proba       = np.column_stack([p_home, p_draw, p_away])
        return np.clip(proba, 0.02, 0.96)

    def _apply_isotonic(self, proba: np.ndarray) -> np.ndarray:
        """Apply per-market isotonic regression."""
        if not self.calibrators:
            return proba
        cal = proba.copy()
        for i, market in enumerate(self.classes_):
            if market in self.calibrators:
                cal[:, i] = self.calibrators[market].predict(proba[:, i])
        # Renormalize
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.maximum(row_sums, 1e-9)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Full calibrated inference pipeline:
        raw outputs → isotonic calibration → temperature scaling → normalize
        """
        raw_proba  = self._raw_predict_proba(X)
        iso_proba  = self._apply_isotonic(raw_proba)

        if self.temp_fitted:
            # Convert to logits for temperature scaling
            logits = np.log(np.clip(iso_proba, 1e-6, 1 - 1e-6))
            logits = logits - logits.mean(axis=1, keepdims=True)
            final  = self.temp_scaler.transform(logits)
        else:
            final = iso_proba

        # Final validity enforcement
        final = np.clip(final, 0.02, 0.96)
        final = final / final.sum(axis=1, keepdims=True)
        return final

    def compute_ece(self, X: np.ndarray, y: np.ndarray,
                     n_bins: int = 10) -> float:
        proba = self.predict_proba(X)
        ece   = 0.0
        for i, market in enumerate(self.classes_):
            y_bin = (y == market).astype(float)
            p     = proba[:, i]
            bins  = np.linspace(0, 1, n_bins + 1)
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() == 0:
                    continue
                acc  = y_bin[mask].mean()
                conf = p[mask].mean()
                ece  += (mask.sum() / len(y)) * abs(acc - conf)
        return ece / 3


# ─── FIT PIPELINE ─────────────────────────────────────────────────────────────
def fit_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray, draw_gate_params=None) -> TwoHeadMetaLearner:
    """
    Fit meta-learner with tighter calibration split (75/25 instead of 85/15).
    oof_preds: (N, 6) — 3 ML base-learner probs + 3 DC probs concatenated.
    """
    n = len(y_oof)
    cal_split = int(n * 0.75)   # ← 25% for calibration (was 15%)

    model = TwoHeadMetaLearner(draw_gate_params=draw_gate_params)
    model.fit(oof_preds[:cal_split], y_oof[:cal_split])
    model.fit_calibration(oof_preds[cal_split:], y_oof[cal_split:])

    # Report ECE on calibration set
    ece = model.compute_ece(oof_preds[cal_split:], y_oof[cal_split:])
    ll  = log_loss(y_oof[cal_split:],
                   model.predict_proba(oof_preds[cal_split:]),
                   labels=model.classes_)
    print(f"  Meta-Learner | ECE={ece:.4f} | Log-Loss={ll:.4f} "
          f"(target ECE<0.08, LL<1.08)")
    return model


def predict_with_draw_threshold(model: TwoHeadMetaLearner,
                                  X: np.ndarray,
                                  draw_thresh: float = 0.30) -> tuple:
    """
    Lower threshold (0.30) because temperature scaling correctly spreads
    draw probabilities — they no longer need to be 0.38+ to fire.
    """
    proba    = model.predict_proba(X)
    draw_idx = model.classes_.index('Draw')
    preds    = []
    for p in proba:
        if p[draw_idx] >= draw_thresh:
            preds.append('Draw')
        else:
            non_draw = [(i, v) for i, v in enumerate(p) if i != draw_idx]
            preds.append(model.classes_[max(non_draw, key=lambda x: x[1])[0]])
    return np.array(preds), proba
