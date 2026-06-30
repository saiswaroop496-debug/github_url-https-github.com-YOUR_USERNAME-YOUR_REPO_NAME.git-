import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
import warnings

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ─── SafeSMOTE (at module scope for pickling) ─────────────────────────────────
class SafeSMOTE:
    """Picklable SMOTE wrapper — safe for Streamlit Cloud deserialization."""
    def fit_resample(self, X, y):
        if not SMOTE_AVAILABLE:
            return X, y
        counts = {c: (y == c).sum() for c in np.unique(y)}
        min_count = min(counts.values())
        if min_count < 4:
            return X, y
        k = min(3, min_count - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        return smote.fit_resample(X, y)
        
    def transform(self, X, y=None):
        """Pass-through for old Pipeline unpickling on Streamlit Cloud."""
        return X

    def fit(self, X, y=None):
        """Pass-through for old Pipeline unpickling."""
        return self


# ─── TWO-HEAD ARCHITECTURE ────────────────────────────────────────────────────
class TwoHeadMetaLearner:
    """
    Head 1: Draw Gate — binary classifier for P(Draw)
    Head 2: Direction Gate — binary classifier for P(Home Win | not Draw)
    This completely decouples draw prediction from the H/A competition.
    """
    def __init__(self, C_draw=0.3, C_direction=0.5):
        self.draw_gate = LogisticRegression(
            C=C_draw,
            class_weight={0: 1.0, 1: 3.5},  # 3.5x weight on draws
            solver='lbfgs', max_iter=1000, random_state=42
        )
        self.direction_gate = LogisticRegression(
            C=C_direction,
            class_weight='balanced',
            solver='lbfgs', max_iter=1000, random_state=42
        )
        self.calibrators = {}   # per-market isotonic calibrators
        self.classes_ = ['Home Win', 'Draw', 'Away Win']
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit both heads on OOF training split with SMOTE."""
        smote = SafeSMOTE()
        X_bal, y_bal = smote.fit_resample(X, y)

        y_draw     = (y_bal == 'Draw').astype(int)
        self.draw_gate.fit(X_bal, y_draw)

        non_draw_mask = y_bal != 'Draw'
        if non_draw_mask.sum() > 10:
            y_dir = (y_bal[non_draw_mask] == 'Home Win').astype(int)
            self.direction_gate.fit(X_bal[non_draw_mask], y_dir)
        else:
            warnings.warn("Insufficient non-draw samples for direction gate.")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Produce calibrated 3-class probabilities.
        p_draw from draw_gate, p_home/p_away from direction_gate.
        """
        # Patch for Scikit-Learn version mismatch during unpickling
        if not hasattr(self.draw_gate, 'multi_class'):
            self.draw_gate.multi_class = 'auto'
        if not hasattr(self.direction_gate, 'multi_class'):
            self.direction_gate.multi_class = 'auto'

        p_draw      = self.draw_gate.predict_proba(X)[:, 1]
        p_hw_cond   = self.direction_gate.predict_proba(X)[:, 1]
        p_home      = (1 - p_draw) * p_hw_cond
        p_away      = (1 - p_draw) * (1 - p_hw_cond)
        proba       = np.column_stack([p_home, p_draw, p_away])

        # Apply per-market isotonic calibration if fitted
        if self.calibrators:
            proba = self._apply_calibration(proba)

        # Ensure valid probability simplex
        proba = np.clip(proba, 0.02, 0.96)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def _apply_calibration(self, proba: np.ndarray) -> np.ndarray:
        cal = proba.copy()
        for i, market in enumerate(self.classes_):
            if market in self.calibrators:
                cal[:, i] = self.calibrators[market].predict(proba[:, i])
        return cal

    def fit_calibration(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Fit per-market Isotonic Regression calibrators on held-out data.
        Call AFTER fit(), on a separate chronological slice.
        """
        proba_cal = self.predict_proba(X_cal)
        for i, market in enumerate(self.classes_):
            iso = IsotonicRegression(out_of_bounds='clip',
                                     y_min=0.05, y_max=0.90)
            y_bin = (y_cal == market).astype(float)
            iso.fit(proba_cal[:, i], y_bin)
            self.calibrators[market] = iso
        return self

    def compute_ece(self, X: np.ndarray, y: np.ndarray,
                     n_bins: int = 10) -> float:
        """Expected Calibration Error across all 3 markets."""
        proba = self.predict_proba(X)
        ece_total = 0.0
        for i, market in enumerate(self.classes_):
            y_bin = (y == market).astype(float)
            p = proba[:, i]
            bins = np.linspace(0, 1, n_bins + 1)
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() == 0:
                    continue
                acc  = y_bin[mask].mean()
                conf = p[mask].mean()
                ece_total += (mask.sum() / len(y)) * abs(acc - conf)
        return ece_total / 3   # average over 3 markets


def fit_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray) -> TwoHeadMetaLearner:
    """
    Full meta-learner training:
    1. Train on first 85% of OOF (chronological)
    2. Calibrate isotonic on last 15% of OOF
    """
    n = len(y_oof)
    cal_split = int(n * 0.85)

    model = TwoHeadMetaLearner(C_draw=0.3, C_direction=0.5)
    model.fit(oof_preds[:cal_split], y_oof[:cal_split])
    model.fit_calibration(oof_preds[cal_split:], y_oof[cal_split:])

    ece = model.compute_ece(oof_preds[cal_split:], y_oof[cal_split:])
    print(f"  Meta-learner ECE (cal set): {ece:.4f} (target: <0.04)")
    return model


def predict_with_draw_threshold(model: TwoHeadMetaLearner,
                                  X: np.ndarray,
                                  draw_thresh: float = 0.30) -> tuple:
    """
    With the two-head architecture, draw_thresh is lower (0.30 vs old 0.38)
    because draw probabilities are now correctly distributed, not suppressed.
    """
    proba = model.predict_proba(X)
    draw_idx = model.classes_.index('Draw')
    preds = []
    for p in proba:
        if p[draw_idx] >= draw_thresh:
            preds.append('Draw')
        else:
            non_draw = [(i, v) for i, v in enumerate(p) if i != draw_idx]
            preds.append(model.classes_[max(non_draw, key=lambda x: x[1])[0]])
    return np.array(preds), proba
