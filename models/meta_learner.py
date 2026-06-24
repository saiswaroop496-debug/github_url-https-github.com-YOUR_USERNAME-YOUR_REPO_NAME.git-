import numpy as np
from sklearn.linear_model import LogisticRegression

class BetaCalibration:
    def __init__(self):
        self.coefs = {}
        self.intercepts = {}
        
    def _log_odds(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
        
    def fit(self, probs, y_true):
        # probs: shape (n_samples, 3)
        # y_true: shape (n_samples,) integers 0, 1, 2
        
        for class_idx in range(3):
            # Binary target for this class
            y_binary = (y_true == class_idx).astype(int)
            
            # Predictor is the log-odds of the predicted probability for this class
            X = self._log_odds(probs[:, class_idx]).reshape(-1, 1)
            
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(X, y_binary)
            
            self.coefs[class_idx] = lr.coef_[0][0]
            self.intercepts[class_idx] = lr.intercept_[0]
            
    def predict_proba(self, probs):
        calibrated = np.zeros_like(probs)
        for class_idx in range(3):
            X = self._log_odds(probs[:, class_idx])
            z = self.coefs[class_idx] * X + self.intercepts[class_idx]
            calibrated[:, class_idx] = 1 / (1 + np.exp(-z))
            
        # Normalize back to sum=1.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / row_sums

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

class MetaLearner:
    def __init__(self):
        self.meta_model = LogisticRegression(C=0.5, class_weight='balanced', solver='lbfgs', max_iter=1000)
        self.calibrator = BetaCalibration()
        
    def fit_predict(self, X, y):
        n = len(X)
        cal_start = int(n * 0.75)   # last 25% chronologically
        
        X_train, y_train = X[:cal_start], y[:cal_start]
        X_calib, y_calib = X[cal_start:], y[cal_start:]
        
        # Train
        self.meta_model.fit(X_train, y_train)
        
        # Predict on Calib and fit calibrator
        calib_probs = self.meta_model.predict_proba(X_calib)
        self.calibrator.fit(calib_probs, y_calib)
        
        # Return full calibrated predictions for the entire set
        raw_probs = self.meta_model.predict_proba(X)
        final_probs = self.calibrator.predict_proba(raw_probs)
        
        # Hard constraints
        final_probs = np.clip(final_probs, 0.05, 0.95)
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        
        return final_probs

def predict_with_draw_threshold(model, X: np.ndarray, classes, draw_thresh: float = 0.28):
    """
    Override argmax for Draw class: if Draw prob >= draw_thresh, predict Draw.
    This unlocks draws that the meta-learner suppresses at the decision boundary.
    """
    proba = model.predict_proba(X)
    draw_idx = list(classes).index('Draw')
    preds = []
    for p in proba:
        if p[draw_idx] >= draw_thresh:
            preds.append('Draw')
        else:
            preds.append(classes[np.argmax(p)])
    return np.array(preds), proba

def build_ordinal_meta_learner(X_oof: np.ndarray, y_oof_encoded: np.ndarray):
    """
    Ordered logit respects Away Win < Draw < Home Win natural ordering.
    Use as challenger against the balanced LR in champion/challenger gate.
    
    y_oof_encoded must be integer: 0=Away Win, 1=Draw, 2=Home Win
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        ord_model = OrderedModel(y_oof_encoded, X_oof, distr='logit')
        result = ord_model.fit(method='bfgs', disp=False)
        return result
    except ImportError:
        print("statsmodels not installed. Run: pip install statsmodels")
        return None
