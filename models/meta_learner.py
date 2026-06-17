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

class MetaLearner:
    def __init__(self):
        self.meta_model = LogisticRegression(C=0.3, solver='lbfgs', multi_class='multinomial', max_iter=1000)
        self.calibrator = BetaCalibration()
        
    def fit_predict(self, X, y):
        # 3-Way Chronological Split
        n = len(X)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_calib, y_calib = X[val_end:], y[val_end:]
        
        # Train
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate on Val (for reporting, omitted logic to tune C for simplicity here as C=0.3 is hardcoded)
        
        # Predict on Calib and fit calibrator
        calib_probs = self.meta_model.predict_proba(X_calib)
        self.calibrator.fit(calib_probs, y_calib)
        
        # Return full calibrated predictions for the entire set (or just what's needed)
        raw_probs = self.meta_model.predict_proba(X)
        final_probs = self.calibrator.predict_proba(raw_probs)
        
        # Hard constraints
        final_probs = np.clip(final_probs, 0.05, 0.95)
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        
        return final_probs
