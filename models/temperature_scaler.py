import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

class TemperatureScaler:
    """
    Single-parameter post-hoc calibration.
    Divides pre-softmax logits by temperature T before final softmax.
    T > 1 = softer (fixes overconfidence).
    T < 1 = sharper (fixes underconfidence — rare).
    Bounds (1.0, 5.0) — never artificially sharpen.

    USAGE:
        scaler = TemperatureScaler()
        scaler.fit(logits_cal, y_cal, classes)    # fit on held-out cal set
        proba  = scaler.transform(logits_test)    # apply during inference
    """
    def __init__(self):
        self.T       = 1.0
        self.fitted  = False
        self.classes = None

    def fit(self, logits: np.ndarray, y_true: np.ndarray, classes: list):
        """
        Find optimal T that minimises NLL on calibration set.
        logits: (N, 3) raw pre-softmax values — NOT probabilities.
        y_true: (N,) string labels e.g. ['Home Win', 'Draw', ...]
        """
        self.classes = classes

        def nll(T):
            proba = self._softmax(logits / T)
            return log_loss(y_true, proba, labels=classes)

        result = minimize_scalar(nll, bounds=(1.0, 5.0), method='bounded')
        self.T      = float(result.x)
        self.fitted = True

        pre_nll  = nll(1.0)
        post_nll = nll(self.T)
        print(f"  TemperatureScaler: T={self.T:.4f} | "
              f"NLL before={pre_nll:.4f} -> after={post_nll:.4f} "
              f"({'improved' if post_nll < pre_nll else 'no improvement'})")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities."""
        if not self.fitted:
            # Fallback: return raw softmax without scaling
            return self._softmax(logits)
        return self._softmax(logits / self.T)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def get_params(self) -> dict:
        return {"T": self.T, "fitted": self.fitted}
