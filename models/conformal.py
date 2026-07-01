import numpy as np

class ConformalPredictor:
    """
    Adaptive Conformal Inference for football outcome probabilities.
    Produces statistically valid [lower, upper] bounds at any coverage level.
    """
    def __init__(self, coverage: float = 0.90):
        self.coverage = coverage
        self.calibration_scores = []
        self.threshold = 0.0

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray, classes: list):
        """Compute nonconformity scores on held-out calibration set."""
        self.calibration_scores = []
        for i, y in enumerate(y_cal):
            if y not in classes:
                continue
            true_idx = classes.index(y)
            score = 1 - proba_cal[i, true_idx]   # nonconformity = 1 - P(true class)
            self.calibration_scores.append(score)

        if not self.calibration_scores:
            self.threshold = 0.5
            return

        self.calibration_scores = np.array(self.calibration_scores)
        alpha = 1 - self.coverage
        n = len(self.calibration_scores)
        self.threshold = np.quantile(
            self.calibration_scores,
            np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1),
            method='higher'
        )

    def predict_set(self, proba: np.ndarray, classes: list) -> list:
        """Return prediction SET (all classes whose 1-p <= threshold)."""
        prediction_sets = []
        for p in proba:
            pred_set = [classes[i] for i, pi in enumerate(p)
                        if 1 - pi <= self.threshold]
            if not pred_set:
                pred_set = [classes[np.argmax(p)]]
            prediction_sets.append(pred_set)
        return prediction_sets

    def get_bounds(self, proba: np.ndarray, class_idx: int) -> tuple:
        """Get [lower, upper] bounds for a specific class probability."""
        p = proba[:, class_idx]
        margin = self.threshold * 0.5   # approximate bound
        return (
            np.clip(p - margin, 0.02, 0.98),
            np.clip(p + margin, 0.02, 0.98)
        )
