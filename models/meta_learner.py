import numpy as np
import numpy as np

# Dummy SafeSMOTE to allow old models to unpickle cleanly during inference
class SafeSMOTE:
    def __init__(self, *args, **kwargs):
        pass
    def fit_resample(self, X, y):
        return X, y
    def transform(self, X):
        return X

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfpl = tfp.layers

    def prior_fn(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
                    reinterpreted_batch_ndims=1
                )
            )
        ])

    def posterior_fn(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n)
        ])
    
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    tf = None
    tfpl = None
    tfd = None

class BNNMetaLearner:
    """
    Bayesian Neural Network for 1X2 market probability forecasting.
    """
    def __init__(self, input_dim: int, num_train_samples: int, hidden_units: list = [64, 32]):
        self.input_dim = input_dim
        self.num_train_samples = num_train_samples
        self.hidden_units = hidden_units
        self.kl_weight = 1.0 / max(1, num_train_samples)
        self.classes_ = ['Home Win', 'Draw', 'Away Win']
        self.model = self._build_model() if TF_AVAILABLE else None
        
    def _build_model(self):
        layers = [tf.keras.layers.InputLayer(input_shape=(self.input_dim,))]
        for units in self.hidden_units:
            layers.append(
                tfpl.DenseVariational(
                    units=units,
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=self.kl_weight,
                    activation='relu'
                )
            )
        layers.append(
            tfpl.DenseVariational(
                units=3,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=self.kl_weight,
                activation=None  # Outputs logits
            )
        )
        return tf.keras.Sequential(layers)

    def compile(self, learning_rate: float = 0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        def negative_log_likelihood(y_true, logits):
            return tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
            )
        self.model.compile(optimizer=optimizer, loss=negative_log_likelihood, metrics=['accuracy'])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: int = 0):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Standard interface for train_test.py. Returns the mean probability."""
        mean_p, _ = self.predict_distribution(X, num_samples=50)
        return mean_p

    def predict_distribution(self, X: np.ndarray, num_samples: int = 50) -> tuple:
        """
        Estimates the probability distribution over the 1X2 outcomes by sampling 
        from the posterior of the weights. Returns (mean_probs, var_probs).
        """
        # Shape: (T, N, 3)
        logits_samples = tf.stack([self.model(X) for _ in range(num_samples)])
        prob_samples = tf.nn.softmax(logits_samples, axis=-1)
        mean_probs = tf.reduce_mean(prob_samples, axis=0)
        var_probs = tf.math.reduce_variance(prob_samples, axis=0)
        return mean_probs.numpy(), var_probs.numpy()

def fit_meta_learner(X_train: np.ndarray, y_train_str: np.ndarray, draw_gate_params=None) -> BNNMetaLearner:
    """
    Wrapper for train_test.py. Replaces the two-head XGBoost with a Bayesian Neural Network.
    """
    # y_train_str contains 'Home Win', 'Draw', 'Away Win'
    y_map = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
    y_idx = np.array([y_map[v] for v in y_train_str])
    y_onehot = np.eye(3)[y_idx].astype(np.float32)
    X = X_train.astype(np.float32)

    bnn = BNNMetaLearner(input_dim=X.shape[1], num_train_samples=len(X))
    bnn.compile(learning_rate=0.01)
    
    print(f"  [BNN] Training Stochastic Meta-Learner on {len(X)} samples...")
    # epochs can be adjusted based on needs, kept low here for speed during tests
    bnn.fit(X, y_onehot, epochs=50, batch_size=64, verbose=0)
    
    return bnn


# --- LEGACY V7 CLASS FOR UNPICKLING ---
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from models.temperature_scaler import TemperatureScaler

class TwoHeadMetaLearner:
    """
    Head 1 (draw_gate):      P(Draw) ΓÇö XGBoost (nonlinear draw signals)
    Head 2 (direction_gate): P(Home Win | not Draw) ΓÇö Logistic Regression

    Calibration pipeline:
        raw outputs ΓåÆ Isotonic calibration per market ΓåÆ Temperature Scaling
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

    # ΓöÇΓöÇ TRAINING ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
        Uses the same X_cal slice ΓÇö chronologically AFTER training data.
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

        # Temperature scaling ΓÇö convert calibrated proba to pseudo-logits
        iso_proba = self._apply_isotonic(raw_proba)
        # Convert probabilities back to logits for temperature fitting
        # log(p) with small epsilon guard
        logits = np.log(np.clip(iso_proba, 1e-6, 1 - 1e-6))
        # Normalize to zero-mean (proper logit form)
        logits = logits - logits.mean(axis=1, keepdims=True)

        self.temp_scaler.fit(logits, y_cal, self.classes_)
        self.temp_fitted = True
        return self

    # ΓöÇΓöÇ INFERENCE ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Raw probabilities from both gates ΓÇö no calibration applied."""
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
        raw outputs ΓåÆ isotonic calibration ΓåÆ temperature scaling ΓåÆ normalize
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


# ΓöÇΓöÇΓöÇ FIT PIPELINE ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def fit_meta_learner(oof_preds: np.ndarray, y_oof: np.ndarray, draw_gate_params=None) -> TwoHeadMetaLearner:
    """
    Fit meta-learner with tighter calibration split (75/25 instead of 85/15).
    oof_preds: (N, 6) ΓÇö 3 ML base-learner probs + 3 DC probs concatenated.
    """
    n = len(y_oof)
    cal_split = int(n * 0.75)   # ΓåÉ 25% for calibration (was 15%)

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
    draw probabilities ΓÇö they no longer need to be 0.38+ to fire.
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
