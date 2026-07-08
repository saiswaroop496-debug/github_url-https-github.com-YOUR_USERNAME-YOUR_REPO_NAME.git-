import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

# Dummy SafeSMOTE to allow old models to unpickle cleanly if they leak
class SafeSMOTE:
    def __init__(self, *args, **kwargs):
        pass
    def fit_resample(self, X, y):
        return X, y
    def transform(self, X):
        return X

class BNNMetaLearner:
    """
    Bayesian Neural Network approximation for 1X2 market probability forecasting.
    Uses Deep Ensembles (Lakshminarayanan et al., 2017) via scikit-learn MLPClassifiers.
    This guarantees execution on constrained environments where TF/PyTorch DLLs fail.
    """
    def __init__(self, input_dim: int, num_train_samples: int, hidden_units: list = None, n_ensembles: int = 10):
        if hidden_units is None:
            hidden_units = [64, 32]
        self.input_dim = input_dim
        self.n_ensembles = n_ensembles
        self.classes_ = ['Home Win', 'Draw', 'Away Win']
        self.hidden_units = tuple(hidden_units)
        self.models = []
        self.learning_rate_init = 0.001

    def compile(self, learning_rate: float = 0.001):
        self.learning_rate_init = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: int = 0):
        self.models = []
        # y is one-hot, convert to class indices
        y_idx = np.argmax(y, axis=1)
        
        for i in range(self.n_ensembles):
            # Bootstrap sampling for the ensemble
            X_b, y_b = resample(X, y_idx, random_state=42 + i)
            
            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_units,
                activation='relu',
                solver='adam',
                alpha=0.001, # L2 penalty (weight decay)
                batch_size=min(batch_size, len(X_b)),
                learning_rate_init=self.learning_rate_init,
                max_iter=epochs,
                random_state=42 + i,
                early_stopping=False
            )
            # Suppress ConvergenceWarnings during short epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X_b, y_b)
            self.models.append(mlp)

    def predict_distribution(self, X: np.ndarray, num_samples: int = 50) -> tuple:
        """
        Estimates the probability distribution over the 1X2 outcomes by querying
        the deep ensemble. Returns (mean_probs, var_probs).
        """
        preds = []
        for model in self.models:
            preds.append(model.predict_proba(X))
        
        preds = np.stack(preds) # (n_ensembles, N, 3)
        mean_probs = np.mean(preds, axis=0)
        var_probs = np.var(preds, axis=0)
        return mean_probs, var_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Standard interface for train_test.py. Returns the mean probability."""
        mean_p, _ = self.predict_distribution(X)
        return mean_p

def fit_meta_learner(X_train: np.ndarray, y_train_str: np.ndarray, draw_gate_params=None) -> BNNMetaLearner:
    """
    Wrapper for train_test.py. Replaces the two-head XGBoost with a Deep Ensemble BNN.
    """
    # y_train_str contains 'Home Win', 'Draw', 'Away Win'
    y_map = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
    y_idx = np.array([y_map[v] for v in y_train_str])
    y_onehot = np.eye(3)[y_idx].astype(np.float32)
    X = X_train.astype(np.float32)

    bnn = BNNMetaLearner(input_dim=X.shape[1], num_train_samples=len(X), n_ensembles=15)
    bnn.compile(learning_rate=0.002)
    
    print(f"  [BNN] Training Stochastic Meta-Learner (Deep Ensemble) on {len(X)} samples...")
    # epochs can be adjusted based on needs
    bnn.fit(X, y_onehot, epochs=150, batch_size=64, verbose=0)
    
    return bnn
