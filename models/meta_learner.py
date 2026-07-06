import numpy as np
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
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
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
