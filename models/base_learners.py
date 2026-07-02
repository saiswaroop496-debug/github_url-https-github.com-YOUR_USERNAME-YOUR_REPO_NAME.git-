import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

def xg_to_probs(pred_h, pred_a, rho=-0.13):
    """
    Convert expected goals to 1X2 probabilities via DC score matrix.
    Moved here from train_test.py to prevent circular imports.
    Handles both scalar and array inputs.
    """
    from models.poisson_dixon_coles import score_probability_matrix, outcome_probs
    
    if isinstance(pred_h, (list, np.ndarray, pd.Series)):
        n = len(pred_h)
        probs = np.zeros((n, 3))
        for i in range(n):
            lam_h, lam_a = max(pred_h[i], 0.01), max(pred_a[i], 0.01)
            matrix = score_probability_matrix(lam_h, lam_a, rho=rho)
            h, d, a = outcome_probs(matrix)
            probs[i] = [h, d, a]
        return probs
    else:
        lam_h, lam_a = max(pred_h, 0.01), max(pred_a, 0.01)
        matrix = score_probability_matrix(lam_h, lam_a, rho=rho)
        return outcome_probs(matrix)


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, embargo_gap=4):
        self.n_splits = n_splits
        self.embargo_gap = embargo_gap
        
    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.embargo_gap
            test_end = (i + 2) * fold_size
            
            if test_start >= n_samples: break
            if test_end > n_samples: test_end = n_samples
                
            yield indices[:train_end], indices[test_start:test_end]

def compute_match_weights(dates: pd.Series, xi: float = 0.003) -> np.ndarray:
    """
    Exponential time-decay: w(t) = exp(-xi * days_ago)
    """
    latest = pd.to_datetime(dates).max()
    days_ago = (latest - pd.to_datetime(dates)).dt.days.values
    weights = np.exp(-xi * days_ago)
    return weights / weights.sum() * len(weights)

class BaseLearnerStack:
    def __init__(self):
        self.cat = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10, iterations=250, learning_rate=0.04, subsample=0.8, colsample_bylevel=0.7, random_seed=42, verbose=0)
        self.xgb = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0, n_estimators=200, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, random_state=42, verbosity=0)
        self.ridge = Ridge(alpha=15.0)
        
    def generate_oof(self, X, y_home, y_away, dates):
        cv = PurgedTimeSeriesSplit(n_splits=5, embargo_gap=4)
        
        oof_home = np.zeros(len(X))
        oof_away = np.zeros(len(X))
        counts = np.zeros(len(X))
        
        weights = compute_match_weights(dates)
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            w_train = weights[train_idx]
            
            # Home
            y_train_h = y_home.iloc[train_idx]
            p_home = np.zeros(len(test_idx))
            
            self.cat.fit(X_train, y_train_h, sample_weight=w_train)
            self.xgb.fit(X_train, y_train_h, sample_weight=w_train)
            self.ridge.fit(X_train, y_train_h, sample_weight=w_train)
            
            p_home += self.cat.predict(X_test)
            p_home += self.xgb.predict(X_test)
            p_home += self.ridge.predict(X_test)
            
            oof_home[test_idx] += p_home / 3.0
            
            # Away
            y_train_a = y_away.iloc[train_idx]
            p_away = np.zeros(len(test_idx))
            
            self.cat.fit(X_train, y_train_a, sample_weight=w_train)
            self.xgb.fit(X_train, y_train_a, sample_weight=w_train)
            self.ridge.fit(X_train, y_train_a, sample_weight=w_train)
            
            p_away += self.cat.predict(X_test)
            p_away += self.xgb.predict(X_test)
            p_away += self.ridge.predict(X_test)
            
            oof_away[test_idx] += p_away / 3.0
            
            counts[test_idx] += 1
            
        counts[counts == 0] = 1 # Avoid division by zero
        oof_home = oof_home / counts
        oof_away = oof_away / counts
        
        oof_home = np.clip(oof_home, 0.3, 4.0)
        oof_away = np.clip(oof_away, 0.3, 4.0)
        return oof_home, oof_away
