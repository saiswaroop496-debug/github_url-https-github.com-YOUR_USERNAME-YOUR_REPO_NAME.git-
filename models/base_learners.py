import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

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

class BaseLearnerStack:
    def __init__(self):
        self.cat = CatBoostRegressor(depth=3, l2_leaf_reg=15.0, min_data_in_leaf=10, iterations=250, learning_rate=0.04, subsample=0.8, colsample_bylevel=0.7, random_seed=42, verbose=0)
        self.xgb = XGBRegressor(max_depth=3, min_child_weight=10, reg_lambda=12.0, n_estimators=200, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, random_state=42, verbosity=0)
        self.ridge = Ridge(alpha=15.0)
        
    def generate_oof(self, X, y_home, y_away):
        cv = PurgedTimeSeriesSplit(n_splits=5, embargo_gap=4)
        
        oof_home = np.zeros(len(X))
        oof_away = np.zeros(len(X))
        counts = np.zeros(len(X))
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Home
            y_train_h = y_home.iloc[train_idx]
            p_home = np.zeros(len(test_idx))
            for model in [self.cat, self.xgb, self.ridge]:
                model.fit(X_train, y_train_h)
                p_home += model.predict(X_test)
            oof_home[test_idx] += p_home / 3.0
            
            # Away
            y_train_a = y_away.iloc[train_idx]
            p_away = np.zeros(len(test_idx))
            for model in [self.cat, self.xgb, self.ridge]:
                model.fit(X_train, y_train_a)
                p_away += model.predict(X_test)
            oof_away[test_idx] += p_away / 3.0
            
            counts[test_idx] += 1
            
        counts[counts == 0] = 1 # Avoid division by zero
        oof_home = oof_home / counts
        oof_away = oof_away / counts
        
        oof_home = np.clip(oof_home, 0.3, 4.0)
        oof_away = np.clip(oof_away, 0.3, 4.0)
        return oof_home, oof_away
