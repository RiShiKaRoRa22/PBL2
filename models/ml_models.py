"""
Traditional Machine Learning Models (LightGBM, CatBoost)
"""
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import pickle
from pathlib import Path


class MLModelWrapper:
    """Wrapper for traditional ML models"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.demand_model = None
        self.load_factor_model = None
        self.utilization_model = None
    
    def train_lightgbm(self, X_train, y_train_reg, y_train_cls, 
                       X_val, y_val_reg, y_val_cls):
        """Train LightGBM models"""
        print("\nTraining LightGBM models...")
        
        # Demand prediction
        print("  Training demand model...")
        self.demand_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.demand_model.fit(
            X_train, y_train_reg[:, 0],
            eval_set=[(X_val, y_val_reg[:, 0])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Load factor prediction
        print("  Training load factor model...")
        self.load_factor_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.load_factor_model.fit(
            X_train, y_train_reg[:, 1],
            eval_set=[(X_val, y_val_reg[:, 1])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Utilization classification
        print("  Training utilization classifier...")
        self.utilization_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.utilization_model.fit(
            X_train, y_train_cls,
            eval_set=[(X_val, y_val_cls)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Save models
        self.save_models('lightgbm')
        print("✓ LightGBM training complete")
    
    def train_catboost(self, X_train, y_train_reg, y_train_cls,
                       X_val, y_val_reg, y_val_cls):
        """Train CatBoost models"""
        print("\nTraining CatBoost models...")
        
        # Demand prediction
        print("  Training demand model...")
        self.demand_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=100
        )
        self.demand_model.fit(
            X_train, y_train_reg[:, 0],
            eval_set=(X_val, y_val_reg[:, 0]),
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Load factor prediction
        print("  Training load factor model...")
        self.load_factor_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=100
        )
        self.load_factor_model.fit(
            X_train, y_train_reg[:, 1],
            eval_set=(X_val, y_val_reg[:, 1]),
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Utilization classification
        print("  Training utilization classifier...")
        self.utilization_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=100
        )
        self.utilization_model.fit(
            X_train, y_train_cls,
            eval_set=(X_val, y_val_cls),
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Save models
        self.save_models('catboost')
        print("✓ CatBoost training complete")
    
    def predict(self, X):
        """Make predictions"""
        demand_pred = self.demand_model.predict(X).reshape(-1, 1)
        load_pred = self.load_factor_model.predict(X).reshape(-1, 1)
        util_pred = self.utilization_model.predict(X)
        
        reg_pred = np.hstack([demand_pred, load_pred])
        
        return reg_pred, util_pred
    
    def save_models(self, prefix):
        """Save trained models"""
        with open(self.model_dir / f'{prefix}_demand.pkl', 'wb') as f:
            pickle.dump(self.demand_model, f)
        with open(self.model_dir / f'{prefix}_load_factor.pkl', 'wb') as f:
            pickle.dump(self.load_factor_model, f)
        with open(self.model_dir / f'{prefix}_utilization.pkl', 'wb') as f:
            pickle.dump(self.utilization_model, f)
    
    def load_models(self, prefix):
        """Load trained models"""
        with open(self.model_dir / f'{prefix}_demand.pkl', 'rb') as f:
            self.demand_model = pickle.load(f)
        with open(self.model_dir / f'{prefix}_load_factor.pkl', 'rb') as f:
            self.load_factor_model = pickle.load(f)
        with open(self.model_dir / f'{prefix}_utilization.pkl', 'rb') as f:
            self.utilization_model = pickle.load(f)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from models"""
        importance = {
            'demand': dict(zip(feature_names, self.demand_model.feature_importances_)),
            'load_factor': dict(zip(feature_names, self.load_factor_model.feature_importances_)),
            'utilization': dict(zip(feature_names, self.utilization_model.feature_importances_))
        }
        return importance
