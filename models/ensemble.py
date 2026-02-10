"""
Stacked Meta-Ensemble using XGBoost
"""
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path


class StackedEnsemble:
    """
    Stacking ensemble that combines:
    - Deep Learning predictions
    - LightGBM predictions
    - CatBoost predictions
    
    Meta-learner: XGBoost
    """
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.demand_meta = None
        self.load_factor_meta = None
        self.utilization_meta = None
    
    def prepare_meta_features(self, dl_reg, dl_cls, lgb_reg, lgb_cls, cb_reg, cb_cls):
        """
        Combine predictions from all base models into meta-features
        
        Args:
            dl_reg: Deep learning regression predictions (n_samples, 2)
            dl_cls: Deep learning classification predictions (n_samples,)
            lgb_reg: LightGBM regression predictions (n_samples, 2)
            lgb_cls: LightGBM classification predictions (n_samples,)
            cb_reg: CatBoost regression predictions (n_samples, 2)
            cb_cls: CatBoost classification predictions (n_samples,)
        
        Returns:
            meta_features_reg: For regression tasks (n_samples, 6)
            meta_features_cls: For classification task (n_samples, 3)
        """
        # Meta-features for regression (demand and load factor)
        meta_features_reg = np.hstack([
            dl_reg,      # 2 features
            lgb_reg,     # 2 features
            cb_reg       # 2 features
        ])
        
        # Meta-features for classification (utilization)
        meta_features_cls = np.column_stack([
            dl_cls,
            lgb_cls,
            cb_cls
        ])
        
        return meta_features_reg, meta_features_cls
    
    def train(self, meta_features_reg, y_train_reg, meta_features_cls, y_train_cls):
        """
        Train meta-learners (XGBoost)
        """
        print("\nTraining Stacked Ensemble (XGBoost meta-learners)...")
        
        # Meta-learner for demand prediction
        print("  Training demand meta-learner...")
        self.demand_meta = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=1
        )
        self.demand_meta.fit(meta_features_reg, y_train_reg[:, 0])
        
        # Meta-learner for load factor prediction
        print("  Training load factor meta-learner...")
        self.load_factor_meta = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=1
        )
        self.load_factor_meta.fit(meta_features_reg, y_train_reg[:, 1])
        
        # Meta-learner for utilization classification
        print("  Training utilization meta-learner...")
        self.utilization_meta = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=1
        )
        self.utilization_meta.fit(meta_features_cls, y_train_cls)
        
        # Save models
        self.save_models()
        print("✓ Ensemble training complete")
    
    def predict(self, meta_features_reg, meta_features_cls):
        """
        Make ensemble predictions
        """
        demand_pred = self.demand_meta.predict(meta_features_reg).reshape(-1, 1)
        load_pred = self.load_factor_meta.predict(meta_features_reg).reshape(-1, 1)
        util_pred = self.utilization_meta.predict(meta_features_cls)
        
        reg_pred = np.hstack([demand_pred, load_pred])
        
        return reg_pred, util_pred
    
    def save_models(self):
        """Save meta-learners"""
        with open(self.model_dir / 'ensemble_demand.pkl', 'wb') as f:
            pickle.dump(self.demand_meta, f)
        with open(self.model_dir / 'ensemble_load_factor.pkl', 'wb') as f:
            pickle.dump(self.load_factor_meta, f)
        with open(self.model_dir / 'ensemble_utilization.pkl', 'wb') as f:
            pickle.dump(self.utilization_meta, f)
    
    def load_models(self):
        """Load meta-learners"""
        with open(self.model_dir / 'ensemble_demand.pkl', 'rb') as f:
            self.demand_meta = pickle.load(f)
        with open(self.model_dir / 'ensemble_load_factor.pkl', 'rb') as f:
            self.load_factor_meta = pickle.load(f)
        with open(self.model_dir / 'ensemble_utilization.pkl', 'rb') as f:
            self.utilization_meta = pickle.load(f)
