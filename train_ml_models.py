"""
Train Traditional ML Models (LightGBM, CatBoost)
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from utils.data_loader import load_data, prepare_features_targets, split_train_validation
from utils.metrics import compute_regression_metrics, compute_classification_metrics, print_metrics
from models.ml_models import MLModelWrapper


def train_ml_models():
    """Train LightGBM and CatBoost models"""
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare features and targets
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, feature_names = \
        prepare_features_targets(train_df, test_df)
    
    # Split train into train/validation
    X_tr, X_val, y_tr_reg, y_val_reg, y_tr_cls, y_val_cls = \
        split_train_validation(X_train, y_train_reg, y_train_cls, VALIDATION_SPLIT)
    
    results = {}
    
    # =====================================================
    # LIGHTGBM
    # =====================================================
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    
    lgb_wrapper = MLModelWrapper(MODEL_DIR / 'lightgbm')
    lgb_wrapper.train_lightgbm(X_tr, y_tr_reg, y_tr_cls, X_val, y_val_reg, y_val_cls)
    
    # Evaluate
    lgb_reg_pred, lgb_cls_pred = lgb_wrapper.predict(X_test)
    
    lgb_reg_metrics = compute_regression_metrics(
        y_test_reg, lgb_reg_pred,
        target_names=['passenger_demand', 'load_factor']
    )
    lgb_cls_metrics = compute_classification_metrics(
        y_test_cls, lgb_cls_pred,
        target_name='utilization_status'
    )
    
    print_metrics(lgb_reg_metrics, "LightGBM - Regression")
    print_metrics(lgb_cls_metrics, "LightGBM - Classification")
    
    # Save predictions
    np.save(MODEL_DIR / 'lightgbm' / 'test_predictions_reg.npy', lgb_reg_pred)
    np.save(MODEL_DIR / 'lightgbm' / 'test_predictions_cls.npy', lgb_cls_pred)
    
    results['lightgbm'] = {
        'reg_pred': lgb_reg_pred,
        'cls_pred': lgb_cls_pred,
        'reg_metrics': lgb_reg_metrics,
        'cls_metrics': lgb_cls_metrics
    }
    
    # =====================================================
    # CATBOOST
    # =====================================================
    print("\n" + "="*60)
    print("TRAINING CATBOOST")
    print("="*60)
    
    cb_wrapper = MLModelWrapper(MODEL_DIR / 'catboost')
    cb_wrapper.train_catboost(X_tr, y_tr_reg, y_tr_cls, X_val, y_val_reg, y_val_cls)
    
    # Evaluate
    cb_reg_pred, cb_cls_pred = cb_wrapper.predict(X_test)
    
    cb_reg_metrics = compute_regression_metrics(
        y_test_reg, cb_reg_pred,
        target_names=['passenger_demand', 'load_factor']
    )
    cb_cls_metrics = compute_classification_metrics(
        y_test_cls, cb_cls_pred,
        target_name='utilization_status'
    )
    
    print_metrics(cb_reg_metrics, "CatBoost - Regression")
    print_metrics(cb_cls_metrics, "CatBoost - Classification")
    
    # Save predictions
    np.save(MODEL_DIR / 'catboost' / 'test_predictions_reg.npy', cb_reg_pred)
    np.save(MODEL_DIR / 'catboost' / 'test_predictions_cls.npy', cb_cls_pred)
    
    results['catboost'] = {
        'reg_pred': cb_reg_pred,
        'cls_pred': cb_cls_pred,
        'reg_metrics': cb_reg_metrics,
        'cls_metrics': cb_cls_metrics
    }
    
    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    lgb_importance = lgb_wrapper.get_feature_importance(feature_names)
    
    print("\nTop 10 features for demand prediction:")
    demand_imp = sorted(lgb_importance['demand'].items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in demand_imp:
        print(f"  {feat}: {imp:.2f}")
    
    print("\nTop 10 features for load factor prediction:")
    load_imp = sorted(lgb_importance['load_factor'].items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in load_imp:
        print(f"  {feat}: {imp:.2f}")
    
    print("\n✓ ML models training complete")
    
    return results


if __name__ == "__main__":
    train_ml_models()
