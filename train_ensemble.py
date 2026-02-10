"""
Train Stacked Ensemble Model
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from utils.data_loader import load_data, prepare_features_targets, split_train_validation
from utils.metrics import compute_regression_metrics, compute_classification_metrics, print_metrics
from models.deep_learning_model import MultiTaskTransportModel
from models.ml_models import MLModelWrapper
from models.ensemble import StackedEnsemble


def train_ensemble():
    """Train stacked ensemble combining DL + LightGBM + CatBoost"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_df, test_df = load_data()
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, feature_names = \
        prepare_features_targets(train_df, test_df)
    
    X_tr, X_val, y_tr_reg, y_val_reg, y_tr_cls, y_val_cls = \
        split_train_validation(X_train, y_train_reg, y_train_cls, VALIDATION_SPLIT)
    
    print("\n" + "="*60)
    print("GENERATING BASE MODEL PREDICTIONS")
    print("="*60)
    
    # =====================================================
    # Load trained models and get predictions
    # =====================================================
    
    # Deep Learning
    print("\nLoading Deep Learning model...")
    input_dim = X_train.shape[1]
    dl_model = MultiTaskTransportModel(
        input_dim=input_dim,
        hidden_dims=DL_CONFIG['hidden_dims'],
        dropout_rate=DL_CONFIG['dropout_rate'],
        num_classes=3
    ).to(device)
    
    checkpoint = torch.load(MODEL_DIR / 'deep_learning' / 'best_model.pt', map_location=device)
    dl_model.load_state_dict(checkpoint['model_state_dict'])
    dl_model.eval()
    
    with torch.no_grad():
        # Validation predictions (for meta-training)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        dl_val_reg, dl_val_cls_logits = dl_model(X_val_tensor)
        dl_val_reg = dl_val_reg.cpu().numpy()
        dl_val_cls = torch.argmax(dl_val_cls_logits, dim=1).cpu().numpy()
        
        # Test predictions
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        dl_test_reg, dl_test_cls_logits = dl_model(X_test_tensor)
        dl_test_reg = dl_test_reg.cpu().numpy()
        dl_test_cls = torch.argmax(dl_test_cls_logits, dim=1).cpu().numpy()
    
    print("✓ Deep Learning predictions generated")
    
    # LightGBM
    print("\nLoading LightGBM models...")
    lgb_wrapper = MLModelWrapper(MODEL_DIR / 'lightgbm')
    lgb_wrapper.load_models('lightgbm')
    
    lgb_val_reg, lgb_val_cls = lgb_wrapper.predict(X_val)
    lgb_test_reg, lgb_test_cls = lgb_wrapper.predict(X_test)
    print("✓ LightGBM predictions generated")
    
    # CatBoost
    print("\nLoading CatBoost models...")
    cb_wrapper = MLModelWrapper(MODEL_DIR / 'catboost')
    cb_wrapper.load_models('catboost')
    
    cb_val_reg, cb_val_cls = cb_wrapper.predict(X_val)
    cb_test_reg, cb_test_cls = cb_wrapper.predict(X_test)
    print("✓ CatBoost predictions generated")
    
    # =====================================================
    # Train Stacked Ensemble
    # =====================================================
    
    print("\n" + "="*60)
    print("TRAINING STACKED ENSEMBLE")
    print("="*60)
    
    ensemble = StackedEnsemble(MODEL_DIR / 'ensemble')
    
    # Prepare meta-features from validation set
    meta_val_reg, meta_val_cls = ensemble.prepare_meta_features(
        dl_val_reg, dl_val_cls,
        lgb_val_reg, lgb_val_cls,
        cb_val_reg, cb_val_cls
    )
    
    # Train meta-learners
    ensemble.train(meta_val_reg, y_val_reg, meta_val_cls, y_val_cls)
    
    # =====================================================
    # Evaluate Ensemble on Test Set
    # =====================================================
    
    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE ON TEST SET")
    print("="*60)
    
    # Prepare meta-features from test set
    meta_test_reg, meta_test_cls = ensemble.prepare_meta_features(
        dl_test_reg, dl_test_cls,
        lgb_test_reg, lgb_test_cls,
        cb_test_reg, cb_test_cls
    )
    
    # Ensemble predictions
    ensemble_reg_pred, ensemble_cls_pred = ensemble.predict(meta_test_reg, meta_test_cls)
    
    # Compute metrics
    ensemble_reg_metrics = compute_regression_metrics(
        y_test_reg, ensemble_reg_pred,
        target_names=['passenger_demand', 'load_factor']
    )
    ensemble_cls_metrics = compute_classification_metrics(
        y_test_cls, ensemble_cls_pred,
        target_name='utilization_status'
    )
    
    print_metrics(ensemble_reg_metrics, "Ensemble - Regression")
    print_metrics(ensemble_cls_metrics, "Ensemble - Classification")
    
    # Save predictions
    np.save(MODEL_DIR / 'ensemble' / 'test_predictions_reg.npy', ensemble_reg_pred)
    np.save(MODEL_DIR / 'ensemble' / 'test_predictions_cls.npy', ensemble_cls_pred)
    
    # =====================================================
    # Compare All Models
    # =====================================================
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Compute metrics for all base models on test set
    dl_reg_metrics = compute_regression_metrics(
        y_test_reg, dl_test_reg,
        target_names=['passenger_demand', 'load_factor']
    )
    lgb_reg_metrics = compute_regression_metrics(
        y_test_reg, lgb_test_reg,
        target_names=['passenger_demand', 'load_factor']
    )
    cb_reg_metrics = compute_regression_metrics(
        y_test_reg, cb_test_reg,
        target_names=['passenger_demand', 'load_factor']
    )
    
    print("\nPassenger Demand Prediction:")
    print(f"  Deep Learning - RMSE: {dl_reg_metrics['passenger_demand']['rmse']:.4f}, "
          f"MAE: {dl_reg_metrics['passenger_demand']['mae']:.4f}, "
          f"R²: {dl_reg_metrics['passenger_demand']['r2']:.4f}")
    print(f"  LightGBM      - RMSE: {lgb_reg_metrics['passenger_demand']['rmse']:.4f}, "
          f"MAE: {lgb_reg_metrics['passenger_demand']['mae']:.4f}, "
          f"R²: {lgb_reg_metrics['passenger_demand']['r2']:.4f}")
    print(f"  CatBoost      - RMSE: {cb_reg_metrics['passenger_demand']['rmse']:.4f}, "
          f"MAE: {cb_reg_metrics['passenger_demand']['mae']:.4f}, "
          f"R²: {cb_reg_metrics['passenger_demand']['r2']:.4f}")
    print(f"  Ensemble      - RMSE: {ensemble_reg_metrics['passenger_demand']['rmse']:.4f}, "
          f"MAE: {ensemble_reg_metrics['passenger_demand']['mae']:.4f}, "
          f"R²: {ensemble_reg_metrics['passenger_demand']['r2']:.4f}")
    
    print("\nLoad Factor Prediction:")
    print(f"  Deep Learning - RMSE: {dl_reg_metrics['load_factor']['rmse']:.4f}, "
          f"MAE: {dl_reg_metrics['load_factor']['mae']:.4f}, "
          f"R²: {dl_reg_metrics['load_factor']['r2']:.4f}")
    print(f"  LightGBM      - RMSE: {lgb_reg_metrics['load_factor']['rmse']:.4f}, "
          f"MAE: {lgb_reg_metrics['load_factor']['mae']:.4f}, "
          f"R²: {lgb_reg_metrics['load_factor']['r2']:.4f}")
    print(f"  CatBoost      - RMSE: {cb_reg_metrics['load_factor']['rmse']:.4f}, "
          f"MAE: {cb_reg_metrics['load_factor']['mae']:.4f}, "
          f"R²: {cb_reg_metrics['load_factor']['r2']:.4f}")
    print(f"  Ensemble      - RMSE: {ensemble_reg_metrics['load_factor']['rmse']:.4f}, "
          f"MAE: {ensemble_reg_metrics['load_factor']['mae']:.4f}, "
          f"R²: {ensemble_reg_metrics['load_factor']['r2']:.4f}")
    
    print("\n✓ Ensemble training and evaluation complete")
    
    return ensemble, ensemble_reg_metrics, ensemble_cls_metrics


if __name__ == "__main__":
    train_ensemble()
