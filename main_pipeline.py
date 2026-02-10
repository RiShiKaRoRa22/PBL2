"""
Main Training Pipeline for Transport Analytics System

This script orchestrates the complete training pipeline:
1. Deep Learning Model
2. Traditional ML Models (LightGBM, CatBoost)
3. Stacked Ensemble
4. Uncertainty Estimation

IMPORTANT: This implementation stops at the prediction layer.
Future extensions (multi-agent systems, optimization, simulation)
should be built on top of these trained models.
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

from train_deep_learning import train_dl_model
from train_ml_models import train_ml_models
from train_ensemble import train_ensemble
from uncertainty_estimation import estimate_uncertainty


def main():
    """Execute complete training pipeline"""
    
    print("="*70)
    print(" TRANSPORT ANALYTICS ML/DL PREDICTION SYSTEM")
    print(" Research-Grade Hybrid Prediction Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    # =====================================================
    # STAGE 1: DEEP LEARNING MODEL
    # =====================================================
    print("\n" + "="*70)
    print("STAGE 1: DEEP LEARNING MODEL")
    print("="*70)
    
    try:
        dl_model, dl_reg_pred, dl_cls_pred, dl_reg_metrics, dl_cls_metrics = train_dl_model()
        print("✓ Stage 1 complete")
    except Exception as e:
        print(f"✗ Stage 1 failed: {e}")
        return
    
    # =====================================================
    # STAGE 2: TRADITIONAL ML MODELS
    # =====================================================
    print("\n" + "="*70)
    print("STAGE 2: TRADITIONAL ML MODELS")
    print("="*70)
    
    try:
        ml_results = train_ml_models()
        print("✓ Stage 2 complete")
    except Exception as e:
        print(f"✗ Stage 2 failed: {e}")
        return
    
    # =====================================================
    # STAGE 3: STACKED ENSEMBLE
    # =====================================================
    print("\n" + "="*70)
    print("STAGE 3: STACKED ENSEMBLE")
    print("="*70)
    
    try:
        ensemble, ensemble_reg_metrics, ensemble_cls_metrics = train_ensemble()
        print("✓ Stage 3 complete")
    except Exception as e:
        print(f"✗ Stage 3 failed: {e}")
        return
    
    # =====================================================
    # STAGE 4: UNCERTAINTY ESTIMATION
    # =====================================================
    print("\n" + "="*70)
    print("STAGE 4: UNCERTAINTY ESTIMATION")
    print("="*70)
    
    try:
        reg_mean, reg_std, cls_probs, cls_entropy = estimate_uncertainty()
        print("✓ Stage 4 complete")
    except Exception as e:
        print(f"✗ Stage 4 failed: {e}")
        return
    
    # =====================================================
    # PIPELINE COMPLETE
    # =====================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    print("\n" + "="*70)
    print(" DELIVERABLES")
    print("="*70)
    print("\n✓ Trained Models:")
    print("  - Deep Learning (PyTorch)")
    print("  - LightGBM")
    print("  - CatBoost")
    print("  - Stacked Ensemble (XGBoost)")
    
    print("\n✓ Predictions:")
    print("  - Passenger Demand (regression)")
    print("  - Load Factor (regression)")
    print("  - Utilization Status (classification)")
    
    print("\n✓ Uncertainty Estimates:")
    print("  - MC Dropout predictions")
    print("  - Confidence intervals")
    print("  - Prediction entropy")
    
    print("\n✓ Evaluation Metrics:")
    print("  - RMSE, MAE, R² (regression)")
    print("  - Accuracy, F1-score (classification)")
    
    print("\n" + "="*70)
    print(" FUTURE EXTENSION POINTS")
    print("="*70)
    print("\nThis implementation provides trained prediction models.")
    print("Future phases can build upon these models:")
    print("  - Multi-agent decision systems")
    print("  - Fleet optimization (GA, NSGA-II)")
    print("  - Scenario simulation engines")
    print("  - Reinforcement learning schedulers")
    print("  - Real-time decision support")
    
    print("\n" + "="*70)
    print(" SYSTEM READY FOR DEPLOYMENT")
    print("="*70)


if __name__ == "__main__":
    main()
