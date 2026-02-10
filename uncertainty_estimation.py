"""
Uncertainty Estimation using Monte Carlo Dropout
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from utils.data_loader import load_data, prepare_features_targets, create_dataloaders
from models.deep_learning_model import MultiTaskTransportModel
from models.uncertainty import UncertaintyEstimator


def estimate_uncertainty():
    """Perform uncertainty estimation on test set"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_df, test_df = load_data()
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, feature_names = \
        prepare_features_targets(train_df, test_df)
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    from utils.data_loader import TransportDataset
    
    test_dataset = TransportDataset(X_test, y_test_reg, y_test_cls)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Load trained model
    print("\nLoading trained Deep Learning model...")
    input_dim = X_train.shape[1]
    model = MultiTaskTransportModel(
        input_dim=input_dim,
        hidden_dims=DL_CONFIG['hidden_dims'],
        dropout_rate=DL_CONFIG['dropout_rate'],
        num_classes=3
    ).to(device)
    
    checkpoint = torch.load(MODEL_DIR / 'deep_learning' / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize uncertainty estimator
    estimator = UncertaintyEstimator(
        model, device, 
        n_samples=DL_CONFIG['mc_dropout_samples']
    )
    
    # Perform MC Dropout inference
    print("\n" + "="*60)
    print("UNCERTAINTY ESTIMATION")
    print("="*60)
    
    reg_mean, reg_std, cls_probs, cls_entropy = estimator.predict_with_uncertainty(test_loader)
    
    # Compute confidence intervals
    lower_95, upper_95 = estimator.get_confidence_intervals(reg_mean, reg_std, confidence=0.95)
    
    # Analysis
    print("\n" + "="*60)
    print("UNCERTAINTY ANALYSIS")
    print("="*60)
    
    print("\nRegression Uncertainty (Standard Deviation):")
    print(f"  Passenger Demand:")
    print(f"    Mean: {np.mean(reg_std[:, 0]):.4f}")
    print(f"    Median: {np.median(reg_std[:, 0]):.4f}")
    print(f"    Max: {np.max(reg_std[:, 0]):.4f}")
    
    print(f"  Load Factor:")
    print(f"    Mean: {np.mean(reg_std[:, 1]):.4f}")
    print(f"    Median: {np.median(reg_std[:, 1]):.4f}")
    print(f"    Max: {np.max(reg_std[:, 1]):.4f}")
    
    print("\nClassification Uncertainty (Entropy):")
    print(f"  Mean: {np.mean(cls_entropy):.4f}")
    print(f"  Median: {np.median(cls_entropy):.4f}")
    print(f"  Max: {np.max(cls_entropy):.4f}")
    
    # Identify high uncertainty samples
    high_uncertainty_threshold = np.percentile(reg_std[:, 0], 90)
    high_uncertainty_indices = np.where(reg_std[:, 0] > high_uncertainty_threshold)[0]
    
    print(f"\nHigh Uncertainty Samples (top 10%):")
    print(f"  Count: {len(high_uncertainty_indices)}")
    print(f"  Threshold: {high_uncertainty_threshold:.4f}")
    
    # Save results
    output_dir = RESULTS_DIR / 'uncertainty'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'regression_mean.npy', reg_mean)
    np.save(output_dir / 'regression_std.npy', reg_std)
    np.save(output_dir / 'classification_probs.npy', cls_probs)
    np.save(output_dir / 'classification_entropy.npy', cls_entropy)
    np.save(output_dir / 'confidence_interval_lower.npy', lower_95)
    np.save(output_dir / 'confidence_interval_upper.npy', upper_95)
    
    estimator.save_uncertainty_results(
        reg_mean, reg_std, cls_entropy,
        output_dir / 'uncertainty_summary.json'
    )
    
    print(f"\n✓ Uncertainty estimation complete")
    print(f"Results saved in: {output_dir}")
    
    return reg_mean, reg_std, cls_probs, cls_entropy


if __name__ == "__main__":
    estimate_uncertainty()
