"""
Train Deep Learning Model
"""
import torch
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from utils.data_loader import (
    load_data, prepare_features_targets, 
    split_train_validation, create_dataloaders
)
from utils.metrics import compute_regression_metrics, compute_classification_metrics, print_metrics
from models.deep_learning_model import MultiTaskTransportModel, MultiTaskLoss
from models.trainer import ModelTrainer


def train_dl_model():
    """Train deep learning model"""
    
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare features and targets
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, feature_names = \
        prepare_features_targets(train_df, test_df)
    
    # Split train into train/validation
    X_tr, X_val, y_tr_reg, y_val_reg, y_tr_cls, y_val_cls = \
        split_train_validation(X_train, y_train_reg, y_train_cls, VALIDATION_SPLIT)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_tr, y_tr_reg, y_tr_cls,
        X_val, y_val_reg, y_val_cls,
        batch_size=DL_CONFIG['batch_size']
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = MultiTaskTransportModel(
        input_dim=input_dim,
        hidden_dims=DL_CONFIG['hidden_dims'],
        dropout_rate=DL_CONFIG['dropout_rate'],
        num_classes=3
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=DL_CONFIG['learning_rate'])
    
    # Trainer
    trainer = ModelTrainer(model, criterion, optimizer, device, MODEL_DIR / 'deep_learning')
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        epochs=DL_CONFIG['epochs'],
        early_stopping_patience=DL_CONFIG['early_stopping_patience']
    )
    
    # Load best model for evaluation
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        reg_pred, cls_logits = model(X_test_tensor)
        
        reg_pred = reg_pred.cpu().numpy()
        cls_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()
    
    # Compute metrics
    reg_metrics = compute_regression_metrics(
        y_test_reg, reg_pred,
        target_names=['passenger_demand', 'load_factor']
    )
    cls_metrics = compute_classification_metrics(
        y_test_cls, cls_pred,
        target_name='utilization_status'
    )
    
    # Print results
    print_metrics(reg_metrics, "Deep Learning - Regression")
    print_metrics(cls_metrics, "Deep Learning - Classification")
    
    # Save predictions
    np.save(MODEL_DIR / 'deep_learning' / 'test_predictions_reg.npy', reg_pred)
    np.save(MODEL_DIR / 'deep_learning' / 'test_predictions_cls.npy', cls_pred)
    
    print("\n✓ Deep learning model training complete")
    print(f"Models saved in: {MODEL_DIR / 'deep_learning'}")
    
    return model, reg_pred, cls_pred, reg_metrics, cls_metrics


if __name__ == "__main__":
    train_dl_model()
