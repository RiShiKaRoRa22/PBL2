"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *


class TransportDataset(Dataset):
    """PyTorch Dataset for transport data"""
    
    def __init__(self, features, targets_reg, targets_cls):
        self.features = torch.FloatTensor(features)
        self.targets_reg = torch.FloatTensor(targets_reg)
        self.targets_cls = torch.LongTensor(targets_cls)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets_reg[idx],
            self.targets_cls[idx]
        )


def load_data():
    """
    Load and prepare train/test datasets
    
    Returns:
        train_df, test_df, feature_names, label_encoders
    """
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Handle missing values
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    return train_df, test_df


def prepare_features_targets(train_df, test_df):
    """
    Separate features and targets, handle encoding
    
    Returns:
        X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, feature_names
    """
    # Identify available features
    all_features = [col for col in train_df.columns 
                   if col not in ['passenger_demand', 'load_factor', 
                                 'utilization_encoded', 'utilization_status', 'date']]
    
    print(f"\nAvailable features: {len(all_features)}")
    print(f"Features: {all_features}")
    
    # Extract features
    X_train = train_df[all_features].values
    X_test = test_df[all_features].values
    
    # Extract regression targets
    y_train_reg = train_df[['passenger_demand', 'load_factor']].values
    y_test_reg = test_df[['passenger_demand', 'load_factor']].values
    
    # Extract classification target
    y_train_cls = train_df['utilization_encoded'].values.astype(int)
    y_test_cls = test_df['utilization_encoded'].values.astype(int)
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train_reg shape: {y_train_reg.shape}")
    print(f"y_train_cls shape: {y_train_cls.shape}")
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, all_features


def create_dataloaders(X_train, y_train_reg, y_train_cls, 
                       X_val, y_val_reg, y_val_cls, 
                       batch_size=256):
    """
    Create PyTorch DataLoaders
    """
    train_dataset = TransportDataset(X_train, y_train_reg, y_train_cls)
    val_dataset = TransportDataset(X_val, y_val_reg, y_val_cls)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def split_train_validation(X_train, y_train_reg, y_train_cls, val_split=0.2):
    """
    Split training data into train and validation (chronological)
    """
    split_idx = int(len(X_train) * (1 - val_split))
    
    X_tr = X_train[:split_idx]
    X_val = X_train[split_idx:]
    
    y_tr_reg = y_train_reg[:split_idx]
    y_val_reg = y_train_reg[split_idx:]
    
    y_tr_cls = y_train_cls[:split_idx]
    y_val_cls = y_train_cls[split_idx:]
    
    print(f"\nTrain split: {X_tr.shape[0]} samples")
    print(f"Validation split: {X_val.shape[0]} samples")
    
    return X_tr, X_val, y_tr_reg, y_val_reg, y_tr_cls, y_val_cls
