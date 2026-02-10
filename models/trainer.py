"""
Training utilities for deep learning model
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelTrainer:
    """Trainer for multi-task deep learning model"""
    
    def __init__(self, model, criterion, optimizer, device, model_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_history = {
            'loss': [], 'val_loss': [],
            'demand_loss': [], 'load_loss': [], 'class_loss': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_demand_loss = 0
        total_load_loss = 0
        total_class_loss = 0
        
        for features, targets_reg, targets_cls in train_loader:
            features = features.to(self.device)
            targets_reg = targets_reg.to(self.device)
            targets_cls = targets_cls.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reg_pred, cls_pred = self.model(features)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                reg_pred, targets_reg, cls_pred, targets_cls
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_demand_loss += loss_dict['demand_loss']
            total_load_loss += loss_dict['load_loss']
            total_class_loss += loss_dict['class_loss']
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'demand_loss': total_demand_loss / n_batches,
            'load_loss': total_load_loss / n_batches,
            'class_loss': total_class_loss / n_batches
        }
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, targets_reg, targets_cls in val_loader:
                features = features.to(self.device)
                targets_reg = targets_reg.to(self.device)
                targets_cls = targets_cls.to(self.device)
                
                reg_pred, cls_pred = self.model(features)
                loss, _ = self.criterion(
                    reg_pred, targets_reg, cls_pred, targets_cls
                )
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=15):
        """Full training loop with early stopping"""
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        print("\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_loss)
            self.train_history['demand_loss'].append(train_metrics['demand_loss'])
            self.train_history['load_loss'].append(train_metrics['load_loss'])
            self.train_history['class_loss'].append(train_metrics['class_loss'])
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Demand: {train_metrics['demand_loss']:.4f} | "
                      f"Load: {train_metrics['load_loss']:.4f} | "
                      f"Class: {train_metrics['class_loss']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
            
            # Early stopping
            if early_stopping(val_loss, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best epoch: {early_stopping.best_epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch, val_loss)
        self.save_history()
        
        return self.train_history
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history
        }
        torch.save(checkpoint, self.model_dir / filename)
    
    def save_history(self):
        """Save training history"""
        with open(self.model_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        return checkpoint['epoch'], checkpoint['val_loss']
