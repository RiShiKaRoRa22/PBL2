"""
Multi-task Deep Learning Model with Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature importance"""
    
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x):
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        # Apply attention
        attended = x * attention_weights
        return attended, attention_weights


class MultiTaskTransportModel(nn.Module):
    """
    Multi-task deep learning model for transport prediction
    
    Architecture:
    - Input layer with feature encoding
    - Attention mechanism for feature importance
    - Shared hidden layers with residual connections
    - Three output heads:
        1. Passenger demand (regression)
        2. Load factor (regression)
        3. Utilization status (classification)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rate=0.3, num_classes=3):
        super(MultiTaskTransportModel, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dims[0])
        
        # Shared hidden layers with residual connections
        self.shared_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.shared_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        
        # Task-specific heads
        final_dim = hidden_dims[-1]
        
        # Regression head 1: Passenger demand
        self.demand_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # Regression head 2: Load factor
        self.load_factor_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # Classification head: Utilization status
        self.utilization_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        # Input projection
        x = self.input_layer(x)
        
        # Apply attention
        x, attention_weights = self.attention(x)
        
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific predictions
        demand_pred = self.demand_head(x)
        load_factor_pred = self.load_factor_head(x)
        utilization_logits = self.utilization_head(x)
        
        # Combine regression outputs
        regression_output = torch.cat([demand_pred, load_factor_pred], dim=1)
        
        if return_attention:
            return regression_output, utilization_logits, attention_weights
        
        return regression_output, utilization_logits
    
    def enable_dropout(self):
        """Enable dropout for MC Dropout inference"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning with learnable weights
    """
    
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # Learnable loss weights (log variance technique)
        self.log_var_demand = nn.Parameter(torch.zeros(1))
        self.log_var_load = nn.Parameter(torch.zeros(1))
        self.log_var_class = nn.Parameter(torch.zeros(1))
    
    def forward(self, reg_pred, reg_true, cls_pred, cls_true):
        # Regression losses
        mse_demand = F.mse_loss(reg_pred[:, 0], reg_true[:, 0])
        mse_load = F.mse_loss(reg_pred[:, 1], reg_true[:, 1])
        
        # Classification loss
        ce_loss = F.cross_entropy(cls_pred, cls_true)
        
        # Weighted combination (uncertainty weighting)
        precision_demand = torch.exp(-self.log_var_demand)
        loss_demand = precision_demand * mse_demand + self.log_var_demand
        
        precision_load = torch.exp(-self.log_var_load)
        loss_load = precision_load * mse_load + self.log_var_load
        
        precision_class = torch.exp(-self.log_var_class)
        loss_class = precision_class * ce_loss + self.log_var_class
        
        total_loss = loss_demand + loss_load + loss_class
        
        return total_loss, {
            'demand_loss': mse_demand.item(),
            'load_loss': mse_load.item(),
            'class_loss': ce_loss.item()
        }
