"""
Evaluation metrics for regression and classification
"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)


def compute_regression_metrics(y_true, y_pred, target_names=None):
    """
    Compute regression metrics for multiple targets
    
    Args:
        y_true: Ground truth (n_samples, n_targets)
        y_pred: Predictions (n_samples, n_targets)
        target_names: List of target names
    
    Returns:
        Dictionary of metrics
    """
    if target_names is None:
        target_names = [f"target_{i}" for i in range(y_true.shape[1])]
    
    metrics = {}
    
    for i, name in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        
        metrics[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    
    return metrics


def compute_classification_metrics(y_true, y_pred, target_name="classification"):
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_name: Name of the classification target
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    metrics = {
        target_name: {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "classification_report": classification_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }
    }
    
    return metrics


def print_metrics(metrics_dict, model_name="Model"):
    """
    Pretty print metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    
    for target, metrics in metrics_dict.items():
        print(f"\n{target.upper()}:")
        for metric_name, value in metrics.items():
            if metric_name not in ["classification_report", "confusion_matrix"]:
                print(f"  {metric_name}: {value:.4f}")
            elif metric_name == "classification_report":
                print(f"\n{value}")
