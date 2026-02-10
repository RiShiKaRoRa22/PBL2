"""
Monte Carlo Dropout for Uncertainty Estimation
"""
import torch
import numpy as np
from tqdm import tqdm


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using Monte Carlo Dropout
    """
    
    def __init__(self, model, device, n_samples=50):
        """
        Args:
            model: Trained deep learning model
            device: torch device
            n_samples: Number of MC dropout samples
        """
        self.model = model
        self.device = device
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, dataloader):
        """
        Make predictions with uncertainty estimates
        
        Returns:
            reg_mean: Mean regression predictions (n_samples, 2)
            reg_std: Std of regression predictions (n_samples, 2)
            cls_probs: Classification probabilities (n_samples, n_classes)
            cls_entropy: Prediction entropy (n_samples,)
        """
        self.model.eval()
        self.model.enable_dropout()  # Enable dropout for MC sampling
        
        all_reg_preds = []
        all_cls_preds = []
        
        print(f"\nPerforming MC Dropout inference ({self.n_samples} samples)...")
        
        with torch.no_grad():
            for _ in tqdm(range(self.n_samples), desc="MC Samples"):
                batch_reg_preds = []
                batch_cls_preds = []
                
                for features, _, _ in dataloader:
                    features = features.to(self.device)
                    
                    reg_pred, cls_logits = self.model(features)
                    cls_probs = torch.softmax(cls_logits, dim=1)
                    
                    batch_reg_preds.append(reg_pred.cpu().numpy())
                    batch_cls_preds.append(cls_probs.cpu().numpy())
                
                all_reg_preds.append(np.vstack(batch_reg_preds))
                all_cls_preds.append(np.vstack(batch_cls_preds))
        
        # Stack predictions: (n_samples, n_data, n_outputs)
        all_reg_preds = np.array(all_reg_preds)
        all_cls_preds = np.array(all_cls_preds)
        
        # Compute statistics
        reg_mean = np.mean(all_reg_preds, axis=0)
        reg_std = np.std(all_reg_preds, axis=0)
        
        cls_probs_mean = np.mean(all_cls_preds, axis=0)
        cls_entropy = -np.sum(cls_probs_mean * np.log(cls_probs_mean + 1e-10), axis=1)
        
        return reg_mean, reg_std, cls_probs_mean, cls_entropy
    
    def get_confidence_intervals(self, reg_mean, reg_std, confidence=0.95):
        """
        Compute confidence intervals
        
        Args:
            reg_mean: Mean predictions
            reg_std: Standard deviation
            confidence: Confidence level (default 95%)
        
        Returns:
            lower_bound, upper_bound
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * reg_std
        
        lower_bound = reg_mean - margin
        upper_bound = reg_mean + margin
        
        return lower_bound, upper_bound
    
    def save_uncertainty_results(self, reg_mean, reg_std, cls_entropy, output_path):
        """
        Save uncertainty estimation results
        """
        results = {
            'regression_mean': reg_mean.tolist(),
            'regression_std': reg_std.tolist(),
            'classification_entropy': cls_entropy.tolist()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Uncertainty results saved to {output_path}")
