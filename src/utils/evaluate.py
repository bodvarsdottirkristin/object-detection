import torch
import numpy as np
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    def __init__(self, accuracy, auc, map_score, loss, tp, tn, fp, fn):
        self.accuracy = accuracy
        self.auc = auc
        self.map_score = map_score
        self.loss = loss
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    @property
    def precision(self):
        """Precision: TP / (TP + FP)"""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self):
        """Recall (Sensitivity): TP / (TP + FN)"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self):
        """F1 Score: 2 * (Precision * Recall) / (Precision + Recall)"""
        p = self.precision
        r = self.recall
        denominator = p + r
        return 2 * (p * r) / denominator if denominator > 0 else 0.0

    @property
    def specificity(self):
        """Specificity: TN / (TN + FP)"""
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0 else 0.0

    def __str__(self):
        """Readable string representation of metrics"""
        return (f"Loss: {self.loss:.4f} | Acc: {self.accuracy:.2%} | AUC: {self.auc:.4f}\n"
                f"Prec: {self.precision:.4f} | Rec: {self.recall:.4f} | F1: {self.f1_score:.4f}")


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    
    running_loss = 0.0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Containers for AUC and mAP calculation
    all_targets = []
    all_probs = []
    
    # Confusion matrix counters
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass with automatic mixed precision
            with autocast(device_type=device_type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (pothole)
            preds = torch.argmax(outputs, dim=1)  # Predicted class (0 or 1)
            
            # Store for scikit-learn metrics (AUC and mAP)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update confusion matrix counters
            tp += ((preds == 1) & (targets == 1)).sum().item()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()
            
            total_samples += inputs.size(0)

    # Calculate average loss
    avg_loss = running_loss / total_samples
    
    # Calculate accuracy
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    # Calculate AUC and mAP using scikit-learn
    try:
        auc = roc_auc_score(all_targets, all_probs)
        map_score = average_precision_score(all_targets, all_probs)
    except ValueError as e:
        # This can happen if only one class is present in the batch
        auc = 0.0
        map_score = 0.0
        logger.warning(f"Could not calculate AUC/mAP: {e}")

    # Create and return metrics object
    metrics = EvaluationMetrics(
        accuracy=accuracy,
        auc=auc,
        map_score=map_score,
        loss=avg_loss,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn
    )
    
    logger.info(f"Eval {metrics}")
    
    return metrics
