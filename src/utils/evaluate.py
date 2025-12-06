import torch
import numpy as np
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    """Stores only raw counters and base metrics."""
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
        """TP / (TP + FP)"""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self):
        """TP / (TP + FN)"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self):
        """2 * (Precision * Recall) / (Precision + Recall)"""
        p = self.precision
        r = self.recall
        denominator = p + r
        return 2 * (p * r) / denominator if denominator > 0 else 0.0

    @property
    def specificity(self):
        """TN / (TN + FP)"""
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0 else 0.0

    def __str__(self):
        return (f"Loss: {self.loss:.4f} | Acc: {self.accuracy:.2%} | AUC: {self.auc:.4f}\n"
                f"Prec: {self.precision:.4f} | Rec: {self.recall:.4f} | F1: {self.f1_score:.4f}")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    device = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Containers for global metric calculation
    all_targets = []
    all_probs = []
    
    # Raw counters
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass with AMP
            with autocast(device_type=device):
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
            
            running_loss += loss.item() * inputs.size(0)

            # Calculate Probabilities and Predictions (Assuming Binary Classification)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            # Store for Scikit-Learn metrics (AUC/mAP)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update Raw Counters
            tp += ((preds == 1) & (targets == 1)).sum().item()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()
            
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    # Calculate AUC and mAP using sklearn
    try:
        auc = roc_auc_score(all_targets, all_probs)
        map_score = average_precision_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
        map_score = 0.0
        logger.warning("Only one class present in y_true. AUC and mAP are set to 0.0.")

    return EvaluationMetrics(
        accuracy=accuracy,
        auc=auc,
        map_score=map_score,
        loss=avg_loss,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn
    )