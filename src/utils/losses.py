import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for Dense Object Detection."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: (N, C) where C = number of classes. Raw logits (not softmax).
        targets: (N) where each value is 0 <= targets[i] <= C-1.
        """
        # 1. Compute standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # 2. Get the probability of the true class (p_t)
        p_t = torch.exp(-ce_loss)
        
        # 3. Compute the Focal Term
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        # 4. Apply Alpha Weighting
        if self.alpha is not None:
            # Move alpha to same device as targets before indexing
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss

        # 5. Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss