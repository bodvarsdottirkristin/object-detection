import time
import torch
from torch.amp import autocast

from src.utils.evaluate import EvaluationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, log_interval=10):
    model.train()
    
    # Performance timers
    start_time = time.time()
    batch_start_time = time.time()
    
    # Accumulators for the epoch
    running_loss = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    total_samples = 0
    
    steps = len(dataloader)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type):
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update running loss
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Calculate raw stats for this batch (Binary Classification logic)
        with torch.no_grad():
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            # Simple vector addition for speed
            tp += ((preds == 1) & (targets == 1)).sum().item()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()

        # Interval Logging
        if (i + 1) % log_interval == 0:
            batch_duration = time.time() - batch_start_time
            img_per_sec = (batch_size * log_interval) / batch_duration
            current_loss = running_loss / total_samples
            current_acc = (tp + tn) / total_samples

            logger.info(f"Batch {i+1}/{steps}] ")
            logger.info(f"Loss: {current_loss:.4f} | ")
            logger.info(f"Acc: {current_acc:.2%} | ")
            logger.info(f"Speed: {img_per_sec:.1f} samples/s")
            
            batch_start_time = time.time()

    epoch_duration = time.time() - start_time
    avg_loss = running_loss / total_samples
    accuracy = (tp + tn) / total_samples
    
    # Note: We skip AUC/mAP calculation for training to save time (expensive sorting)
    metrics = EvaluationMetrics(
        accuracy=accuracy,
        auc=0.0,
        map_score=0.0,
        loss=avg_loss,
        tp=tp, tn=tn, fp=fp, fn=fn
    )
    
    return metrics, epoch_duration
