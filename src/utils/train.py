import time
import torch
from torch.amp import autocast

from src.utils.evaluate import EvaluationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, log_interval=10):
    """Train the model for one epoch."""
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
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with automatic mixed precision
        with autocast(device_type=device_type):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update running loss
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Calculate confusion matrix statistics
        with torch.no_grad():
            # Get predicted class (0=background, 1=pothole)
            preds = torch.argmax(outputs, dim=1)
            
            # Update confusion matrix counters
            tp += ((preds == 1) & (targets == 1)).sum().item()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()

        # Periodic logging
        if (i + 1) % log_interval == 0:
            batch_duration = time.time() - batch_start_time
            samples_per_sec = (batch_size * log_interval) / batch_duration
            current_loss = running_loss / total_samples
            current_acc = (tp + tn) / total_samples

            logger.info(
                f"Batch {i+1}/{steps} | "
                f"Loss: {current_loss:.4f} | "
                f"Acc: {current_acc:.2%} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
            
            batch_start_time = time.time()

    # Calculate epoch statistics
    epoch_duration = time.time() - start_time
    avg_loss = running_loss / total_samples
    accuracy = (tp + tn) / total_samples
    
    # Create metrics object (AUC/mAP not computed during training for efficiency)
    metrics = EvaluationMetrics(
        accuracy=accuracy,
        auc=0.0,  # Not computed during training
        map_score=0.0,  # Not computed during training
        loss=avg_loss,
        tp=tp, 
        tn=tn, 
        fp=fp, 
        fn=fn
    )
    
    logger.info(f"Epoch completed in {epoch_duration:.2f}s")
    logger.info(f"Train {metrics}")
    
    return metrics, epoch_duration