"""
Main Training Script for Pothole Classifier
Optimized for NVIDIA L40S GPU (48GB)
"""

import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from pathlib import Path

from src.models.pothole_classifier import PotholeClassifier
from src.datasets.potholes import Potholes
from src.utils.losses import FocalLoss
from src.utils.train import train_one_epoch
from src.utils.evaluate import evaluate_model
from src.utils.early_stopping import EarlyStopping
from src.utils.checkpoints import save_checkpoint
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.transforms import get_train_transforms, get_val_transforms


# Paths
TRAIN_DB_PATH = "src/datasets/proposals/train_db.pkl"
VAL_DB_PATH = "src/datasets/proposals/val_db.pkl"
CHECKPOINT_DIR = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Model Architecture
NUM_CLASSES = 2
PRETRAINED = True
FREEZE_BACKBONE = False  # Fine-tune entire network. If PRETRAINED is False, this won't work.
DROPOUT_P = 0.5

# Training Configuration
NUM_EPOCHS = 100
BATCH_SIZE = 128  # Optimized for L40S (48GB)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# FocalLoss Configuration
FOCAL_ALPHA = [0.25, 0.75]  # Upweight minority class (potholes)
FOCAL_GAMMA = 2.0

# Learning Rate Scheduler (CosineAnnealingWarmRestarts)
LR_T_0 = 10  # First restart at epoch 10
LR_T_MULT = 2  # Double restart period each time
LR_ETA_MIN = 1e-6  # Minimum learning rate

# Early Stopping Configuration
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MODE = "max"  # Maximize mAP
EARLY_STOP_METRIC = "map_score"
EARLY_STOP_DELTA = 0.0

# Data Configuration
IMG_SIZE = (224, 224)
NUM_WORKERS = 8  # Take advantage of multi-core CPU
PIN_MEMORY = True

# Logging
LOG_INTERVAL = 10

# Reproducibility
SEED = 67

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = get_logger(__name__)


def main():
    """Main training function."""
    
    # Set random seed for reproducibility
    set_seed(SEED)
    logger.info(f"Random seed set to {SEED}")
    
    # Log device info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    logger.info("Loading datasets...")

    # Create transforms
    train_transforms = get_train_transforms(img_size=IMG_SIZE)
    val_transforms = get_val_transforms(img_size=IMG_SIZE)
    
    # Create datasets
    train_dataset = Potholes(TRAIN_DB_PATH, transform=train_transforms)
    val_dataset = Potholes(VAL_DB_PATH, transform=val_transforms)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    logger.info("Initializing model...")
    model = PotholeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        dropout_p=DROPOUT_P
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean")
    logger.info(f"Using FocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    logger.info(f"Using AdamW(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_T_0,
        T_mult=LR_T_MULT,
        eta_min=LR_ETA_MIN
    )
    logger.info(f"Using CosineAnnealingWarmRestarts(T_0={LR_T_0}, T_mult={LR_T_MULT}, eta_min={LR_ETA_MIN})")
    
    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda')
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        mode=EARLY_STOP_MODE,
        delta=EARLY_STOP_DELTA,
        verbose=True
    )
    logger.info(f"Early stopping on {EARLY_STOP_METRIC} with patience={EARLY_STOP_PATIENCE}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    best_map = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        logger.info("-" * 80)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Train for one epoch
        train_metrics, train_time = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
            log_interval=LOG_INTERVAL
        )
        
        # Validate
        logger.info("Running validation...")
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        
        # Step the scheduler
        scheduler.step()
        
        # Get metric for early stopping
        if EARLY_STOP_METRIC == "map_score":
            current_metric = val_metrics.map_score
        elif EARLY_STOP_METRIC == "auc":
            current_metric = val_metrics.auc
        elif EARLY_STOP_METRIC == "loss":
            current_metric = val_metrics.loss
        else:
            current_metric = val_metrics.map_score
        
        # Check if this is the best model
        is_best = early_stopping(current_metric, epoch)
        
        # Update best mAP
        if val_metrics.map_score > best_map:
            best_map = val_metrics.map_score
        
        # Save checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_metrics': {
                'loss': train_metrics.loss,
                'accuracy': train_metrics.accuracy,
            },
            'val_metrics': {
                'loss': val_metrics.loss,
                'accuracy': val_metrics.accuracy,
                'auc': val_metrics.auc,
                'map_score': val_metrics.map_score,
                'precision': val_metrics.precision,
                'recall': val_metrics.recall,
                'f1_score': val_metrics.f1_score,
            },
            'best_map': best_map,
        }
        
        save_checkpoint(
            state=checkpoint_state,
            checkpoint_dir=checkpoint_dir,
            filename=f"latest.pth",
            is_best=is_best
        )
        
        # Log epoch summary
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_metrics.loss:.4f} | Train Acc: {train_metrics.accuracy:.2%}")
        logger.info(f"  Val Loss: {val_metrics.loss:.4f} | Val Acc: {val_metrics.accuracy:.2%}")
        logger.info(f"  Val AUC: {val_metrics.auc:.4f} | Val mAP: {val_metrics.map_score:.4f}")
        logger.info(f"  Val Precision: {val_metrics.precision:.4f} | Val Recall: {val_metrics.recall:.4f} | Val F1: {val_metrics.f1_score:.4f}")
        logger.info(f"  Best mAP so far: {best_map:.4f}")
        
        # Check for early stopping
        if early_stopping.stop:
            logger.info(f"\nEarly stopping triggered at epoch {epoch}")
            logger.info(f"Best {EARLY_STOP_METRIC}: {early_stopping.best_metric:.4f} at epoch {early_stopping.best_epoch}")
            break
    

    logger.info("Training completed!")
    logger.info(f"Best validation mAP: {best_map:.4f} at epoch {early_stopping.best_epoch}")
    logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()