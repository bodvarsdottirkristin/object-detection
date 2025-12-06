import sys
from pathlib import Path

# Add project root to Python path (needed for batch job execution)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader

from src.models.pothole_classifier import PotholeClassifier
from src.datasets.potholes import Potholes
from src.utils.losses import FocalLoss
from src.utils.evaluate import evaluate_model
from src.utils.checkpoints import load_checkpoint
from src.utils.logger import get_logger
from src.utils.transforms import get_val_transforms


# Paths
TEST_DB_PATH = "src/datasets/proposals/test_db.pkl"
CHECKPOINT_PATH = "checkpoints/20251206_195106/best_model.pth"  # Best model from training
RESULTS_DIR = "results/"

# Model Architecture (must match training)
NUM_CLASSES = 2
DROPOUT_P = 0.5

# Loss Function (must match training)
FOCAL_ALPHA = [0.25, 0.75]
FOCAL_GAMMA = 2.0

# Data Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 512
NUM_WORKERS = 4
PIN_MEMORY = True

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = get_logger(__name__)


def save_results(metrics, results_dir, filename="test_results.json"):
    """Save evaluation results to JSON file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CHECKPOINT_PATH),
        "test_set": str(TEST_DB_PATH),
        "metrics": {
            "loss": float(metrics.loss),
            "accuracy": float(metrics.accuracy),
            "auc": float(metrics.auc),
            "map_score": float(metrics.map_score),
            "precision": float(metrics.precision),
            "recall": float(metrics.recall),
            "f1_score": float(metrics.f1_score),
            "specificity": float(metrics.specificity),
        },
        "confusion_matrix": {
            "true_positives": int(metrics.tp),
            "true_negatives": int(metrics.tn),
            "false_positives": int(metrics.fp),
            "false_negatives": int(metrics.fn),
        }
    }
    
    filepath = results_dir / (results["timestamp"] + "_" + filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def print_results_table(metrics):
    """Print nicely formatted results table."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Main metrics
    logger.info("\n📊 Main Metrics:")
    logger.info(f"  Loss:           {metrics.loss:.4f}")
    logger.info(f"  Accuracy:       {metrics.accuracy:.2%}")
    logger.info(f"  AUC:            {metrics.auc:.4f}")
    logger.info(f"  mAP:            {metrics.map_score:.4f}")
    
    # Classification metrics
    logger.info("\n📈 Classification Metrics:")
    logger.info(f"  Precision:      {metrics.precision:.4f}")
    logger.info(f"  Recall:         {metrics.recall:.4f}")
    logger.info(f"  F1 Score:       {metrics.f1_score:.4f}")
    logger.info(f"  Specificity:    {metrics.specificity:.4f}")
    
    # Confusion matrix
    logger.info("\n🔢 Confusion Matrix:")
    logger.info(f"  True Positives (TP):   {metrics.tp:>8,}")
    logger.info(f"  True Negatives (TN):   {metrics.tn:>8,}")
    logger.info(f"  False Positives (FP):  {metrics.fp:>8,}")
    logger.info(f"  False Negatives (FN):  {metrics.fn:>8,}")
    
    total = metrics.tp + metrics.tn + metrics.fp + metrics.fn
    logger.info(f"  Total Samples:         {total:>8,}")
    
    # Additional insights
    logger.info("\n💡 Insights:")
    logger.info(f"  Correctly classified:  {metrics.tp + metrics.tn:>8,} ({(metrics.tp + metrics.tn)/total:.2%})")
    logger.info(f"  Misclassified:         {metrics.fp + metrics.fn:>8,} ({(metrics.fp + metrics.fn)/total:.2%})")
    logger.info(f"  Pothole detection rate: {metrics.tp/(metrics.tp + metrics.fn):.2%} (Recall)")
    logger.info(f"  Background rejection rate: {metrics.tn/(metrics.tn + metrics.fp):.2%} (Specificity)")
    
    logger.info("=" * 80 + "\n")


def main():
    """Main evaluation function."""
    
    logger.info("=" * 80)
    logger.info("Pothole Classifier - Test Set Evaluation")
    logger.info("=" * 80)
    
    # Log device info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")

    # Load test dataset    
    logger.info(f"\nLoading test dataset from: {TEST_DB_PATH}")
    
    # Create transforms (no augmentation for test)
    test_transforms = get_val_transforms(img_size=IMG_SIZE)
    
    # Create test dataset
    test_dataset = Potholes(TEST_DB_PATH, transform=test_transforms)
    logger.info(f"Test samples: {len(test_dataset):,}")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle test set
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Load model checkpoint
    logger.info(f"\nLoading model from: {CHECKPOINT_PATH}")
    
    # Initialize model
    model = PotholeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,  # We're loading trained weights
        freeze_backbone=False,
        dropout_p=DROPOUT_P
    )
    
    # Load checkpoint
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please ensure training has completed and best_model.pth exists.")
        return
    
    epoch, best_map = load_checkpoint(checkpoint_path, model, device=DEVICE)
    logger.info(f"Loaded model from epoch {epoch}")
    if best_map is not None:
        logger.info(f"Model's validation mAP was: {best_map:.4f}")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Use same loss as training for consistency
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean")
    logger.info(f"Using FocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE
    )
    
    # Print formatted results
    print_results_table(test_metrics)
    
    # Save results to JSON
    results_file = save_results(test_metrics, RESULTS_DIR)
    logger.info(f"Evaluation complete. Results saved to: {results_file}")


if __name__ == "__main__":
    main()