"""Main training script."""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from pathlib import Path
from src.models.pothole_ResNet import PotholeClassifier
from src.datasets.potholes import ProposalDataset
from src.training.train import train_model
from src.training.early_stopping import EarlyStopping

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


if __name__ == "__main__":
    # Configuration (read from environment or use defaults)
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "scratch/proposals"
    SPLITS_PATH = "splits.json"
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    CHECKPOINT_DIR = "models/checkpoints"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    print(f"Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, _ = ProposalDataset.get_dataloaders(
        DATASET_PATH, PROPOSALS_DIR, SPLITS_PATH, batch_size=BATCH_SIZE
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches\n")
    
    # Create model
    print("Creating model...")
    model = PotholeClassifier(num_classes=2, pretrained=True).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=10, mode="max", delta=0.01, verbose=True)
    
    # Train
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        early_stopping=early_stopping
    )
    
    # Save history
    history_dict = {
        'train_loss': history.train_loss,
        'train_acc': history.train_acc,
        'val_loss': history.val_loss,
        'val_acc': history.val_acc
    }
    with open("training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    print("✓ Training complete!")
    print("✓ History saved to training_history.json")