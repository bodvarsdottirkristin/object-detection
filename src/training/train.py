"""Main training loop."""
import torch
from dataclasses import dataclass, field
from tqdm import tqdm
from .evaluate import evaluate
from .helpers import save_checkpoint


@dataclass
class TrainingHistory:
    """Training history across epochs."""
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), 100 * correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_dir,
    early_stopping=None,
):
    """
    Train model with validation and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        early_stopping: EarlyStopping object (optional)
        
    Returns:
        TrainingHistory: History of training metrics
    """
    history = TrainingHistory()
    best_val_acc = 0
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        val_loss = val_metrics['loss']
        
        # Log
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"  Val Precision: {val_metrics.get('precision', 0):.4f} | Recall: {val_metrics.get('recall', 0):.4f} | F1: {val_metrics.get('f1', 0):.4f}")
        
        # Store history
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                f"{checkpoint_dir}/best_model.pt",
            )
        
        # Save last model
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            f"{checkpoint_dir}/last_model.pt",
            verbose=False,
        )
        
        # Early stopping
        if early_stopping:
            early_stopping(val_acc, epoch=epoch)
            if early_stopping.stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print()
    
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return history