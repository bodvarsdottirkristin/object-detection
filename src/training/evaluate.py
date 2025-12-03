"""Evaluation metrics computation."""
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set with comprehensive metrics.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        dict: Metrics including loss, accuracy, precision, recall, F1
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Eval"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels
    }