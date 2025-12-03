
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, path, verbose=True):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Path to save checkpoint
        verbose: Print message if True
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)
    
    if verbose:
        print(f"  ✓ Best model saved: {path}")
    else:
        print(f"  Last model saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint
        device: Device to load on
        
    Returns:
        tuple: (epoch, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']