import torch
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth", is_best=False):
    """Save model checkpoint. state should include:
    'model_state_dict', 'optimizer_state_dict', 'epoch', and optionally 'best_map'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    logger.debug(f"Saved checkpoint to {filepath}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """Load model checkpoint and optionally resume optimizer state.
    Returns (epoch, best_map) where best_map may be None if not present.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_map = checkpoint.get("best_map", None)

    logger.info(f"Resumed from epoch {epoch} with best_map={best_map}")
    return epoch, best_map
