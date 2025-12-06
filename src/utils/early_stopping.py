from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stops training if validation metric does not improve after a given patience."""

    def __init__(self, patience=10, mode="min", delta=0.0, verbose=False):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose

        self.best_metric = None
        self.counter = 0
        self.stop = False
        self.best_epoch = 0

    def __call__(self, current_metric, epoch=None):
        """
        Returns:
            bool: True if this is the best metric so far
        """
        is_best = False

        if self.best_metric is None:
            self.best_metric = current_metric
            is_best = True
            if self.verbose:
                logger.info(f"Epoch {epoch}: Initial best metric: {current_metric:.4f}")

        elif self._is_improvement(current_metric):
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: Metric improved from {self.best_metric:.4f} to {current_metric:.4f}"
                )
            self.best_metric = current_metric
            self.counter = 0
            self.best_epoch = epoch if epoch is not None else 0
            is_best = True

        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: No improvement. Counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping! Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}"
                    )

        return is_best

    def _is_improvement(self, current_metric):
        """Check if current metric is an improvement."""
        if self.mode == "min":
            return current_metric < self.best_metric - self.delta
        else:  # mode == "max"
            return current_metric > self.best_metric + self.delta


if __name__ == "__main__":
    print("Testing EarlyStopping with mode='min' (like loss)")
    print("=" * 60)

    early_stop = EarlyStopping(patience=3, mode="min", delta=0.01, verbose=True)

    # Simulate training epochs with decreasing then plateauing loss
    losses = [1.0, 0.8, 0.7, 0.69, 0.685, 0.68, 0.682]

    for epoch, loss in enumerate(losses, start=1):
        is_best = early_stop(loss, epoch=epoch)
        print(
            f"Epoch {epoch}: Loss={loss:.3f}, Is Best={is_best}, Stop={early_stop.stop}"
        )
        if early_stop.stop:
            break

    print(f"\nBest loss: {early_stop.best_metric:.4f} at epoch {early_stop.best_epoch}")
    print("\n" + "=" * 60)
    print("Testing EarlyStopping with mode='max' (like Dice)")
    print("=" * 60)

    early_stop = EarlyStopping(patience=3, mode="max", delta=0.01, verbose=True)

    # Simulate training epochs with increasing then plateauing Dice
    dice_scores = [0.5, 0.6, 0.7, 0.71, 0.705, 0.70, 0.698]

    for epoch, dice in enumerate(dice_scores, start=1):
        is_best = early_stop(dice, epoch=epoch)
        print(
            f"Epoch {epoch}: Dice={dice:.3f}, Is Best={is_best}, Stop={early_stop.stop}"
        )
        if early_stop.stop:
            break

    print(f"\nBest Dice: {early_stop.best_metric:.4f} at epoch {early_stop.best_epoch}")
