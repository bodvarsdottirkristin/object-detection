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