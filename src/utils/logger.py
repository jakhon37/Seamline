"""
Logging Utilities
"""
import json
import logging
import os
import sys
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


def setup_logger(
    name: str, log_dir: str = None, level: int = logging.INFO, rank: int = 0
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        rank: Process rank (only rank 0 logs to console)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only for main process)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """Track and log metrics during training."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
    ):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
            self.writer = SummaryWriter(log_dir=tb_dir)
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not installed, skipping TB logging")

        # Weights & Biases
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or "semantic_segmentation", name=experiment_name
                )
            except ImportError:
                print("wandb not installed, skipping W&B logging")

        # History for plotting
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "metrics": [],
        }

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

        if self.wandb_run:
            import wandb

            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)

        if self.wandb_run:
            import wandb

            wandb.log({f"{main_tag}/{k}": v for k, v in values.items()}, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        metrics: dict = None,
    ):
        """Log epoch statistics."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["learning_rate"].append(learning_rate)
        self.history["metrics"].append(metrics or {})

        # Log to TensorBoard
        self.log_scalar("loss/train", train_loss, epoch)
        self.log_scalar("loss/val", val_loss, epoch)
        self.log_scalar("learning_rate", learning_rate, epoch)

        if metrics:
            for name, value in metrics.items():
                self.log_scalar(f"metrics/{name}", value, epoch)

        # Save history and plots
        self.save_history()
        self.save_plots()

    def save_history(self, filepath: str = None):
        """Save training history to JSON."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_history.json")

        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def save_plots(self, plot_dir: str = None):
        """Save training metric plots as images."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if plot_dir is None:
            plot_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        epochs = self.history["epoch"]
        if len(epochs) < 2:
            return

        # Plot 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax.plot(epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training & Validation Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add current values annotation
        ax.annotate(
            f'Train: {self.history["train_loss"][-1]:.5f}',
            xy=(epochs[-1], self.history["train_loss"][-1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )
        ax.annotate(
            f'Val: {self.history["val_loss"][-1]:.5f}',
            xy=(epochs[-1], self.history["val_loss"][-1]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "loss_curves.png"), dpi=150)
        plt.close(fig)

        # Plot 2: Metrics (F1, IoU, etc.)
        if self.history["metrics"] and self.history["metrics"][0]:
            metric_names = list(self.history["metrics"][0].keys())
            n_metrics = len(metric_names)

            if n_metrics > 0:
                fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
                if n_metrics == 1:
                    axes = [axes]

                for ax, metric_name in zip(axes, metric_names):
                    values = [m.get(metric_name, 0) for m in self.history["metrics"]]
                    ax.plot(epochs, values, "g-", linewidth=2, marker="o", markersize=3)
                    ax.set_xlabel("Epoch", fontsize=12)
                    ax.set_ylabel(metric_name.upper(), fontsize=12)
                    ax.set_title(f"{metric_name.upper()} over Epochs", fontsize=14)
                    ax.grid(True, alpha=0.3)

                    # Annotate current value
                    if values:
                        ax.annotate(
                            f"{values[-1]:.4f}",
                            xy=(epochs[-1], values[-1]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=10,
                        )

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "metrics.png"), dpi=150)
                plt.close(fig)

        # Plot 3: Learning Rate
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, self.history["learning_rate"], "purple", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "learning_rate.png"), dpi=150)
        plt.close(fig)

        # Plot 4: Combined summary plot
        fig = plt.figure(figsize=(14, 10))

        # Loss subplot
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train", linewidth=2)
        ax1.plot(epochs, self.history["val_loss"], "r-", label="Val", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1 subplot
        if self.history["metrics"] and "f1" in self.history["metrics"][0]:
            ax2 = fig.add_subplot(2, 2, 2)
            f1_values = [m.get("f1", 0) for m in self.history["metrics"]]
            ax2.plot(epochs, f1_values, "g-", linewidth=2, marker="o", markersize=3)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("F1 Score")
            ax2.set_title(f"F1 Score (Best: {max(f1_values):.4f})")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])

        # IoU subplot
        if self.history["metrics"] and "iou" in self.history["metrics"][0]:
            ax3 = fig.add_subplot(2, 2, 3)
            iou_values = [m.get("iou", 0) for m in self.history["metrics"]]
            ax3.plot(epochs, iou_values, "orange", linewidth=2, marker="o", markersize=3)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("IoU")
            ax3.set_title(f"IoU (Best: {max(iou_values):.4f})")
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])

        # LR subplot
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(epochs, self.history["learning_rate"], "purple", linewidth=2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate")
        ax4.set_yscale("log")
        ax4.grid(True, alpha=0.3)

        # Add summary text
        best_val_loss = min(self.history["val_loss"])
        best_epoch = self.history["val_loss"].index(best_val_loss)
        fig.suptitle(
            f"Training Summary - Epoch {epochs[-1]} | Best Val Loss: {best_val_loss:.5f} @ Epoch {best_epoch}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_summary.png"), dpi=150)
        plt.close(fig)

    def close(self):
        """Close all loggers."""
        if self.writer:
            self.writer.close()

        if self.wandb_run:
            import wandb

            wandb.finish()


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
