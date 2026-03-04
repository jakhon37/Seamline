"""
Checkpoint Utilities
"""
import os
import torch
from typing import Optional


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    metrics: dict,
    filepath: str,
    is_distributed: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        filepath: Save path
        is_distributed: Whether using distributed training
    """
    # Get model state (handle DDP wrapper)
    if is_distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    scheduler=None,
    device: torch.device = None
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint info dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }


def save_model(model, filepath: str, is_distributed: bool = False):
    """
    Save model only (for inference).
    
    Args:
        model: Model to save
        filepath: Save path
        is_distributed: Whether using distributed training
    """
    if is_distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model_state, filepath)


class CheckpointManager:
    """Manage multiple checkpoints with top-k saving."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        mode: str = "min"
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.mode = mode
        self.checkpoints = []  # List of (metric, filepath) tuples
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def is_best(self, metric: float) -> bool:
        """Check if metric is the best so far."""
        if not self.checkpoints:
            return True
        
        if self.mode == "min":
            return metric < min(m for m, _ in self.checkpoints)
        else:
            return metric > max(m for m, _ in self.checkpoints)
    
    def should_save(self, metric: float) -> bool:
        """Check if checkpoint should be saved."""
        if len(self.checkpoints) < self.max_checkpoints:
            return True
        
        if self.mode == "min":
            worst = max(m for m, _ in self.checkpoints)
            return metric < worst
        else:
            worst = min(m for m, _ in self.checkpoints)
            return metric > worst
    
    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        loss: float,
        metrics: dict,
        metric_for_best: float,
        is_distributed: bool = False
    ) -> Optional[str]:
        """
        Save checkpoint if it's in top-k.
        
        Returns:
            Saved filepath or None
        """
        if not self.should_save(metric_for_best):
            return None
        
        filename = f"checkpoint_epoch{epoch:03d}_loss{loss:.5f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        save_checkpoint(
            model, optimizer, scheduler,
            epoch, loss, metrics, filepath, is_distributed
        )
        
        self.checkpoints.append((metric_for_best, filepath))
        
        # Remove worst checkpoint if over limit
        if len(self.checkpoints) > self.max_checkpoints:
            if self.mode == "min":
                self.checkpoints.sort(key=lambda x: x[0])
            else:
                self.checkpoints.sort(key=lambda x: x[0], reverse=True)
            
            _, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)
        
        return filepath
