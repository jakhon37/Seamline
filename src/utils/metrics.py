"""
Segmentation Metrics
"""
import torch
import torch.nn.functional as F
from torchmetrics import Metric, F1Score, Precision, Recall, JaccardIndex


class SegmentationMetrics:
    """Compute segmentation metrics."""
    
    def __init__(
        self,
        num_classes: int = 2,
        threshold: float = 0.5,
        device: torch.device = None,
        compute_on_cpu: bool = True
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            threshold: Threshold for binary classification
            device: Device to place metrics on (default: CPU for memory efficiency)
            compute_on_cpu: If True, compute metrics on CPU to save GPU memory
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.compute_on_cpu = compute_on_cpu
        # Always use CPU for metrics to free up GPU memory
        self.device = torch.device('cpu') if compute_on_cpu else (device or torch.device('cpu'))
        
        # Always use multiclass for segmentation (even for 2 classes)
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        ).to(self.device)
        
        self.precision = Precision(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        ).to(self.device)
        
        self.recall = Recall(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        ).to(self.device)
        
        self.iou = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        ).to(self.device)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions.
        
        Args:
            preds: Predictions [B, C, H, W] or [B, H, W]
            targets: Ground truth [B, C, H, W] (one-hot) or [B, H, W]
        """
        # Move to CPU if computing on CPU (saves GPU memory)
        if self.compute_on_cpu:
            preds = preds.detach().cpu()
            targets = targets.detach().cpu()
        
        # Convert predictions to class indices (do once)
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
        
        # Convert targets to class indices (do once)
        if targets.dim() == 4:
            targets = torch.argmax(targets, dim=1)
        
        # Flatten once
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        
        # Update all metrics with same tensors (more efficient)
        self.f1.update(preds_flat, targets_flat)
        self.precision.update(preds_flat, targets_flat)
        self.recall.update(preds_flat, targets_flat)
        self.iou.update(preds_flat, targets_flat)
    
    def compute(self) -> dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with metric values
        """
        return {
            "f1": self.f1.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "iou": self.iou.compute().item()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.iou.reset()


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient.
    
    Args:
        preds: Predictions [B, C, H, W]
        targets: Ground truth [B, C, H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    preds = preds.flatten(2)
    targets = targets.flatten(2)
    
    intersection = (preds * targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()
