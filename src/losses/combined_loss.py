"""
Combined Loss Implementation
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Combined loss function (e.g., Focal + Dice).
    """
    
    def __init__(
        self,
        primary_loss: nn.Module,
        secondary_loss: nn.Module,
        alpha: float = 0.5
    ):
        """
        Args:
            primary_loss: First loss function
            secondary_loss: Second loss function
            alpha: Weight for primary loss (1-alpha for secondary)
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.__name__ = f"combined_{primary_loss.__name__}_{secondary_loss.__name__}"
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [B, C, H, W]
            targets: Ground truth [B, C, H, W] or [B, H, W]
            
        Returns:
            Combined loss value
        """
        loss1 = self.primary_loss(inputs, targets)
        loss2 = self.secondary_loss(inputs, targets)
        return self.alpha * loss1 + (1 - self.alpha) * loss2


class DiceLoss(nn.Module):
    """
    Dice Loss wrapper for multi-class segmentation.
    """
    
    def __init__(self, mode: str = "multiclass", from_logits: bool = True):
        """
        Args:
            mode: Loss mode (binary, multiclass, multilabel)
            from_logits: Whether inputs are logits
        """
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=mode, from_logits=from_logits)
        self.__name__ = "dice_loss"
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handle one-hot encoded targets
        if targets.dim() == 4 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return self.dice(inputs, targets)
