"""
Focal Loss Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = "mean",
        ignore_index: int = -100
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            weight: Class weights tensor (will be stored as list for device safety)
            reduction: Reduction method (mean, sum, none)
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.gamma = gamma
        # Store weights as a list to avoid device issues in DDP
        self._weight_values = weight.tolist() if weight is not None else None
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.__name__ = "focal_loss"
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [B, C, H, W]
            targets: Ground truth [B, C, H, W] (one-hot) or [B, H, W] (indices)
            
        Returns:
            Focal loss value
        """
        # Handle one-hot encoded targets
        if targets.dim() == 4 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        
        # Create weight tensor on the same device as inputs
        weight = None
        if self._weight_values is not None:
            weight = torch.tensor(self._weight_values, device=inputs.device, dtype=inputs.dtype)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
