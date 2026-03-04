"""
Loss Function Factory
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .focal_loss import FocalLoss
from .combined_loss import CombinedLoss, DiceLoss


def create_loss(config: dict, device: torch.device = None) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place loss function on (not used for weights - they move in forward pass)
        
    Returns:
        Loss function module
    """
    loss_cfg = config['loss']
    loss_type = loss_cfg['type'].lower()
    
    # Class weights - keep on CPU, will be moved to correct device in forward pass
    class_weights = None
    if 'class_weights' in loss_cfg and loss_cfg['class_weights']:
        class_weights = torch.FloatTensor(loss_cfg['class_weights'])
    
    if loss_type == "focal":
        loss_fn = FocalLoss(
            gamma=loss_cfg.get('focal_gamma', 2.0),
            weight=class_weights,
            reduction="mean"
        )
    
    elif loss_type == "dice":
        loss_fn = DiceLoss(mode="multiclass", from_logits=True)
    
    elif loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss_fn.__name__ = "cross_entropy_loss"
    
    elif loss_type == "combined":
        focal_loss = FocalLoss(
            gamma=loss_cfg.get('focal_gamma', 2.0),
            weight=class_weights,
            reduction="mean"
        )
        dice_loss = DiceLoss(mode="multiclass", from_logits=True)
        loss_fn = CombinedLoss(
            focal_loss, dice_loss,
            alpha=loss_cfg.get('combined_alpha', 0.5)
        )
    
    elif loss_type == "tversky":
        loss_fn = smp.losses.TverskyLoss(
            mode="multiclass",
            from_logits=True,
            alpha=0.3,
            beta=0.7
        )
        loss_fn.__name__ = "tversky_loss"
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn
