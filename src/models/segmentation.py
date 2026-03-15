"""
Segmentation Models
"""
import segmentation_models_pytorch as smp
import torch


def create_model(
    architecture: str = "Unet",
    encoder: str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    activation: str = "softmax2d",
    gradient_checkpointing: bool = False
):
    """
    Create a segmentation model using segmentation_models_pytorch.
    
    Args:
        architecture: Model architecture (Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, FPN, PSPNet, etc.)
        encoder: Encoder backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pretrained weights (imagenet, ssl, swsl, or None)
        num_classes: Number of segmentation classes
        activation: Output activation (softmax2d, sigmoid, None)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency (20-30% memory savings)
        
    Returns:
        Segmentation model
    """
    model_classes = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
        "PAN": smp.PAN,
        "Linknet": smp.Linknet,
        "MAnet": smp.MAnet,
    }
    
    if architecture not in model_classes:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(model_classes.keys())}")
    
    model_class = model_classes[architecture]
    
    model = model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation
    )
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model, encoder)
    
    return model


def _enable_gradient_checkpointing(model, encoder_name: str):
    """
    Enable gradient checkpointing for supported encoders.
    
    Args:
        model: Segmentation model
        encoder_name: Name of the encoder
    """
    # ResNet encoders
    if 'resnet' in encoder_name.lower():
        if hasattr(model.encoder, 'layer1'):
            for layer in [model.encoder.layer1, model.encoder.layer2, 
                         model.encoder.layer3, model.encoder.layer4]:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
                # For newer PyTorch versions
                elif hasattr(layer, 'set_grad_checkpointing'):
                    layer.set_grad_checkpointing(True)
    
    # EfficientNet encoders
    elif 'efficientnet' in encoder_name.lower():
        if hasattr(model.encoder, 'blocks'):
            for block in model.encoder.blocks:
                if hasattr(block, 'set_grad_checkpointing'):
                    block.set_grad_checkpointing(True)
    
    # For other encoders, try to enable on the main encoder
    if hasattr(model.encoder, 'gradient_checkpointing'):
        model.encoder.gradient_checkpointing = True
    elif hasattr(model.encoder, 'set_grad_checkpointing'):
        model.encoder.set_grad_checkpointing(True)
    
    print(f"[Model] Gradient checkpointing enabled for {encoder_name}")



def get_model_info(model) -> dict:
    """
    Get model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6
    }
