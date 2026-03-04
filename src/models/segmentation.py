"""
Segmentation Models
"""
import segmentation_models_pytorch as smp


def create_model(
    architecture: str = "Unet",
    encoder: str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    activation: str = "softmax2d"
):
    """
    Create a segmentation model using segmentation_models_pytorch.
    
    Args:
        architecture: Model architecture (Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, FPN, PSPNet, etc.)
        encoder: Encoder backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pretrained weights (imagenet, ssl, swsl, or None)
        num_classes: Number of segmentation classes
        activation: Output activation (softmax2d, sigmoid, None)
        
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
    
    return model


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
