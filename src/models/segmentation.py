"""
Segmentation Models
"""
import inspect

import segmentation_models_pytorch as smp

ARCHITECTURES = {
    name: getattr(smp, name)
    for name in (
        "Unet",
        "UnetPlusPlus",
        "DeepLabV3",
        "DeepLabV3Plus",
        "FPN",
        "PSPNet",
        "PAN",
        "Linknet",
        "MAnet",
        "UPerNet",
        "Segformer",
    )
    # UPerNet and Segformer came with smp 0.4.0; an older smp still builds the others
    if hasattr(smp, name)
}

# The arguments that set each decoder's width, all scaled by decoder_width. Linknet's decoder
# takes its widths from the encoder, so it has none.
DECODER_WIDTHS = {
    "Unet": ("decoder_channels",),
    "UnetPlusPlus": ("decoder_channels",),
    "MAnet": ("decoder_channels",),
    "DeepLabV3": ("decoder_channels",),
    "DeepLabV3Plus": ("decoder_channels",),
    "PAN": ("decoder_channels",),
    "UPerNet": ("decoder_channels",),
    "FPN": ("decoder_pyramid_channels", "decoder_segmentation_channels"),
    "PSPNet": ("psp_out_channels",),
    "Segformer": ("decoder_segmentation_channels",),
}

# Decoders with an attention block: scSE, squeeze and excitation over channels and space
ATTENTION_TYPES = {"Unet": ("scse",), "UnetPlusPlus": ("scse",)}
# scSE squeezes a block's channels 16 to 1, so a block narrower than that has none left
ATTENTION_MIN_CHANNELS = 16

# Encoder depths smp builds for every architecture: the number of stages, each halving the size
ENCODER_DEPTHS = (3, 4, 5)
# Heads that upsample by a fixed factor, so only their own depth gives masks the image's size
FIXED_DEPTH = ("FPN", "PSPNet")

# A model's shape beyond its architecture and encoder, as a config's model section keeps it
SHAPE_SETTINGS = ("encoder_depth", "decoder_width", "decoder_attention_type")


def shape_of(model_cfg: dict) -> dict:
    """The shape settings in a config's model section, for create_model. Configs from before there
    were these settings have none of them, so their models keep smp's defaults, as they had."""
    return {key: model_cfg.get(key) for key in SHAPE_SETTINGS}


def default_argument(architecture: str, name: str):
    """smp's default for one of the architecture's arguments"""
    return inspect.signature(ARCHITECTURES[architecture].__init__).parameters[name].default


def depth_kwargs(architecture: str, encoder_depth: int = None) -> dict:
    """The encoder depth asked for, or nothing for the architecture's own"""
    if encoder_depth is None:
        return {}
    if encoder_depth not in ENCODER_DEPTHS:
        raise ValueError(f"encoder_depth must be one of {ENCODER_DEPTHS}, got {encoder_depth}")
    own = default_argument(architecture, "encoder_depth")
    if architecture in FIXED_DEPTH and encoder_depth != own:
        raise ValueError(
            f"{architecture} gives masks the size of the image only at its own encoder depth, {own}"
        )
    return {"encoder_depth": int(encoder_depth)}


def scaled(channels, width: float):
    if isinstance(channels, tuple):
        return tuple(max(1, round(c * width)) for c in channels)
    return max(1, round(channels * width))


def width_kwargs(architecture: str, depth: int, decoder_width: float = None) -> dict:
    """The decoder widths, smp's scaled by decoder_width, that differ from smp's own"""
    widths = DECODER_WIDTHS.get(architecture, ())
    change = decoder_width is not None and decoder_width != 1
    if change and not widths:
        raise ValueError(f"{architecture}'s decoder has no width to change")
    if change and not decoder_width > 0:
        raise ValueError(f"decoder_width must be positive, got {decoder_width}")
    kwargs = {}
    for name in widths:
        own = default_argument(architecture, name)
        channels = own
        if isinstance(own, (tuple, list)):
            # One width per decoder block, deepest first: a shallower encoder drops the deepest
            # blocks, so each block left keeps its width at its resolution
            channels = tuple(own)[len(own) - depth :]
        if change:
            channels = scaled(channels, decoder_width)
        if channels != own:
            kwargs[name] = channels
    return kwargs


def attention_kwargs(architecture: str, widths: dict, decoder_attention_type: str = None) -> dict:
    """The decoder's attention: it needs its blocks, as widths sets them, wide enough to squeeze"""
    if decoder_attention_type is None:
        return {}
    if decoder_attention_type not in ATTENTION_TYPES.get(architecture, ()):
        raise ValueError(f"{architecture}'s decoder can't use {decoder_attention_type} attention")
    channels = widths.get("decoder_channels") or default_argument(architecture, "decoder_channels")
    if min(channels) < ATTENTION_MIN_CHANNELS:
        raise ValueError(
            f"{decoder_attention_type} attention needs decoder blocks of "
            f"{ATTENTION_MIN_CHANNELS} channels or more, and this decoder's narrowest has "
            f"{min(channels)}"
        )
    return {"decoder_attention_type": decoder_attention_type}


def model_kwargs(
    architecture: str,
    encoder_depth: int = None,
    decoder_width: float = None,
    decoder_attention_type: str = None,
) -> dict:
    """The arguments, beyond smp's defaults, that give the architecture this depth, decoder width
    and attention. None, or a width of 1, keeps smp's default, so a model built without them is
    built as it always was."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. Available: {list(ARCHITECTURES.keys())}"
        )
    kwargs = depth_kwargs(architecture, encoder_depth)
    depth = kwargs.get("encoder_depth", default_argument(architecture, "encoder_depth"))
    kwargs.update(width_kwargs(architecture, depth, decoder_width))
    kwargs.update(attention_kwargs(architecture, kwargs, decoder_attention_type))
    return kwargs


def create_model(
    architecture: str = "Unet",
    encoder: str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    activation: str = "softmax2d",
    gradient_checkpointing: bool = False,
    encoder_depth: int = None,
    decoder_width: float = None,
    decoder_attention_type: str = None,
):
    """
    Create a segmentation model using segmentation_models_pytorch.

    Args:
        architecture: Model architecture (Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, FPN,
            PSPNet, PAN, Linknet, MAnet, UPerNet, Segformer)
        encoder: Encoder backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pretrained weights (imagenet, ssl, swsl, or None)
        num_classes: Number of segmentation classes
        activation: Output activation (softmax2d, sigmoid, None)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        encoder_depth: Encoder stages used, 3 to 5; None for the architecture's default
        decoder_width: Multiplies the decoder's default widths; None or 1 keeps them
        decoder_attention_type: None, or "scse" for Unet and UnetPlusPlus

    Returns:
        Segmentation model
    """
    kwargs = model_kwargs(architecture, encoder_depth, decoder_width, decoder_attention_type)
    model = ARCHITECTURES[architecture](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
        **kwargs,
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
    if "resnet" in encoder_name.lower():
        if hasattr(model.encoder, "layer1"):
            for layer in [
                model.encoder.layer1,
                model.encoder.layer2,
                model.encoder.layer3,
                model.encoder.layer4,
            ]:
                if hasattr(layer, "gradient_checkpointing"):
                    layer.gradient_checkpointing = True
                # For newer PyTorch versions
                elif hasattr(layer, "set_grad_checkpointing"):
                    layer.set_grad_checkpointing(True)

    # EfficientNet encoders
    elif "efficientnet" in encoder_name.lower():
        if hasattr(model.encoder, "blocks"):
            for block in model.encoder.blocks:
                if hasattr(block, "set_grad_checkpointing"):
                    block.set_grad_checkpointing(True)

    # For other encoders, try to enable on the main encoder
    if hasattr(model.encoder, "gradient_checkpointing"):
        model.encoder.gradient_checkpointing = True
    elif hasattr(model.encoder, "set_grad_checkpointing"):
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
        "trainable_params_m": trainable_params / 1e6,
    }
