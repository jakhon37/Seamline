#!/usr/bin/env python3
"""
Export Model to ONNX Format

Usage:
    python export_onnx.py --checkpoint best_model.pth --output model.onnx
"""
import argparse
import inspect
import os
import sys

import onnx
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 12,
    dynamic_batch: bool = True,
    device: torch.device = None,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model
        output_path: Output ONNX file path
        input_size: Input tensor size (B, C, H, W)
        opset_version: ONNX opset version
        dynamic_batch: Whether to enable dynamic batch size
        device: Device for export
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)

    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Newer PyTorch defaults to an exporter that needs onnxscript and treats
    # opset_version and dynamic_axes differently, so keep the TorchScript exporter.
    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ ONNX export successful: {output_path}")
    print(f"  Input shape: {input_size}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")


def main(args):
    """Main export function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Default model config
    architecture = "Unet"
    encoder = "resnet18"
    num_classes = 2
    activation = "softmax2d"
    image_size = 224

    # Override with config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        model_cfg = config["model"]
        architecture = model_cfg["architecture"]
        encoder = model_cfg["encoder"]
        num_classes = model_cfg["num_classes"]
        activation = model_cfg["activation"]
        image_size = config["data"]["image_size"]

    # Override image size if specified
    if args.image_size:
        image_size = args.image_size

    # Create model
    model = create_model(
        architecture=architecture,
        encoder=encoder,
        encoder_weights=None,
        num_classes=num_classes,
        activation=activation,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint: {args.checkpoint}")

    # Export
    export_to_onnx(
        model,
        args.output,
        input_size=(1, 3, image_size, image_size),
        opset_version=args.opset,
        dynamic_batch=not args.static_batch,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument(
        "--image-size", type=int, default=None, help="Input image size (overrides config)"
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--static-batch", action="store_true", help="Use static batch size (default: dynamic)"
    )

    args = parser.parse_args()
    main(args)
