"""
Configuration Validation
"""
import os
from typing import Any, Dict


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    # Required sections and their keys
    required_structure = {
        "experiment": ["name", "seed"],
        "data": ["base_dir", "dataset_dir", "image_size", "num_classes"],
        "training": ["epochs", "batch_size", "learning_rate"],
        "model": ["architecture", "encoder", "num_classes"],
        "loss": ["type"],
        "scheduler": ["type"],
    }

    # Check required sections
    for section, keys in required_structure.items():
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")

        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing required config key: '{section}.{key}'")

    # Validate specific values
    _validate_data_config(config["data"])
    _validate_training_config(config["training"])
    _validate_model_config(config["model"])
    _validate_loss_config(config["loss"])

    print("[Config] Validation passed ✓")


def _validate_data_config(data_cfg: Dict[str, Any]) -> None:
    """Validate data configuration."""
    # Expand environment variables in base_dir
    if "base_dir" in data_cfg:
        data_cfg["base_dir"] = os.path.expandvars(data_cfg["base_dir"])

    # Check image size
    if data_cfg["image_size"] <= 0:
        raise ValueError(f"image_size must be positive, got {data_cfg['image_size']}")

    # Check num_classes
    if data_cfg["num_classes"] < 2:
        raise ValueError(f"num_classes must be >= 2, got {data_cfg['num_classes']}")


def _validate_training_config(train_cfg: Dict[str, Any]) -> None:
    """Validate training configuration."""
    # Check epochs
    if train_cfg["epochs"] <= 0:
        raise ValueError(f"epochs must be positive, got {train_cfg['epochs']}")

    # Check batch size
    if train_cfg["batch_size"] <= 0:
        raise ValueError(f"batch_size must be positive, got {train_cfg['batch_size']}")

    # Check learning rate
    if train_cfg["learning_rate"] <= 0:
        raise ValueError(f"learning_rate must be positive, got {train_cfg['learning_rate']}")

    # Check num_workers
    num_workers = train_cfg.get("num_workers", 0)
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    # Warn if num_workers is too low
    if num_workers < 4:
        print(
            f"[Config] Warning: num_workers={num_workers} is low. "
            f"Consider increasing to 8+ for better performance."
        )


def _validate_model_config(model_cfg: Dict[str, Any]) -> None:
    """Validate model configuration."""
    valid_architectures = [
        "Unet",
        "UnetPlusPlus",
        "DeepLabV3",
        "DeepLabV3Plus",
        "FPN",
        "PSPNet",
        "PAN",
        "Linknet",
        "MAnet",
    ]

    if model_cfg["architecture"] not in valid_architectures:
        raise ValueError(
            f"Invalid architecture '{model_cfg['architecture']}'. "
            f"Must be one of: {valid_architectures}"
        )

    # Check num_classes matches data
    # This will be checked in the caller by comparing with data.num_classes


def _validate_loss_config(loss_cfg: Dict[str, Any]) -> None:
    """Validate loss configuration."""
    valid_loss_types = ["focal", "dice", "cross_entropy", "combined", "tversky"]

    if loss_cfg["type"].lower() not in valid_loss_types:
        raise ValueError(
            f"Invalid loss type '{loss_cfg['type']}'. " f"Must be one of: {valid_loss_types}"
        )

    # Validate combined loss parameters
    if loss_cfg["type"].lower() == "combined":
        alpha = loss_cfg.get("combined_alpha", 0.5)
        if not 0 <= alpha <= 1:
            raise ValueError(f"combined_alpha must be in [0, 1], got {alpha}")


def expand_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in config strings.

    Args:
        config: Configuration dictionary

    Returns:
        Config with expanded environment variables
    """
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
        elif isinstance(value, dict):
            config[key] = expand_environment_variables(value)

    return config
