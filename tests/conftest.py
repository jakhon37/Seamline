"""
Pytest Configuration and Fixtures
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_image():
    """Generate sample image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Generate sample mask."""
    return np.random.randint(0, 2, (224, 224), dtype=np.uint8)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "experiment": {"name": "test_experiment", "seed": 42},
        "data": {
            "base_dir": "/tmp/test_data",
            "dataset_dir": "test_dataset",
            "train_dir": "train",
            "val_dir": "val",
            "test_dir": "test",
            "image_size": 224,
            "num_classes": 2,
            "val_split": 0.1,
            "use_onehot_encoding": False,
            "cache_validation": False,
        },
        "model": {
            "architecture": "Unet",
            "encoder": "resnet18",
            "encoder_weights": "imagenet",
            "activation": "softmax2d",
            "num_classes": 2,
            "gradient_checkpointing": False,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_workers": 0,  # No workers for tests
            "prefetch_factor": None,
            "persistent_workers": False,
            "mixed_precision": False,  # Disabled for CPU tests
            "gradient_accumulation_steps": 1,
            "metrics_on_cpu": True,
            "track_memory": False,
            "enable_profiling": False,
        },
        "loss": {"type": "cross_entropy", "class_weights": None},
        "scheduler": {"type": "reduce_on_plateau", "factor": 0.1, "patience": 5, "min_lr": 1e-7},
        "early_stopping": {"patience": 10, "min_delta": 0.0001},
        "augmentation": {
            "train": {
                "rotate_limit": 45,
                "rotate_p": 0.5,
                "h_flip_p": 0.5,
                "v_flip_p": 0.5,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "brightness_contrast_p": 0.3,
                "gamma_limit": [80, 120],
                "gamma_p": 0.3,
            },
            "normalize": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "max_pixel_value": 255.0,
            },
        },
        "distributed": {"backend": "gloo", "find_unused_parameters": False},
        "logging": {"log_interval": 10, "save_top_k": 3, "tensorboard": False, "wandb": False},
        "paths": {
            "output_dir": "/tmp/outputs",
            "checkpoint_dir": "/tmp/checkpoints",
            "log_dir": "/tmp/logs",
        },
    }


@pytest.fixture
def mock_dataset(tmp_path, sample_image, sample_mask):
    """Create mock dataset directory structure."""
    from PIL import Image

    dataset_dir = tmp_path / "dataset"

    # Create directory structure
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        # Create sample files
        for i in range(5):
            img_path = dataset_dir / split / "images" / f"sample_{i}.png"
            mask_path = dataset_dir / split / "masks" / f"sample_{i}.png"

            Image.fromarray(sample_image).save(img_path)
            Image.fromarray(sample_mask).save(mask_path)

    return dataset_dir


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="function")
def cleanup_cuda():
    """Cleanup CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip GPU tests if no GPU available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
