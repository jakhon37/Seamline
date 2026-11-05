"""End-to-end test: train, resume, evaluate, infer and export on synthetic data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(*args):
    """Run a repo script and fail with its output if it exits non-zero."""
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert (
        result.returncode == 0
    ), f"{args[0]} failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Small dataset of noise images with a bright square, and a config that fits it."""
    root = tmp_path_factory.mktemp("pipeline")
    rng = np.random.default_rng(0)
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        images = root / "dataset" / split / "images"
        masks = root / "dataset" / split / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for i in range(count):
            image = rng.integers(0, 80, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            x, y = rng.integers(5, 35, 2)
            image[y : y + 20, x : x + 20] = 220
            mask[y : y + 20, x : x + 20] = 1
            Image.fromarray(image).save(images / f"{i:03d}.png")
            Image.fromarray(mask).save(masks / f"{i:03d}.png")

    config = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    config["data"].update(base_dir=str(root), image_size=64)
    config["model"]["encoder_weights"] = None
    config["training"].update(
        epochs=2,
        batch_size=4,
        num_workers=0,
        persistent_workers=False,
        mixed_precision=False,
        gradient_accumulation_steps=1,
    )
    config["logging"].update(wandb=False, tensorboard=False)
    config["paths"] = {
        "output_dir": str(root / "outputs"),
        "checkpoint_dir": str(root / "checkpoints"),
        "log_dir": str(root / "logs"),
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return root, str(config_path)


@pytest.mark.integration
def test_full_pipeline(workspace):
    root, config = workspace

    run_script("scripts/train.py", "--config", config)
    best = next((root / "checkpoints").rglob("best_model.pth"))
    full = sorted((root / "checkpoints").rglob("checkpoint_epoch*.pth"))[0]

    run_script("scripts/train.py", "--config", config, "--resume", str(full))
    run_script("scripts/evaluate.py", "--config", config, "--checkpoint", str(best))

    image = root / "dataset" / "test" / "images" / "000.png"
    single = root / "single.png"
    run_script(
        "scripts/inference.py",
        "--checkpoint",
        str(best),
        "--config",
        config,
        "--input",
        str(image),
        "--output",
        str(single),
    )
    assert np.array(Image.open(single)).shape == (64, 64)

    chunked = root / "chunked.png"
    run_script(
        "scripts/inference.py",
        "--checkpoint",
        str(best),
        "--config",
        config,
        "--input",
        str(image),
        "--output",
        str(chunked),
        "--chunk",
        "--chunk-size",
        "32",
        "--overlap",
        "8",
    )
    assert np.array(Image.open(chunked)).shape == (64, 64)

    onnx_path = root / "model.onnx"
    run_script(
        "scripts/export_onnx.py",
        "--checkpoint",
        str(best),
        "--config",
        config,
        "--output",
        str(onnx_path),
    )
    assert onnx_path.stat().st_size > 0
