#!/usr/bin/env python3
"""Build the training environment on a GPU machine. Run with the machine's python3.

It sees nothing of the machine's other packages, which can conflict, except links to its
PyTorch and its requirements: that already matches the GPU driver and is too big to copy to a full disk.
"""
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "segmentation-models-pytorch", "torchmetrics", "albumentations",
    "opencv-python-headless", "PyYAML", "tqdm", "tensorboard", "scikit-learn", "numpy<2",
]
CHECK = "import segmentation_models_pytorch, torchmetrics, albumentations, cv2, yaml, torch"
LINKED = ["torch", "torchvision"]
LINKED_PREFIXES = ("torch", "nvidia-", "triton")


def linked_distributions() -> list:
    """PyTorch and its big requirements, or pip would download them all (CUDA is GBs).

    Only these: pip writes through a link when it upgrades, so anything it might upgrade,
    like numpy, gets its own copy in the environment instead.
    """
    found, pending = {}, list(LINKED)
    while pending:
        name = pending.pop().lower().replace("_", "-")
        if name in found or not name.startswith(LINKED_PREFIXES):
            continue
        try:
            found[name] = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for requirement in found[name].requires or []:
            if "extra ==" not in requirement:
                pending.append(re.match(r"[\w.-]+", requirement).group())
    return list(found.values())


def main(venv: Path):
    python = venv / "bin" / "python"
    if python.exists() and subprocess.call([python, "-c", CHECK], stderr=subprocess.DEVNULL) == 0:
        return
    subprocess.check_call([sys.executable, "-m", "venv", "--clear", venv])
    site = next((venv / "lib").glob("python*/site-packages"))
    for dist in linked_distributions():
        for root in {f.parts[0] for f in dist.files or [] if f.parts[0] != ".."}:
            source = Path(dist.locate_file(root))
            target = site / root
            if source.exists() and not target.exists():
                target.symlink_to(source)
    subprocess.check_call([python, "-m", "pip", "install", "-q", "--no-cache-dir", *PACKAGES])
    subprocess.check_call([python, "-c", CHECK])


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
