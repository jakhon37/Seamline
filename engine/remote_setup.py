#!/usr/bin/env python3
"""Build the training environment on a GPU machine. Run with the machine's python3.

It sees nothing of the machine's other packages, which can conflict, except links to its
PyTorch and its requirements: that already matches the GPU driver, and is too big to copy to a
full disk.
"""
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "segmentation-models-pytorch",
    "torchmetrics",
    "albumentations",
    "opencv-python-headless",
    "PyYAML",
    "tqdm",
    "tensorboard",
    "scikit-learn",
]
# An environment made before Seamline offered UPerNet and Segformer can have an older smp: it is
# made again, with the newest
CHECK = (
    "import segmentation_models_pytorch as smp, torchmetrics, albumentations, cv2, yaml, torch; "
    "smp.UPerNet, smp.Segformer"
)
LINKED = ["torch", "torchvision"]
# PyTorch 2.11 and later bring most of CUDA through cuda-toolkit's extras, as
# cuda-toolkit[cublas,...]
LINKED_PREFIXES = ("torch", "nvidia-", "cuda-", "triton")


def parse(requirement: str) -> tuple:
    """Name, extras and extra marker of a requirement, such as
    'cuda-toolkit[cublas]==13.2.2; platform_system == "Linux"'."""
    found = re.match(r"\s*([\w.-]+)\s*(?:\[([^\]]*)\])?", requirement)
    extras = {e.strip().lower() for e in (found.group(2) or "").split(",") if e.strip()}
    marker = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", requirement.partition(";")[2])
    return (
        found.group(1).lower().replace("_", "-"),
        extras,
        marker.group(1).lower() if marker else None,
    )


def linked_distributions() -> list:
    """PyTorch and its big requirements, or pip would download them all (CUDA is GBs).

    Only these: pip writes through a link when it upgrades, so anything it might upgrade,
    like numpy, gets its own copy in the environment instead. A requirement asked for with
    extras brings what those extras need too.
    """
    found, walked = {}, {}
    pending = [(name, set()) for name in LINKED]
    while pending:
        name, extras = pending.pop()
        name = name.lower().replace("_", "-")
        if not name.startswith(LINKED_PREFIXES):
            continue
        if name not in found:
            try:
                found[name] = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
        # Its own requirements once, and those of each extra the first time it's asked for
        first = name not in walked
        new = extras - walked.get(name, set())
        if not first and not new:
            continue
        walked[name] = walked.get(name, set()) | extras
        for requirement in found[name].requires or []:
            needed, needed_extras, extra = parse(requirement)
            if (extra is None and first) or (extra is not None and extra in new):
                pending.append((needed, needed_extras))
    return list(found.values())


def numpy_requirement() -> str:
    """PyTorch builds older than 2.3 were made against numpy 1, which has no builds past
    Python 3.12."""
    try:
        major, minor = (
            int(n) for n in re.match(r"(\d+)\.(\d+)", metadata.version("torch")).groups()
        )
    except (metadata.PackageNotFoundError, AttributeError):
        return "numpy"
    return "numpy" if (major, minor) >= (2, 3) else "numpy<2"


def link_roots(dist) -> set:
    """What of a distribution to link: its top-level folders and files, but inside a namespace
    folder that several share, such as nvidia/ or cuda/, its own subfolders, so that the folder
    itself is the environment's own and pip installs nothing through a link into the machine's
    Python."""
    roots = set()
    for file in dist.files or []:
        parts = file.parts
        if parts[0] == "..":
            continue
        top = Path(dist.locate_file(parts[0]))
        shared = (
            len(parts) > 1
            and top.is_dir()
            and not (top / "__init__.py").exists()
            and not parts[0].endswith((".dist-info", ".data"))
        )
        roots.add(parts[:2] if shared else parts[:1])
    return roots


def main(venv: Path):
    python = venv / "bin" / "python"
    if python.exists() and subprocess.call([python, "-c", CHECK], stderr=subprocess.DEVNULL) == 0:
        return
    subprocess.check_call([sys.executable, "-m", "venv", "--clear", venv])
    site = next((venv / "lib").glob("python*/site-packages"))
    for dist in linked_distributions():
        for root in link_roots(dist):
            source = Path(dist.locate_file(Path(*root)))
            target = site.joinpath(*root)
            if source.exists() and not target.exists() and not target.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
    packages = [*PACKAGES, numpy_requirement()]
    subprocess.check_call([python, "-m", "pip", "install", "-q", "--no-cache-dir", *packages])
    subprocess.check_call([python, "-c", CHECK])


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
