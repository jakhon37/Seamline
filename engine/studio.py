#!/usr/bin/env python3
"""Backend commands for the desktop app. Each prints JSON or writes JSONL events.

prepare: project folder with LabelMe JSON files -> train/val images and index masks
train:   prepare, write a run config and call scripts/train.py
predict: run a trained model on images and print its polygons as LabelMe shapes
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_project(root: Path) -> dict:
    return json.loads((root / ".myseg" / "project.json").read_text())


def list_images(root: Path) -> list:
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise SystemExit(f"{path.name} can't be read: {error}")


def rasterize(shapes: list, size: tuple, class_ids: dict) -> np.ndarray:
    """Draw shapes into an index mask, in file order, so later shapes win."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        class_id = class_ids.get(shape.get("label"))
        points = [tuple(p) for p in shape.get("points", [])]
        if class_id is None:
            continue
        if shape.get("shape_type") == "rectangle" and len(points) == 2:
            # Rectangles can be drawn in any direction, and PIL needs the top left first
            (x0, y0), (x1, y1) = points
            draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=class_id)
        elif len(points) >= 3:
            draw.polygon(points, fill=class_id)
    return np.array(mask)


def is_val(name: str, val_split: float) -> bool:
    """Stable split: an image stays on the same side as the dataset grows."""
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_split


def prepare(root: Path, val_split: float, tile: int) -> dict:
    project = load_project(root)
    class_ids = {c["name"]: i + 1 for i, c in enumerate(project["classes"])}
    out = root / ".myseg" / "dataset"
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "masks").mkdir(parents=True)

    labelled, unlabelled, tiles = [], 0, {"train": 0, "val": 0}
    pixels = np.zeros(len(class_ids) + 1, dtype=np.int64)
    unknown = set()
    for path in list_images(root):
        doc = read_labels(path.with_suffix(".json"))
        # AI proposals only count once a person has accepted them
        shapes = [s for s in doc.get("shapes", []) if not s.get("flags", {}).get("ai")]
        if not shapes:
            unlabelled += 1
            continue
        unknown |= {s.get("label") for s in shapes} - set(class_ids)
        labelled.append((path, shapes))

    # Small sets can land all on one side; both sides need at least one image
    forced = {}
    if len(labelled) >= 2:
        sides = [is_val(p.stem, val_split) for p, _ in labelled]
        if all(sides):
            forced[labelled[0][0]] = "train"
        elif not any(sides):
            forced[labelled[-1][0]] = "val"

    for path, shapes in labelled:
        split = forced.get(path) or ("val" if is_val(path.stem, val_split) else "train")
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        mask = rasterize(shapes, image.size, class_ids)
        pixels += np.bincount(mask.ravel(), minlength=len(pixels))[: len(pixels)]
        width, height = image.size
        for top in range(0, max(height - tile // 2, 1), tile):
            for left in range(0, max(width - tile // 2, 1), tile):
                box = (left, top, min(left + tile, width), min(top + tile, height))
                piece = mask[box[1] : box[3], box[0] : box[2]]
                # Skip empty tiles of big images, or background swamps the set
                if max(width, height) > tile and not piece.any():
                    continue
                name = f"{path.stem}__{left}_{top}.png"
                image.crop(box).save(out / split / "images" / name)
                Image.fromarray(piece).save(out / split / "masks" / name)
                tiles[split] += 1

    summary = {
        "labelled": len(labelled),
        "unlabelled": unlabelled,
        "train": tiles["train"],
        "val": tiles["val"],
        "unknown_labels": sorted(l for l in unknown if l),
        "class_pixels": {
            name: int(pixels[i]) for name, i in {"background": 0, **class_ids}.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary))
    return summary


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def train(root: Path, options: dict, run: str = None, prepared: bool = False) -> int:
    """Train one run. With prepared=True the dataset was built and uploaded by the app."""
    root = root.resolve()
    run_dir = root / ".myseg" / "runs" / (run or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "options.json").write_text(json.dumps(options, indent=2))
    events = run_dir / "events.jsonl"

    def emit(**event):
        with events.open("a") as f:
            f.write(json.dumps(event) + "\n")

    print(json.dumps({"run": run_dir.name}), flush=True)
    if prepared:
        summary = json.loads((root / ".myseg" / "dataset" / "summary.json").read_text())
    else:
        summary = prepare(root, options.get("val_split", 0.2), options.get("tile", 1024))
    emit(type="prepared", **summary)
    if summary["train"] == 0 or summary["val"] == 0:
        emit(type="error", message="Need labelled images for both train and validation.")
        return 1

    project = load_project(root)
    config = copy.deepcopy(yaml.safe_load((REPO / "configs" / "default.yaml").read_text()))
    num_classes = len(project["classes"]) + 1
    config["experiment"]["name"] = "run"
    config["data"].update(
        base_dir=str(root / ".myseg"),
        dataset_dir="dataset",
        image_size=options["image_size"],
        num_classes=num_classes,
    )
    config["model"].update(
        architecture=options["architecture"],
        encoder=options["encoder"],
        num_classes=num_classes,
        encoder_weights=options.get("encoder_weights", "imagenet"),
    )
    config["training"].update(
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        learning_rate=options["learning_rate"],
        num_workers=options.get("num_workers", 2),
        gradient_accumulation_steps=1,
    )
    if config["training"]["num_workers"] == 0:
        config["training"].update(persistent_workers=False, prefetch_factor=None)
    config["loss"]["type"] = options["loss"]
    config["loss"]["class_weights"] = [1.0] * num_classes
    config["logging"].update(wandb=False, tensorboard=False)
    config["paths"].update(
        output_dir=str(run_dir), checkpoint_dir=str(run_dir), log_dir=str(run_dir)
    )
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    emit(type="start", epochs=options["epochs"], device=device_name())
    command = [
        sys.executable, str(REPO / "scripts" / "train.py"),
        "--config", str(config_path), "--events", str(events),
    ]
    code = subprocess.call(command, cwd=REPO)
    if code != 0:
        emit(type="error", message=f"Training stopped with exit code {code}.")
    return code


def predict_labels(model, transform, image: np.ndarray, tile: int, device: str) -> tuple:
    """Class index and confidence (0-255) per pixel. Big images go tile by tile, as in training."""
    import cv2
    import torch

    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.uint8)
    step = tile if max(height, width) > tile else max(height, width)
    with torch.inference_mode():
        for top in range(0, height, step):
            for left in range(0, width, step):
                piece = image[top : top + step, left : left + step]
                batch = transform(image=piece)["image"].unsqueeze(0).to(device)
                best, index = model(batch).softmax(1).max(1)
                size = (piece.shape[1], piece.shape[0])
                labels[top : top + step, left : left + step] = cv2.resize(
                    index[0].cpu().numpy().astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
                )
                confidence[top : top + step, left : left + step] = cv2.resize(
                    (best[0].cpu().numpy() * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR
                )
    return labels, confidence


def shapes_from_labels(labels: np.ndarray, confidence: np.ndarray, classes: list, min_area: int) -> list:
    """Polygons for each class. Score is the mean confidence inside the shape: use it to rank, not as a probability."""
    import cv2

    shapes = []
    for index, cls in enumerate(classes, start=1):
        contours, _ = cv2.findContours(
            (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            points = cv2.approxPolyDP(contour, epsilon, True)[:, 0].tolist()
            if len(points) < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            inside = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inside, [contour], -1, 1, -1, offset=(-x, -y))
            score = confidence[y : y + h, x : x + w][inside > 0].mean() / 255
            shapes.append(
                {
                    "label": cls["name"],
                    "points": points,
                    "shape_type": "polygon",
                    "group_id": None,
                    "description": "",
                    "flags": {"ai": True},
                    "score": round(float(score), 3),
                }
            )
    return shapes


def predict(root: Path, run: str, images: list, min_area: int):
    """Yield the device once, then one result per image. A bad image doesn't stop the rest."""
    import torch

    from src.data.transforms import get_val_transforms
    from src.models import create_model

    run_dir = root / ".myseg" / "runs" / run
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    checkpoints = list(run_dir.glob("run_*/best_model.pth"))
    if not checkpoints:
        raise SystemExit(f"Run {run} has no saved model yet.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = config["model"]
    model = create_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        encoder_weights=None,
        num_classes=model_cfg["num_classes"],
        activation=None,
    )
    state = torch.load(checkpoints[0], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    size = config["data"]["image_size"]
    normalize = config["augmentation"]["normalize"]
    transform = get_val_transforms(size, size, **normalize)
    tile = json.loads((run_dir / "options.json").read_text()).get("tile", 1024)
    classes = load_project(root)["classes"]
    yield {"device": device}
    for name in images:
        try:
            image = np.array(ImageOps.exif_transpose(Image.open(root / name)).convert("RGB"))
            labels, confidence = predict_labels(model, transform, image, tile, device)
            yield {
                "name": name,
                "shapes": shapes_from_labels(labels, confidence, classes, min_area),
                "size": [image.shape[1], image.shape[0]],
            }
        except Exception as error:
            yield {"name": name, "error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("project", type=Path)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--tile", type=int, default=1024)
    p = sub.add_parser("train")
    p.add_argument("project", type=Path)
    p.add_argument("--options", required=True, help="JSON training options")
    p.add_argument("--run", help="Run name; defaults to the current time")
    p.add_argument("--prepared", action="store_true", help="Use the dataset already in place")
    p = sub.add_parser("predict")
    p.add_argument("project", type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--images", nargs="+", help="Image names; without it, a JSON list on stdin")
    p.add_argument("--min-area", type=int, default=25)
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare(args.project, args.val_split, args.tile)))
    elif args.command == "train":
        sys.exit(train(args.project, json.loads(args.options), args.run, args.prepared))
    else:
        images = args.images or json.load(sys.stdin)
        for result in predict(args.project, args.run, images, args.min_area):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
