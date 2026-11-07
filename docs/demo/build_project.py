"""VisA pcb1 to a LabelMe project: 70% of each defect type labelled from its true masks, the rest held back.

Usage: build_project.py VISA/pcb1 PROJECT TRUTH
Images with more than one defect type are left out, as their masks don't say which region is which.
"""
import csv, json, random, shutil, sys
from pathlib import Path
import cv2
import numpy as np

src, project, truth = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
project.mkdir(parents=True); truth.mkdir(parents=True)
rows = [r for r in csv.DictReader(open(src / "image_anno.csv")) if r["mask"] and "," not in r["label"]]
random.seed(37)
by_class, held = {}, set()
for r in rows:
    by_class.setdefault(r["label"], []).append(r)
for label, group in by_class.items():
    random.shuffle(group)
    held.update(r["image"] for r in group[: round(len(group) * 0.3)])
counts = {"train": {}, "held": {}}
for r in rows:
    name = "sensor_" + Path(r["image"]).stem
    shutil.copy(src.parent / r["image"], project / f"{name}.jpg")
    mask = cv2.imread(str(src.parent / r["mask"]), 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        if cv2.contourArea(c) < 25:
            continue
        p = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist()
        if len(p) >= 3:
            shapes.append({"label": r["label"], "points": p, "shape_type": "polygon", "flags": {}})
    h, w = mask.shape
    doc = {"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": f"{name}.jpg", "imageData": None, "imageHeight": h, "imageWidth": w}
    side = "held" if r["image"] in held else "train"
    counts[side][r["label"]] = counts[side].get(r["label"], 0) + 1
    if side == "held":
        (truth / f"{name}.json").write_text(json.dumps(doc))
        cv2.imwrite(str(truth / f"{name}.png"), (mask > 0).astype(np.uint8))
    else:
        (project / f"{name}.json").write_text(json.dumps(doc))
print(counts, "held back", sum(counts["held"].values()), "labelled", sum(counts["train"].values()))
