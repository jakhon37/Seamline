"""Score the app's pre-labels on the held-back images against their true labels. Usage: evaluate.py PROJECT TRUTH OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.studio import rasterize  # noqa: E402

project, truth, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
classes = ["bent", "melt", "missing", "scratch"]
ids = {c: i + 1 for i, c in enumerate(classes)}
inter, union = np.zeros(5), np.zeros(5)
shapes, images, found, total_true = [], [], 0, 0

for path in sorted(truth.glob("*.json")):
    doc = json.loads(path.read_text())
    size = (doc["imageWidth"], doc["imageHeight"])
    true_mask = rasterize(doc["shapes"], size, ids)
    pred_file = project / path.name
    pred_shapes = json.loads(pred_file.read_text())["shapes"] if pred_file.exists() else []
    ai = [s for s in pred_shapes if s.get("flags", {}).get("ai")]
    pred_mask = rasterize(ai, size, ids)
    for c in range(1, 5):
        inter[c] += np.logical_and(true_mask == c, pred_mask == c).sum()
        union[c] += np.logical_or(true_mask == c, pred_mask == c).sum()
    # A true defect counts as found when a predicted shape of its class covers at least half of it
    for t in doc["shapes"]:
        m = rasterize([t], size, ids) > 0
        total_true += 1
        if (pred_mask[m] == ids[t["label"]]).mean() >= 0.5:
            found += 1
    for s in ai:
        m = rasterize([s], size, ids) > 0
        same = true_mask == ids[s["label"]]
        iou = np.logical_and(m, same).sum() / max(np.logical_or(m, same).sum(), 1)
        any_true = np.logical_and(m, true_mask > 0).sum() / max(m.sum(), 1)
        shapes.append({"image": path.stem, "label": s["label"], "score": s.get("score"), "iou": float(iou),
                       "on_a_defect": float(any_true), "area": int(m.sum())})
    images.append({"image": path.stem, "shapes": len(ai), "true": len(doc["shapes"])})

iou_per_class = {c: float(inter[i] / union[i]) if union[i] else None for c, i in ids.items()}
result = {
    "held_back_images": len(images),
    "iou_per_class": iou_per_class,
    "mean_iou": float(np.mean([v for v in iou_per_class.values() if v is not None])),
    "true_defects": total_true,
    "true_defects_found": found,
    "predicted_shapes": len(shapes),
    "images_with_no_prediction": sum(1 for i in images if not i["shapes"]),
    "shapes": shapes,
}
scored = [s for s in shapes if s["score"] is not None]
if len(scored) >= 8:
    scored.sort(key=lambda s: s["score"])
    q = len(scored) // 4
    result["spearman_score_iou"] = float(spearmanr([s["score"] for s in scored], [s["iou"] for s in scored]).statistic)
    result["least_certain_quarter_iou"] = float(np.mean([s["iou"] for s in scored[:q]]))
    result["most_certain_quarter_iou"] = float(np.mean([s["iou"] for s in scored[-q:]]))
    wrong = [i for i, s in enumerate(scored) if s["iou"] < 0.5]
    half = len(scored) // 2
    result["wrong_shapes"] = len(wrong)
    result["wrong_in_least_certain_half"] = sum(1 for i in wrong if i < half)
out.write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "shapes"}, indent=1))
