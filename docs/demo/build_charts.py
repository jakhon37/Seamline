"""Draw the README's results: bar charts of IoU by defect type and by certainty, and an example of each defect
type, its true label beside the model's proposals. Pages are drawn in HTML and photographed by headless Chrome,
in the app's colours and font.

Usage: build_charts.py RESULTS.json PROJECT TRUTH OUTDIR
Writes results.webp and examples.webp. Needs Chrome, and app/node_modules for the Inter font.
"""
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from engine.studio import rasterize  # noqa: E402

FONT = REPO / "app/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"
CLASSES = ["bent", "melt", "missing", "scratch"]
# The app's colours for the demo project's classes, in their order
COLOURS = {"bent": (79, 140, 255), "melt": (255, 107, 107), "missing": (62, 207, 142), "scratch": (247, 183, 49)}
NAMES = {"melt": "Melted solder", "scratch": "Scratches", "missing": "Missing sensor", "bent": "Bent pin"}
STYLE = f"""
@font-face {{ font-family: Inter; src: url('{FONT.as_uri()}'); font-weight: 100 900; }}
* {{ margin: 0; box-sizing: border-box; }}
body {{ background: #16181d; color: #e7e9ee; font-family: Inter, sans-serif; }}
.muted {{ color: #8f96a3; }}
"""


def shoot(html: str, width: int, height: int, out: Path, trim: bool = False):
    """The page as Chrome draws it, with a profile of its own and the GPU left alone. Trimmed, it ends where
    its content does, with the page's own margin below."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    with tempfile.TemporaryDirectory() as work:
        page = Path(work) / "page.html"
        page.write_text(f"<!doctype html><meta charset=utf-8><style>{STYLE}</style>{html}")
        png = Path(work) / "page.png"
        subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--use-angle=swiftshader",
             f"--user-data-dir={work}/profile", "--hide-scrollbars", "--allow-file-access-from-files",
             f"--window-size={width},{height}", f"--screenshot={png}", page.as_uri()],
            capture_output=True, timeout=180, check=True,
        )
        image = Image.open(png).convert("RGB")
        if trim:
            rows = np.where((np.asarray(image).astype(int) - (22, 24, 29)).any(axis=2).any(axis=1))[0]
            image = image.crop((0, 0, width, min(height, rows[-1] + 1 + 44)))
        image.save(out, quality=90, method=6)


def bars(title: str, note: str, rows: list) -> str:
    """One chart: bars from 0 to 1, each with its value above and its name, and any detail, below"""
    grid = "".join(
        f'<div style="position:absolute;left:64px;right:0;bottom:{100 + v * 500}px;border-top:1px solid #2e333b">'
        f'<span class="muted" style="position:absolute;right:calc(100% + 14px);top:-12px;font-size:21px">'
        f"{v:g}</span></div>"
        for v in (0, 0.25, 0.5, 0.75, 1)
    )
    columns = "".join(
        f'<div style="flex:1;position:relative;height:100%">'
        f'<div style="position:absolute;left:50%;bottom:100px;width:30px;margin-left:-15px;height:{max(v, 0.004) * 500}px;'
        f'background:#4f8cff;border-radius:3px 3px 0 0"></div>'
        f'<div style="position:absolute;left:0;right:0;bottom:{108 + v * 500}px;text-align:center;font-size:22px;'
        f'font-weight:600">{v:.2f}</div>'
        f'<div style="position:absolute;left:0;right:0;top:calc(100% - 86px);text-align:center;font-size:23px">{name}'
        f'<div class="muted" style="font-size:19px;margin-top:6px">{detail}</div></div></div>'
        for name, detail, v in rows
    )
    return (
        f'<div style="width:940px"><div style="font-size:32px;font-weight:600">{title}</div>'
        f'<div class="muted" style="font-size:22px;margin-top:10px">{note}</div>'
        f'<div style="position:relative;height:630px;margin-top:40px">{grid}'
        f'<div style="position:absolute;left:64px;right:0;top:0;bottom:0;display:flex">{columns}</div></div></div>'
    )


def results_page(results: dict, out: Path):
    by_class = sorted(results["iou_per_class"].items(), key=lambda kv: -(kv[1] or 0))
    left = bars(
        "IoU on unseen images, by defect type",
        f"Over all pixels of each type, on the {results['held_back_images']} images held back from training",
        [(name, "", v or 0) for name, v in by_class],
    )
    quarters = results["iou_by_certainty_quarter"]
    # Whole percents, unless both ends round to the same one
    ranges = [
        f"{round(lo * 100)}–{round(hi * 100)}%" if round(lo * 100) != round(hi * 100)
        else f"{lo * 100:.1f}–{hi * 100:.1f}%"
        for lo, hi in results["score_range"]
    ]
    names = ["Least certain", "Quarter 2", "Quarter 3", "Most certain"]
    rising = all(a <= b for a, b in zip(quarters, quarters[1:]))
    right = bars(
        "Least certain first finds the weak shapes" if rising else "IoU by how certain the model was",
        f"Mean IoU of the {results['predicted_shapes']} predicted shapes, by how certain the model was",
        [(n, r, v) for n, r, v in zip(names, ranges, quarters)],
    )
    shoot(f'<div style="display:flex;gap:120px;padding:44px 50px">{left}{right}</div>', 2000, 850, out)


def draw(image: Image.Image, shapes: list, box: tuple, scale: float) -> str:
    """The crop with the shapes filled and outlined in their class colours, as a data URL"""
    crop = image.crop(box).resize((round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale)), Image.LANCZOS)
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    for s in shapes:
        colour = COLOURS[s["label"]]
        points = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in s["points"]]
        if len(points) >= 3:
            pen.polygon(points, fill=colour + (90,))
            pen.line(points + points[:1], fill=colour + (255,), width=3, joint="curve")
    crop = Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")
    buffer = io.BytesIO()
    crop.save(buffer, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def examples_page(results: dict, project: Path, truth: Path, out: Path):
    """For each type, the held-back image with that type where the model did best, so the examples are better
    than average; a type it never found shows one of its images, found or not"""
    ids = {c: i + 1 for i, c in enumerate(CLASSES)}
    rows, missed = [], []
    for label in ("melt", "scratch", "missing", "bent"):
        best = None
        for doc_file in sorted(truth.glob("*.json")):
            doc = json.loads(doc_file.read_text())
            if not doc["shapes"] or doc["shapes"][0]["label"] != label:
                continue
            pred_file = project / doc_file.name
            pred = json.loads(pred_file.read_text())["shapes"] if pred_file.exists() else []
            pred = [s for s in pred if (s.get("flags") or {}).get("ai")]
            size = (doc["imageWidth"], doc["imageHeight"])
            true_mask = rasterize(doc["shapes"], size, ids) == ids[label]
            pred_mask = rasterize(pred, size, ids) == ids[label]
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = np.logical_and(true_mask, pred_mask).sum() / union if union else 0.0
            if best is None or iou > best[0]:
                best = (iou, doc_file, doc, pred)
        iou, doc_file, doc, pred = best
        # The defect and its surroundings, as wide as the column allows
        points = np.array([p for s in doc["shapes"] + pred for p in s["points"]])
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max(x1 - x0, (y1 - y0) * 1.6) * 1.9
        w = min(max(w, 380), doc["imageWidth"])
        h = w / 1.6
        left = min(max(cx - w / 2, 0), doc["imageWidth"] - w)
        top = min(max(cy - h / 2, 0), doc["imageHeight"] - h)
        box = (round(left), round(top), round(left + w), round(top + h))
        image = Image.open(project / doc["imagePath"]).convert("RGB")
        scale = 614 / (box[2] - box[0])
        regions = len(doc["shapes"])
        caption = f"IoU {iou:.2f}" if pred else "Found nothing"
        if pred and not iou:
            caption = f"{len(pred)} shape{'s' * (len(pred) > 1)} elsewhere"
        rows.append((NAMES[label], draw(image, doc["shapes"], box, scale), f"{regions} region{'s' * (regions > 1)}",
                     draw(image, pred, box, scale), caption))
        if not iou:
            missed.append(NAMES[label].lower())
    found_none = "".join(f" It found no {name} on any of them." for name in missed)
    cells = "".join(
        f'<div style="font-size:31px;font-weight:600;padding-top:30px">{name}</div>'
        f'<div><img src="{true_img}" style="width:614px;border-radius:12px;display:block">'
        f'<div class="muted" style="font-size:24px;margin:16px 0 26px">{true_caption}</div></div>'
        f'<div><img src="{pred_img}" style="width:614px;border-radius:12px;display:block">'
        f'<div class="muted" style="font-size:24px;margin:16px 0 26px">{pred_caption}</div></div>'
        for name, true_img, true_caption, pred_img, pred_caption in rows
    )
    html = (
        '<div style="padding:44px 56px">'
        '<div style="display:grid;grid-template-columns:266px 614px 614px;column-gap:30px">'
        '<div></div><div style="font-size:31px;font-weight:600;margin-bottom:24px">True label</div>'
        f'<div style="font-size:31px;font-weight:600;margin-bottom:24px">Model</div>{cells}</div>'
        '<div class="muted" style="font-size:21px;margin-top:8px">Images the model never saw, from VisA (CC BY 4.0): '
        f"for each defect type, the one it did best on.{found_none}</div></div>"
    )
    shoot(html, 1755, 2400, out, trim=True)


if __name__ == "__main__":
    results_file, project, truth, out = (Path(a) for a in sys.argv[1:5])
    results = json.loads(results_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    results_page(results, out / "results.webp")
    examples_page(results, project, truth, out / "examples.webp")
    print("wrote", out / "results.webp", out / "examples.webp")
