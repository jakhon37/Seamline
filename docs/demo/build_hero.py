"""Build the README banner: images from three public datasets, unlabelled on the light side of the seam and
labelled on the dark side, with the app's training on a channel along the seam between them. The channel's
light edge is the logo's S, with its handles and shadow.

Shapes are the datasets' own labels, except the tiles marked AI, which are the demo model's proposals as the app
saved them (banner-proposals.json); the training tiles show that model's run (banner-training.json):
- VisA (CC BY 4.0): foods and circuit boards, defect masks
- MVTec AD (CC BY-NC-SA 4.0, so the banner files are too): materials and parts, defect masks
- D-Fire (CC0; its PublicDataset images are left out): fire and smoke, boxes

Writes banner.webp, banner-animated.webp and banner-phone.webp. Needs Chrome, and app/node_modules for the
Inter font.

Usage: python docs/demo/build_hero.py VisA mvtec-ad d-fire docs/media
"""
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
FONT = REPO / "app/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"
W = 2000
TILE_PX = 488
# The logo's two sides: paper for the data, ink for training
PAPER = (243, 245, 249)
EDGE = (218, 223, 231)
BACKDROP = (27, 31, 40)
INK = (20, 22, 26)
# The logo's seam, from docs/brand/seamline-mark.svg
SEAM = [(322, -24), (230, 18), (148, 72), (148, 150), (148, 216), (200, 236), (256, 256),
        (312, 276), (364, 296), (364, 362), (364, 440), (282, 494), (190, 536)]
HANDLES = [(148, 150), (256, 256), (364, 362)]

# Dataset labels, shortened for chips: (words in the label, short name), first match wins
SHORT = [("chunk of wax", "wax missing"), ("melded", "wax melt"), ("contamination", "dirt"), ("imprint", "print"),
         ("pill_type", "mix-up"), ("swap", "swap"), ("poke", "poke"), ("split", "split"), ("squeez", "squeeze"),
         ("fabric", "fabric"), ("manipulated", "damage"), ("gray_stroke", "stroke"), ("flip", "flipped"),
         ("defective", "defect"), ("broken", "broken"), ("burnt", "burnt"), ("bubble", "bubble"), ("hole", "hole"),
         ("stuck", "stuck"), ("crack", "crack"), ("chip", "chip"), ("breakage", "breakage"), ("missing", "missing"),
         ("spot", "spot"), ("scratch", "scratch"), ("partical", "particle"), ("damage", "damage"), ("wick", "wick"),
         ("extra", "extra"), ("wrong place", "misplaced"), ("misplaced", "misplaced"), ("melt", "melt"),
         ("bent", "bent"), ("cut", "cut"), ("glue", "glue"), ("oil", "oil"), ("liquid", "liquid"), ("fold", "fold"),
         ("thread", "thread"), ("rough", "rough"), ("color", "colour"), ("print", "print")]
RED, GREEN, BLUE, AMBER = (255, 92, 108), (52, 211, 153), (79, 140, 255), (251, 191, 36)
ORANGE, PINK, VIOLET, CYAN, SLATE = (255, 138, 76), (244, 114, 182), (167, 139, 250), (34, 211, 238), (170, 184, 204)
COLOURS = {RED: "melt wax_melt", GREEN: "missing wax_missing", BLUE: "bent flipped swap misplaced mix-up",
           AMBER: "scratch thread fold rough fabric", ORANGE: "burnt fire",
           PINK: "breakage chip crack cut broken split squeeze damage defect",
           VIOLET: "spot particle colour stroke print oil liquid glue dirt", SLATE: "smoke"}
COLOUR = {name.replace("_", " "): rgb for rgb, names in COLOURS.items() for name in names.split()}

# Hand-picked from contact sheets of each dataset's clearest labels: no faces or crowds, no watermarks,
# no scenes too dark to read. VisA: object/image, MVTec AD: object/defect/image, D-Fire: image name.
WALL = dict(name="banner", cols=8, rows=3, height=750, pad=3, pitch=250, tile=244, scale=2,
            seam=(750, 250 / 108, 225 / 106, 375), sweep=(450, 1250), band=260, chip=34, clear=30, seed=3,
            train=["split", "run", "loss", "gpu", "lr", "settings", "epoch", "valloss",
                   "lr", "data", "best", "iou", "iou", "loss", "run", "best",
                   "epoch", "valloss", "iou", "lr", "prelabel", "loss", "data", "gpu"],
            picks=["capsules/017.JPG", "cashew/035.JPG", "macaroni1/018.JPG",
                   "bottle/broken_small/002.png", "cable/missing_cable/000.png", "capsule/scratch/001.png",
                   "carpet/color/000.png", "grid/thread/001.png", "hazelnut/hole/000.png", "leather/glue/003.png",
                   "metal_nut/bent/000.png", "screw/scratch_neck/001.png", "toothbrush/defective/000.png",
                   "wood/scratch/000.png", "zipper/split_teeth/003.png", "tile/oil/002.png",
                   "transistor/cut_lead/002.png", "WEB04561", "WEB06559", "WEB05005", "WEB05375"],
            ai={(0, 7): "pcb1/035.JPG", (1, 6): "pcb1/046.JPG", (2, 7): "pcb1/024.JPG"},
            chips=[(0, 4), (2, 5)])
PHONE = dict(name="banner-phone", cols=5, rows=2, height=800, pad=3, pitch=400, tile=394, scale=1,
             seam=(1000, 200 / 108, 200 / 106, 400), sweep=None, band=380, chip=56, clear=40, seed=5,
             train=["data", "run", "loss", "gpu", "best", "lr", "best", "iou", "prelabel", "epoch"],
             picks=["macaroni1/018.JPG", "cable/missing_cable/000.png", "carpet/color/000.png",
                    "hazelnut/hole/000.png", "wood/scratch/000.png", "WEB05005", "WEB04561", "WEB09149"],
             ai={(0, 4): "pcb1/046.JPG", (1, 3): "pcb1/035.JPG"},
             chips=[(0, 3), (1, 4)])
# Real proposals from the model trained in the demo (docs/demo/README.md), as the app saved them
PROPOSALS = Path(__file__).with_name("banner-proposals.json")
# Per-epoch numbers from that model's training run, as the app recorded them
TRAINING = Path(__file__).with_name("banner-training.json")


def bgr(rgb):
    return rgb[::-1]


def short(label):
    return next((name for words, name in SHORT if words in label), label.split("_")[0])


def colour(name):
    return COLOUR.get(name, CYAN)


def square(h, w, x0, y0, x1, y1, grow, least):
    """A square crop around a region, big enough to show what the object is."""
    side = int(min(max(max(x1 - x0, y1 - y0) * grow, least * min(h, w)), h, w))
    ax = int(np.clip((x0 + x1) // 2 - side // 2, 0, w - side))
    ay = int(np.clip((y0 + y1) // 2 - side // 2, 0, h - side))
    return ax, ay, side


def mask_item(source, obj, label, image, mask, least, pick):
    img = cv2.imread(str(image))
    mask = (cv2.imread(str(mask), 0) > 0).astype("uint8") * 255
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    ax, ay, side = square(*img.shape[:2], xs.min(), ys.min(), xs.max(), ys.max(), 1.8, least)
    size = (TILE_PX, TILE_PX)
    photo = cv2.resize(img[ay : ay + side, ax : ax + side], size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask[ay : ay + side, ax : ax + side], size, interpolation=cv2.INTER_LINEAR)
    # Smooth the label's pixel edge a little; its shape stays the label's
    m = (cv2.GaussianBlur(m, (0, 0), 4.5) > 120).astype("uint8") * 255
    name = short(label)
    return dict(id=pick, source=source, object=obj, label=name, photo=photo, mask=m, boxes=None, score=clarity(m, name))


def ai_item(visa, pick, shapes):
    """A tile of the demo model's proposals on a VisA board, drawn the way the app shows them."""
    img = cv2.imread(str(visa / pick.split("/")[0] / "Data/Images/Anomaly" / pick.split("/")[1]))
    polys = [np.array(s["points"], float) for s in shapes]
    top = max(shapes, key=lambda s: cv2.contourArea(np.array(s["points"], np.float32)))
    pts = np.array(top["points"], float)
    ax, ay, side = square(*img.shape[:2], *pts.min(0), *pts.max(0), 1.8, 0.26)
    k = TILE_PX / side
    photo = cv2.resize(img[ay : ay + side, ax : ax + side], (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA)
    mask = np.zeros((TILE_PX, TILE_PX), "uint8")
    tile_polys = []
    for s, p in zip(shapes, polys):
        q = np.round((p - (ax, ay)) * k).astype(np.int32)
        cv2.fillPoly(mask, [q], 255)
        tile_polys.append((s["label"], q))
    return dict(id=pick, source="ai", object="ai", label=top["label"], score=1.0, confidence=top["score"],
                photo=photo, mask=mask, boxes=None, polys=tile_polys)


def box_item(image, labels):
    """A D-Fire scene: a square around its largest box, with the boxes that fit inside it."""
    img = cv2.imread(str(image))
    h, w = img.shape[:2]
    boxes = []
    for line in labels.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            boxes.append(("smoke" if int(c) == 0 else "fire", (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    big = max(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
    ax, ay, side = square(h, w, *big[1:], 1.5, 0.6)
    k = TILE_PX / side
    kept = []
    for name, x0, y0, x1, y1 in boxes:
        cx0, cy0, cx1, cy1 = max(x0, ax), max(y0, ay), min(x1, ax + side), min(y1, ay + side)
        inside = max(cx1 - cx0, 0) * max(cy1 - cy0, 0) / max((x1 - x0) * (y1 - y0), 1)
        if inside > 0.9 and (cx1 - cx0) * k > 18 and (cy1 - cy0) * k > 18:
            kept.append((name, *(round(v) for v in ((cx0 - ax) * k, (cy0 - ay) * k, (cx1 - ax) * k, (cy1 - ay) * k))))
    photo = cv2.resize(img[ay : ay + side, ax : ax + side], (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA if side > TILE_PX else cv2.INTER_CUBIC)
    mask = np.zeros((TILE_PX, TILE_PX), "uint8")
    for _, x0, y0, x1, y1 in kept:
        mask[y0:y1, x0:x1] = 255
    area = mask.sum() / 255 / mask.size
    score = min(1.0, area / 0.05) * (1 if area < 0.6 else 0.3) / (1 + 0.35 * max(len(kept) - 2, 0)) if kept else 0.0
    label = big[0] if kept else "fire"
    return dict(id=image.stem, source="dfire", object=label, label=label, photo=photo, mask=mask, boxes=kept, score=score)


def clarity(mask, name):
    """How cleanly a label reads at banner size: few pieces, compact, not tiny."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    if not areas or areas[0] < 1:
        return 0.0
    biggest = max(contours, key=cv2.contourArea)
    solidity = areas[0] / max(cv2.contourArea(cv2.convexHull(biggest)), 1)
    size = min(1.0, sum(areas) / mask.size / 0.02) * (1 if sum(areas) / mask.size < 0.5 else 0.3)
    if name == "bent":  # several pins or wires are one defect
        return size * solidity / (1 + 0.15 * max(len(areas) - 3, 0))
    return size * solidity * areas[0] / sum(areas) / (1 + 0.25 * (len(areas) - 1))


def load(visa, mvtec, dfire, picks, ai_picks):
    proposals = json.loads(PROPOSALS.read_text())
    items = [ai_item(visa, pick, proposals[pick]) for pick in sorted(ai_picks)]
    for anno in sorted(visa.glob("*/image_anno.csv")):
        for row in csv.DictReader(open(anno)):
            pick = f"{anno.parent.name}/{Path(row['image']).name}"
            if pick in picks and row["mask"]:
                items.append(mask_item("visa", anno.parent.name, row["label"], visa / row["image"], visa / row["mask"], 0.42, pick))
    for pick in sorted(picks):
        parts = pick.split("/")
        if len(parts) == 3:
            obj, defect, name = parts
            mask = mvtec / obj / "ground_truth" / defect / f"{Path(name).stem}_mask.png"
            items.append(mask_item("mvtec", obj, defect, mvtec / obj / "test" / defect / name, mask, 0.55, pick))
        elif len(parts) == 1:
            items.append(box_item(dfire / "train/images" / f"{pick}.jpg", dfire / "train/labels" / f"{pick}.txt"))
    missing = (set(picks) | set(ai_picks)) - {i["id"] for i in items}
    if missing:
        sys.exit(f"Not found: {', '.join(sorted(missing))}")
    return items


def duotone(photo):
    """The raw side, as quiet as the logo's data side: pale slate greys on paper."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(float) / 255
    g = np.clip((g - 0.08) / 0.92, 0, 1)[..., None]
    low, high = (np.array(bgr(c), float) for c in ((170, 176, 188), (247, 248, 251)))
    return (low + (high - low) * g).astype("uint8")


def handle(out, p, c, k):
    cv2.circle(out, p, round(9 * k), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, p, round(9 * k), c, round(3 * k), cv2.LINE_AA)


def dashed(out, poly, c, k):
    """A closed dashed outline, as the app draws a shape it proposed."""
    pts = np.vstack([poly, poly[:1]]).astype(float)
    dash, gap, carry, on = 16 * k, 10 * k, 0.0, True
    for a, b in zip(pts[:-1], pts[1:]):
        length = np.linalg.norm(b - a)
        t = 0.0
        while t < length:
            step = min((dash if on else gap) - carry, length - t)
            if on:
                p, q = (a + (b - a) * t / length).round().astype(int), (a + (b - a) * (t + step) / length).round().astype(int)
                cv2.line(out, tuple(p), tuple(q), (255, 255, 255), round(8 * k), cv2.LINE_AA)
                cv2.line(out, tuple(p), tuple(q), c, round(4 * k), cv2.LINE_AA)
            t += step
            carry += step
            if carry >= (dash if on else gap) - 1e-6:
                carry, on = 0.0, not on


def labelled(item, photo, mask):
    k = photo.shape[0] / 388
    if item["source"] == "ai":
        out = photo.copy()
        s = photo.shape[0] / TILE_PX
        for name, poly in item["polys"]:
            c = bgr(colour(name))
            m = np.zeros(photo.shape[:2], "uint8")
            q = np.round(poly * s).astype(np.int32)
            cv2.fillPoly(m, [q], 255)
            a = (m > 0)[..., None] * 0.3
            out = (out * (1 - a) + np.array(c, float) * a).astype("uint8")
            dashed(out, q, c, k)
            points = cv2.approxPolyDP(q, 14 * k, True)[:, 0]
            for p in points[:14]:
                handle(out, tuple(int(v) for v in p), c, k)
        return out
    if item["boxes"] is not None:
        # Boxes, as the app's rectangle tool draws them
        out = photo.copy()
        s = photo.shape[0] / TILE_PX
        for name, *b in item["boxes"]:
            x0, y0, x1, y1 = (round(v * s) for v in b)
            c = bgr(colour(name))
            tint = out[y0:y1, x0:x1].astype(float)
            out[y0:y1, x0:x1] = (tint * 0.82 + np.array(c, float) * 0.18).astype("uint8")
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), round(9 * k), cv2.LINE_AA)
            cv2.rectangle(out, (x0, y0), (x1, y1), c, round(5 * k), cv2.LINE_AA)
            for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                handle(out, p, c, k)
        return out
    c = bgr(colour(item["label"]))
    alpha = (mask > 0)[..., None] * 0.36
    out = (photo * (1 - alpha) + np.array(c, float) * alpha).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [x for x in contours if cv2.contourArea(x) > 60 * k * k]
    cv2.drawContours(out, contours, -1, (255, 255, 255), round(9 * k), cv2.LINE_AA)
    cv2.drawContours(out, contours, -1, c, round(5 * k), cv2.LINE_AA)
    # Vertices on the largest shape, as the app draws them
    biggest = max(contours, key=cv2.contourArea)
    epsilon = 18 * k
    points = cv2.approxPolyDP(biggest, epsilon, True)[:, 0]
    while len(points) > 12:
        epsilon *= 1.3
        points = cv2.approxPolyDP(biggest, epsilon, True)[:, 0]
    for p in points:
        handle(out, tuple(int(v) for v in p), c, k)
    return out


def place(cfg, points, cx=None):
    cx0, sx, sy, cy = cfg["seam"]
    cx = cx0 if cx is None else cx
    return [(cx + (x - 256) * sx, cy + (y - 256) * sy) for x, y in points]


def seam_curve(cfg, cx=None):
    """The seam's Bezier points, plus straight runs off both ends so it always crosses the whole banner."""
    p = place(cfg, SEAM, cx)
    ends = []
    for a, b in ((p[0], p[1]), (p[12], p[11])):
        d = np.subtract(a, b)
        ends.append(tuple(np.add(a, d / np.linalg.norm(d) * 600)))
    return ends[0], p, ends[1]


def sample(cfg, cx=None, steps=60):
    start, p, end = seam_curve(cfg, cx)
    points = [np.array(start)]
    for i in range(0, 12, 3):
        a, b, c, d = (np.array(p[i + k]) for k in range(4))
        for k in range(steps):
            t = k / steps
            points.append((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t**3 * d)
    return np.array(points + [np.array(p[12]), np.array(end)])


def path_d(cfg):
    start, p, end = seam_curve(cfg)
    f = lambda q: f"{q[0]:.1f} {q[1]:.1f}"
    return f"M{f(start)}L{f(p[0])}" + "".join(f"C{f(p[i])} {f(p[i + 1])} {f(p[i + 2])}" for i in range(1, 12, 3)) + f"L{f(end)}"


def band(cfg, cxs, width):
    out = np.zeros((cfg["height"], W), "uint8")
    for cx in cxs:
        cv2.polylines(out, [np.round(sample(cfg, cx)).astype(np.int32)], False, 255, int(width))
    return out


def chip_text(item):
    return f"{round(item['confidence'] * 100)}%" if item["source"] == "ai" else item["label"]


def chip_size(cfg, item):
    badge = 2.5 if item["source"] == "ai" else 0
    return int(cfg["chip"] * (0.58 * len(chip_text(item)) + 1.9 + badge)), int(cfg["chip"] * 1.8)


def chip_html(cfg, item, x, y):
    f = cfg["chip"]
    lead = f'<i style="background:rgb{colour(item["label"])};width:{f * .55}px;height:{f * .55}px;border-radius:{f * .16}px"></i>'
    if item["source"] == "ai":
        # The class colour, then the app's badge for a proposed shape and its score
        lead += (f'<b style="background:rgb{AMBER};color:#14161a;font-size:{f * .62}px;font-weight:700;'
                 f'padding:{f * .12}px {f * .22}px;border-radius:{f * .16}px">AI</b>')
    return (f'<div class=chip style="left:{x}px;top:{y}px;height:{f * 1.8}px;font-size:{f}px;gap:{f * .35}px;'
            f'padding:0 {f * .5}px 0 {f * .42}px;border-radius:{f * .35}px">{lead}{chip_text(item)}</div>')


def chip_spot(cfg, item, origin, seam):
    """The corner where the chip covers least of the shape and stays clear of the seam. Returns (spot, hidden share)."""
    t, inset = cfg["tile"], 8
    w, h = chip_size(cfg, item)
    m = cv2.resize(item["mask"], (t, t), interpolation=cv2.INTER_NEAREST) > 0
    x, y = origin
    best = None
    for sx, sy in ((inset, t - h - inset), (t - w - inset, t - h - inset), (inset, inset), (t - w - inset, inset)):
        if seam[y + sy : y + sy + h, x + sx : x + sx + w].any():
            continue
        hidden = m[sy : sy + h, sx : sx + w].sum() / max(m.sum(), 1)
        if best is None or hidden < best[1]:
            best = ((sx, sy), hidden)
    return best or (None, 1.0)


def arrange(cfg, items, rng):
    """Pick which image goes in which cell: shapes clear of the seam, no object twice side by side."""
    t, pad, pitch = cfg["tile"], cfg["pad"], cfg["pitch"]
    pool = [i for i in items if i["id"] in cfg["picks"]]
    rng.shuffle(pool)
    near = band(cfg, [cfg["seam"][0]], cfg["band"] + cfg["clear"] * 2)
    chip_band = band(cfg, [cfg["seam"][0]] + list(cfg["sweep"] or []), cfg["band"] + 30)
    origin = lambda cell: (pad + cell[1] * pitch, pad + cell[0] * pitch)

    def touches(item, cell):
        x, y = origin(cell)
        m = cv2.resize(item["mask"], (t, t), interpolation=cv2.INTER_NEAREST) > 0
        return bool((m & (near[y : y + t, x : x + t] > 0)).any())

    def beside(item, cell, grid):
        r, c = cell
        return sum(grid[n]["source"] == item["source"] for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if n in grid)

    grid, chips = {}, {}
    # Proposal tiles sit far right, so their chips stay in view through the whole sweep
    for cell, pick in cfg["ai"].items():
        grid[cell] = next(i for i in items if i["id"] == pick)
        chips[cell] = chip_spot(cfg, grid[cell], origin(cell), chip_band)[0]
    static = band(cfg, [cfg["seam"][0]], cfg["band"] + 30)
    for cell in cfg["chips"]:
        labels, sources = {g["label"] for g in grid.values()}, {g["source"] for g in grid.values()}
        options = [i for i in pool if len(i["label"]) <= 8 and i["label"] not in labels]
        pick = min(options, key=lambda i: (i["source"] in sources, round(chip_spot(cfg, i, origin(cell), static)[1] * 10), -i["score"]))
        grid[cell], chips[cell] = pick, chip_spot(cfg, pick, origin(cell), static)[0]
        pool = [i for i in pool if i is not pick]

    cells = [(r, c) for r in range(cfg["rows"]) for c in range(cfg["cols"]) if (r, c) not in grid]
    rng.shuffle(cells)
    crossing = lambda cell: near[origin(cell)[1] : origin(cell)[1] + t, origin(cell)[0] : origin(cell)[0] + t].any()
    cells.sort(key=lambda cell: not crossing(cell))
    for cell in cells:
        grid[cell] = min(pool, key=lambda i: (touches(i, cell), beside(i, cell, grid), -i["score"]))
        pool = [i for i in pool if i is not grid[cell]]
        if touches(grid[cell], cell):
            print(f"{cfg['name']}: the shape in cell {cell} touches the seam")
    return grid, chips, origin


def rounded_paste(canvas, tile, x, y, radius):
    h, w = tile.shape[:2]
    m = np.zeros((h, w), "uint8")
    cv2.rectangle(m, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(m, (0, radius), (w, h - radius), 255, -1)
    for p in ((radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(m, p, radius, 255, -1, cv2.LINE_AA)
    a = m[..., None] / 255
    canvas[y : y + h, x : x + w] = (tile * a + canvas[y : y + h, x : x + w] * (1 - a)).astype("uint8")


def walls(cfg, grid, chips, origin, tmp):
    s, t, h = cfg["scale"], cfg["tile"], cfg["height"]
    raw = np.full((h * s, W * s, 3), bgr(PAPER), "uint8")
    lab = np.full((h * s, W * s, 3), bgr(BACKDROP), "uint8")
    plain = lab.copy()
    html, marks = "", []
    for cell, item in grid.items():
        x, y = origin(cell)
        size = (t * s, t * s)
        photo = cv2.resize(item["photo"], size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(item["mask"], size, interpolation=cv2.INTER_NEAREST)
        o = max(1, round(1.5 * s))
        rounded_paste(raw, np.full((t * s + 2 * o, t * s + 2 * o, 3), bgr(EDGE), "uint8"), x * s - o, y * s - o, 10 * s + o)
        rounded_paste(raw, duotone(photo), x * s, y * s, 10 * s)
        drawn = labelled(item, photo, mask)
        rounded_paste(lab, drawn, x * s, y * s, 10 * s)
        rounded_paste(plain, photo, x * s, y * s, 10 * s)
        # Where the label is drawn, with its chip, so the label can show only when the channel hides none of it
        ys, xs = np.where(np.abs(drawn.astype(int) - photo.astype(int)).sum(-1) > 24)
        box = [x + xs.min() / s, y + ys.min() / s, x + (xs.max() + 1) / s, y + (ys.max() + 1) / s]
        if chips.get(cell):
            cx, cy = chips[cell]
            html += chip_html(cfg, item, x + cx, y + cy)
            cw, ch = chip_size(cfg, item)
            box = [min(box[0], x + cx), min(box[1], y + cy), max(box[2], x + cx + cw), max(box[3], y + cy + ch)]
        marks.append(((x, y, x + t, y + t), box))
    for name, image in (("raw", raw), ("lab", lab), ("plain", plain)):
        cv2.imwrite(str(tmp / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return html, marks


def shoot(cfg, tmp, name, body, css_extra="", dpr=1, dump=False):
    h = cfg["height"]
    css = (f"@font-face{{font-family:Inter;src:url('file://{FONT}');font-weight:100 900}}*{{margin:0;box-sizing:border-box}}"
           f"html,body{{width:{W}px;height:{h}px;background:transparent;overflow:hidden;font-family:Inter}}"
           f".b{{position:absolute;inset:0;border-radius:28px;overflow:hidden;background:rgb{BACKDROP}}}"
           f".l{{position:absolute;inset:0;background-size:{W}px {h}px;background-repeat:no-repeat}}"
           f"svg{{position:absolute;inset:0;width:{W}px;height:{h}px}}"
           ".chip{position:absolute;display:flex;align-items:center;background:rgba(12,14,20,.94);color:#fff;font-weight:560;"
           "letter-spacing:-.01em;box-shadow:0 4px 18px rgba(0,0,0,.5);white-space:nowrap}.chip i{display:block;flex:none}" + css_extra)
    page = tmp / f"{name}.html"
    page.write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style><div class=b>{body}</div>")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if dump:
        return subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--allow-file-access-from-files",
                               f"--window-size={W},{h}", "--virtual-time-budget=5000", "--dump-dom", f"file://{page}"],
                              capture_output=True, timeout=120, check=True, text=True).stdout
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--default-background-color=00000000", f"--window-size={W},{h}", f"--force-device-scale-factor={dpr}",
                    f"--screenshot={tmp / name}.png", f"file://{page}"], capture_output=True, timeout=120, check=True)
    return Image.open(tmp / f"{name}.png").convert("RGBA")


def chart(values, w, h, colour, s, top=None, log=False):
    v = np.log10(values) if log else np.array(values, float)
    lo, hi = (v.min(), v.max()) if log else (0.0, top or v.max())
    pts = " ".join(f"{i / (len(v) - 1) * w:.1f},{h - (x - lo) / (hi - lo) * h:.1f}" for i, x in enumerate(v))
    return f'<polyline points="{pts}" fill="none" stroke="rgb{colour}" stroke-width="{3.2 * s}" stroke-linejoin="round"/>'


def meter(share, colour):
    """A bar for a share the run recorded."""
    return f'<div class=meter><i style="width:{share * 100:.1f}%;background:rgb{colour}"></i></div>'


def tile_html(kind, run, s):
    """One training tile in the app's panel style, from the run's own numbers. Its words carry the group they show and
    hide with (G is the tile's number): the head, the foot, or a settings row."""
    cw = 208 * s
    svg = lambda h, body, mt=10: (f'<svg width="{cw}" height="{h * s}" style="position:static;width:{cw}px;height:{h * s}px;'
                                  f'flex:none;margin-top:{mt * s}px;overflow:visible">{body}</svg>')
    bar = '<div class=bar></div>'
    head, foot = 'data-g="G-h"', 'data-g="G-f"'
    best = max(run["iou"])
    images = run["labelled"] + run["unlabelled"]
    rows = (f'input {run["input"]}', f'tile {run["tile"]}', f'batch {run["batch"]}', f'{run["loss"]} loss')
    return {
        "run": f'<div class=k {head}>TRAINING</div><div class=m {head}>{run["model"]}</div>{bar}<div class=u {foot}>{run["epochs"]} epochs</div>',
        "loss": f'<div class=k {head}>LOSS <span style="float:right;letter-spacing:0;font-weight:500"><i style="color:rgb{BLUE}">■</i> train '
                f'<i style="color:rgb{ORANGE}">■</i> val</span></div>'
                + svg(130, chart(run["train_loss"], cw, 130 * s, BLUE, s, max(run["train_loss"]))
                      + chart(run["val_loss"], cw, 130 * s, ORANGE, s, max(run["train_loss"])), 22),
        "best": f'<div class=k {head}>BEST IOU</div><div class=v {head}>{best:.3f}</div>' + svg(54, chart(run["iou"], cw, 54 * s, GREEN, s, 1.0), 14),
        "iou": f'<div class=k {head}>VALIDATION IOU</div>' + svg(140, chart(run["iou"], cw, 140 * s, GREEN, s, 1.0), 26)
               + f'<div class=u {foot}>{best:.3f} at best</div>',
        "data": f'<div class=k {head}>LABELLED</div><div class=v {head}>{run["labelled"]}</div>{meter(run["labelled"] / images, BLUE)}'
                f'<div class=u {foot}>images, {run["unlabelled"]} left for the model</div>',
        "split": f'<div class=k {head}>TILES</div><div class=v {head}>{run["train_tiles"]}</div>'
                 f'{meter(run["train_tiles"] / (run["train_tiles"] + run["val_tiles"]), BLUE)}'
                 f'<div class=u {foot}>for training, {run["val_tiles"]} to validate</div>',
        "gpu": f'<div class=k {head}>GPU</div><div class=m {head}>{run["device"]}</div><div class=u {foot}>over SSH, peak {run["peak_gb"]} GB</div>',
        "lr": f'<div class=k {head}>LEARNING RATE</div>' + svg(110, chart(run["lr"], cw, 110 * s, AMBER, s, log=True), 22)
              + f'<div class=u {foot}>{run["learning_rate"]:.0e} to {min(run["lr"]):.0e}</div>'.replace("e-0", "e-"),
        "settings": f'<div class=k {head}>SETTINGS</div>' + "".join(f'<div class=field><span data-g="G-r{n}">{row}</span></div>'
                                                                   for n, row in enumerate(rows)),
        "epoch": f'<div class=k {head}>EPOCH</div><div class=v {head}>{run["epochs"]}/{run["epochs"]}</div>{bar}<div class=u {foot}>finished</div>',
        "valloss": f'<div class=k {head}>VAL LOSS</div><div class=v {head}>{min(run["val_loss"]):.3f}</div>'
                   + svg(54, chart(run["val_loss"], cw, 54 * s, ORANGE, s), 14),
        "prelabel": f'<div class=k {head}>PRE-LABEL</div><div class=v {head}>{run["proposals"]}</div>{meter(run["unlabelled"] / images, AMBER)}'
                    f'<div class=u {foot}>proposals on {run["unlabelled"]} images</div>',
    }[kind]


def training_wall(cfg, run, origin, tmp):
    """Every cell as a training tile, for the channel to show: the tiles without their words, the words alone, and
    where each group of words sits, so the channel can show a group only when it shows all of it."""
    s, t = cfg["tile"] / 244, cfg["tile"]
    css = (f".t{{position:absolute;width:{t}px;height:{t}px;border-radius:10px;background:#161a22;border:{1.5 * s}px solid rgba(255,255,255,.08);"
           f"padding:{18 * s}px;color:#eef1f6;overflow:hidden;display:flex;flex-direction:column}}"
           f".k{{font-size:{14.5 * s}px;letter-spacing:.08em;font-weight:650;color:#8a93a6}}"
           f".v{{font-size:{50 * s}px;font-weight:650;letter-spacing:-.02em;line-height:1.1;margin-top:{8 * s}px}}"
           f".m{{font-size:{21 * s}px;font-weight:600;margin-top:{10 * s}px}}"
           f".u{{font-size:{15.5 * s}px;color:#aab2c2;margin-top:auto;line-height:1.3}}"
           f".bar{{height:{8 * s}px;border-radius:{4 * s}px;background:rgb{BLUE};margin-top:{16 * s}px}}"
           f".meter{{height:{8 * s}px;border-radius:{4 * s}px;background:rgba(255,255,255,.1);margin-top:auto;overflow:hidden;flex:none}}"
           f".meter i{{display:block;height:100%}}.meter+.u{{margin-top:{12 * s}px}}"
           f".field{{border:{1.5 * s}px solid rgba(255,255,255,.12);border-radius:{6 * s}px;padding:0 {10 * s}px;height:{36 * s}px;"
           f"display:flex;align-items:center;margin-top:{8 * s}px;font-size:{17 * s}px;font-weight:550;color:#dfe4ec;flex:none}}")
    cells = [(r, c) for r in range(cfg["rows"]) for c in range(cfg["cols"])]
    body = "".join(f'<div class=t style="left:{origin(cell)[0]}px;top:{origin(cell)[1]}px">{tile_html(kind, run, s).replace("G-", f"{n}-")}</div>'
                   for n, (cell, kind) in enumerate(zip(cells, cfg["train"])))
    tiles = shoot(cfg, tmp, "train", body, css + "[data-g],[data-g] *{color:transparent!important}", dpr=cfg["scale"])
    words = shoot(cfg, tmp, "words", body, css + ".b,.t{background:transparent!important;border-color:transparent!important}"
                  ".field{border-color:transparent!important}svg,.bar,.meter{visibility:hidden}", dpr=cfg["scale"])
    script = ("<script>document.fonts.ready.then(() => { const g = {}; document.querySelectorAll('[data-g]').forEach(e => {"
              "const r = e.getBoundingClientRect(), b = g[e.dataset.g] || [1e9, 1e9, -1e9, -1e9];"
              "g[e.dataset.g] = [Math.min(b[0], r.left), Math.min(b[1], r.top), Math.max(b[2], r.right), Math.max(b[3], r.bottom)]; });"
              "document.body.insertAdjacentHTML('beforeend', '<pre id=boxes>' + JSON.stringify(Object.values(g)) + '</pre>'); });</script>")
    dom = shoot(cfg, tmp, "boxes", body + script, css, dump=True)
    # the script's own source holds the same tag, so take the one it added at the end
    return tiles, words, json.loads(re.findall(r"<pre id=\"?boxes\"?>(.*?)</pre>", dom)[-1])


def corners(size, radius):
    w, h = size
    m = np.zeros((h * 3, w * 3), "uint8")
    cv2.rectangle(m, (radius * 3, 0), (w * 3 - radius * 3, h * 3), 255, -1)
    cv2.rectangle(m, (0, radius * 3), (w * 3, h * 3 - radius * 3), 255, -1)
    for p in ((radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(m, (p[0] * 3, p[1] * 3), radius * 3, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, size, interpolation=cv2.INTER_AREA)


def smooth(mask, radius):
    """Fill the sharp notch a wide channel leaves on the inside of a tight bend."""
    h, w = mask.shape
    m = (mask * 255).astype("uint8")
    half = cv2.resize(m, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(cv2.copyMakeBorder(half, radius, radius, radius, radius, cv2.BORDER_REPLICATE), cv2.MORPH_CLOSE, disk)
    # Half resolution leaves steps on the filled edge: soften them there, scale up, then steepen the edge again
    closed = cv2.GaussianBlur(closed[radius:-radius, radius:-radius], (0, 0), 1.2)
    closed = cv2.resize(closed, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    closed = np.clip((closed - 127.5) * 2 + 127.5, 0, 255)
    return np.maximum(m, closed) / 255


def edge_handles(ink, cy):
    """The logo's three handles on the channel's paper edge: at the tips of its two bowls, as on the logo, and midway between."""
    m = (ink > 0.5).astype("uint8")
    h, w = m.shape
    contour = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0]
    inside = (contour[:, 0] > 2) & (contour[:, 0] < w - 3) & (contour[:, 1] > 2) & (contour[:, 1] < h - 3)
    # The edge is the longest run of contour points off the frame; the contour may start partway along it
    start = int(np.argmin(inside)) if not inside.all() else 0
    contour, inside = np.roll(contour, -start, axis=0), np.roll(inside, -start)
    runs, run = [], []
    for p, ok in zip(contour, inside):
        if ok:
            run.append(p)
        elif run:
            runs.append(run)
            run = []
    edge = np.array(max(runs + [run], key=len), float)
    upper, lower = np.flatnonzero(edge[:, 1] < cy), np.flatnonzero(edge[:, 1] >= cy)
    tips = int(upper[np.argmin(edge[upper, 0])]), int(lower[np.argmax(edge[lower, 0])])
    a, z = sorted(tips)
    length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(edge[a:z + 1], axis=0), axis=1))])
    mid = a + int(np.searchsorted(length, length[-1] / 2))
    return [tuple(edge[i].round().astype(int)) for i in (tips[0], mid, tips[1])]


def compose(cfg, layers, cx, size):
    """Raw images on paper left of the channel, training tiles inside it, labelled images on ink right of it.
    The channel's paper edge is the logo's S: no line, just the two sides meeting, with its handles and shadow."""
    ow, oh = size
    k, ss = ow / W, 3
    raw, train, lab, plain, words = (cv2.resize(layers[n], size, interpolation=cv2.INTER_AREA).astype(float)
                                     for n in ("raw", "train", "lab", "plain", "words"))
    pts = np.round(sample(cfg, cx) * k * ss).astype(np.int32)

    def stroke(width):
        m = np.zeros((oh * ss, ow * ss), "uint8")
        cv2.polylines(m, [pts], False, 255, max(1, round(width * k * ss)), cv2.LINE_AA)
        return cv2.resize(m, size, interpolation=cv2.INTER_AREA)[..., None] / 255

    right = np.zeros((oh * ss, ow * ss), "uint8")
    cv2.fillPoly(right, [np.vstack([pts, [[(ow + 600) * ss, (oh + 600) * ss], [(ow + 600) * ss, -600 * ss]]])], 255)
    right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)[..., None] / 255
    inner = stroke(cfg["band"])
    radius = max(1, round(cfg["band"] * 0.15 * k))
    ink = smooth(np.maximum(inner, right)[..., 0], radius)[..., None]
    # The same rounding on the labelled side, so no pointed sliver of it pokes into the channel
    labelled_part = 1 - smooth(1 - (right * (1 - inner))[..., 0], radius)[..., None]
    channel = np.clip(ink - labelled_part, 0, 1)
    # The ink side casts a soft shadow onto the paper, as in the logo
    cast = cv2.warpAffine(ink[..., 0], np.float32([[1, 0, -7 * k], [0, 1, 5 * k]]), size, borderMode=cv2.BORDER_REPLICATE)
    shade = cv2.GaussianBlur(cast, (0, 0), 14 * k)[..., None]
    img = raw * (1 - 0.3 * shade) * (1 - ink) + train * channel + lab * labelled_part
    # A label shows only when the channel hides none of it; until then its image is plain
    for tile, box in layers["marks"]:
        a, b = max(int(box[0] * k) - 3, 0), max(int(box[1] * k) - 3, 0)
        c, d = min(int(np.ceil(box[2] * k)) + 3, ow), min(int(np.ceil(box[3] * k)) + 3, oh)
        weight = float(labelled_part[b:d, a:c].mean() > 0.995) if c > a and d > b else 1
        if weight < 1:
            x0, y0, x1, y1 = (int(round(v * k)) for v in tile)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, ow), min(y1, oh)
            img[y0:y1, x0:x1] += (plain - lab)[y0:y1, x0:x1] * labelled_part[y0:y1, x0:x1] * (1 - weight)
    # Words show only where the channel shows all of them, so none is ever cut in half
    for x0, y0, x1, y1 in layers["boxes"]:
        a, b = max(int(x0 * k) - 3, 0), max(int(y0 * k) - 3, 0)
        c, d = min(int(np.ceil(x1 * k)) + 3, ow), min(int(np.ceil(y1 * k)) + 3, oh)
        weight = float(channel[b:d, a:c].mean() > 0.995) if c > a and d > b else 0
        if weight > 0:
            alpha = words[b:d, a:c, 3:] / 255 * weight
            img[b:d, a:c] = img[b:d, a:c] * (1 - alpha) + words[b:d, a:c, :3] * alpha
    out = np.ascontiguousarray(img.astype("uint8")[..., ::-1])
    r, w = max(1, round(18 * k)), max(1, round(7 * k))
    for p in edge_handles(ink[..., 0], cfg["seam"][3] * k):
        cv2.circle(out, p, r, INK, -1, cv2.LINE_AA)
        cv2.circle(out, p, r - w, (255, 255, 255), -1, cv2.LINE_AA)
    # A faint rim, as on the logo, so the frame shows on pages the colour of either side
    alpha = corners(size, round(28 * k))
    rim = np.clip(alpha.astype(float) - cv2.erode(alpha, np.ones((3, 3), "uint8"), iterations=max(1, round(2 * k))), 0, 255)[..., None] / 255
    out = (out * (1 - 0.35 * rim) + np.array((138, 147, 163)) * 0.35 * rim).astype("uint8")
    return np.dstack([out, alpha])


def animate(cfg, layers, out):
    """The channel sweeps across the wall, so raw images pass through training and come out labelled, and back."""
    size = (1400, round(cfg["height"] * 1400 / W))
    left, right = cfg["sweep"]
    frames, n = [], 120
    for f in range(n):
        u = f / n
        # Hold mostly labelled, sweep back, hold mostly raw, sweep again
        e = 1 if u < 0.2 else 1 - (u - 0.2) / 0.3 if u < 0.5 else 0 if u < 0.65 else (u - 0.65) / 0.35
        e = e * e * (3 - 2 * e)
        frames.append(Image.fromarray(compose(cfg, layers, right + (left - right) * e, size), "RGBA"))
    frames[0].save(out / "banner-animated.webp", save_all=True, append_images=frames[1:], duration=66, loop=0, quality=72, method=4)


def build(cfg, items, run, out):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        grid, chips, origin = arrange(cfg, items, random.Random(cfg["seed"]))
        html, marks = walls(cfg, grid, chips, origin, tmp)
        lab = shoot(cfg, tmp, "wall", f'<div class=l style="background-image:url(lab.jpg)"></div>{html}', dpr=cfg["scale"])
        train, words, boxes = training_wall(cfg, run, origin, tmp)
        layers = dict(raw=cv2.imread(str(tmp / "raw.jpg")), train=np.array(train.convert("RGB"))[..., ::-1],
                      lab=np.array(lab.convert("RGB"))[..., ::-1], words=np.array(words)[..., [2, 1, 0, 3]], boxes=boxes,
                      plain=cv2.imread(str(tmp / "plain.jpg")), marks=marks)
        size = (W // 2, cfg["height"] // 2) if cfg["scale"] == 1 else (W, cfg["height"])
        Image.fromarray(compose(cfg, layers, cfg["seam"][0], size), "RGBA").save(out / f"{cfg['name']}.webp", quality=88, method=6)
        if cfg["sweep"]:
            animate(cfg, layers, out)


def main(visa, mvtec, dfire, out):
    picks = set(WALL["picks"]) | set(PHONE["picks"])
    items = load(Path(visa), Path(mvtec), Path(dfire), picks, set(WALL["ai"].values()) | set(PHONE["ai"].values()))
    run = json.loads(TRAINING.read_text())
    build(WALL, items, run, Path(out))
    build(PHONE, items, run, Path(out))


if __name__ == "__main__":
    main(*sys.argv[1:5])
