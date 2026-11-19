"""Writes label files with labelme 7.9.1's own code, one for each kind of shape, and the index
masks labelme's shapes_to_label makes of them, for tests/unit/test_studio_shapes.py: Seamline must
train on the pixels labelme gives each shape. From this folder:

    uv run --no-project --python 3.12 --with labelme==7.9.1 --with pillow==13.0.0 \
        python make_masks.py

shapes/ holds each case's label file beside its 64x48 image, and masks/ labelme's mask of it,
each pixel the number masks/classes.json gives its class. labelme draws lines, linestrips and
points too, and can't draw multi-points; Seamline trains on none of them.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
from labelme._label_file import Annotation, read_label_file, write_label_file
from labelme._utils import shapes_to_label
from PIL import Image

here = Path(__file__).parent
WIDTH, HEIGHT = 64, 48
CLASSES = {"a": 1, "b": 2, "c": 3}
rng = np.random.default_rng(0)


def shape(label, shape_type, points, mask=None):
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "shape_type": shape_type,
        "flags": {},
        "description": "",
        "group_id": None,
        "mask": mask,
        "locked": False,
        "other_data": {},
    }


def pixels(rows, cols):
    """A mask shape's pixels: a scatter, so a pixel out of place shows"""
    return rng.random((rows, cols)) > 0.35


def turned(cx, cy, width, height, degrees):
    """An oriented rectangle's four corners"""
    a = math.radians(degrees)
    corners = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2)]
    corners.append((-width / 2, height / 2))
    return [
        [
            round(cx + x * math.cos(a) - y * math.sin(a), 3),
            round(cy + x * math.sin(a) + y * math.cos(a), 3),
        ]
        for x, y in corners
    ]


CASES = {
    # Concave, with fractions
    "polygon": [
        shape("a", "polygon", [[5.5, 4.2], [40.7, 8.1], [30.2, 20.5], [55.9, 40.3], [10.1, 35.6]])
    ],
    "rectangle": [shape("b", "rectangle", [[8.4, 6.6], [50.2, 30.7]])],
    # Drawn from the bottom right
    "rectangle_backwards": [shape("a", "rectangle", [[50.2, 40.7], [8.6, 5.3]])],
    "oriented_rectangle": [shape("c", "oriented_rectangle", turned(32.3, 24.6, 36.0, 16.0, 30))],
    # The centre, then a point on the edge
    "circle": [shape("b", "circle", [[30.5, 22.3], [42.1, 27.9]])],
    # Whole numbers: the box's corners, both inside it, 20 x 20 here
    "mask_whole": [shape("a", "mask", [[10, 8], [29, 27]], pixels(20, 20))],
    # A PNG bigger than its whole-numbered box is cut to the box
    "mask_whole_cut": [shape("b", "mask", [[40, 30], [49, 37]], pixels(12, 14))],
    # Fractions: at the first point, rounded to even (12.5 to 12, 9.5 to 10), the PNG's own size
    "mask_fractional": [shape("c", "mask", [[12.5, 9.5], [30.4, 25.6]], pixels(15, 17))],
    # Past the image's edges, cut to the image
    "mask_edges": [
        shape("a", "mask", [[-5, -4], [14, 11]], pixels(16, 20)),
        shape("b", "mask", [[55.2, 40.7], [70.1, 52.3]], pixels(12, 15)),
    ],
    # Later shapes win where they overlap
    "overlap": [
        shape("a", "rectangle", [[5, 5], [40, 30]]),
        shape("b", "polygon", [[20, 10], [60, 15], [35, 45]]),
        shape("c", "circle", [[45.5, 30.5], [53.5, 30.5]]),
        shape("a", "mask", [[2.5, 20.5], [18.5, 44.5]], pixels(20, 14)),
    ],
    # No area. labelme draws lines 10 pixels wide and points as discs, and can't draw multi-points.
    "lines_and_points": [
        shape("a", "line", [[2, 2], [60, 10]]),
        shape("b", "linestrip", [[5, 20], [30, 25], [55, 20], [60, 40]]),
        shape("c", "point", [[50.5, 10.5]]),
        shape("a", "points", [[5, 40], [10, 42], [15, 44]]),
    ],
}

for sub in ("shapes", "masks"):
    shutil.rmtree(here / sub, ignore_errors=True)
    (here / sub).mkdir()
for case, shapes in CASES.items():
    Image.new("RGB", (WIDTH, HEIGHT), (0x44, 0x55, 0x66)).save(here / "shapes" / f"{case}.png")
    label_file = str(here / "shapes" / f"{case}.json")
    annotation = Annotation(
        image_path=f"{case}.png", image_data=b"", shapes=shapes, flags={}, other_data={}
    )
    write_label_file(
        filename=label_file,
        annotation=annotation,
        image_height=HEIGHT,
        image_width=WIDTH,
        save_image_data=False,
    )
    # Read back as labelme reads it, masks decoded by labelme too
    read = read_label_file(filename=label_file).shapes
    drawable = [s for s in read if s["shape_type"] != "points"]
    mask, _ = shapes_to_label(
        img_shape=(HEIGHT, WIDTH, 3),
        shapes=drawable,
        label_name_to_value={"_background_": 0, **CLASSES},
    )
    Image.fromarray(mask.astype(np.uint8)).save(here / "masks" / f"{case}.png")
(here / "masks" / "classes.json").write_text(json.dumps(CLASSES, indent=2) + "\n")
print("wrote", len(CASES), "cases")
