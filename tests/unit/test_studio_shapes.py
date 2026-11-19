"""Seamline's training masks against labelme 7.9.1's, for every kind of shape Seamtag saves."""
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from engine.studio import place_mask, prepare, rasterize

# Label files written by labelme 7.9.1 and the masks its shapes_to_label makes of them:
# tests/fixtures/seamtag/make_masks.py
FIXTURES = Path(__file__).parent.parent / "fixtures" / "seamtag"
CLASS_IDS = json.loads((FIXTURES / "masks" / "classes.json").read_text())
AREA_CASES = [
    "polygon",
    "rectangle",
    "rectangle_backwards",
    "oriented_rectangle",
    "circle",
    "mask_whole",
    "mask_whole_cut",
    "mask_fractional",
    "mask_edges",
    "overlap",
]


def load(case):
    doc = json.loads((FIXTURES / "shapes" / f"{case}.json").read_text())
    return doc["shapes"], (doc["imageWidth"], doc["imageHeight"])


def labelme_mask(case):
    return np.array(Image.open(FIXTURES / "masks" / f"{case}.png"))


def png(pixels):
    data = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()


def mask_shape(points, pixels, label="a"):
    return {"label": label, "shape_type": "mask", "points": points, "mask": png(pixels)}


@pytest.mark.parametrize("case", AREA_CASES)
def test_area_shapes_cover_the_pixels_labelme_gives_them(case):
    shapes, size = load(case)
    assert (rasterize(shapes, size, CLASS_IDS) == labelme_mask(case)).all()


@pytest.mark.parametrize("case", ["polygon", "rectangle", "oriented_rectangle", "circle"])
def test_labelmes_outline_adds_no_pixels(case):
    # labelme draws these with outline=1 as well as fill=1, and Seamline with the fill alone.
    # Pillow draws no outline in the fill's own colour, so both give the same pixels.
    shapes, size = load(case)
    (x0, y0), (x1, y1), *_ = shapes[0]["points"]
    if case == "circle":
        r = math.dist((x0, y0), (x1, y1))
        method, xy = "ellipse", [(x0 - r, y0 - r), (x0 + r, y0 + r)]
    elif case == "rectangle":
        method, xy = "rectangle", [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
    else:
        method, xy = "polygon", [tuple(p) for p in shapes[0]["points"]]
    masks = []
    for outline in (None, 1):
        mask = Image.new("L", size)
        getattr(ImageDraw.Draw(mask), method)(xy, fill=1, outline=outline)
        masks.append(np.array(mask))
    assert (masks[0] == masks[1]).all() and masks[0].any()


def test_lines_and_points_have_no_area():
    shapes, size = load("lines_and_points")
    # labelme gives lines a stroke 10 pixels wide and points a disc; Seamline trains on neither
    assert labelme_mask("lines_and_points").any()
    assert not rasterize(shapes, size, CLASS_IDS).any()


def test_a_linestrip_is_not_filled_as_a_polygon():
    strip = {"label": "a", "shape_type": "linestrip", "points": [[1, 1], [9, 1], [9, 9], [1, 9]]}
    assert not rasterize([strip], (10, 10), {"a": 1}).any()


def test_a_shape_without_a_shape_type_is_a_polygon():
    points = [[1, 1], [8, 2], [5, 9]]
    without = rasterize([{"label": "a", "points": points}], (10, 10), {"a": 1})
    polygon = rasterize(
        [{"label": "a", "points": points, "shape_type": "polygon"}], (10, 10), {"a": 1}
    )
    assert without.any() and (without == polygon).all()


def test_a_rectangle_given_by_its_four_corners_is_filled_as_before():
    corners = [[1, 1], [8, 1], [8, 6], [1, 6]]
    mask = rasterize(
        [{"label": "a", "shape_type": "rectangle", "points": corners}], (10, 10), {"a": 1}
    )
    assert mask[1:7, 1:9].all() and mask.sum() == 8 * 6


def test_a_mask_png_smaller_than_its_box_leaves_the_rest_outside():
    # labelme 7.9.1 stops here, unable to fit the PNG to the box; the PNG's pixels are all it has
    box, inside = place_mask(png(np.ones((2, 3))), [(2, 1), (7, 5)], (10, 10))
    assert box == (2, 1, 8, 6)
    assert np.array(inside).sum() == 6 and np.array(inside)[:2, :3].all()


def test_a_mask_drawn_backwards_with_whole_numbers_covers_nothing():
    # As in labelme: the box runs from the first point, so it has no size
    assert place_mask(png(np.ones((4, 4))), [(5, 5), (2, 2)], (10, 10)) is None


def test_mask_pixels_are_inside_wherever_they_are_not_zero():
    pixels = [[0, 1, 255], [7, 0, 0]]
    mask = rasterize([mask_shape([[0.5, 0.5], [2.5, 1.5]], pixels)], (4, 3), {"a": 1})
    # Fractional corners: at the first one, rounded to even (0.5 to 0), the PNG's own size
    assert mask[:2, :3].tolist() == [[0, 1, 1], [1, 0, 0]] and not mask[2].any()


def make_project(root, docs, classes=("a", "b", "c")):
    (root / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    for name, shapes in docs.items():
        Image.new("RGB", (64, 48)).save(root / f"{name}.png")
        (root / f"{name}.json").write_text(json.dumps({"shapes": shapes}))


def test_prepare_counts_the_shapes_it_leaves_out(tmp_path):
    lines, _ = load("lines_and_points")
    polygon, _ = load("polygon")
    make_project(tmp_path, {"lines": lines, "polygon": polygon + lines[:1]})
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["skipped_shapes"] == {"line": 2, "linestrip": 1, "point": 1, "points": 1}
    # An image with nothing to train on counts as unlabelled
    assert summary["labelled"] == 1 and summary["unlabelled"] == 1


def test_prepare_trains_on_every_area_shape(tmp_path):
    make_project(tmp_path, {case: load(case)[0] for case in AREA_CASES})
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["labelled"] == len(AREA_CASES) and summary["skipped_shapes"] == {}
    for case in AREA_CASES:
        tile = next((tmp_path / ".myseg" / "dataset").glob(f"*/masks/{case}__0_0.png"))
        assert (np.array(Image.open(tile)) == labelme_mask(case)).all()


def test_a_mask_without_pixels_is_left_out(tmp_path):
    make_project(
        tmp_path, {"x": [{"label": "a", "shape_type": "mask", "points": [[1, 1], [5, 5]]}]}
    )
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["skipped_shapes"] == {"mask": 1} and summary["labelled"] == 0


def test_a_mask_png_with_colour_channels_stops_prepare(tmp_path):
    rgb = mask_shape([[1, 1], [2, 2]], [[0, 1], [1, 1]])
    rgb["mask"] = png(np.ones((2, 2, 3)))
    make_project(tmp_path, {"x": [rgb]})
    with pytest.raises(SystemExit, match="x.json can't be read: A mask shape's PNG must have one"):
        prepare(tmp_path, 0.5, 1024)
