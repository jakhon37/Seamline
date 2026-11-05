"""Tests for the desktop app's dataset preparation."""
import json

import numpy as np
import pytest
from PIL import Image

from engine.studio import is_val, list_images, prepare, rasterize


def make_project(root, images, classes=("a", "b")):
    (root / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    for name, size, shapes in images:
        Image.new("RGB", size).save(root / f"{name}.png")
        if shapes is not None:
            (root / f"{name}.json").write_text(json.dumps({"shapes": shapes}))


def box(label, x0, y0, x1, y1, **flags):
    return {"label": label, "shape_type": "rectangle", "points": [[x0, y0], [x1, y1]], "flags": flags}


def test_rasterize_uses_class_index_and_later_shape_wins():
    shapes = [box("a", 0, 0, 9, 9), box("b", 5, 5, 9, 9)]
    mask = rasterize(shapes, (10, 10), {"a": 1, "b": 2})
    assert mask[0, 0] == 1 and mask[6, 6] == 2


def test_prepare_pairs_exact_names_and_counts_unlabelled(tmp_path):
    images = [(f"i{n}", (32, 32), [box("a", 0, 0, 9, 9)]) for n in range(6)]
    make_project(tmp_path, images + [("nolabel", (32, 32), None)])
    summary = prepare(tmp_path, 0.3, 1024)
    assert summary["labelled"] == 6 and summary["unlabelled"] == 1
    assert summary["train"] + summary["val"] == 6 and summary["train"] and summary["val"]
    for split in ("train", "val"):
        base = tmp_path / ".myseg" / "dataset" / split
        names = sorted(p.name for p in (base / "images").iterdir())
        assert names == sorted(p.name for p in (base / "masks").iterdir())


def test_prepare_skips_unaccepted_ai_shapes(tmp_path):
    make_project(tmp_path, [("x", (32, 32), [box("a", 0, 0, 9, 9, ai=True)])])
    assert prepare(tmp_path, 0.2, 1024)["labelled"] == 0


def test_prepare_reports_unknown_labels(tmp_path):
    make_project(tmp_path, [("x", (32, 32), [box("ghost", 0, 0, 9, 9)])])
    assert prepare(tmp_path, 0.2, 1024)["unknown_labels"] == ["ghost"]


@pytest.mark.parametrize("val_split", [0.0001, 0.9999])
def test_prepare_always_fills_both_splits(tmp_path, val_split):
    images = [(f"i{n}", (32, 32), [box("a", 0, 0, 9, 9)]) for n in range(3)]
    make_project(tmp_path, images)
    summary = prepare(tmp_path, val_split, 1024)
    assert summary["train"] and summary["val"]


def test_big_image_tiles_keep_alignment_and_skip_empty(tmp_path):
    make_project(tmp_path, [("big", (400, 200), [box("a", 250, 50, 350, 150)]), ("b2", (32, 32), [box("a", 0, 0, 9, 9)])])
    prepare(tmp_path, 0.5, 100)
    files = list((tmp_path / ".myseg" / "dataset").glob("*/masks/big__*.png"))
    assert files and all(np.array(Image.open(f)).any() for f in files)
    tile = next(f for f in files if f.name == "big__300_100.png")
    # Shape spans x 250-350, y 50-150: this tile holds its lower right part
    mask = np.array(Image.open(tile))
    assert mask[0, 0] == 1 and mask[60, 60] == 0


def test_split_is_stable():
    assert is_val("abc", 0.5) == is_val("abc", 0.5)


def test_list_images_ignores_other_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    assert [p.name for p in list_images(tmp_path)] == ["a.png"]
