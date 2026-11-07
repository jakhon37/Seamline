"""Tests for the desktop app's dataset preparation."""
import json

import numpy as np
import pytest
from PIL import Image

from engine.studio import (
    is_val,
    list_images,
    predict_labels,
    prepare,
    rasterize,
    shapes_from_labels,
    tile_starts,
)


def make_project(root, images, classes=("a", "b")):
    (root / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    for name, size, shapes in images:
        Image.new("RGB", size).save(root / f"{name}.png")
        if shapes is not None:
            (root / f"{name}.json").write_text(json.dumps({"shapes": shapes}))


def box(label, x0, y0, x1, y1, **flags):
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": [[x0, y0], [x1, y1]],
        "flags": flags,
    }


def test_rasterize_uses_class_index_and_later_shape_wins():
    shapes = [box("a", 0, 0, 9, 9), box("b", 5, 5, 9, 9)]
    mask = rasterize(shapes, (10, 10), {"a": 1, "b": 2})
    assert mask[0, 0] == 1 and mask[6, 6] == 2


def test_rasterize_accepts_a_rectangle_drawn_backwards():
    mask = rasterize([box("a", 9, 9, 0, 0)], (10, 10), {"a": 1})
    assert mask[5, 5] == 1


def test_prepare_names_a_label_file_it_cannot_read(tmp_path):
    make_project(tmp_path, [("broken", (32, 32), None)])
    (tmp_path / "broken.json").write_text('{"shapes": [')
    with pytest.raises(SystemExit, match="broken.json"):
        prepare(tmp_path, 0.2, 1024)


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
    make_project(
        tmp_path,
        [
            ("big", (400, 200), [box("a", 250, 50, 350, 150)]),
            ("b2", (32, 32), [box("a", 0, 0, 9, 9)]),
        ],
    )
    prepare(tmp_path, 0.5, 100)
    files = list((tmp_path / ".myseg" / "dataset").glob("*/masks/big__*.png"))
    assert files and all(np.array(Image.open(f)).any() for f in files)
    tile = next(f for f in files if f.name == "big__300_100.png")
    # Shape spans x 250-350, y 50-150: this tile holds its lower right part
    mask = np.array(Image.open(tile))
    assert mask[0, 0] == 1 and mask[60, 60] == 0


@pytest.mark.parametrize(
    "size, tile, starts",
    [
        (500, 1024, [0]),
        (1024, 1024, [0]),
        (1404, 1024, [0, 380]),
        (2048, 1024, [0, 1024]),
        (3000, 1024, [0, 1024, 1976]),
    ],
)
def test_tiles_cover_the_whole_side_at_full_size(size, tile, starts):
    assert tile_starts(size, tile) == starts


def test_tiles_reach_a_shape_past_the_last_full_tile(tmp_path):
    # 40 px past the last full tile: under half a tile, and the only shape is there
    shape = box("a", 120, 10, 135, 100)
    make_project(
        tmp_path,
        [("big", (140, 107), [shape]), ("b2", (32, 32), [box("a", 0, 0, 9, 9)])],
    )
    prepare(tmp_path, 0.5, 100)
    files = list((tmp_path / ".myseg" / "dataset").glob("*/masks/big__*.png"))
    assert files and all(Image.open(f).size == (100, 100) for f in files)
    covered = np.zeros((107, 140), dtype=bool)
    for f in files:
        left, top = map(int, f.stem.split("__")[1].split("_"))
        covered[top : top + 100, left : left + 100] |= np.array(Image.open(f)) > 0
    assert (covered == (rasterize([shape], (140, 107), {"a": 1}) > 0)).all()


def test_predict_covers_the_image_and_keeps_the_surer_tile_where_they_overlap():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    answers = iter([[0.2, 0.6, 0.2], [0.05, 0.05, 0.9]])

    def model(batch):
        logits = torch.tensor(next(answers)).log()
        return logits[None, :, None, None].expand(1, 3, 8, 8)

    def transform(image):
        return {"image": torch.zeros(3, 8, 8)}

    image = np.zeros((100, 140, 3), dtype=np.uint8)
    labels, confidence = predict_labels(model, transform, image, 100, "cpu")
    # The tiles start at x 0 and x 40, and the second is surer
    assert (labels[:, :40] == 1).all() and (labels[:, 40:] == 2).all()
    assert confidence[0, 0] == pytest.approx(0.6 * 255, abs=2)
    assert confidence[0, 139] == pytest.approx(0.9 * 255, abs=2)


def test_split_is_stable():
    assert is_val("abc", 0.5) == is_val("abc", 0.5)


def test_list_images_ignores_other_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    assert [p.name for p in list_images(tmp_path)] == ["a.png"]


def test_shape_score_is_mean_confidence_inside_each_shape():
    pytest.importorskip("cv2")
    labels = np.zeros((60, 100), dtype=np.uint8)
    labels[10:30, 10:30] = 1
    labels[10:30, 60:80] = 2
    confidence = np.full((60, 100), 255, dtype=np.uint8)
    confidence[10:30, 10:30] = 128
    confidence[10:30, 60:80] = 230
    shapes = shapes_from_labels(labels, confidence, [{"name": "a"}, {"name": "b"}], 25)
    assert [s["label"] for s in shapes] == ["a", "b"]
    assert shapes[0]["score"] == pytest.approx(128 / 255, abs=0.01)
    assert shapes[1]["score"] == pytest.approx(230 / 255, abs=0.01)
    assert all(s["flags"] == {"ai": True} for s in shapes)


def test_shapes_below_min_area_are_dropped():
    pytest.importorskip("cv2")
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[2:5, 2:5] = 1
    confidence = np.full((20, 20), 200, dtype=np.uint8)
    assert shapes_from_labels(labels, confidence, [{"name": "a"}], 25) == []


def test_extra_shape_keys_do_not_change_masks():
    shape = {**box("a", 0, 0, 9, 9), "score": 0.5}
    assert rasterize([shape], (10, 10), {"a": 1})[5, 5] == 1
