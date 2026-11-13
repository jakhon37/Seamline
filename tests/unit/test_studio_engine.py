"""Tests for the desktop app's dataset preparation."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from engine import studio
from engine.studio import (
    TILE_STEM_BYTES,
    is_val,
    list_images,
    predict_labels,
    prepare,
    rasterize,
    run_classes,
    shapes_from_labels,
    tile_starts,
    tile_stem,
    train,
)
from src.data.dataset import get_image_mask_pairs

OPTIONS = {
    "image_size": 64,
    "architecture": "Unet",
    "encoder": "resnet18",
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 0.001,
    "loss": "dice",
    "val_split": 0.5,
}


def make_project(root, images, classes=("a", "b")):
    (root / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    for name, size, shapes in images:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
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


def test_list_images_walks_subfolders_as_seamtag_does(tmp_path):
    names = ["a.png", "d.avif", "e.jfif", "f.gif", "sub/b.JPG", "sub/deep/c.webp", "notes.txt"]
    # Hidden files and folders, such as .myseg, hold no images to train on
    hidden = [".myseg/dataset/x.png", ".cache/y.png", "sub/.z.png"]
    for name in names + hidden:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    # A folder linked from elsewhere isn't followed, so nothing is listed twice
    (tmp_path / "linked").symlink_to(tmp_path / "sub", target_is_directory=True)
    found = [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path)]
    assert found == names[:-1]


def test_tile_names_keep_each_images_folder(tmp_path):
    shape = [box("a", 0, 0, 9, 9)]
    images = [("img_007", (32, 32), shape), ("line2/img_007", (32, 32), shape)]
    make_project(tmp_path, images + [("100%", (32, 32), shape)])
    prepare(tmp_path, 0.5, 1024)
    tiles = {p.name for p in (tmp_path / ".myseg" / "dataset").glob("*/images/*.png")}
    assert tiles == {"img_007__0_0.png", "line2%2Fimg_007__0_0.png", "100%25__0_0.png"}


def test_the_split_hashes_names_without_their_extension(tmp_path):
    names = [f"i{n}" for n in range(6)] + [f"line2/i{n}" for n in range(6)]
    make_project(tmp_path, [(name, (32, 32), [box("a", 0, 0, 9, 9)]) for name in names])
    prepare(tmp_path, 0.5, 1024)
    sides = {name: "val" if is_val(name, 0.5) else "train" for name in names}
    assert set(sides.values()) == {"train", "val"}
    # At the top of the folder that's the file's stem, as before, so projects keep their split
    assert all(is_val(Path(f"{n}.png").stem, 0.5) == (sides[n] == "val") for n in names[:6])
    for name, side in sides.items():
        tile = name.replace("/", "%2F") + "__0_0.png"
        assert (tmp_path / ".myseg" / "dataset" / side / "images" / tile).exists()


def test_summary_counts_skipped_shapes_and_formats(tmp_path):
    line = {"label": "lane", "shape_type": "line", "points": [[0, 0], [9, 9]]}
    shapes = [box("a", 0, 0, 9, 9), box("ghost", 0, 0, 9, 9), line]
    make_project(tmp_path, [("x", (32, 32), shapes)])
    summary = prepare(tmp_path, 0.2, 1024)
    assert list(summary) == [
        "labelled",
        "unlabelled",
        "train",
        "val",
        "unknown_labels",
        "class_pixels",
        "skipped_shapes",
        "formats",
    ]
    # Only labels that could train are unknown: a line trains whatever its label
    assert summary["unknown_labels"] == ["ghost"] and summary["skipped_shapes"] == {"line": 1}
    assert summary["formats"] == {"labelme": 1, "voc": 0, "yolo": 0, "createml": 0}


def test_prepare_refuses_a_shape_seamtag_cannot_read_either(tmp_path):
    make_project(tmp_path, [("odd", (32, 32), [{**box("a", 0, 0, 9, 9), "shape_type": "blob"}])])
    with pytest.raises(
        SystemExit, match="odd.json can't be read: Shape 1 has an unknown shape_type"
    ):
        prepare(tmp_path, 0.2, 1024)


def test_train_writes_the_runs_classes_before_training_starts(tmp_path, monkeypatch):
    images = [(f"i{n}", (32, 32), [box("a", 0, 0, 9, 9)]) for n in range(4)]
    make_project(tmp_path, images, classes=("a", "b"))
    seen = []

    def call(command, cwd):
        config = Path(command[command.index("--config") + 1])
        seen.append(json.loads((config.parent / "classes.json").read_text()))
        return 0

    monkeypatch.setattr("engine.studio.subprocess.call", call)
    assert train(tmp_path, OPTIONS, run="r1") == 0
    assert seen == [["a", "b"]]


def test_a_run_that_stops_before_training_keeps_the_reason(tmp_path, monkeypatch):
    odd = {**box("a", 0, 0, 9, 9), "shape_type": "ellipse"}
    make_project(tmp_path, [("a", (32, 32), [box("a", 0, 0, 9, 9)]), ("line2/c", (32, 32), [odd])])
    monkeypatch.setattr("engine.studio.subprocess.call", lambda *args, **kwargs: pytest.fail())
    message = "line2/c.json can't be read: Shape 1 has an unknown shape_type: ellipse"
    with pytest.raises(SystemExit, match=message):
        train(tmp_path, OPTIONS, run="r1")
    # The app lists the run with its events once the live log is gone
    events = (tmp_path / ".myseg" / "runs" / "r1" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in events] == [{"type": "error", "message": message}]


def test_shapes_whose_label_is_not_a_class_are_left_out(tmp_path):
    # Labels typed in Seamtag go into its label files, not onto Seamline's class list
    images = [
        ("a", (32, 32), [box("crack", 0, 0, 9, 9)]),
        ("b", (32, 32), [box("crack", 0, 0, 9, 9)]),
        ("dent_only", (32, 32), [box("dent", 0, 0, 9, 9)]),
        ("both", (32, 32), [box("crack", 0, 0, 9, 9), box("dent", 20, 20, 29, 29)]),
    ]
    make_project(tmp_path, images, classes=("crack",))
    summary = prepare(tmp_path, 0.5, 1024)
    # An image with only those isn't labelled, rather than all background, and their labels are
    # named, so they can be added as classes
    assert summary["labelled"] == 3 and summary["unlabelled"] == 1
    assert summary["unknown_labels"] == ["dent"]
    dataset = tmp_path / ".myseg" / "dataset"
    assert not list(dataset.glob("*/images/dent_only__*"))
    # Beside a class's shapes they're left out as if not drawn
    mask = np.array(Image.open(next(dataset.glob("*/masks/both__0_0.png"))))
    assert mask[5, 5] == 1 and mask[25, 25] == 0


def test_long_names_get_tile_names_a_file_can_have(tmp_path):
    # Each part is 63 bytes or less, but a tile name holding all of them would be 258 bytes
    folder = "/".join(
        [
            "2025년_3월_12일_2공장_라인2_왼쪽카메라_오전근무",
            "품질팀_확인완료_불량품_스크래치_샘플_모음",
            "자르기_전_고해상도_원본_이미지_보관용",
            "1차_검수",
        ]
    )
    names = [f"{folder}/IMG_20250312_103455_0001_CAM1_LINE2_LOT2503120001_{n}" for n in "AB"]
    make_project(tmp_path, [(name, (32, 32), [box("a", 0, 0, 9, 9)]) for name in names])
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["train"] + summary["val"] == 2
    tiles = sorted(p.name for p in (tmp_path / ".myseg" / "dataset").glob("*/images/*.png"))
    # They keep a start to know them by, and stay apart, though only their ends differ
    assert len(tiles) == 2 and all(t.startswith(tile_stem(folder)[:40]) for t in tiles)
    assert all(len(t.encode()) <= TILE_STEM_BYTES + len("__0_0.png") for t in tiles)
    # Names up to the limit keep the tile names they had
    assert tile_stem("x" * TILE_STEM_BYTES + ".png") == "x" * TILE_STEM_BYTES


def test_training_pairs_each_tile_with_its_own_mask(tmp_path):
    # Names that are the same once _mask is left out, which pairing also accepts
    labels = {"x/a": "a", "x_mask/a": "b", "p": "a", "p_mask": "b", "z": "a"}
    images = [(name, (16, 16), [box(label, 0, 0, 15, 15)]) for name, label in labels.items()]
    make_project(tmp_path, images)
    # All but the last image train, so each pair of look-alike names is in one folder
    summary = prepare(tmp_path, 0.0001, 1024)
    assert (summary["train"], summary["val"]) == (4, 1)
    for split in ("train", "val"):
        base = tmp_path / ".myseg" / "dataset" / split
        pairs = zip(*get_image_mask_pairs(str(base / "images"), str(base / "masks")))
        assert all(Path(image).name == Path(mask).name for image, mask in pairs)


def test_yolo_boxes_train_as_they_were_counted(tmp_path, monkeypatch):
    # Seamtag can move a label file to the trash while prepare cuts the images into tiles
    make_project(tmp_path, [], classes=("cat",))
    for name in ("a", "b"):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "classes.txt").write_text("cat\n")
    opened = studio.open_image

    def trash_then_open(path):
        path.with_suffix(".txt").unlink()
        return opened(path)

    monkeypatch.setattr("engine.studio.open_image", trash_then_open)
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["labelled"] == 2 and summary["class_pixels"]["cat"] == 2 * 17 * 17


def test_16_bit_greys_train_on_the_pixels_seamtag_reads(tmp_path):
    make_project(tmp_path, [], classes=("a",))
    ramp = (np.arange(32 * 32, dtype=np.uint32).reshape(32, 32) * 64).astype(np.uint16)
    Image.fromarray(ramp).save(tmp_path / "grey16.png")
    (tmp_path / "grey16.json").write_text(json.dumps({"shapes": [box("a", 0, 0, 9, 9)]}))
    prepare(tmp_path, 0.5, 1024)
    tile = next((tmp_path / ".myseg" / "dataset").glob("*/images/grey16__0_0.png"))
    # Seamtag's pre-labelling keeps each pixel's high byte, where straight to RGB is all white
    assert (np.array(Image.open(tile)) == (ramp >> 8)[..., None]).all()


def test_turned_tiffs_train_upright_at_the_size_both_apps_show(tmp_path):
    # Uncompressed, as cameras write them: stored 48 wide and 32 high, shown turned (EXIF 6)
    make_project(tmp_path, [], classes=("a",))
    ramp = (np.arange(32 * 48).reshape(32, 48) * 43).astype(np.uint16)
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(ramp).save(tmp_path / "turned.tif", exif=exif)
    # Drawn on the upright picture, 32 wide and 48 high
    (tmp_path / "turned.json").write_text(json.dumps({"shapes": [box("a", 2, 36, 11, 45)]}))
    summary = prepare(tmp_path, 0.5, 1024)
    tile = np.array(
        Image.open(next((tmp_path / ".myseg" / "dataset").glob("*/images/turned__0_0.png")))
    )
    assert tile.shape == (48, 32, 3)
    assert (tile == np.rot90(ramp >> 8, -1)[..., None]).all()
    assert summary["class_pixels"]["a"] == 100


def swapped_profile() -> bytes:
    """sRGB's ICC profile with red and green swapped, as a photo's own profile can say"""
    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    tags = {
        bytes(data[132 + 12 * i : 136 + 12 * i]): 136 + 12 * i
        for i in range(int.from_bytes(data[128:132], "big"))
    }
    red, green = tags[b"rXYZ"], tags[b"gXYZ"]
    data[red : red + 8], data[green : green + 8] = data[green : green + 8], data[red : red + 8]
    return bytes(data)


def test_colours_go_through_the_images_own_profile_as_in_seamtag(tmp_path):
    make_project(tmp_path, [], classes=("a",))
    pixels = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "tagged.png", icc_profile=swapped_profile())
    (tmp_path / "tagged.json").write_text(json.dumps({"shapes": [box("a", 0, 0, 9, 9)]}))
    prepare(tmp_path, 0.5, 1024)
    tile = np.array(
        Image.open(next((tmp_path / ".myseg" / "dataset").glob("*/images/tagged__0_0.png")))
    )
    # libvips reads such a file with red and green swapped, for Seamtag and for the window
    assert np.abs(tile.astype(int) - pixels[..., [1, 0, 2]]).max() <= 1
    # In sRGB, as most photos come, the pixels are as stored
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(pixels).save(tmp_path / "plain.png", icc_profile=srgb)
    assert (np.array(studio.open_image(tmp_path / "plain.png")) == pixels).all()


def test_requirements_ask_for_a_pillow_that_reads_images_as_seamtag_does(tmp_path):
    # Older Pillow opens 16-bit grey PNGs as 32-bit integers, which turn white in RGB, and can't
    # open AVIF at all; pip keeps an older Pillow the line allows
    lines = (studio.REPO / "requirements.txt").read_text().splitlines()
    floor = next(line for line in lines if line.startswith("Pillow>=")).split(">=")[1]
    assert tuple(int(part) for part in floor.split(".")) >= (12, 0)
    make_project(tmp_path, [("a", (16, 16), [box("a", 0, 0, 7, 7)])], classes=("a",))
    Image.new("RGB", (16, 16), (90, 120, 150)).save(tmp_path / "b.avif")
    (tmp_path / "b.json").write_text(json.dumps({"shapes": [box("a", 0, 0, 7, 7)]}))
    assert prepare(tmp_path, 0.5, 1024)["labelled"] == 2


def test_runs_name_their_classes_from_their_own_list_first(tmp_path):
    make_project(tmp_path, [], classes=("a", "b"))
    run_dir = tmp_path / ".myseg" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    # Runs from before runs kept their classes take the project's
    assert run_classes(tmp_path, run_dir) == ["a", "b"]
    (run_dir / "classes.json").write_text(json.dumps(["x", "y"]))
    assert run_classes(tmp_path, run_dir) == ["x", "y"]
