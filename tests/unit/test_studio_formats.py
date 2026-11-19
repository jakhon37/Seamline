"""Seamline finds and reads label files as Seamtag does: LabelMe, Pascal VOC, YOLO and CreateML."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.studio import (
    find_labels,
    list_images,
    looks_like_createml,
    prepare,
    read_classes,
    read_createml,
    read_label_file,
    read_labelme,
    read_text,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "seamtag"
# LabelImg's own files, other tools' and broken ones, in folders as a project has them
FOLDER = FIXTURES / "labelimg"
# What Seamtag's own findLabels and readLabels give for each image there: see the README
SEAMTAG = json.loads((FIXTURES / "labelimg.json").read_text())


def format_of(file):
    if file is None:
        return None
    if file.suffix == ".json":
        return "createml" if looks_like_createml(read_text(file)) else "labelme"
    return {".xml": "voc", ".txt": "yolo"}[file.suffix]


def write_project(root, classes):
    (root / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))


def test_the_same_images_are_listed():
    names = [p.relative_to(FOLDER).as_posix() for p in list_images(FOLDER)]
    assert sorted(names) == sorted(SEAMTAG)
    # In a stable order: by folder, then by name
    assert names == sorted(names, key=lambda name: name.split("/"))


@pytest.mark.parametrize("name", sorted(SEAMTAG))
def test_labels_are_found_and_read_as_seamtag_reads_them(name):
    expected, image = SEAMTAG[name], FOLDER / name
    file = find_labels(image)
    assert format_of(file) == expected["format"]
    if file is None:
        return
    size = Image.open(image).size
    if "error" in expected:
        with pytest.raises(ValueError) as error:
            read_label_file(file, image, size)
        # Each language's JSON parser words its own errors
        if not expected["error"].startswith("It isn't valid JSON"):
            assert str(error.value) == expected["error"]
        return
    fmt, shapes = read_label_file(file, image, size)
    assert fmt == expected["format"]
    # A CreateML list that doesn't name the image gives Seamtag null: no labels
    keys = ("label", "shape_type", "points", "flags")
    got = [[s[k] for k in keys] for s in shapes]
    want = [[s[k] for k in keys] for s in expected["shapes"] or []]
    assert got == want


def test_boxes_keep_their_fractions_and_difficult_marks():
    by_hand = FOLDER / "other" / "by-hand"
    _, shapes = read_label_file(by_hand.with_suffix(".xml"), by_hand.with_suffix(".png"))
    assert shapes[0]["points"] == [[10.5, 20], [30, 40.25]]
    assert shapes[0]["flags"] == {"difficult": True}


def test_a_createml_list_that_does_not_name_the_image_holds_none_of_its_labels():
    text = (FOLDER / "other" / "elsewhere.json").read_text()
    assert read_createml(text, "elsewhere.png") is None
    assert read_createml(text, "somewhere.png")[0]["points"] == [[8.0, 8.0], [12.0, 12.0]]


# LabelMe files and what Seamtag says of them, which the app's e2e/shared.spec.ts reads too
LABELME_CHECKS = json.loads((FIXTURES / "labelme_checks.json").read_text(encoding="utf-8"))
# Each language's JSON parser words its own errors after these words
NOT_JSON = "It isn't valid JSON"


@pytest.mark.parametrize("case", LABELME_CHECKS, ids=[case["name"] for case in LABELME_CHECKS])
def test_labelme_files_are_read_or_refused_in_seamtags_words(case):
    if case["error"] is None:
        assert read_labelme(case["text"]) == json.loads(case["text"])["shapes"]
        return
    with pytest.raises(ValueError) as refused:
        read_labelme(case["text"])
    if case["error"] == NOT_JSON:
        assert str(refused.value).startswith(f"{NOT_JSON} (")
    else:
        assert str(refused.value) == case["error"]


def test_a_text_file_is_yolos_only_beside_classes_txt(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.png")
    (tmp_path / "photo.txt").write_text("a cat on a mat\n")
    assert find_labels(tmp_path / "photo.png") is None
    (tmp_path / "classes.txt").write_text("cat\n")
    assert find_labels(tmp_path / "photo.png") == tmp_path / "photo.txt"


def test_classes_txt_is_never_an_images_labels(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "classes.png")
    (tmp_path / "classes.txt").write_text("cat\n")
    assert find_labels(tmp_path / "classes.png") is None


def test_byte_order_marks_are_read_past():
    # XML and YOLO files read past one, and so does classes.txt, as in Seamtag 3.3, although
    # LabelImg and Seamtag 3.2.0 keep it in the first name: there it would make the dogs a class
    # of their own, which looks like dog in every list
    _, voc = read_label_file(FOLDER / "bom" / "voc.xml", FOLDER / "bom" / "voc.png")
    assert voc[0]["label"] == "car"
    box = FOLDER / "bom-yolo" / "box"
    _, yolo = read_label_file(box.with_suffix(".txt"), box.with_suffix(".png"), (64, 48))
    assert [s["label"] for s in yolo] == ["dog", "cat"]
    assert read_classes("\N{BYTE ORDER MARK}\ndog\n") == ["", "dog"]
    # JSON doesn't: JavaScript's parser refuses one, and so does Python's
    for name in ("labelme", "createml"):
        with pytest.raises(ValueError, match="It isn't valid JSON"):
            read_label_file(FOLDER / "bom" / f"{name}.json", FOLDER / "bom" / f"{name}.png")


def test_prepare_trains_on_every_format_in_subfolders(tmp_path):
    shutil.copytree(FOLDER, tmp_path, dirs_exist_ok=True)
    for unreadable in ("broken", "bom"):
        shutil.rmtree(tmp_path / unreadable)
    write_project(tmp_path, ["car", "dog", "cat"])
    summary = prepare(tmp_path, 0.5, 1024)
    assert summary["formats"] == {"labelme": 1, "voc": 2, "yolo": 5, "createml": 3}
    # Empty files, a caption, a CreateML list of another image's labels, and other/by-hand,
    # whose labels are none of the classes
    assert summary["labelled"] == 11 and summary["unlabelled"] == 6
    assert summary["unknown_labels"] == sorted(["a & <b>", "고양이", "cat & dog", "café & bar&x;"])


def test_a_label_file_that_cannot_be_read_stops_prepare_and_is_named(tmp_path):
    (tmp_path / "line2").mkdir()
    for name in ("classes.txt", "bad-class.txt", "bad-class.png"):
        shutil.copy(FOLDER / "broken" / name, tmp_path / "line2" / name)
    write_project(tmp_path, ["dog"])
    message = "line2/bad-class.txt can't be read: Line 1: class 5 isn't in classes.txt"
    with pytest.raises(SystemExit, match=message):
        prepare(tmp_path, 0.5, 1024)


def test_yolo_boxes_are_shares_of_the_upright_image(tmp_path):
    # Stored 64 wide and 48 high, and shown turned a quarter: 48 wide and 64 high
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (64, 48)).save(tmp_path / "turned.jpg", exif=exif)
    (tmp_path / "classes.txt").write_text("cat\n")
    (tmp_path / "turned.txt").write_text("0 0.5 0.25 0.5 0.5\n")
    write_project(tmp_path, ["cat"])
    prepare(tmp_path, 0.5, 1024)
    tile = next((tmp_path / ".myseg" / "dataset").glob("*/masks/turned__0_0.png"))
    mask = np.array(Image.open(tile))
    rows, cols = np.nonzero(mask)
    assert mask.shape == (64, 48)
    assert (cols.min(), rows.min(), cols.max(), rows.max()) == (12, 0, 36, 32)
