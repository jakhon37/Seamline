"""Tests for docs/demo/evaluate.py, which scores the app's pre-labels against the true labels."""
import json
import runpy
import sys
from pathlib import Path

EVALUATE = Path(__file__).resolve().parents[2] / "docs" / "demo" / "evaluate.py"
SIZE = (40, 30)


def square(x0, y0, x1, y1, ai=False):
    shape = {
        "label": "melt",
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "shape_type": "polygon",
        "flags": {"ai": True} if ai else {},
    }
    if ai:
        shape["score"] = 0.9
    return shape


def labels(image, shapes):
    return json.dumps(
        {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image,
            "imageData": None,
            "imageHeight": SIZE[1],
            "imageWidth": SIZE[0],
        }
    )


def test_pre_labels_are_found_beside_their_images_in_subfolders(tmp_path, monkeypatch, capsys):
    project, truth = tmp_path / "pcb", tmp_path / "truth"
    truth.mkdir()
    for rel in ["top.jpg", "line2/deep.jpg"]:
        image = project / rel
        image.parent.mkdir(parents=True, exist_ok=True)
        # The script reads the sizes from the label files, not the images
        image.write_bytes(b"")
        (truth / f"{image.stem}.json").write_text(labels(image.name, [square(5, 5, 15, 15)]))
        image.with_suffix(".json").write_text(labels(image.name, [square(5, 5, 15, 15, ai=True)]))
    # Not an image to score: Seamline's own folder can hold files of the same name
    (project / ".myseg" / "cache").mkdir(parents=True)
    (project / ".myseg" / "cache" / "deep.jpg").write_bytes(b"")
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(project), str(truth), str(out)])
    # It puts the repo on the path for itself
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(EVALUATE), run_name="__main__")
    capsys.readouterr()
    result = json.loads(out.read_text())
    assert result["true_defects_found"] == 2
    assert result["images_with_no_prediction"] == 0
    assert result["iou_per_class"]["melt"] == 1.0
    # By their paths in the project, without the extension: just the name at the top
    assert [s["image"] for s in result["shapes"]] == ["line2/deep", "top"]
