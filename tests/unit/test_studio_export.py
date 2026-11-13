"""Exporting runs for Seamtag's pre-labelling, and predicting with them, on a small real model."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml
from PIL import Image

from engine import studio
from engine.studio import export_classes, predict, seamtag_normalize
from src.data.transforms import get_val_transforms

REPO = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """A small real model's weights, made once: as created, and changed so every pixel scores
    class 1 highest"""
    from src.models import create_model

    model = create_model(
        architecture="Unet",
        encoder="resnet18",
        encoder_weights=None,
        num_classes=3,
        activation=None,
    )
    folder = tmp_path_factory.mktemp("weights")
    torch.save(model.state_dict(), folder / "random.pth")
    head = model.segmentation_head[0]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 4.0, 0.0]))
    torch.save(model.state_dict(), folder / "class_one.pth")
    return {"random": folder / "random.pth", "class_one": folder / "class_one.pth"}


def make_run(
    root, weights=None, classes=("a", "b"), saved=None, architecture="Unet", encoder="resnet18"
):
    """A run folder as training leaves it: its options, config, best model and, for runs made
    since runs kept them, its classes"""
    run_dir = root / ".myseg" / "runs" / "r1"
    (run_dir / "run_x").mkdir(parents=True)
    if weights:
        shutil.copy(weights, run_dir / "run_x" / "best_model.pth")
    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    config["data"]["image_size"] = 64
    config["model"].update(
        architecture=architecture, encoder=encoder, encoder_weights=None, num_classes=3
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "options.json").write_text(json.dumps({"image_size": 64, "tile": 512}))
    if saved is not None:
        (run_dir / "classes.json").write_text(json.dumps(saved))
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in classes]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    return run_dir


def dims(value):
    return [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]


def test_export_writes_the_model_then_the_sidecar_seamtag_reads(
    tmp_path, weights, monkeypatch, capfd
):
    run_dir = make_run(tmp_path, weights["random"], saved=["cat", "dog"])
    monkeypatch.setattr(sys, "argv", ["studio.py", "export", str(tmp_path), "--run", "r1"])
    studio.main()
    lines = capfd.readouterr().out.strip().splitlines()
    model, sidecar = run_dir / "model.onnx", run_dir / "model.json"
    assert len(lines) == 1 and json.loads(lines[0]) == {
        "model": str(model),
        "sidecar": str(sidecar),
        "classes": ["cat", "dog"],
        "input_size": [64, 64],
        "tile": 512,
    }
    # Seamline's runs take pixels scaled to 0-1, not ImageNet's mean and spread
    assert json.loads(sidecar.read_text()) == {
        "classes": ["cat", "dog"],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "tile": 512,
        "input_size": [64, 64],
    }
    # Each is written whole and renamed into place, the sidecar last: Seamtag lists a model once
    # both are there
    assert not [p.name for p in run_dir.iterdir() if p.name.endswith(".tmp")]
    assert sidecar.stat().st_mtime_ns >= model.stat().st_mtime_ns

    # Raw scores for background and both classes, any number of images at once, each 64 x 64
    graph = onnx.load(str(model))
    onnx.checker.check_model(graph)
    assert [(i.name, dims(i)) for i in graph.graph.input] == [("input", ["batch_size", 3, 64, 64])]
    assert [(o.name, dims(o)) for o in graph.graph.output] == [
        ("output", ["batch_size", 3, 64, 64])
    ]
    assert graph.opset_import[0].version == 17
    try:
        import onnxruntime
    except ImportError:
        return
    from src.models import create_model

    built = create_model(
        architecture="Unet",
        encoder="resnet18",
        encoder_weights=None,
        num_classes=3,
        activation=None,
    )
    built.load_state_dict(torch.load(weights["random"], weights_only=True))
    images = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = built.eval()(images).numpy()
    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    scores = session.run(None, {"input": images.numpy()})[0]
    assert scores.shape == (2, 3, 64, 64) and np.abs(scores - expected).max() < 1e-3


def test_a_run_without_a_saved_model_cannot_be_exported(tmp_path):
    make_run(tmp_path)
    with pytest.raises(SystemExit, match="Run r1 has no saved model yet."):
        studio.export(tmp_path, "r1")


@pytest.mark.parametrize("encoder", ["resnet18", "efficientnet-b0"])
def test_pspnet_exports_although_torchs_exporter_refuses_its_pools(tmp_path, encoder):
    # At 64, as at the app's 512, PSPNet pools features 1/8 the image's size into 3 x 3 and
    # 6 x 6 cells, which can't all be one size, and behind EfficientNet the exporter can't see
    # their size at all
    from src.models import create_model

    model = create_model(
        architecture="PSPNet",
        encoder=encoder,
        encoder_weights=None,
        num_classes=3,
        activation=None,
    )
    torch.save(model.state_dict(), tmp_path / "psp.pth")
    run_dir = make_run(
        tmp_path, tmp_path / "psp.pth", saved=["a", "b"], architecture="PSPNet", encoder=encoder
    )
    pool = torch.nn.functional.adaptive_avg_pool2d
    studio.export(tmp_path, "r1")
    assert torch.nn.functional.adaptive_avg_pool2d is pool
    graph = onnx.load(str(run_dir / "model.onnx"))
    assert [dims(o) for o in graph.graph.output] == [["batch_size", 3, 64, 64]]
    onnxruntime = pytest.importorskip("onnxruntime")
    images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model.eval()(images).numpy()
    session = onnxruntime.InferenceSession(
        str(run_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert np.abs(session.run(None, {"input": images.numpy()})[0] - expected).max() < 1e-3


def test_a_failed_export_says_why_in_one_line_and_leaves_nothing(tmp_path, weights, monkeypatch):
    run_dir = make_run(tmp_path, weights["random"], saved=["a", "b"])
    before = sorted(p.name for p in run_dir.iterdir())

    def export(model, sample, path, **kwargs):
        Path(path).write_bytes(b"half a model")
        raise RuntimeError("Unsupported: an operator\nONNX graph:\n    )  (type 'Tensor')")

    monkeypatch.setattr(torch.onnx, "export", export)
    with pytest.raises(SystemExit) as stopped:
        studio.export(tmp_path, "r1")
    # The app shows the last line the engine writes to stderr, which is this one
    assert str(stopped.value) == "Run r1's model can't be written as ONNX: Unsupported: an operator"
    assert sorted(p.name for p in run_dir.iterdir()) == before


def same_model(file, shape):
    """An ONNX model that gives back its input"""
    names = ["input", "output"]
    tensors = [onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, shape) for n in names]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", names[:1], names[1:])], "same", tensors[:1], tensors[1:]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(file))


def test_the_runtime_check_allows_for_rounding_in_large_scores(tmp_path):
    pytest.importorskip("onnxruntime")
    same_model(tmp_path / "same.onnx", [1, 3, 4, 4])
    # PAN's scores reach tens of thousands, where float32's rounding alone passes 0.001
    scores = np.full((1, 3, 4, 4), 1.5e5, dtype=np.float32)
    studio.check_runtime(tmp_path / "same.onnx", scores + np.float32(0.17), scores)
    with pytest.raises(SystemExit, match="scores are 1.5e\\+03 off PyTorch's"):
        studio.check_runtime(tmp_path / "same.onnx", scores * np.float32(1.01), scores)
    # Small scores still need to be within 0.001
    small = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
    with pytest.raises(SystemExit, match="scores are 0.002 off PyTorch's"):
        studio.check_runtime(tmp_path / "same.onnx", small + np.float32(0.002), small)


def test_runs_from_before_runs_kept_their_classes_take_the_projects_while_they_fit(tmp_path):
    run_dir = make_run(tmp_path, classes=("a", "b"))
    assert export_classes(tmp_path, run_dir, "r1", 3) == ["a", "b"]
    with pytest.raises(SystemExit, match="Run r1 doesn't say which classes it learned"):
        export_classes(tmp_path, run_dir, "r1", 4)
    (run_dir / "classes.json").write_text(json.dumps(["x", "y", "z"]))
    assert export_classes(tmp_path, run_dir, "r1", 4) == ["x", "y", "z"]
    with pytest.raises(SystemExit, match="classes.json names 3 classes, but its model has 1"):
        export_classes(tmp_path, run_dir, "r1", 2)


@pytest.mark.parametrize(
    "normalize",
    [
        {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "max_pixel_value": 255.0},
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "max_pixel_value": 255.0},
        {"mean": [0.5, 0.25, 0.0], "std": [0.25, 0.5, 2.0], "max_pixel_value": 1.0},
        {"mean": 0.5, "std": 0.25},
        {},
    ],
)
def test_the_sidecars_mean_and_std_feed_the_model_what_training_fed_it(normalize):
    pixels = np.array([[[0, 1, 2], [127, 128, 129]], [[250, 254, 255], [3, 77, 200]]], np.uint8)
    trained = get_val_transforms(2, 2, **normalize)(image=pixels)["image"].numpy()
    mean, std = seamtag_normalize(normalize)
    # As Seamtag's pre-labelling feeds each pixel to the model
    seamtag = (pixels / 255 - np.array(mean)) / np.array(std)
    assert np.allclose(seamtag.transpose(2, 0, 1), trained, rtol=1e-5, atol=1e-5)


def test_predict_reads_images_in_subfolders_and_names_classes_as_the_run_does(tmp_path, weights):
    make_run(tmp_path, weights["class_one"], classes=("a", "b"), saved=["crack", "dent"])
    (tmp_path / "line2").mkdir()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(tmp_path / "line2" / "img_007.png")
    device, result = predict(tmp_path, "r1", ["line2/img_007.png"], 25)
    assert set(device) == {"device"}
    assert result["name"] == "line2/img_007.png" and result["size"] == [64, 48]
    # The model says class 1 everywhere: the run's first class, whatever the project says now
    assert [s["label"] for s in result["shapes"]] == ["crack"]


def test_predict_names_classes_from_the_project_for_older_runs(tmp_path, weights):
    make_run(tmp_path, weights["class_one"], classes=("a", "b"))
    Image.new("RGB", (64, 48)).save(tmp_path / "x.png")
    _, result = predict(tmp_path, "r1", ["x.png"], 25)
    assert [s["label"] for s in result["shapes"]] == ["a"]
