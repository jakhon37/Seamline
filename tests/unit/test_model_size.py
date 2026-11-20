"""A model's depth, decoder width and attention, from the run's options to its export, and the
estimate of what a model costs before it trains."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch
import yaml
from PIL import Image

from engine import studio
from src.models import create_model
from src.models.estimate import activation_bytes, build_meta, count_flops, estimate, zeroed
from src.models.segmentation import model_kwargs

REPO = Path(__file__).resolve().parents[2]


def params(model):
    return sum(p.numel() for p in model.parameters())


def test_a_model_without_the_new_settings_is_smps_own():
    for architecture, cls in (("Unet", smp.Unet), ("PSPNet", smp.PSPNet), ("FPN", smp.FPN)):
        with torch.device("meta"):
            ours = create_model(architecture, "resnet18", encoder_weights=None, num_classes=3)
            theirs = cls(encoder_name="resnet18", encoder_weights=None, classes=3)
        assert params(ours) == params(theirs)
        assert model_kwargs(architecture) == {}


def test_a_shallower_encoder_keeps_each_decoder_blocks_width_at_its_resolution():
    # Unet's blocks go from 1/16 to full size at 256, 128, 64, 32, 16 channels. Without the
    # deepest stage the first block is at 1/8, with 128.
    assert model_kwargs("Unet", encoder_depth=4) == {
        "encoder_depth": 4,
        "decoder_channels": (128, 64, 32, 16),
    }
    assert model_kwargs("UnetPlusPlus", encoder_depth=3)["decoder_channels"] == (64, 32, 16)
    # Widths that don't follow the depth stay as they are
    assert model_kwargs("DeepLabV3", encoder_depth=4) == {"encoder_depth": 4}
    # PSPNet's own depth is 3
    assert model_kwargs("PSPNet", encoder_depth=3) == {"encoder_depth": 3}


def test_decoder_width_scales_every_width_of_the_decoder():
    assert model_kwargs("Unet", decoder_width=0.5) == {"decoder_channels": (128, 64, 32, 16, 8)}
    assert model_kwargs("Unet", encoder_depth=4, decoder_width=2)["decoder_channels"] == (
        256,
        128,
        64,
        32,
    )
    assert model_kwargs("FPN", decoder_width=1.5) == {
        "decoder_pyramid_channels": 384,
        "decoder_segmentation_channels": 192,
    }
    assert model_kwargs("PAN", decoder_width=0.5) == {"decoder_channels": 16}
    assert model_kwargs("PSPNet", decoder_width=2) == {"psp_out_channels": 1024}
    assert model_kwargs("Segformer", decoder_width=0.5) == {"decoder_segmentation_channels": 128}
    with torch.device("meta"):
        narrow = create_model("Unet", "resnet18", encoder_weights=None, decoder_width=0.5)
        own = create_model("Unet", "resnet18", encoder_weights=None)
    assert params(narrow) < params(own)


def test_settings_an_architecture_cannot_take_are_refused():
    # FPN's and PSPNet's heads upsample by a fixed factor: at another depth their masks would
    # come out another size than the image
    with pytest.raises(ValueError, match="FPN gives masks the size of the image only at its own"):
        model_kwargs("FPN", encoder_depth=4)
    with pytest.raises(ValueError, match="encoder depth, 3"):
        model_kwargs("PSPNet", encoder_depth=5)
    # scSE squeezes 16 channels to 1: a half-width Unet's last block has 8
    with pytest.raises(ValueError, match="this decoder's narrowest has 8"):
        model_kwargs("Unet", decoder_width=0.5, decoder_attention_type="scse")
    assert model_kwargs("Unet", encoder_depth=3, decoder_attention_type="scse") == {
        "encoder_depth": 3,
        "decoder_channels": (64, 32, 16),
        "decoder_attention_type": "scse",
    }
    with pytest.raises(ValueError, match="Linknet's decoder has no width to change"):
        model_kwargs("Linknet", decoder_width=2)
    with pytest.raises(ValueError, match="FPN's decoder can't use scse attention"):
        model_kwargs("FPN", decoder_attention_type="scse")
    with pytest.raises(ValueError, match="encoder_depth must be one of"):
        model_kwargs("Unet", encoder_depth=6)
    with pytest.raises(ValueError, match="Unknown architecture: DPT"):
        model_kwargs("DPT")
    # A width of 1 is no change, wherever
    assert model_kwargs("Linknet", decoder_width=1) == {}


def make_run(root, settings, weights=None):
    """A run folder as training leaves it, its config written from these options"""
    options = {
        "architecture": "Unet",
        "encoder": "resnet18",
        "encoder_weights": None,
        "loss": "dice",
        "epochs": 1,
        "batch_size": 2,
        "image_size": 64,
        "learning_rate": 0.001,
        "tile": 512,
        **settings,
    }
    (root / ".myseg").mkdir(exist_ok=True)
    project = {"name": "t", "classes": [{"name": c, "color": "#fff"} for c in ("a", "b")]}
    (root / ".myseg" / "project.json").write_text(json.dumps(project))
    run_dir = root / ".myseg" / "runs" / "r1"
    (run_dir / "run_1").mkdir(parents=True)
    config = studio.run_config(root, options, run_dir, 3)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "options.json").write_text(json.dumps(options))
    (run_dir / "classes.json").write_text(json.dumps(["a", "b"]))
    if weights is not None:
        torch.save(weights, run_dir / "run_1" / "best_model.pth")
    return run_dir, config


def test_predict_and_export_build_the_network_the_run_trained(tmp_path):
    settings = {"encoder_depth": 4, "decoder_width": 1.5, "decoder_attention_type": "scse"}
    model = create_model("Unet", "resnet18", encoder_weights=None, num_classes=3, **settings)
    run_dir, config = make_run(tmp_path, settings, model.state_dict())
    assert {k: config["model"][k] for k in settings} == settings
    # The saved weights fit only that network
    with pytest.raises(RuntimeError):
        create_model("Unet", "resnet18", encoder_weights=None, num_classes=3).load_state_dict(
            model.state_dict()
        )
    loaded, _ = studio.load_model(run_dir, "r1", "cpu")
    assert params(loaded) == params(model)
    exported = studio.export(tmp_path, "r1")
    assert exported["input_size"] == [64, 64] and exported["classes"] == ["a", "b"]
    assert sorted(json.loads((run_dir / "model.json").read_text())) == [
        "classes",
        "input_size",
        "mean",
        "std",
        "tile",
    ]


def test_a_config_from_before_the_settings_loads_as_it_did(tmp_path):
    # As runs were saved until now: no depth, width or attention in the config
    model = create_model("PSPNet", "resnet18", encoder_weights=None, num_classes=3)
    run_dir, config = make_run(tmp_path, {"architecture": "PSPNet"}, model.state_dict())
    for key in ("encoder_depth", "decoder_width", "decoder_attention_type"):
        config["model"].pop(key)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    loaded, _ = studio.load_model(run_dir, "r1", "cpu")
    assert params(loaded) == params(model)
    assert studio.export(tmp_path, "r1")["input_size"] == [64, 64]


def test_run_config_takes_the_advanced_settings(tmp_path):
    options = {
        "architecture": "FPN",
        "encoder": "resnet34",
        "loss": "focal",
        "epochs": 12,
        "batch_size": 4,
        "image_size": 256,
        "learning_rate": 0.0003,
        "encoder_depth": 4,
        "decoder_width": 1.5,
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "augmentation": "strong",
        "seed": 7,
        "early_stopping_patience": 5,
    }
    config = studio.run_config(tmp_path, options, tmp_path, 3)
    assert config["model"]["encoder_depth"] == 4 and config["model"]["decoder_width"] == 1.5
    assert config["model"]["decoder_attention_type"] is None
    assert config["scheduler"]["type"] == "cosine" and config["scheduler"]["warmup_epochs"] == 2
    assert config["experiment"]["seed"] == 7 and config["early_stopping"]["patience"] == 5
    assert config["augmentation"]["strength"] == "strong"
    assert config["augmentation"]["train"]["rotate_p"] == 0.8
    # None turns every augmentation off, and the others keep the config's own keys
    none = studio.run_config(tmp_path, {**options, "augmentation": "none"}, tmp_path, 3)
    assert all(v == 0 for k, v in none["augmentation"]["train"].items() if k.endswith("_p"))
    default = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    for strength in studio.AUGMENTATIONS:
        built = studio.run_config(tmp_path, {**options, "augmentation": strength}, tmp_path, 3)
        assert set(built["augmentation"]["train"]) == set(default["augmentation"]["train"])


@pytest.mark.parametrize(
    "change,message",
    [
        ({"scheduler": "step"}, "scheduler must be one of reduce_on_plateau, cosine, constant"),
        ({"augmentation": "wild"}, "augmentation must be one of none, light, standard, strong"),
        ({"warmup_epochs": -1}, "warmup_epochs must be a whole number of at least 0"),
        ({"early_stopping_patience": 0}, "early_stopping_patience must be a whole number"),
        ({"seed": 1.5}, "seed must be a whole number"),
        ({"architecture": "DeepLabV3", "batch_size": 1}, "DeepLabV3 trains on batches of 2"),
        ({"image_size": 32}, "Unet needs an input size of 64 or more"),
        ({"architecture": "PAN", "image_size": 96}, "PAN needs an input size of 128 or more"),
    ],
)
def test_run_config_refuses_settings_training_cannot_use(tmp_path, change, message):
    options = {
        "architecture": "Unet",
        "encoder": "resnet18",
        "loss": "dice",
        "epochs": 1,
        "batch_size": 2,
        "image_size": 64,
        "learning_rate": 0.001,
        **change,
    }
    with pytest.raises(SystemExit, match=message):
        studio.run_config(tmp_path, options, tmp_path, 3)


@pytest.mark.parametrize(
    "architecture,encoder,shape",
    [
        ("Unet", "resnet18", {}),
        ("Unet", "efficientnet-b0", {"encoder_depth": 4, "decoder_attention_type": "scse"}),
        ("FPN", "resnet34", {"decoder_width": 0.5}),
        ("DeepLabV3Plus", "mobilenet_v2", {}),
    ],
)
def test_the_memory_counted_without_data_is_what_a_real_step_keeps(architecture, encoder, shape):
    """On the meta device nothing is allocated, yet the tensors kept for the backward pass are
    those of a real forward pass, byte for byte. Their sum is the bytes of every distinct storage
    a real step keeps alive until its backward pass."""
    settings = {"architecture": architecture, "encoder": encoder, **shape}
    # As training builds it, giving logits
    real = create_model(
        architecture, encoder, encoder_weights=None, num_classes=3, activation=None, **shape
    )
    counted = activation_bytes(build_meta(settings, 3), 64, 2)
    assert counted == activation_bytes(real, 64, 2)

    # Counted another way on the real model: the storages the graph still holds at its end,
    # weights left out
    held = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: held.append(t) or t, lambda t: t):
        output = real.train()(torch.zeros(2, 3, 64, 64))
    alive = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in held}
    for p in real.parameters():
        alive.pop(p.untyped_storage().data_ptr(), None)
    # Holding them all here keeps the dead branches' tensors too, so this is the most there is
    assert counted <= sum(alive.values())
    del output


def test_the_estimate_counts_the_model_as_built():
    settings = {"architecture": "Unet", "encoder": "resnet18", "encoder_depth": 4}
    result = estimate(settings, 128, 4, 3, timing=False)
    with torch.device("meta"):
        built = create_model(
            "Unet", "resnet18", encoder_weights=None, num_classes=3, encoder_depth=4
        )
    assert result["params"] == params(built)
    assert result["ms_per_tile"] is None and result["device"] is None
    # Convolutions' work grows with the pixels, and kept tensors with the batch
    larger = estimate(settings, 256, 8, 3, timing=False)
    assert larger["gflops"] == pytest.approx(4 * result["gflops"], rel=0.01)
    held = 16 * result["params"]
    assert larger["memory"]["float32"] - held == pytest.approx(
        8 * (result["memory"]["float32"] - held), rel=0.01
    )
    # Mixed precision keeps them in 16 bits: about half
    assert 0.4 < (result["memory"]["mixed"] - held) / (result["memory"]["float32"] - held) < 0.6


def test_flops_are_two_per_multiply_add():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1, bias=False))
    # 8 x 3 x 3 x 3 multiply-adds at each of 16 x 16 pixels
    assert count_flops(model, 16) == 2 * 8 * 3 * 3 * 3 * 16 * 16


def test_the_estimate_times_a_tile_on_this_machine():
    result = estimate({"architecture": "Unet", "encoder": "resnet18"}, 64, 2, 2)
    assert result["device"] == "CPU" and result["ms_per_tile"] > 0 and result["runs"] >= 2


def test_timing_runs_on_zeros_without_initialising_weights():
    model = zeroed(
        build_meta({"architecture": "Unet", "encoder": "resnet18"}, 3), torch.device("cpu")
    )
    assert all(not p.is_meta and not p.any() for p in model.parameters())
    assert model.eval()(torch.rand(1, 3, 64, 64)).shape == (1, 3, 64, 64)


def test_estimates_are_served_one_a_line_each_with_its_id():
    asked = [
        {
            "id": 1,
            "options": {"architecture": "Unet", "encoder": "resnet18", "image_size": 64},
            "time": False,
        },
        "not json",
        {
            "id": 3,
            "options": {"architecture": "Linknet", "encoder": "resnet18", "decoder_width": 2},
        },
        {
            "id": 4,
            "options": {
                "architecture": "Unet",
                "encoder": "resnet18",
                "image_size": 64,
                "batch_size": 2,
            },
        },
    ]
    answers = io.StringIO()
    lines = [a if isinstance(a, str) else json.dumps(a) for a in asked]
    studio.serve_estimates(io.StringIO("\n".join(lines) + "\n\n"), answers)
    got = [json.loads(line) for line in answers.getvalue().splitlines()]
    assert [a["id"] for a in got] == [1, None, 3, 4]
    assert got[0]["params"] > 0 and got[0]["ms_per_tile"] is None
    assert "error" in got[1]
    assert got[2] == {"id": 3, "error": "Linknet's decoder has no width to change"}
    assert got[3]["ms_per_tile"] > 0 and got[3]["batch_size"] == 2


def test_the_estimate_command_prints_one_answer(tmp_path):
    options = {"architecture": "FPN", "encoder": "resnet18", "image_size": 64, "num_classes": 3}
    done = subprocess.run(
        [sys.executable, str(REPO / "engine" / "studio.py"), "estimate", "--options"]
        + [json.dumps(options), "--no-time"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    [line] = done.stdout.splitlines()
    answer = json.loads(line)
    assert answer["params"] == params(
        create_model("FPN", "resnet18", encoder_weights=None, num_classes=3)
    )


def test_a_run_trains_with_every_advanced_setting(tmp_path):
    """The whole way, as the app starts a run: options to config, scripts/train.py in its own
    process, the run's events and its export"""
    (tmp_path / ".myseg").mkdir()
    project = {"name": "t", "classes": [{"name": "a", "color": "#fff"}]}
    (tmp_path / ".myseg" / "project.json").write_text(json.dumps(project))
    for i in range(4):
        image = Image.new("RGB", (64, 64), (40 * i, 90, 90))
        image.save(tmp_path / f"i{i}.png")
        shape = {"label": "a", "shape_type": "rectangle", "points": [[8, 8], [40, 40]]}
        (tmp_path / f"i{i}.json").write_text(json.dumps({"shapes": [shape]}))
    options = {
        "architecture": "Unet",
        "encoder": "resnet18",
        "encoder_weights": None,
        "loss": "combined",
        "epochs": 3,
        "batch_size": 2,
        "image_size": 64,
        "learning_rate": 0.003,
        "val_split": 0.5,
        "tile": 1024,
        "num_workers": 0,
        "encoder_depth": 3,
        "decoder_width": 1.5,
        "decoder_attention_type": "scse",
        "scheduler": "constant",
        "warmup_epochs": 2,
        "augmentation": "light",
        "seed": 7,
        "early_stopping_patience": 10,
    }
    assert studio.train(tmp_path, options, run="r1") == 0
    run_dir = tmp_path / ".myseg" / "runs" / "r1"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    [cost] = [e for e in events if e["type"] == "model"]
    assert cost["device"] == "CPU" and cost["image_size"] == 64 and cost["ms_per_tile"] > 0
    assert cost["params"] == params(
        create_model(
            "Unet",
            "resnet18",
            encoder_weights=None,
            num_classes=2,
            encoder_depth=3,
            decoder_width=1.5,
            decoder_attention_type="scse",
        )
    )
    # Warm-up climbs to the rate set, which stays: a third, two thirds, then all of it
    rates = [e["lr"] for e in events if e["type"] == "epoch"]
    assert rates == pytest.approx([0.001, 0.002, 0.003])
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert config["experiment"]["seed"] == 7
    assert config["model"]["decoder_attention_type"] == "scse"
    assert studio.export(tmp_path, "r1")["input_size"] == [64, 64]
