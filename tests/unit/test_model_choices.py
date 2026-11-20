"""Every network the Train page offers trains and goes to Seamtag.

The page offers what app/src/shared/models.json lists. Each architecture and encoder pair there,
and each encoder depth, decoder width and attention it offers, is built from a run's config as
training builds it, trained a step on a tiny dataset, exported to ONNX as the app exports it and
checked in onnxruntime against PyTorch. Each architecture with resnet18 and each encoder with Unet,
and each architecture's depths, attention and widest decoder with resnet18, run everywhere; the
rest are marked slow.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
import yaml
from PIL import Image

from engine import studio
from src.data import create_dataloaders
from src.losses import create_loss
from src.models import create_model
from src.models.segmentation import (
    ARCHITECTURES,
    ATTENTION_TYPES,
    DECODER_WIDTHS,
    ENCODER_DEPTHS,
    model_kwargs,
    shape_of,
)

REPO = Path(__file__).resolve().parents[2]
CHOICES = json.loads((REPO / "app" / "src" / "shared" / "models.json").read_text())
OFFERED = CHOICES["architectures"]
ENCODERS = CHOICES["encoders"]
CLASSES = ["a", "b"]


def quick(architecture, encoder, shape):
    """Run everywhere: each architecture with resnet18 and each encoder with Unet, and with
    resnet18 each depth, attention and the widest decoder an architecture offers"""
    if shape:
        return encoder == "resnet18" and shape.get("decoder_width", 2) == 2
    return encoder == "resnet18" or architecture == "Unet"


def param(architecture, encoder, **shape):
    marks = () if quick(architecture, encoder, shape) else (pytest.mark.slow,)
    name = "-".join([architecture, encoder, *(f"{k}={v}" for k, v in shape.items())])
    return pytest.param(architecture, encoder, shape, marks=marks, id=name)


def pairs():
    """Each architecture with each encoder it takes, at its own depth and width"""
    for architecture, info in OFFERED.items():
        for encoder in ENCODERS:
            if encoder not in info["without"]:
                yield param(architecture, encoder)


def shapes():
    """Each depth, decoder width and attention an architecture offers, with each encoder"""
    for architecture, info in OFFERED.items():
        for encoder in ENCODERS:
            if encoder in info["without"]:
                continue
            for depth in info["depths"]:
                if depth != info["depth"]:
                    yield param(architecture, encoder, encoder_depth=depth)
            for attention in info["attention"]:
                yield param(architecture, encoder, decoder_attention_type=attention)
        if info["decoderWidth"]:
            for width in CHOICES["decoderWidths"]:
                if width != 1:
                    yield param(architecture, "resnet18", decoder_width=width)
                # Attention with each width wide enough for it, at the narrowest depth
                if width != 1 and width >= info.get("attentionMinWidth", width + 1):
                    for attention in info["attention"]:
                        shape = {"encoder_depth": min(info["depths"]), "decoder_width": width}
                        yield param(
                            architecture, "resnet18", decoder_attention_type=attention, **shape
                        )


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """Tiny prepared datasets, as the app's prepare leaves them, at each size a test needs:
    two tiles each for training and validation, with both classes and background"""
    made = {}

    def at(size):
        if size not in made:
            root = tmp_path_factory.mktemp(f"tiny{size}")
            rng = np.random.default_rng(size)
            for split in ("train", "val"):
                for folder in ("images", "masks"):
                    (root / ".myseg" / "dataset" / split / folder).mkdir(parents=True)
                for i in range(2):
                    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
                    mask = np.zeros((size, size), dtype=np.uint8)
                    mask[: size // 2, : size // 2] = 1
                    mask[size // 2 :, size // 2 :] = 2
                    tile = root / ".myseg" / "dataset" / split
                    Image.fromarray(image).save(tile / "images" / f"t{i}.png")
                    Image.fromarray(mask).save(tile / "masks" / f"t{i}.png")
            project = {"name": "tiny", "classes": [{"name": c, "color": "#fff"} for c in CLASSES]}
            (root / ".myseg" / "project.json").write_text(json.dumps(project))
            made[size] = root
        return made[size]

    return at


def options_for(architecture, encoder, size, **shape):
    """The app's options for a short run of this network, without weights to download"""
    return {
        "architecture": architecture,
        "encoder": encoder,
        "encoder_weights": None,
        "loss": "combined",
        "epochs": 1,
        "batch_size": 2,
        "image_size": size,
        "learning_rate": 0.001,
        "val_split": 0.5,
        "tile": 1024,
        "num_workers": 0,
        **shape,
    }


def train_a_step_and_export(root, options, run):
    """A run as the app makes one, trained one step: the run's config from the options, the
    model built from that config as training builds it, a step on a batch from the dataset, the
    model saved as training saves its best, and exported as the app exports a run. Gives the
    trained model and the export's answer."""
    run_dir = root / ".myseg" / "runs" / run
    run_dir.mkdir(parents=True)
    config = studio.run_config(root, options, run_dir, len(CLASSES) + 1)
    # In this process, without workers or a GPU
    config["training"].update(mixed_precision=False, persistent_workers=False, prefetch_factor=None)
    config["data"]["cache_validation"] = False
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "options.json").write_text(json.dumps(options))
    (run_dir / "classes.json").write_text(json.dumps(CLASSES))

    torch.manual_seed(0)
    model_cfg = config["model"]
    model = create_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        num_classes=model_cfg["num_classes"],
        activation=None,
        **shape_of(model_cfg),
    )
    train_loader = create_dataloaders(config)[0]
    images, masks = next(iter(train_loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"])
    model.train()
    loss = create_loss(config)(model(images), masks)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    (run_dir / "run_1").mkdir()
    torch.save(model.state_dict(), run_dir / "run_1" / "best_model.pth")
    return model, studio.export(root, run)


def check_round_trip(root, architecture, encoder, shape, run):
    size = max(64, OFFERED[architecture]["minSize"])
    options = options_for(architecture, encoder, size, **shape)
    model, exported = train_a_step_and_export(root, options, run)
    run_dir = root / ".myseg" / "runs" / run

    # Seamtag's side is as it was: the sidecar keeps its keys, and the model its input and output
    sidecar = json.loads((run_dir / "model.json").read_text())
    assert sorted(sidecar) == ["classes", "input_size", "mean", "std", "tile"]
    assert sidecar["classes"] == CLASSES and sidecar["input_size"] == [size, size]
    assert exported["model"] == str(run_dir / "model.onnx")

    # export checked one tile in onnxruntime; any number of tiles at once score as in PyTorch
    images = torch.rand(2, 3, size, size, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model.eval()(images).numpy()
    session = onnxruntime.InferenceSession(
        str(run_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    scores = session.run(None, {"input": images.numpy()})[0]
    assert scores.shape == (2, len(CLASSES) + 1, size, size)
    assert np.abs(scores - expected).max() <= 1e-3 * max(1.0, float(np.abs(expected).max()))


@pytest.mark.parametrize("architecture,encoder,shape", list(pairs()))
def test_each_offered_network_trains_a_step_and_exports_for_seamtag(
    datasets, architecture, encoder, shape
):
    root = datasets(max(64, OFFERED[architecture]["minSize"]))
    check_round_trip(root, architecture, encoder, shape, f"{architecture}-{encoder}")


@pytest.mark.parametrize("architecture,encoder,shape", list(shapes()))
def test_each_offered_depth_width_and_attention_trains_and_exports(
    datasets, architecture, encoder, shape
):
    root = datasets(max(64, OFFERED[architecture]["minSize"]))
    name = "-".join([architecture, encoder, *(f"{k}{v}" for k, v in shape.items())])
    check_round_trip(root, architecture, encoder, shape, name)


def test_the_page_offers_only_architectures_the_engine_builds():
    assert set(OFFERED) <= set(ARCHITECTURES)
    # DPT takes only timm's ViT encoders, none of which the page offers
    assert "DPT" not in OFFERED
    for architecture, info in OFFERED.items():
        assert info["depth"] == smps_own_depth(architecture), architecture
        assert info["depth"] in info["depths"]
        assert info["decoderWidth"] == (architecture in DECODER_WIDTHS), architecture
        assert info["attention"] == list(ATTENTION_TYPES.get(architecture, ())), architecture
        assert info["minBatch"] == (2 if architecture in studio.POOLED else 1), architecture
        assert info["minSize"] == studio.MIN_SIZE.get(architecture, studio.SMALLEST), architecture
        assert set(info["without"]) <= set(ENCODERS)
        # The engine takes exactly the depths offered, and refuses the others
        for depth in ENCODER_DEPTHS:
            if depth in info["depths"]:
                model_kwargs(architecture, encoder_depth=depth)
            else:
                with pytest.raises(ValueError):
                    model_kwargs(architecture, encoder_depth=depth)
        # and attention only with a decoder wide enough for it, at every depth
        for attention in info["attention"]:
            for depth in info["depths"]:
                for width in CHOICES["decoderWidths"]:
                    shape = dict(encoder_depth=depth, decoder_width=width)
                    if width >= info["attentionMinWidth"]:
                        model_kwargs(architecture, decoder_attention_type=attention, **shape)
                    else:
                        with pytest.raises(ValueError, match="attention needs decoder blocks"):
                            model_kwargs(architecture, decoder_attention_type=attention, **shape)


def smps_own_depth(architecture):
    from src.models.segmentation import default_argument

    return default_argument(architecture, "encoder_depth")


def test_no_offered_weights_are_for_non_commercial_use_only():
    # NVIDIA's MiT weights, SegFormer's encoder, are for non-commercial use only
    assert not [e for e in ENCODERS if e.startswith("mit_")]
    for encoder, info in ENCODERS.items():
        assert info["licence"] in ("BSD-3-Clause", "Apache-2.0", "MIT"), encoder
        assert info["weights"] and info["family"], encoder


def test_presets_grow_within_one_family_from_the_recipe():
    recipe = CHOICES["recipe"]
    presets = [CHOICES["presets"][size] for size in ("fast", "balanced", "accurate")]
    assert len({ENCODERS[p["encoder"]]["family"] for p in presets}) == 1
    sizes = []
    for preset in presets:
        assert preset["encoder"] not in OFFERED[recipe["architecture"]]["without"]
        model = meta_model(recipe["architecture"], preset["encoder"])
        sizes.append(sum(p.numel() for p in model.parameters()))
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def meta_model(architecture, encoder):
    with torch.device("meta"):
        return create_model(architecture, encoder, encoder_weights=None, num_classes=3)


def test_the_recipe_is_what_runs_had_before_there_was_a_choice(tmp_path):
    """A run's options without the new settings train as the page's recipe does, so a run from
    before them is duplicated as it trained"""
    recipe = CHOICES["recipe"]
    old = studio.run_config(tmp_path, options_for("Unet", "resnet34", 512), tmp_path, 3)
    chosen = {k: v for k, v in recipe.items() if k != "architecture"}
    new = studio.run_config(tmp_path, options_for("Unet", "resnet34", 512, **chosen), tmp_path, 3)
    # Without a depth a run takes the architecture's own, which the recipe names
    assert old["model"].pop("encoder_depth") is None
    assert new["model"].pop("encoder_depth") == recipe["encoder_depth"] == smps_own_depth("Unet")
    assert old == new
    assert new["scheduler"]["type"] == recipe["scheduler"]
    assert new["augmentation"]["strength"] == recipe["augmentation"]
    assert new["experiment"]["seed"] == recipe["seed"]
    assert new["early_stopping"]["patience"] == recipe["early_stopping_patience"]


@pytest.mark.parametrize("architecture", list(OFFERED))
def test_the_smallest_batch_and_input_each_architecture_trains_on(architecture):
    info = OFFERED[architecture]
    size = info["minSize"]

    def step(batch):
        torch.manual_seed(0)
        model = create_model(architecture, "resnet18", encoder_weights=None, num_classes=3)
        model.train()(torch.rand(batch, 3, size, size)).sum().backward()

    step(info["minBatch"])
    if info["minBatch"] > 1:
        with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
            step(1)
