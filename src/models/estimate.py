"""
What a model costs before it is trained: its parameters, its work and time per tile, and the
memory training it takes.

The model is built on PyTorch's meta device, where tensors have shapes but no data, so nothing is
initialised or allocated and counting takes a fraction of a second whatever the model's size.
Only the timing runs the model for real, with zeros for weights: a convolution takes as long
whatever its numbers, and initialising weights at random takes seconds on an older CPU.
"""
import itertools
import statistics
import time
import weakref

import torch
from torch.utils.flop_counter import FlopCounterMode

from .segmentation import create_model

# Training keeps float32 weights, their gradients and AdamW's two moments: 4 numbers of 4 bytes
BYTES_PER_PARAMETER = 16


def build_meta(settings: dict, num_classes: int):
    """The model the settings describe, on the meta device"""
    with torch.device("meta"):
        return create_model(
            architecture=settings["architecture"],
            encoder=settings["encoder"],
            encoder_weights=None,
            num_classes=num_classes,
            activation=None,
            encoder_depth=settings.get("encoder_depth"),
            decoder_width=settings.get("decoder_width"),
            decoder_attention_type=settings.get("decoder_attention_type"),
        )


def count_flops(model, size: int) -> int:
    """FLOPs for one tile: two per multiply-add in convolutions and matrix products, as
    torch.utils.flop_counter counts them. Normalisation, activations and resizing are left out."""
    device = next(model.parameters()).device
    model.eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=device))
    return counter.get_total_flops()


def activation_bytes(model, size: int, batch: int, dtype=torch.float32) -> int:
    """Bytes of the tensors one training step keeps for its backward pass: what grows with the
    batch and the tile. A tensor kept by several layers, or seen through views, counts once, and
    one kept only by a part of the graph the output no longer reaches, such as the encoder stage
    a shallower model computes but doesn't use, is gone before the backward pass and isn't
    counted."""
    device = next(model.parameters()).device
    model.to(dtype).train()
    packed = []

    def pack(tensor):
        packed.append(weakref.ref(tensor))
        return tensor

    images = torch.zeros(batch, 3, size, size, device=device, dtype=dtype)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model(images)
        weights = {p.untyped_storage()._cdata for p in model.parameters()}
        kept = {}
        for ref in packed:
            tensor = ref()
            if tensor is not None and tensor.untyped_storage()._cdata not in weights:
                kept[tensor.untyped_storage()._cdata] = tensor.untyped_storage().nbytes()
        del output
    finally:
        model.float()
    return sum(kept.values())


def time_per_tile(model, size: int, device: torch.device, budget: float = 1.0) -> tuple:
    """Milliseconds the model takes for one tile on the device, the median of at least two runs
    after a first that sets things up, and as many more as fit in the budget, up to ten"""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    # Pixels as an image's: zeros through trained weights can leave numbers too small for a
    # float, which some CPUs work through many times slower
    tile = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(0)).to(device)
    times = []
    model.eval()
    with torch.inference_mode():
        model(tile)
        sync()
        start = time.perf_counter()
        while len(times) < 2 or (len(times) < 10 and time.perf_counter() - start < budget):
            began = time.perf_counter()
            model(tile)
            sync()
            times.append(time.perf_counter() - began)
    return statistics.median(times) * 1000, len(times)


def zeroed(model, device: torch.device):
    """The meta model made real on the device, every weight and buffer zero"""
    model.to_empty(device=device)
    with torch.no_grad():
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            tensor.zero_()
    return model


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def measure(model, settings: dict, image_size: int, num_classes: int, device) -> dict:
    """For a model about to train: its parameters, its FLOPs per tile, and its time per tile on
    the device it trains on. It goes back to training mode after."""
    device = torch.device(device)
    flops = count_flops(build_meta(settings, num_classes), image_size)
    ms, runs = time_per_tile(model, image_size, device)
    model.train()
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "gflops": round(flops / 1e9, 2),
        "ms_per_tile": round(ms, 1),
        "runs": runs,
        "device": device_name(device),
        "image_size": image_size,
    }


def estimate(
    settings: dict,
    image_size: int,
    batch_size: int,
    num_classes: int = 2,
    timing: bool = True,
    device: str = None,
) -> dict:
    """A model's size and costs, before training it. Memory is for training at the batch size:
    in float32, as on a CPU, and with mixed precision, as on a GPU, where the kept tensors are
    16-bit. It leaves out what the device itself needs, such as a GPU's own context."""
    started = time.perf_counter()
    model = build_meta(settings, num_classes)
    params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model, image_size)
    weights = BYTES_PER_PARAMETER * params
    full = activation_bytes(model, image_size, batch_size)
    half = activation_bytes(model, image_size, batch_size, torch.float16)
    result = {
        "params": params,
        "gflops": round(flops / 1e9, 2),
        "memory": {"float32": weights + full, "mixed": weights + half},
        "image_size": image_size,
        "batch_size": batch_size,
        "ms_per_tile": None,
        "device": None,
    }
    if timing:
        where = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ms, runs = time_per_tile(zeroed(model, where), image_size, where)
        result.update(ms_per_tile=round(ms, 1), runs=runs, device=device_name(where))
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result
