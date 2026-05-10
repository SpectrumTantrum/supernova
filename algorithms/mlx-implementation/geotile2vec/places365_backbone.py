"""Native MLX ResNet-18 + Places365 weight loader.

Reference: He et al., "Deep Residual Learning for Image Recognition" (2015) for
the architecture, and Zhou et al., "Places: A 10 million Image Database for
Scene Recognition" (TPAMI 2017) / CSAILVision/places365 repo for the
checkpoint. Used by Geo-Tile2Vec §3.3 for street-view feature extraction.

The PyTorch path of this repo loads the checkpoint via ``torch.hub`` and runs
inference with ``torchvision.models.resnet18``. To keep the MLX path torch-free
at runtime we re-implement the architecture in MLX and convert the PyTorch
checkpoint to MLX arrays once (cached on disk). The conversion step uses torch
lazily — only the first time a user requests real Places365 features on a
machine — so torch is a *build-time* dependency for the weights file, not a
runtime dependency for inference.

The cache file lives at::

    ~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz

If the cache is missing and torch isn't installed, ``load_places365_weights``
raises with the URL of the PyTorch checkpoint and the cache path so users can
side-load by running the converter on a torch-equipped machine and copying the
result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np


PLACES365_RESNET18_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

# Standard ImageNet normalisation — Places365 uses the same preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _default_cache_path() -> Path:
    return Path(os.path.expanduser("~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz"))


# ---------------------------------------------------------------------- Architecture

class _BasicBlock(nn.Module):
    """ResNet-18 BasicBlock — two 3x3 conv-BN-ReLU stacks with a residual skip."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, with_downsample: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        if with_downsample:
            self.ds_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            self.ds_bn = nn.BatchNorm(planes)
        self.with_downsample = with_downsample

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.with_downsample:
            identity = self.ds_bn(self.ds_conv(x))
        return nn.relu(out + identity)


class _ResStage(nn.Module):
    """Pair of BasicBlocks (ResNet-18 has 2 per stage). Mirrors PyTorch's ``layerN``."""

    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        with_ds = (stride != 1) or (in_planes != planes)
        self.b0 = _BasicBlock(in_planes, planes, stride=stride, with_downsample=with_ds)
        self.b1 = _BasicBlock(planes, planes, stride=1, with_downsample=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.b1(self.b0(x))


class MLXResNet18(nn.Module):
    """ResNet-18 up to global-avg-pool (no FC head). Output shape: (B, 512)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _ResStage(64, 64, stride=1)
        self.layer2 = _ResStage(64, 128, stride=2)
        self.layer3 = _ResStage(128, 256, stride=2)
        self.layer4 = _ResStage(256, 512, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 224, 224, 3) — NHWC.
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return mx.mean(x, axis=(1, 2))  # (B, 512)


# ---------------------------------------------------------------------- Preprocessing

def imagenet_preprocess(images: Iterable[np.ndarray]) -> mx.array:
    """Resize-shortest-256 → centre-crop 224 → /255 → ImageNet-normalise.

    Returns an NHWC ``mx.array`` of shape ``(B, 224, 224, 3)``, dtype float32.
    Requires Pillow for the resize/crop step.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to preprocess images for the MLX Places365 backbone. "
            "Install with `pip install pillow`."
        ) from exc

    out: list[np.ndarray] = []
    for img in images:
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Each image must be HxWx3 uint8, got {img.shape}")
        pil = Image.fromarray(img.astype(np.uint8))
        # Resize so the shorter side is 256 (matches torchvision's
        # ``transforms.Resize(256)``).
        w, h = pil.size
        if w < h:
            new_w = 256
            new_h = int(round(h * 256 / w))
        else:
            new_h = 256
            new_w = int(round(w * 256 / h))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        # Centre-crop 224x224.
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        pil = pil.crop((left, top, left + 224, top + 224))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        out.append(arr)
    if not out:
        return mx.zeros((0, 224, 224, 3), dtype=mx.float32)
    return mx.array(np.stack(out, axis=0).astype(np.float32))


# ---------------------------------------------------------------------- Weight conversion

def _map_pt_key(pt_key: str) -> tuple[str | None, bool]:
    """Translate a PyTorch ResNet-18 state-dict key to an MLX module-tree path.

    Returns (mlx_key, is_conv_weight). ``mlx_key`` is None if the entry should
    be dropped. ``is_conv_weight`` flags Conv2d weight tensors that need an
    NCHW→NHWC permutation.
    """
    if pt_key in ("fc.weight", "fc.bias"):
        return None, False
    if pt_key.endswith("num_batches_tracked"):
        return None, False
    parts = pt_key.split(".")
    # conv1.*, bn1.*
    if parts[0] in ("conv1", "bn1"):
        is_conv = parts[0] == "conv1"
        return pt_key, is_conv
    # layerN.idx.{conv1|bn1|conv2|bn2}.{...} or layerN.idx.downsample.{0|1}.{...}
    if parts[0].startswith("layer") and len(parts) >= 4:
        stage = parts[0]              # layer1, layer2, ...
        block_idx = parts[1]          # 0 or 1
        block_attr = parts[2]
        block_name = "b0" if block_idx == "0" else "b1"
        if block_attr == "downsample":
            ds_kind = parts[3]        # "0" (conv) or "1" (bn)
            tail = ".".join(parts[4:])
            mlx_attr = "ds_conv" if ds_kind == "0" else "ds_bn"
            return f"{stage}.{block_name}.{mlx_attr}.{tail}", ds_kind == "0"
        # conv1/bn1/conv2/bn2 inside the block
        tail = ".".join(parts[3:])
        is_conv = block_attr in ("conv1", "conv2")
        return f"{stage}.{block_name}.{block_attr}.{tail}", is_conv
    raise KeyError(f"Unmapped PT key: {pt_key!r}")


def _convert_pt_state_dict(pt_state: dict) -> dict[str, np.ndarray]:
    """Translate a PyTorch ResNet-18 state_dict into a flat MLX-key dict."""
    out: dict[str, np.ndarray] = {}
    for pt_key, tensor in pt_state.items():
        mlx_key, is_conv = _map_pt_key(pt_key)
        if mlx_key is None:
            continue
        # tensor may be a torch.Tensor; convert to numpy.
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if is_conv:
            # PyTorch Conv2d weight: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
            arr = np.transpose(arr, (0, 2, 3, 1))
        out[mlx_key] = arr.astype(np.float32, copy=False)
    return out


def _download_and_convert(cache_path: Path) -> dict[str, np.ndarray]:
    """Lazy-import torch, download CSAIL checkpoint, convert, cache, return."""
    try:
        import torch  # noqa: WPS433 — intentional lazy import
    except ImportError as exc:
        raise RuntimeError(
            "MLX Places365 weights are not cached and PyTorch is not installed. "
            f"Either pip-install torch so this loader can convert the official "
            f"checkpoint from {PLACES365_RESNET18_URL}, or pre-build the MLX "
            f"cache file on a torch-equipped machine and copy it to "
            f"{cache_path}. The cache is a numpy .npz mapping flat MLX-key "
            "strings to float32 arrays."
        ) from exc

    ckpt = torch.hub.load_state_dict_from_url(
        PLACES365_RESNET18_URL, map_location="cpu", weights_only=False,
    )
    raw = ckpt.get("state_dict", ckpt)
    pt_state = {k.replace("module.", ""): v for k, v in raw.items()}
    converted = _convert_pt_state_dict(pt_state)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **converted)
    return converted


def load_places365_weights(cache_path: Path | None = None) -> dict[str, mx.array]:
    """Return the MLX-keyed state dict for ResNet-18 + Places365.

    Reads the disk cache when present; otherwise lazily imports torch,
    downloads the CSAIL checkpoint, converts, and caches. Raises with
    side-load instructions when neither path is available.
    """
    cache = cache_path or _default_cache_path()
    if cache.exists():
        with np.load(cache) as npz:
            converted = {key: npz[key].astype(np.float32, copy=False) for key in npz.files}
    else:
        converted = _download_and_convert(cache)
    return {k: mx.array(v) for k, v in converted.items()}


def assign_state(model: MLXResNet18, state: dict[str, mx.array]) -> None:
    """Inject a flat MLX-key state dict into ``model`` via ``mlx.utils.tree_unflatten``."""
    from mlx.utils import tree_unflatten

    pairs = list(state.items())
    nested = tree_unflatten(pairs)
    model.update(nested)
