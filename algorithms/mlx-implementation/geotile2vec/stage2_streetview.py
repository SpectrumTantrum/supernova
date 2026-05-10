"""Stage 2: street-view features + triplet metric learning.

Reference: Luo et al., "Geo-Tile2Vec", ACM TSAS 2023, §3.3, Eq. (6).

The MLX port runs a native MLX ResNet-18 (see ``places365_backbone.py``) with
weights converted from the official CSAIL Places365 PyTorch checkpoint. The
converted weights are cached on disk after first use; the conversion step
itself is the only place torch is imported, and is gated behind a
"weights-not-cached" branch.

When real weights cannot be loaded (no network, no torch, no cache), the
``allow_synthetic`` flag falls back to deterministic image-statistic features
suitable for the smoke test only — the README documents this clearly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.decomposition import IncrementalPCA

from data import StreetViewShot, TileId
from places365_backbone import (
    PLACES365_RESNET18_URL,
    MLXResNet18,
    assign_state,
    imagenet_preprocess,
    load_places365_weights,
)
from stage1_mobility import _semi_hard_triplet_loss


class Places365PretrainedResNet18:
    """ResNet-18 with Places365 weights, native MLX inference path.

    The 512-dim penultimate-layer features (paper §3.3) are produced by an
    MLX ResNet-18 whose state dict is converted from the official CSAIL
    PyTorch checkpoint at first run. ``allow_synthetic=True`` enables a
    deterministic image-statistic fallback for environments where the
    weights cache cannot be built (no torch, offline, etc.); in that mode
    the smoke test still passes but Stage-2 visual features are NOT the
    paper's Places365 features. Use ``allow_synthetic=False`` to require
    paper-faithful features and surface an explicit error otherwise.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        allow_synthetic: bool = True,
        batch_size: int = 32,
    ) -> None:
        del device  # MLX picks the active device implicitly
        self.allow_synthetic = allow_synthetic
        self.batch_size = batch_size
        self._model: MLXResNet18 | None = None
        self._init_error: Exception | None = None
        try:
            cache = Path(weights_path) if weights_path is not None else None
            state = load_places365_weights(cache)
            model = MLXResNet18()
            assign_state(model, state)
            model.train(False)
            mx.eval(model.parameters())
            self._model = model
        except Exception as exc:  # noqa: BLE001 — re-raised below if not allowed
            self._init_error = exc
            if not allow_synthetic:
                raise

    def features(self, images: Iterable[np.ndarray]) -> np.ndarray:
        images = list(images)
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        if self._model is None:
            if not self.allow_synthetic:
                raise RuntimeError(
                    "Places365 ResNet-18 weights could not be loaded and "
                    f"allow_synthetic is False. Source checkpoint: {PLACES365_RESNET18_URL}. "
                    f"Original error: {self._init_error!r}"
                )
            feats = [_image_statistics(img) for img in images]
            return np.stack(feats, axis=0).astype(np.float32)
        out_chunks: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = imagenet_preprocess(images[i : i + self.batch_size])
            out = self._model(batch)
            mx.eval(out)
            out_chunks.append(np.array(out).astype(np.float32, copy=False))
        return np.concatenate(out_chunks, axis=0)


def _image_statistics(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Street-view images must be HxWx3 arrays, got {arr.shape}")
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    q = np.quantile(arr.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).reshape(-1)
    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=32, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    base = np.concatenate([means, stds, q, *hist_parts]).astype(np.float32)
    reps = int(np.ceil(512 / base.size))
    return np.tile(base, reps)[:512]


def extract_image_features(
    extractor_or_features: Places365PretrainedResNet18 | np.ndarray,
    shots: list[StreetViewShot],
    batch_size: int = 32,
) -> np.ndarray:
    """Return per-image features in [shot0_dir0, ..., shotN_dir3] order.

    Pass a precomputed ndarray shaped `(len(shots) * 4, feature_dim)` to avoid
    image extraction entirely.
    """
    if isinstance(extractor_or_features, np.ndarray):
        expected = len(shots) * 4
        if extractor_or_features.shape[0] != expected:
            raise ValueError(f"Expected {expected} precomputed per-image features, got {extractor_or_features.shape[0]}")
        return extractor_or_features.astype(np.float32, copy=False)

    flat: list[np.ndarray] = []
    for shot in shots:
        flat.extend(shot.images)
    out: list[np.ndarray] = []
    for i in range(0, len(flat), batch_size):
        out.append(extractor_or_features.features(flat[i:i + batch_size]))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)


def fit_and_apply_pca(features: np.ndarray, n_components: int = 128) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n_components = min(n_components, n_samples, n_features)
    if n_components <= 0:
        raise ValueError(f"n_components must be positive after clamping, got {n_components}")
    pca = IncrementalPCA(n_components=n_components, batch_size=max(n_components, 64))
    return pca.fit_transform(features).astype(np.float32)


def concat_per_shot(per_image_features: np.ndarray, n_shots: int) -> np.ndarray:
    if per_image_features.shape[0] != n_shots * 4:
        raise ValueError(f"Expected {n_shots * 4} per-image features, got {per_image_features.shape[0]}")
    return per_image_features.reshape(n_shots, 4 * per_image_features.shape[1]).astype(np.float32, copy=False)


class _Stage2Module(nn.Module):
    def __init__(self, V_init: mx.array, sv_dim: int) -> None:
        super().__init__()
        self.V = mx.array(V_init)
        self.W = nn.Linear(sv_dim, V_init.shape[1], bias=True)

    def project(self, shot_features: mx.array) -> mx.array:
        return self.W(shot_features)


def train_stage2_triplet(
    V_init: mx.array,
    tile_order: list[TileId],
    shot_features: np.ndarray,
    shot_tiles: list[TileId],
    *,
    margin: float = 2.0,
    epochs: int = 5,
    steps_per_epoch: int = 200,
    batch_size: int = 128,
    n_negatives: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[mx.array, list[float]]:
    del device
    rng = random.Random(seed)
    tile_to_row = {t: i for i, t in enumerate(tile_order)}
    tile_to_shot_idxs: dict[TileId, list[int]] = defaultdict(list)
    for k, t in enumerate(shot_tiles):
        if t in tile_to_row:
            tile_to_shot_idxs[t].append(k)

    eligible = [t for t, sh in tile_to_shot_idxs.items() if sh]
    if not eligible:
        raise ValueError("No tiles have street-view shots; can't run stage 2.")
    valid_shot_idxs = [idx for idx, tile in enumerate(shot_tiles) if tile in tile_to_row]
    negative_candidates = {
        t: [idx for idx in valid_shot_idxs if shot_tiles[idx] != t]
        for t in eligible
    }
    empty_negative_tiles = [t for t, candidates in negative_candidates.items() if not candidates]
    if empty_negative_tiles:
        raise ValueError(
            "Need street-view shots from at least two fitted tiles for stage 2; "
            f"{len(empty_negative_tiles)} eligible tile(s) have no different-tile negatives."
        )

    sv_t = mx.array(shot_features.astype(np.float32, copy=False))
    mod = _Stage2Module(V_init, shot_features.shape[1])
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m: _Stage2Module, anc_rows: mx.array, pos_idx: mx.array, neg_idx: mx.array) -> mx.array:
        v_anc = m.V[anc_rows]
        s_pos = m.project(sv_t[pos_idx])
        s_neg = m.project(sv_t[neg_idx])
        return _semi_hard_triplet_loss(v_anc, s_pos, s_neg, margin)

    grad_fn = nn.value_and_grad(mod, loss_fn)
    losses: list[float] = []
    for ep in range(epochs):
        running, n = 0.0, 0
        for _ in range(steps_per_epoch):
            anc_rows, pos_idx, neg_idx = [], [], []
            for _ in range(batch_size):
                t = rng.choice(eligible)
                pos = rng.choice(tile_to_shot_idxs[t])
                negs = rng.choices(negative_candidates[t], k=n_negatives)
                anc_rows.append(tile_to_row[t])
                pos_idx.append(pos)
                neg_idx.append(negs)
            loss, grads = grad_fn(
                mod,
                mx.array(np.asarray(anc_rows, dtype=np.int32)),
                mx.array(np.asarray(pos_idx, dtype=np.int32)),
                mx.array(np.asarray(neg_idx, dtype=np.int32)),
            )
            opt.update(mod, grads)
            mx.eval(mod.parameters(), opt.state)
            running += float(loss)
            n += 1
        ep_loss = running / max(1, n)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage2/triplet]  epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")
    return mx.array(mod.V), losses
