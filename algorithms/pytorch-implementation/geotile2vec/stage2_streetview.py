"""Stage 2: street-view feature extraction + triplet metric learning.

Reference: Luo et al., "Geo-Tile2Vec", ACM TSAS 2023, §3.3, Eq. (6).

Pipeline:
    1. Places365PreTrainedResNet18 — wraps a torchvision ResNet-18 with the
       Places365 checkpoint loaded from torch.hub. Falls back to a clear
       error message when offline so the user can side-load weights.
    2. extract_image_features — 512-dim penultimate-layer features per image.
    3. fit_pca / apply_pca — fit IncrementalPCA on the FULL corpus once,
       then transform every image (paper §3.3 — PCA is fit globally).
    4. concat_per_shot — 4 directional images per shooting point → 512-dim.
    5. train_stage2_triplet — triplet loss with linear projection 512→d,
       updating both V (tile embeddings) and W (projection).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from torchvision import transforms
from torchvision.models import resnet18

from data import StreetViewShot, TileId
from stage1_mobility import _semi_hard_triplet_loss


# --- Places365 ResNet-18 wrapper -------------------------------------------------

# Official checkpoint URL from CSAILVision/places365.
PLACES365_RESNET18_URL = (
    "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
)

# Standard ImageNet normalization — the Places365 weights expect the same
# preprocessing because the model is plain ResNet-18 architecture.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Places365PretrainedResNet18(nn.Module):
    """ResNet-18 with Places365-365 weights, exposing the 512-dim penultimate
    features (i.e. global-average-pool output, one before the FC layer).

    Set `weights_path=None` (default) to auto-download the official checkpoint
    from `PLACES365_RESNET18_URL`. If the download fails, we raise with the
    expected cache path so the user can pre-download manually.
    """

    def __init__(self, weights_path: str | None = None, device: str = "cpu") -> None:
        super().__init__()
        backbone = resnet18(num_classes=365)
        state = self._load_state(weights_path)
        backbone.load_state_dict(state, strict=True)
        backbone.fc = nn.Identity()                      # expose 512-dim features
        backbone.train(False)                            # inference mode
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone.to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @staticmethod
    def _load_state(weights_path: str | None) -> dict:
        if weights_path is not None and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        else:
            try:
                ckpt = torch.hub.load_state_dict_from_url(
                    PLACES365_RESNET18_URL, map_location="cpu", weights_only=False,
                )
            except Exception as e:
                cache = os.path.join(
                    os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints",
                    "resnet18_places365.pth.tar",
                )
                raise RuntimeError(
                    "Failed to download Places365 ResNet-18 weights from "
                    f"{PLACES365_RESNET18_URL}. Pre-download to {cache} and retry. "
                    f"Original error: {e}"
                ) from e
        # Checkpoint keys are prefixed with "module." (DataParallel artifact).
        raw = ckpt.get("state_dict", ckpt)
        return {k.replace("module.", ""): v for k, v in raw.items()}

    @torch.no_grad()
    def features(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Extract 512-dim features from a batch of HxWx3 uint8 images."""
        tensors = [self.transform(Image.fromarray(img)) for img in images]
        batch = torch.stack(tensors).to(self.device)
        feats = self.backbone(batch)
        return feats.cpu().numpy()


# --- Pipeline pieces ------------------------------------------------------------

def extract_image_features(
    extractor: Places365PretrainedResNet18,
    shots: list[StreetViewShot],
    batch_size: int = 32,
) -> np.ndarray:
    """Returns features of shape (n_shots * 4, 512) in flattened order:
    [shot0_dir0, shot0_dir1, shot0_dir2, shot0_dir3, shot1_dir0, ...].
    """
    flat: list[np.ndarray] = []
    for shot in shots:
        flat.extend(shot.images)
    out: list[np.ndarray] = []
    for i in range(0, len(flat), batch_size):
        out.append(extractor.features(flat[i:i + batch_size]))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)


def fit_and_apply_pca(features: np.ndarray, n_components: int = 128) -> np.ndarray:
    """Fit IncrementalPCA on the WHOLE corpus once, then transform.

    `n_components` is clamped to ≤ min(n_samples, n_features) to handle tiny
    smoke-test datasets.
    """
    n_samples, n_features = features.shape
    n_components = min(n_components, n_samples, n_features)
    pca = IncrementalPCA(n_components=n_components, batch_size=max(n_components, 64))
    pca.fit(features)
    return pca.transform(features).astype(np.float32)


def concat_per_shot(per_image_features: np.ndarray, n_shots: int) -> np.ndarray:
    """Group every 4 image vectors into one shooting-point vector.

    Input:  (n_shots * 4, k)   PCA-reduced features
    Output: (n_shots, 4 * k)   per-shot concatenation
    """
    if per_image_features.shape[0] != n_shots * 4:
        raise ValueError(
            f"Expected {n_shots * 4} per-image features, got {per_image_features.shape[0]}"
        )
    return per_image_features.reshape(n_shots, 4 * per_image_features.shape[1])


# --- Stage 2 triplet metric learning (paper Eq. 6) ------------------------------

class _Stage2Module(nn.Module):
    """Holds the trainable tile-embedding matrix V and the linear projection W."""

    def __init__(self, V_init: torch.Tensor, sv_dim: int) -> None:
        super().__init__()
        self.V = nn.Parameter(V_init.clone())                        # (N, d)
        self.W = nn.Linear(sv_dim, V_init.shape[1], bias=True)       # 512 → d


def train_stage2_triplet(
    V_init: torch.Tensor,
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
) -> tuple[torch.Tensor, list[float]]:
    """Triplet loss: anchor = v_i, positive = projected sv inside tile i,
    negative = projected sv from a different tile.

    Returns (final V, per-epoch losses).
    """
    rng = random.Random(seed)
    sv_dim = shot_features.shape[1]
    tile_to_row = {t: i for i, t in enumerate(tile_order)}

    # Group shot indices by tile id (only tiles that appear in tile_order).
    tile_to_shot_idxs: dict[TileId, list[int]] = defaultdict(list)
    for k, t in enumerate(shot_tiles):
        if t in tile_to_row:
            tile_to_shot_idxs[t].append(k)

    eligible = [t for t, sh in tile_to_shot_idxs.items() if sh]
    if not eligible:
        raise ValueError("No tiles have street-view shots; can't run stage 2.")

    sv_t = torch.tensor(shot_features, dtype=torch.float32, device=device)
    mod = _Stage2Module(V_init, sv_dim).to(device)
    mod.train(True)
    opt = torch.optim.Adam(mod.parameters(), lr=lr)

    losses: list[float] = []
    all_shot_idxs = list(range(len(shot_tiles)))
    for ep in range(epochs):
        running, n = 0.0, 0
        for _ in range(steps_per_epoch):
            anc_rows, pos_idx, neg_idx = [], [], []
            for _ in range(batch_size):
                t = rng.choice(eligible)
                pos = rng.choice(tile_to_shot_idxs[t])
                negs: list[int] = []
                while len(negs) < n_negatives:
                    cand = rng.choice(all_shot_idxs)
                    if shot_tiles[cand] != t and shot_tiles[cand] in tile_to_row:
                        negs.append(cand)
                anc_rows.append(tile_to_row[t])
                pos_idx.append(pos)
                neg_idx.append(negs)

            anc_rows = torch.tensor(anc_rows, device=device)
            pos_idx = torch.tensor(pos_idx, device=device)
            neg_idx = torch.tensor(neg_idx, device=device)

            v_anc = mod.V[anc_rows]                                  # (B, d)
            s_pos = mod.W(sv_t[pos_idx])                             # (B, d)
            s_neg = mod.W(sv_t[neg_idx])                             # (B, K, d)

            opt.zero_grad()
            loss = _semi_hard_triplet_loss(v_anc, s_pos, s_neg, margin)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        ep_loss = running / max(1, n)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage2/triplet]  epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")

    return mod.V.detach(), losses
