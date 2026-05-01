"""MLX MHGL orchestrator (paper §3, Algorithm 2)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from data import AttributedNetwork, build_normalized_adj
from gcn import GCNEncoder
from pde import Pattern, fit_pde
from train import anomaly_scores, compute_centres, compute_high_confidence, train_mhgl


@dataclass
class MHGLConfig:
    hidden_dims: tuple[int, ...] = (256, 128, 64, 32)
    activation: str = "relu"
    bias: bool = True
    k_normal: int = 10
    pde_split_threshold_u: int = 30
    pde_max_recursion: int = 3
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 5e-4
    sigma: float = 1.0
    augmentation_alpha: int = 2
    eps_repulsion: float = 1e-6
    high_confidence_t: float = 0.7
    radius_quantile: float = 1.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


class MHGL:
    def __init__(self, config: MHGLConfig | None = None) -> None:
        self.config = config or MHGLConfig()
        self._encoder: Optional[GCNEncoder] = None
        self._centres: Optional[mx.array] = None
        self._patterns: list[Pattern] = []
        self._high_conf: list[np.ndarray] = []
        self._anom_indices: Optional[np.ndarray] = None
        self._history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}
        self._scores: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._n: Optional[int] = None

    def fit(self, network: AttributedNetwork) -> "MHGL":
        cfg = self.config
        if network.train_mask is None or network.label_mask is None or network.labels is None:
            raise ValueError("MHGL.fit needs train_mask, label_mask, and labels.")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)
        A_hat = mx.array(build_normalized_adj(network.edges, network.n))
        X = mx.array(network.X)
        encoder = GCNEncoder(network.f, tuple(cfg.hidden_dims), activation=cfg.activation, bias=cfg.bias)
        H0 = encoder(X, A_hat)
        H0_np = np.array(H0)
        labelled = network.label_mask
        labels = network.labels
        norm_idx = np.nonzero(labelled & (labels == 0))[0].astype(np.int64)
        anom_idx = np.nonzero(labelled & (labels == 1))[0].astype(np.int64)
        if norm_idx.size == 0:
            raise ValueError("No labelled-normal nodes in label_mask — PDE has nothing to cluster.")
        if anom_idx.size == 0:
            raise ValueError("No labelled-anomaly nodes in label_mask — MHGL's repulsion term needs q > 0.")
        patterns = fit_pde(H0_np, norm_idx, k=cfg.k_normal, u=cfg.pde_split_threshold_u, max_recursion=cfg.pde_max_recursion, seed=cfg.seed)
        if not patterns:
            raise RuntimeError("PDE produced no patterns — check k_normal and labelled-normal count.")
        centres = compute_centres(H0, patterns)
        high_conf = [compute_high_confidence(H0, p, centres[i], threshold_t=cfg.high_confidence_t, radius_quantile=cfg.radius_quantile) for i, p in enumerate(patterns)]
        if cfg.verbose:
            print(f"[MHGL-MLX] |labelled_normal|={len(norm_idx)} |labelled_anom|={len(anom_idx)}  patterns={len(patterns)}  |H^i|={[int(h.shape[0]) for h in high_conf]}")
        history = train_mhgl(encoder, X, A_hat, centres, high_conf, anom_idx, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, sigma=cfg.sigma, eps=cfg.eps_repulsion, augmentation_alpha=cfg.augmentation_alpha, seed=cfg.seed, verbose=cfg.verbose)
        self._encoder = encoder
        self._centres = centres
        self._patterns = patterns
        self._high_conf = high_conf
        self._anom_indices = anom_idx
        self._history = history
        self._feat_dim = network.f
        self._n = network.n
        self._scores = anomaly_scores(encoder, X, A_hat, centres)
        return self

    def score(self, network: AttributedNetwork | None = None) -> np.ndarray:
        if self._encoder is None or self._centres is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._scores is not None
            return self._scores
        if network.f != self._feat_dim:
            raise ValueError(f"feature-dim mismatch: trained on {self._feat_dim}, got {network.f}")
        return anomaly_scores(self._encoder, mx.array(network.X), mx.array(build_normalized_adj(network.edges, network.n)), self._centres)

    def predict(self, threshold: float | None = None) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Call fit() first.")
        s = self.score()
        if threshold is None:
            if not self._high_conf:
                raise ValueError("threshold is required because no high-confidence normal pool is available.")
            pool = np.unique(np.concatenate(self._high_conf))
            threshold = float(np.max(s[pool]))
        return (s > threshold).astype(np.int64)

    @property
    def history(self) -> dict[str, list[float]]:
        return self._history

    def patterns(self) -> list[Pattern]:
        if not self._patterns:
            raise RuntimeError("Call fit() first.")
        return list(self._patterns)

    def save(self, path: str) -> None:
        if self._encoder is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez(path, config=np.array([self.config.__dict__], dtype=object), centres=np.array(self._centres), patterns=np.array([self._patterns], dtype=object), high_conf=np.array([self._high_conf], dtype=object), anom_indices=self._anom_indices, history=np.array([self._history], dtype=object), scores=self._scores, feat_dim=np.array(self._feat_dim), n=np.array(self._n), params=np.array([self._encoder.parameters()], dtype=object))

    @classmethod
    def load(cls, path: str) -> "MHGL":
        ckpt = np.load(path, allow_pickle=True)
        cfg = dict(ckpt["config"][0])
        cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        cfg["device"] = "cpu"
        obj = cls(MHGLConfig(**cfg))
        obj._feat_dim = int(ckpt["feat_dim"])
        obj._n = int(ckpt["n"])
        encoder = GCNEncoder(obj._feat_dim, obj.config.hidden_dims, activation=obj.config.activation, bias=obj.config.bias)
        encoder.update(ckpt["params"][0])
        obj._encoder = encoder
        obj._centres = mx.array(ckpt["centres"])
        obj._patterns = list(ckpt["patterns"][0])
        obj._high_conf = list(ckpt["high_conf"][0])
        obj._anom_indices = ckpt["anom_indices"]
        obj._history = ckpt["history"][0]
        obj._scores = ckpt["scores"]
        return obj
