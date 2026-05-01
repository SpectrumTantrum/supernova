"""MLX AAGNN orchestrator (paper §3, Algorithm 1)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from data import AttributedNetwork, k_hop_neighbors
from layer import AbnormalityAwareLayer
from train import anomaly_scores, compute_pseudo_labels, train_aagnn


@dataclass
class AAGNNConfig:
    hidden_dim: int = 256
    aggregator: str = "mean"
    activation: str = "relu"
    k_hop: int = 1
    pseudo_label_pct: float = 50.0
    train_val_split: float = 0.6
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adam"
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


class AAGNN:
    def __init__(self, config: AAGNNConfig | None = None) -> None:
        self.config = config or AAGNNConfig()
        self._layer: Optional[AbnormalityAwareLayer] = None
        self._c: Optional[mx.array] = None
        self._R_idx: Optional[np.ndarray] = None
        self._D_idx: Optional[np.ndarray] = None
        self._T_idx: Optional[np.ndarray] = None
        self._history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}
        self._scores: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._n: Optional[int] = None

    def fit(self, network: AttributedNetwork) -> "AAGNN":
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)
        neigh = k_hop_neighbors(network.edges, network.n, k=cfg.k_hop)
        X = mx.array(network.X)
        layer_obj = AbnormalityAwareLayer(network.f, cfg.hidden_dim, cfg.aggregator, cfg.activation)
        R, D, T, c = compute_pseudo_labels(layer_obj, X, neigh, pseudo_label_pct=cfg.pseudo_label_pct, train_val_split=cfg.train_val_split, seed=cfg.seed)
        if cfg.verbose:
            print(f"[AAGNN-MLX] |R|={len(R)} |D|={len(D)} |T|={len(T)}  c.shape={c.shape}")
        self._history = train_aagnn(layer_obj, X, neigh, R, D, c, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, optimizer=cfg.optimizer, verbose=cfg.verbose)
        self._layer = layer_obj
        self._c = c
        self._R_idx, self._D_idx, self._T_idx = R, D, T
        self._feat_dim = network.f
        self._n = network.n
        self._scores = anomaly_scores(layer_obj, X, neigh, c)
        return self

    def score(self, network: AttributedNetwork | None = None) -> np.ndarray:
        if self._layer is None or self._c is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._scores is not None
            return self._scores
        if network.f != self._feat_dim:
            raise ValueError(f"feature-dim mismatch: trained on {self._feat_dim}, got {network.f}")
        neigh = k_hop_neighbors(network.edges, network.n, k=self.config.k_hop)
        return anomaly_scores(self._layer, mx.array(network.X), neigh, self._c)

    def predict(self, threshold: float | None = None) -> np.ndarray:
        if self._layer is None:
            raise RuntimeError("Call fit() first.")
        s = self.score()
        if threshold is None:
            assert self._R_idx is not None and self._D_idx is not None
            S = np.concatenate([self._R_idx, self._D_idx])
            threshold = float(np.quantile(s[S], 0.95))
        return (s > threshold).astype(np.int64)

    @property
    def history(self) -> dict[str, list[float]]:
        return self._history

    def split_indices(self) -> dict[str, np.ndarray]:
        if self._R_idx is None:
            raise RuntimeError("Call fit() first.")
        return {"R": self._R_idx, "D": self._D_idx, "T": self._T_idx}

    def save(self, path: str) -> None:
        if self._layer is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez(path, config=np.array([self.config.__dict__], dtype=object), c=np.array(self._c), R_idx=self._R_idx, D_idx=self._D_idx, T_idx=self._T_idx, scores=self._scores, feat_dim=np.array(self._feat_dim), n=np.array(self._n), history=np.array([self._history], dtype=object), params=np.array([self._layer.parameters()], dtype=object))

    @classmethod
    def load(cls, path: str) -> "AAGNN":
        ckpt = np.load(path, allow_pickle=True)
        cfg = dict(ckpt["config"][0])
        cfg["device"] = "cpu"
        obj = cls(AAGNNConfig(**cfg))
        obj._feat_dim = int(ckpt["feat_dim"])
        obj._n = int(ckpt["n"])
        layer = AbnormalityAwareLayer(obj._feat_dim, obj.config.hidden_dim, obj.config.aggregator, obj.config.activation)
        layer.update(ckpt["params"][0])
        obj._layer = layer
        obj._c = mx.array(ckpt["c"])
        obj._R_idx = ckpt["R_idx"]
        obj._D_idx = ckpt["D_idx"]
        obj._T_idx = ckpt["T_idx"]
        obj._scores = ckpt["scores"]
        obj._history = ckpt["history"][0]
        return obj
