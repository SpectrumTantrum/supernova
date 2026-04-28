"""AAGNN orchestrator (paper §3, Algorithm 1).

Reference: Zhou et al., "Subtractive Aggregation for Attributed Network Anomaly
Detection", CIKM 2021.

Wires the three sibling modules into a single trainer:
    data.py   -> AttributedNetwork + k_hop_neighbors
    layer.py  -> AbnormalityAwareLayer (subtractive aggregation, §3.1)
    train.py  -> compute_pseudo_labels + train_aagnn + anomaly_scores (§3.2-3.3)

Typical usage:
    from data import SyntheticAttributedNetwork
    from model import AAGNN, AAGNNConfig

    net = SyntheticAttributedNetwork(seed=0).generate()
    model = AAGNN(AAGNNConfig(hidden_dim=64, epochs=80)).fit(net)
    scores = model.score()           # (n,) float64, higher = more anomalous
    labels = model.predict()         # (n,) int64 in {0, 1}
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from data import AttributedNetwork, k_hop_neighbors
from layer import AbnormalityAwareLayer
from train import anomaly_scores, compute_pseudo_labels, train_aagnn


@dataclass
class AAGNNConfig:
    """Hyperparameter bundle for the AAGNN trainer (paper §3.2-3.3 defaults)."""

    hidden_dim: int = 256          # paper d
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
    """End-to-end AAGNN trainer (paper §3, Algorithm 1).

    Stateless until ``fit`` runs; afterwards exposes scores, predicted labels,
    pseudo-label split indices, training history, and save/load.
    """

    def __init__(self, config: AAGNNConfig | None = None) -> None:
        self.config = config or AAGNNConfig()
        self._layer: Optional[AbnormalityAwareLayer] = None
        self._c: Optional[torch.Tensor] = None
        self._R_idx: Optional[np.ndarray] = None
        self._D_idx: Optional[np.ndarray] = None
        self._T_idx: Optional[np.ndarray] = None
        self._history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}
        self._scores: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._n: Optional[int] = None

    # ------------------------------------------------------------------ Fit

    def fit(self, network: AttributedNetwork) -> "AAGNN":
        """Run Algorithm 1 end-to-end on ``network``.

        Lines 1-5 (pseudo-labels) → lines 6-10 (hypersphere SGD) → Eq. 6 scoring.
        """
        cfg = self.config

        # Seed all RNGs touched by data.py / train.py / torch.
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        neigh = k_hop_neighbors(network.edges, network.n, k=cfg.k_hop)
        X = torch.from_numpy(network.X).to(cfg.device)
        layer_obj = AbnormalityAwareLayer(
            in_dim=network.f,
            out_dim=cfg.hidden_dim,
            aggregator=cfg.aggregator,
            activation=cfg.activation,
        ).to(cfg.device)

        R, D, T, c = compute_pseudo_labels(
            layer_obj, X, neigh,
            pseudo_label_pct=cfg.pseudo_label_pct,
            train_val_split=cfg.train_val_split,
            seed=cfg.seed,
        )
        if cfg.verbose:
            print(
                f"[AAGNN] |R|={len(R)} |D|={len(D)} |T|={len(T)}  "
                f"c.shape={tuple(c.shape)}"
            )

        self._history = train_aagnn(
            layer_obj, X, neigh, R, D, c,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            verbose=cfg.verbose,
        )

        self._layer = layer_obj
        self._c = c
        self._R_idx, self._D_idx, self._T_idx = R, D, T
        self._feat_dim = network.f
        self._n = network.n
        self._scores = anomaly_scores(layer_obj, X, neigh, c)
        return self

    # ------------------------------------------------------------------ Outputs

    def score(self, network: AttributedNetwork | None = None) -> np.ndarray:
        """Return Eq. 6 anomaly scores. With no arg, returns cached fit-time scores.

        Passing a different ``network`` re-runs the trained layer on it; the
        feature dimension must match the training network.
        """
        if self._layer is None or self._c is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._scores is not None
            return self._scores
        if network.f != self._feat_dim:
            raise ValueError(
                f"feature-dim mismatch: trained on {self._feat_dim}, got {network.f}"
            )
        neigh = k_hop_neighbors(network.edges, network.n, k=self.config.k_hop)
        X = torch.from_numpy(network.X).to(self.config.device)
        return anomaly_scores(self._layer, X, neigh, self._c)

    def predict(self, threshold: float | None = None) -> np.ndarray:
        """Binarise the fit-time anomaly scores.

        Default threshold = median over the pseudo-normal set S = R ∪ D, i.e.
        "more anomalous than a typical presumed-normal node".
        """
        if self._layer is None:
            raise RuntimeError("Call fit() first.")
        s = self.score()
        if threshold is None:
            assert self._R_idx is not None and self._D_idx is not None
            S = np.concatenate([self._R_idx, self._D_idx])
            threshold = float(np.median(s[S]))
        return (s >= threshold).astype(np.int64)

    @property
    def history(self) -> dict[str, list[float]]:
        """Per-epoch train / val losses recorded during ``fit``."""
        return self._history

    def split_indices(self) -> dict[str, np.ndarray]:
        """Algorithm 1 lines 1-5 partition: R (train), D (val), T (test)."""
        if self._R_idx is None:
            raise RuntimeError("Call fit() first.")
        return {"R": self._R_idx, "D": self._D_idx, "T": self._T_idx}

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Pickle the trained layer, centre c, splits, history, and scores."""
        if self._layer is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        assert self._c is not None
        torch.save(
            {
                "config": self.config.__dict__,
                "layer_state_dict": self._layer.state_dict(),
                "c": self._c.cpu(),
                "R_idx": self._R_idx,
                "D_idx": self._D_idx,
                "T_idx": self._T_idx,
                "history": self._history,
                "scores": self._scores,
                "feat_dim": self._feat_dim,
                "n": self._n,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "AAGNN":
        """Reconstruct an AAGNN from a ``save()`` checkpoint (CPU-loaded)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(AAGNNConfig(**ckpt["config"]))
        obj._feat_dim = ckpt["feat_dim"]
        obj._n = ckpt["n"]
        layer_obj = AbnormalityAwareLayer(
            in_dim=obj._feat_dim,
            out_dim=obj.config.hidden_dim,
            aggregator=obj.config.aggregator,
            activation=obj.config.activation,
        )
        layer_obj.load_state_dict(ckpt["layer_state_dict"])
        obj._layer = layer_obj
        obj._c = ckpt["c"]
        obj._R_idx = ckpt["R_idx"]
        obj._D_idx = ckpt["D_idx"]
        obj._T_idx = ckpt["T_idx"]
        obj._history = ckpt["history"]
        obj._scores = ckpt["scores"]
        return obj
