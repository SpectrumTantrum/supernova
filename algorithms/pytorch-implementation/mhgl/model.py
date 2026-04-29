"""MHGL orchestrator (paper §3, Algorithm 2).

Reference: Zhou et al., "Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning", SIAM SDM 2022.

Wires the four sibling modules into a single trainer:
    data.py   -> AttributedNetwork + build_normalized_adj
    gcn.py    -> GCNEncoder (paper §3 Eq. 3.2)
    pde.py    -> fit_pde (paper §3.1, Algorithm 1)
    train.py  -> compute_centres / compute_high_confidence / train_mhgl /
                 anomaly_scores (paper §3.2, Algorithm 2)

Typical usage:
    from data import SyntheticAttributedNetwork
    from model import MHGL, MHGLConfig

    net = SyntheticAttributedNetwork(seed=0).generate()
    model = MHGL(MHGLConfig(hidden_dims=(64, 32, 16, 16), epochs=80)).fit(net)
    scores = model.score()         # (n,) float64, higher = more anomalous
    labels = model.predict()       # (n,) int64 in {0, 1}
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from data import AttributedNetwork, build_normalized_adj
from gcn import GCNEncoder
from pde import Pattern, fit_pde
from train import (
    anomaly_scores,
    compute_centres,
    compute_high_confidence,
    train_mhgl,
)


@dataclass
class MHGLConfig:
    """Hyperparameter bundle for the MHGL trainer (paper §4.2 defaults)."""

    # Encoder
    hidden_dims: tuple[int, ...] = (256, 128, 64, 32)   # paper §4.2
    activation: str = "relu"                            # paper §4.2
    bias: bool = True                                   # Kipf default; see gcn.py

    # PDE (Pattern Distribution Estimator)
    k_normal: int = 10                                  # paper §4.2 "k = 10"
    pde_split_threshold_u: int = 30
    pde_max_recursion: int = 3

    # Training
    epochs: int = 300                                   # paper §4.2
    lr: float = 1e-3
    weight_decay: float = 5e-4                          # paper §4.2
    sigma: float = 1.0                                  # paper-tuned per dataset; 1.0 synthetic default
    augmentation_alpha: int = 2                         # paper §4.2
    eps_repulsion: float = 1e-6                         # numerical guard, paper silent
    high_confidence_t: float = 0.7                      # threshold t for ξ_j filter
    radius_quantile: float = 1.0                         # paper line 6 max; lower tightens H^i

    # Misc
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


class MHGL:
    """End-to-end MHGL trainer (paper §3, Algorithm 2).

    Stateless until ``fit`` runs; afterwards exposes scores, predicted labels,
    pattern records, training history, and save/load.
    """

    def __init__(self, config: MHGLConfig | None = None) -> None:
        self.config = config or MHGLConfig()
        self._encoder: Optional[GCNEncoder] = None
        self._A_hat: Optional[torch.Tensor] = None
        self._centres: Optional[torch.Tensor] = None
        self._patterns: list[Pattern] = []
        self._high_conf: list[np.ndarray] = []
        self._anom_indices: Optional[np.ndarray] = None
        self._history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}
        self._scores: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._n: Optional[int] = None

    # ------------------------------------------------------------------ Fit

    def fit(self, network: AttributedNetwork) -> "MHGL":
        """Run Algorithm 2 end-to-end on ``network``.

        Lines 1-3 (init + PDE + centres) → lines 4-7 (high-confidence sets) →
        lines 8-16 (mixup + multi-hypersphere SGD) → Eq. 3.7 scoring.

        ``network`` must carry ``train_mask`` and ``label_mask`` populated
        per the paper §2 protocol (see SyntheticAttributedNetwork docstring).
        """
        cfg = self.config

        if network.train_mask is None or network.label_mask is None or network.labels is None:
            raise ValueError(
                "MHGL.fit needs network.train_mask, network.label_mask, and network.labels — "
                "the paper §2 protocol is semi-supervised and depends on this partition."
            )

        # Seed all RNGs.
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # 1. Build A_hat + encoder.
        device = torch.device(cfg.device)
        A_hat = build_normalized_adj(network.edges, network.n, device=device)
        X = torch.from_numpy(network.X).to(device)
        encoder = GCNEncoder(
            in_dim=network.f,
            hidden_dims=tuple(cfg.hidden_dims),
            activation=cfg.activation,
            bias=cfg.bias,
        ).to(device)

        # 2. Random-init forward pass under no_grad → H₀ for PDE + centres.
        was_training = encoder.training
        encoder.train(False)
        with torch.no_grad():
            H0 = encoder(X, A_hat)
        if was_training:
            encoder.train(True)
        H0_np = H0.cpu().numpy()

        # 3. Identify labelled-normal and labelled-anomaly indices.
        labelled = network.label_mask
        labels = network.labels
        norm_idx = np.nonzero(labelled & (labels == 0))[0].astype(np.int64)
        anom_idx = np.nonzero(labelled & (labels == 1))[0].astype(np.int64)
        if norm_idx.size == 0:
            raise ValueError("No labelled-normal nodes in label_mask — PDE has nothing to cluster.")

        # 4. PDE on labelled normals → patterns.
        patterns = fit_pde(
            H0_np,
            norm_idx,
            k=cfg.k_normal,
            u=cfg.pde_split_threshold_u,
            max_recursion=cfg.pde_max_recursion,
            seed=cfg.seed,
        )
        if not patterns:
            raise RuntimeError("PDE produced no patterns — check k_normal and labelled-normal count.")

        # 5. Centres (Eq. 3.5) — frozen for the rest of training.
        centres = compute_centres(H0, patterns)

        # 6. High-confidence sets H^i (Algorithm 2 lines 4-7).
        high_conf = [
            compute_high_confidence(
                H0, p, centres[i],
                threshold_t=cfg.high_confidence_t,
                radius_quantile=cfg.radius_quantile,
            )
            for i, p in enumerate(patterns)
        ]
        if cfg.verbose:
            sizes = [int(hi.shape[0]) for hi in high_conf]
            print(
                f"[MHGL] |labelled_normal|={len(norm_idx)} |labelled_anom|={len(anom_idx)}  "
                f"patterns={len(patterns)}  |H^i|={sizes}"
            )

        # 7. Train loop (Algorithm 2 lines 8-16).
        history = train_mhgl(
            encoder, X, A_hat, centres, high_conf, anom_idx,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            sigma=cfg.sigma,
            eps=cfg.eps_repulsion,
            augmentation_alpha=cfg.augmentation_alpha,
            seed=cfg.seed,
            verbose=cfg.verbose,
        )

        # 8. Cache state + scores.
        self._encoder = encoder
        self._A_hat = A_hat
        self._centres = centres
        self._patterns = patterns
        self._high_conf = high_conf
        self._anom_indices = anom_idx
        self._history = history
        self._feat_dim = network.f
        self._n = network.n
        self._scores = anomaly_scores(encoder, X, A_hat, centres)
        return self

    # ------------------------------------------------------------------ Outputs

    def score(self, network: AttributedNetwork | None = None) -> np.ndarray:
        """Return Eq. 3.7 anomaly scores. With no arg, returns cached fit-time scores."""
        if self._encoder is None or self._centres is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._scores is not None
            return self._scores
        if network.f != self._feat_dim:
            raise ValueError(
                f"feature-dim mismatch: trained on {self._feat_dim}, got {network.f}"
            )
        device = torch.device(self.config.device)
        A_hat = build_normalized_adj(network.edges, network.n, device=device)
        X = torch.from_numpy(network.X).to(device)
        return anomaly_scores(self._encoder, X, A_hat, self._centres)

    def predict(self, threshold: float | None = None) -> np.ndarray:
        """Binarise the fit-time anomaly scores.

        Default threshold = median over the union of all high-confidence sets,
        matching "more anomalous than a typical presumed-normal node".
        """
        if self._encoder is None:
            raise RuntimeError("Call fit() first.")
        s = self.score()
        if threshold is None:
            if self._high_conf:
                pool = np.unique(np.concatenate(self._high_conf))
                threshold = float(np.median(s[pool]))
            else:
                threshold = float(np.median(s))
        return (s >= threshold).astype(np.int64)

    @property
    def history(self) -> dict[str, list[float]]:
        """Per-epoch train / val losses recorded during ``fit``."""
        return self._history

    def patterns(self) -> list[Pattern]:
        """Fine-grained normal patterns from the PDE (paper §3.1)."""
        if not self._patterns:
            raise RuntimeError("Call fit() first.")
        return list(self._patterns)

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Pickle encoder weights, centres, patterns, high-conf sets, history, scores."""
        if self._encoder is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        assert self._centres is not None
        torch.save(
            {
                "config": self.config.__dict__,
                "encoder_state_dict": self._encoder.state_dict(),
                "centres": self._centres.cpu(),
                "patterns": [
                    {"indices": p.indices, "posteriors": p.posteriors}
                    for p in self._patterns
                ],
                "high_conf": self._high_conf,
                "anom_indices": self._anom_indices,
                "history": self._history,
                "scores": self._scores,
                "feat_dim": self._feat_dim,
                "n": self._n,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MHGL":
        """Reconstruct an MHGL from a ``save()`` checkpoint (CPU-loaded)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        # tuple round-trip — torch.save serialises tuples as lists in the dict
        cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
        obj = cls(MHGLConfig(**cfg_dict))
        obj._feat_dim = ckpt["feat_dim"]
        obj._n = ckpt["n"]
        encoder = GCNEncoder(
            in_dim=obj._feat_dim,
            hidden_dims=obj.config.hidden_dims,
            activation=obj.config.activation,
            bias=obj.config.bias,
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        obj._encoder = encoder
        obj._centres = ckpt["centres"]
        obj._patterns = [
            Pattern(indices=p["indices"], posteriors=p["posteriors"])
            for p in ckpt["patterns"]
        ]
        obj._high_conf = ckpt["high_conf"]
        obj._anom_indices = ckpt["anom_indices"]
        obj._history = ckpt["history"]
        obj._scores = ckpt["scores"]
        return obj
