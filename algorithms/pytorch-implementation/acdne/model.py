"""ACDNE orchestrator (paper §3, Algorithm 1).

Reference: Shen, Dai, Chung, Lu, Choi — "Adversarial Deep Network Embedding
for Cross-network Node Classification", AAAI 2020.

Wires the four sibling modules into a single trainer:
    data.py     -> CrossNetwork + ppmi_matrix + neighbour_input
    layers.py   -> EmbeddingModule, NodeClassifier, DomainDiscriminator
    train.py    -> train_acdne (Algorithm 1 joint loop)

Typical usage:
    from data import SyntheticCrossNetwork
    from model import ACDNE, ACDNEConfig

    net = SyntheticCrossNetwork(seed=0).generate()
    model = ACDNE(ACDNEConfig(n_iters=1000)).fit(net)
    y_pred = model.predict()                # target-side label predictions
    e_t    = model.embed("target")          # learned target embeddings
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from data import CrossNetwork, neighbour_input, ppmi_matrix
from layers import DomainDiscriminator, EmbeddingModule, NodeClassifier
from train import train_acdne


@dataclass
class ACDNEConfig:
    """Hyperparameter bundle for ACDNE — defaults follow paper §Implementation Details.

    `mu_0=0.02` and `p_pair=0.1` are the citation-network defaults; for
    Blog-style dense networks the paper uses `mu_0=0.01`, `p_pair=1e-3`.
    `n_iters` replaces `epochs` — Algorithm 1 is iteration-based.
    """

    embed_hidden_dim: int = 512        # paper f(1)
    fe_out_dim: int = 128              # paper f(2)
    embed_dim: int = 128               # paper d
    disc_hidden_dim: int = 128         # paper d(1) = d(2) = 128
    n_iters: int = 1000
    batch_size: int = 100              # paper §Implementation Details
    mu_0: float = 0.02                 # paper §Implementation Details (citation default)
    momentum: float = 0.9              # paper §Implementation Details
    p_pair: float = 0.1                # paper §Implementation Details (sparse default)
    weight_decay: float = 1e-3         # paper §Implementation Details: L2 weight 10^-3
    ppmi_K: int = 3                    # paper §Implementation Details: K-step = 3
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


class ACDNE:
    """End-to-end ACDNE trainer (paper §3, Algorithm 1).

    Stateless until ``fit`` runs; afterwards exposes embeddings, target
    predictions, training history, and save/load.
    """

    def __init__(self, config: ACDNEConfig | None = None) -> None:
        self.config = config or ACDNEConfig()
        self._embed: Optional[EmbeddingModule] = None
        self._classifier: Optional[NodeClassifier] = None
        self._discriminator: Optional[DomainDiscriminator] = None
        self._history: dict[str, list[float]] = {}
        self._cached_e_s: Optional[np.ndarray] = None
        self._cached_e_t: Optional[np.ndarray] = None
        self._cached_y_t_pred: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._n_classes: Optional[int] = None

    # ------------------------------------------------------------------ Fit

    def fit(self, network: CrossNetwork) -> "ACDNE":
        """Run Algorithm 1 end-to-end on ``network`` and cache trained state."""
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        device = torch.device(cfg.device)

        # Precompute PPMI and neighbour-input matrices for both networks.
        A_s = ppmi_matrix(network.edges_s, network.n_s, K=cfg.ppmi_K)
        A_t = ppmi_matrix(network.edges_t, network.n_t, K=cfg.ppmi_K)
        N_s = neighbour_input(network.X_s, A_s)
        N_t = neighbour_input(network.X_t, A_t)

        X_s_t = torch.from_numpy(network.X_s).to(device)
        X_t_t = torch.from_numpy(network.X_t).to(device)
        N_s_t = torch.from_numpy(N_s).to(device)
        N_t_t = torch.from_numpy(N_t).to(device)
        A_s_t = torch.from_numpy(A_s).to(device)
        A_t_t = torch.from_numpy(A_t).to(device)
        y_s_t = torch.from_numpy(network.y_s).to(device)

        embed = EmbeddingModule(
            in_dim=network.feat_dim,
            hidden_dim=cfg.embed_hidden_dim,
            fe_out_dim=cfg.fe_out_dim,
            embed_dim=cfg.embed_dim,
        ).to(device)
        classifier = NodeClassifier(cfg.embed_dim, network.n_classes).to(device)
        discriminator = DomainDiscriminator(
            cfg.embed_dim, hidden_dim=cfg.disc_hidden_dim
        ).to(device)

        if cfg.verbose:
            print(
                f"[ACDNE] n_s={network.n_s} n_t={network.n_t} "
                f"feat_dim={network.feat_dim} n_classes={network.n_classes}  "
                f"PPMI K={cfg.ppmi_K}"
            )

        self._history = train_acdne(
            embed, classifier, discriminator,
            X_s=X_s_t, N_s=N_s_t, A_s_ppmi=A_s_t, y_s=y_s_t,
            X_t=X_t_t, N_t=N_t_t, A_t_ppmi=A_t_t,
            n_iters=cfg.n_iters,
            batch_size=cfg.batch_size,
            mu_0=cfg.mu_0,
            p_pair=cfg.p_pair,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            seed=cfg.seed,
            verbose=cfg.verbose,
        )

        self._embed = embed
        self._classifier = classifier
        self._discriminator = discriminator
        self._feat_dim = network.feat_dim
        self._n_classes = network.n_classes

        # Cache full-network embeddings + target predictions from the
        # final-iteration parameters (paper Algorithm 1, "Output").
        with torch.no_grad():
            embed.train(False)
            classifier.train(False)
            e_s = embed(X_s_t, N_s_t)
            e_t = embed(X_t_t, N_t_t)
            logits_t = classifier(e_t)
            y_t_pred = logits_t.argmax(dim=-1)
            self._cached_e_s = e_s.detach().cpu().numpy().astype(np.float32)
            self._cached_e_t = e_t.detach().cpu().numpy().astype(np.float32)
            self._cached_y_t_pred = y_t_pred.detach().cpu().numpy().astype(np.int64)
        return self

    # ------------------------------------------------------------------ Outputs

    def embed(
        self,
        which: Literal["source", "target", "both"] = "target",
    ) -> np.ndarray:
        """Return cached embeddings from ``fit``."""
        if self._cached_e_s is None or self._cached_e_t is None:
            raise RuntimeError("Call fit() first.")
        if which == "source":
            return self._cached_e_s
        if which == "target":
            return self._cached_e_t
        if which == "both":
            return np.concatenate([self._cached_e_s, self._cached_e_t], axis=0)
        raise ValueError(f"which must be 'source'|'target'|'both', got {which!r}")

    def predict(self, network: CrossNetwork | None = None) -> np.ndarray:
        """Argmax target-node predictions (paper Algorithm 1, Output)."""
        if self._embed is None or self._classifier is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._cached_y_t_pred is not None
            return self._cached_y_t_pred
        return self._predict_fresh(network).argmax(axis=-1).astype(np.int64)

    def predict_proba(self, network: CrossNetwork | None = None) -> np.ndarray:
        """Softmax target-node probabilities."""
        if self._embed is None or self._classifier is None:
            raise RuntimeError("Call fit() first.")
        if network is None:
            assert self._cached_e_t is not None
            with torch.no_grad():
                e_t = torch.from_numpy(self._cached_e_t).to(self.config.device)
                logits = self._classifier(e_t)
                probs = torch.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy().astype(np.float32)
        return self._predict_fresh(network)

    def _predict_fresh(self, network: CrossNetwork) -> np.ndarray:
        """Run the trained model on a (possibly new) target side; return softmax."""
        if network.feat_dim != self._feat_dim:
            raise ValueError(
                f"feat_dim mismatch: trained on {self._feat_dim}, got {network.feat_dim}"
            )
        if network.n_classes != self._n_classes:
            raise ValueError(
                f"n_classes mismatch: trained on {self._n_classes}, got {network.n_classes}"
            )
        device = torch.device(self.config.device)
        A_t = ppmi_matrix(network.edges_t, network.n_t, K=self.config.ppmi_K)
        N_t = neighbour_input(network.X_t, A_t)
        X_t_t = torch.from_numpy(network.X_t).to(device)
        N_t_t = torch.from_numpy(N_t).to(device)
        with torch.no_grad():
            assert self._embed is not None and self._classifier is not None
            self._embed.train(False)
            self._classifier.train(False)
            e_t = self._embed(X_t_t, N_t_t)
            logits = self._classifier(e_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy().astype(np.float32)

    @property
    def history(self) -> dict[str, list[float]]:
        """Per-iteration training metrics from `fit`."""
        return self._history

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Round-trip config + module state_dicts + caches via torch.save."""
        if self._embed is None or self._classifier is None or self._discriminator is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        torch.save(
            {
                "config": self.config.__dict__,
                "embed_state_dict": self._embed.state_dict(),
                "classifier_state_dict": self._classifier.state_dict(),
                "discriminator_state_dict": self._discriminator.state_dict(),
                "history": self._history,
                "cached_e_s": self._cached_e_s,
                "cached_e_t": self._cached_e_t,
                "cached_y_t_pred": self._cached_y_t_pred,
                "feat_dim": self._feat_dim,
                "n_classes": self._n_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ACDNE":
        """Reconstruct an ACDNE from a `save()` checkpoint (CPU-loaded)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ACDNEConfig(**ckpt["config"])
        cfg.device = "cpu"
        obj = cls(cfg)
        obj._feat_dim = ckpt["feat_dim"]
        obj._n_classes = ckpt["n_classes"]
        embed = EmbeddingModule(
            in_dim=obj._feat_dim,
            hidden_dim=obj.config.embed_hidden_dim,
            fe_out_dim=obj.config.fe_out_dim,
            embed_dim=obj.config.embed_dim,
        )
        embed.load_state_dict(ckpt["embed_state_dict"])
        classifier = NodeClassifier(obj.config.embed_dim, obj._n_classes)
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        discriminator = DomainDiscriminator(
            obj.config.embed_dim, hidden_dim=obj.config.disc_hidden_dim
        )
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        obj._embed = embed
        obj._classifier = classifier
        obj._discriminator = discriminator
        obj._history = ckpt["history"]
        obj._cached_e_s = ckpt["cached_e_s"]
        obj._cached_e_t = ckpt["cached_e_t"]
        obj._cached_y_t_pred = ckpt["cached_y_t_pred"]
        return obj
