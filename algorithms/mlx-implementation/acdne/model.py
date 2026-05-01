"""MLX ACDNE orchestrator (paper §3, Algorithm 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from data import CrossNetwork, neighbour_input, ppmi_matrix
from layers import DomainDiscriminator, EmbeddingModule, NodeClassifier
from train import train_acdne


@dataclass
class ACDNEConfig:
    """Hyperparameter bundle for ACDNE; defaults follow paper §Implementation Details."""

    embed_hidden_dim: int = 512
    fe_out_dim: int = 128
    embed_dim: int = 128
    disc_hidden_dim: int = 128
    n_iters: int = 1000
    batch_size: int = 100
    mu_0: float = 0.02
    momentum: float = 0.9
    p_pair: float = 0.1
    weight_decay: float = 1e-3
    ppmi_K: int = 3
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


class ACDNE:
    """End-to-end ACDNE trainer with a NumPy-facing public API."""

    def __init__(self, config: ACDNEConfig | None = None) -> None:
        self.config = config or ACDNEConfig()
        self._embed: EmbeddingModule | None = None
        self._classifier: NodeClassifier | None = None
        self._discriminator: DomainDiscriminator | None = None
        self._history: dict[str, list[float]] = {}
        self._cached_e_s: np.ndarray | None = None
        self._cached_e_t: np.ndarray | None = None
        self._cached_y_t_pred: np.ndarray | None = None
        self._feat_dim: int | None = None
        self._n_classes: int | None = None

    def fit(self, network: CrossNetwork) -> "ACDNE":
        """Run Algorithm 1 on ``network`` and cache trained outputs."""
        cfg = self.config
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)

        if cfg.device != "cpu" and cfg.verbose:
            print("[ACDNE] MLX selects its active device globally; ignoring config.device")

        A_s = ppmi_matrix(network.edges_s, network.n_s, K=cfg.ppmi_K)
        A_t = ppmi_matrix(network.edges_t, network.n_t, K=cfg.ppmi_K)
        N_s = neighbour_input(network.X_s, A_s)
        N_t = neighbour_input(network.X_t, A_t)

        X_s = mx.array(network.X_s)
        X_t = mx.array(network.X_t)
        N_s_mx = mx.array(N_s)
        N_t_mx = mx.array(N_t)
        A_s_mx = mx.array(A_s)
        A_t_mx = mx.array(A_t)
        y_s = mx.array(network.y_s, dtype=mx.int32)

        embed = EmbeddingModule(
            in_dim=network.feat_dim,
            hidden_dim=cfg.embed_hidden_dim,
            fe_out_dim=cfg.fe_out_dim,
            embed_dim=cfg.embed_dim,
        )
        classifier = NodeClassifier(cfg.embed_dim, network.n_classes)
        discriminator = DomainDiscriminator(cfg.embed_dim, hidden_dim=cfg.disc_hidden_dim)
        mx.eval(embed.parameters(), classifier.parameters(), discriminator.parameters())

        if cfg.verbose:
            print(
                f"[ACDNE] n_s={network.n_s} n_t={network.n_t} "
                f"feat_dim={network.feat_dim} n_classes={network.n_classes}  "
                f"PPMI K={cfg.ppmi_K}"
            )

        self._history = train_acdne(
            embed,
            classifier,
            discriminator,
            X_s=X_s,
            N_s=N_s_mx,
            A_s_ppmi=A_s_mx,
            y_s=y_s,
            X_t=X_t,
            N_t=N_t_mx,
            A_t_ppmi=A_t_mx,
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

        embed.eval()
        classifier.eval()
        e_s = embed(X_s, N_s_mx)
        e_t = embed(X_t, N_t_mx)
        logits_t = classifier(e_t)
        mx.eval(e_s, e_t, logits_t)
        self._cached_e_s = np.asarray(e_s, dtype=np.float32)
        self._cached_e_t = np.asarray(e_t, dtype=np.float32)
        self._cached_y_t_pred = np.asarray(mx.argmax(logits_t, axis=-1), dtype=np.int64)
        return self

    def embed(self, which: Literal["source", "target", "both"] = "target") -> np.ndarray:
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
        """Argmax target-node predictions."""
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
            logits = self._classifier(mx.array(self._cached_e_t))
            probs = nn.softmax(logits, axis=-1)
            mx.eval(probs)
            return np.asarray(probs, dtype=np.float32)
        return self._predict_fresh(network)

    def _predict_fresh(self, network: CrossNetwork) -> np.ndarray:
        """Run the trained model on a fresh target side and return softmax."""
        if network.feat_dim != self._feat_dim:
            raise ValueError(
                f"feat_dim mismatch: trained on {self._feat_dim}, got {network.feat_dim}"
            )
        if network.n_classes != self._n_classes:
            raise ValueError(
                f"n_classes mismatch: trained on {self._n_classes}, got {network.n_classes}"
            )
        assert self._embed is not None and self._classifier is not None
        A_t = ppmi_matrix(network.edges_t, network.n_t, K=self.config.ppmi_K)
        N_t = neighbour_input(network.X_t, A_t)
        self._embed.eval()
        self._classifier.eval()
        logits = self._classifier(self._embed(mx.array(network.X_t), mx.array(N_t)))
        probs = nn.softmax(logits, axis=-1)
        mx.eval(probs)
        return np.asarray(probs, dtype=np.float32)

    @property
    def history(self) -> dict[str, list[float]]:
        """Per-iteration training metrics from ``fit``."""
        return self._history

    def save(self, path: str) -> None:
        """Save config, MLX module weights, and cached NumPy outputs."""
        if self._embed is None or self._classifier is None or self._discriminator is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        np.savez(
            path,
            config=np.array(asdict(self.config), dtype=object),
            embed_state=np.array(self._tree_to_numpy(self._embed.state), dtype=object),
            classifier_state=np.array(self._tree_to_numpy(self._classifier.state), dtype=object),
            discriminator_state=np.array(self._tree_to_numpy(self._discriminator.state), dtype=object),
            history=np.array(self._history, dtype=object),
            cached_e_s=self._cached_e_s,
            cached_e_t=self._cached_e_t,
            cached_y_t_pred=self._cached_y_t_pred,
            feat_dim=np.array(self._feat_dim, dtype=np.int64),
            n_classes=np.array(self._n_classes, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "ACDNE":
        """Reconstruct an ``ACDNE`` saved by ``save``."""
        with np.load(path, allow_pickle=True) as ckpt:
            cfg = ACDNEConfig(**ckpt["config"].item())
            obj = cls(cfg)
            obj._feat_dim = int(ckpt["feat_dim"])
            obj._n_classes = int(ckpt["n_classes"])
            embed = EmbeddingModule(
                in_dim=obj._feat_dim,
                hidden_dim=cfg.embed_hidden_dim,
                fe_out_dim=cfg.fe_out_dim,
                embed_dim=cfg.embed_dim,
            )
            classifier = NodeClassifier(cfg.embed_dim, obj._n_classes)
            discriminator = DomainDiscriminator(cfg.embed_dim, hidden_dim=cfg.disc_hidden_dim)
            embed.update(obj._tree_to_mx(ckpt["embed_state"].item()))
            classifier.update(obj._tree_to_mx(ckpt["classifier_state"].item()))
            discriminator.update(obj._tree_to_mx(ckpt["discriminator_state"].item()))
            obj._embed = embed
            obj._classifier = classifier
            obj._discriminator = discriminator
            obj._history = ckpt["history"].item()
            obj._cached_e_s = ckpt["cached_e_s"].astype(np.float32)
            obj._cached_e_t = ckpt["cached_e_t"].astype(np.float32)
            obj._cached_y_t_pred = ckpt["cached_y_t_pred"].astype(np.int64)
        return obj

    @classmethod
    def _tree_to_numpy(cls, value: Any) -> Any:
        if isinstance(value, mx.array):
            return np.asarray(value)
        if isinstance(value, dict):
            return {k: cls._tree_to_numpy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._tree_to_numpy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._tree_to_numpy(v) for v in value)
        return value

    @classmethod
    def _tree_to_mx(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.dtype != object:
            return mx.array(value)
        if isinstance(value, dict):
            return {k: cls._tree_to_mx(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._tree_to_mx(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._tree_to_mx(v) for v in value)
        return value
