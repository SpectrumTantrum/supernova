"""MLX multi-hypersphere training loop for MHGL (paper §3.2)."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from gcn import GCNEncoder
from pde import Pattern


def compute_centres(H: mx.array, patterns: list[Pattern]) -> mx.array:
    if not patterns:
        raise ValueError("patterns is empty — nothing to compute centres for")
    centres = []
    for p in patterns:
        centres.append(mx.mean(H[mx.array(p.indices, dtype=mx.int32)], axis=0))
    return mx.stop_gradient(mx.stack(centres, axis=0))


def compute_high_confidence(H: mx.array, pattern: Pattern, centre: mx.array, *, threshold_t: float, radius_quantile: float = 1.0) -> np.ndarray:
    if not (0.0 < radius_quantile <= 1.0):
        raise ValueError(f"radius_quantile must be in (0, 1], got {radius_quantile}")
    members = pattern.indices
    posteriors = pattern.posteriors
    F_indices = members[posteriors > threshold_t] if (posteriors > threshold_t).any() else members
    H_np = np.array(H)
    centre_np = np.array(centre)
    F_dists = np.linalg.norm(H_np[F_indices] - centre_np, axis=-1)
    r_i = float(F_dists.max() if radius_quantile >= 1.0 else np.quantile(F_dists, radius_quantile))
    all_dists = np.linalg.norm(H_np - centre_np, axis=-1)
    return np.asarray(np.nonzero(all_dists <= r_i)[0], dtype=np.int64)


def mixup_pseudo_labels(H: mx.array, high_conf_indices: np.ndarray, n_samples: int, rng: np.random.Generator) -> mx.array:
    m = high_conf_indices.shape[0]
    if m < 2 or n_samples <= 0:
        return mx.zeros((0, H.shape[1]), dtype=H.dtype)
    a = rng.integers(0, m, size=n_samples)
    b = rng.integers(0, m, size=n_samples)
    a_idx = mx.array(high_conf_indices[a], dtype=mx.int32)
    b_idx = mx.array(high_conf_indices[b], dtype=mx.int32)
    beta = mx.array(rng.random(size=(n_samples, 1)).astype(np.float32))
    return (1.0 - beta) * H[a_idx] + beta * H[b_idx]


def mhgl_loss(H: mx.array, centres: mx.array, high_conf: list[np.ndarray], mixup_per_pattern: list[mx.array], anom_indices: np.ndarray, *, sigma: float, eps: float) -> mx.array:
    p = centres.shape[0]
    terms = []
    for i in range(p):
        chunks = []
        if high_conf[i].size > 0:
            chunks.append(H[mx.array(high_conf[i], dtype=mx.int32)])
        if mixup_per_pattern[i].shape[0] > 0:
            chunks.append(mixup_per_pattern[i])
        if chunks:
            D_i = mx.concatenate(chunks, axis=0)
            terms.append(mx.mean(mx.sum((D_i - centres[i]) ** 2, axis=-1)))
    contraction = mx.mean(mx.stack(terms)) if terms else mx.array(0.0)
    if anom_indices.size > 0:
        H_anom = H[mx.array(anom_indices, dtype=mx.int32)]
        diff = H_anom[:, None, :] - centres[None, :, :]
        d2 = mx.sum(diff * diff, axis=-1)
        repulsion = mx.mean(1.0 / (d2 + eps))
    else:
        repulsion = mx.array(0.0)
    return contraction + sigma * repulsion


def train_mhgl(encoder: GCNEncoder, X: mx.array, A_hat: mx.array, centres: mx.array, high_conf: list[np.ndarray], anom_indices: np.ndarray, *, epochs: int, lr: float, weight_decay: float, sigma: float, eps: float, augmentation_alpha: int, seed: int, verbose: bool) -> dict[str, list[float]]:
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    opt = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(seed)
    train_losses: list[float] = []
    val_losses: list[float] = []
    log_every = max(1, epochs // 10)

    def loss_fn(model: GCNEncoder) -> mx.array:
        H = model(X, A_hat)
        mix = [mixup_pseudo_labels(H, hi, augmentation_alpha * hi.shape[0], rng) for hi in high_conf]
        loss = mhgl_loss(H, centres, high_conf, mix, anom_indices, sigma=sigma, eps=eps)
        if weight_decay:
            l2 = mx.array(0.0)
            def add_params(tree):
                nonlocal l2
                if isinstance(tree, dict):
                    for v in tree.values():
                        add_params(v)
                elif isinstance(tree, list):
                    for v in tree:
                        add_params(v)
                else:
                    l2 = l2 + mx.sum(tree * tree)
            add_params(model.trainable_parameters())
            loss = loss + 0.5 * weight_decay * l2
        return loss

    grad_fn = nn.value_and_grad(encoder, loss_fn)
    for epoch in range(epochs):
        loss, grads = grad_fn(encoder)
        opt.update(encoder, grads)
        mx.eval(encoder.parameters(), opt.state)
        train_losses.append(float(loss))
        H_val = encoder(X, A_hat)
        val_mix = [mixup_pseudo_labels(H_val, hi, augmentation_alpha * hi.shape[0], rng) for hi in high_conf]
        v = mhgl_loss(H_val, centres, high_conf, val_mix, anom_indices, sigma=sigma, eps=eps)
        val_losses.append(float(v))
        if verbose and (epoch < 5 or epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch + 1}/{epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")
    return {"train_losses": train_losses, "val_losses": val_losses}


def anomaly_scores(encoder: GCNEncoder, X: mx.array, A_hat: mx.array, centres: mx.array) -> np.ndarray:
    H = encoder(X, A_hat)
    diff = H[:, None, :] - centres[None, :, :]
    d2 = mx.sum(diff * diff, axis=-1)
    return np.array(mx.min(d2, axis=-1)).astype(np.float64)
