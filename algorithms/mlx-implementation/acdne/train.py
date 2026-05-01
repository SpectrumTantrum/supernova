"""MLX joint training loop for ACDNE (paper §3, Algorithm 1)."""

from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from layers import DomainDiscriminator, EmbeddingModule, NodeClassifier


def lr_at(progress: float, mu_0: float) -> float:
    p = float(min(max(progress, 0.0), 1.0))
    return mu_0 / (1.0 + 10.0 * p) ** 0.75


def grl_lambda_at(progress: float) -> float:
    p = float(min(max(progress, 0.0), 1.0))
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def sample_minibatch(rng: np.random.Generator, n_s: int, n_t: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    half = batch_size // 2
    return (
        np.sort(rng.choice(n_s, size=half, replace=half > n_s)).astype(np.int64),
        np.sort(rng.choice(n_t, size=half, replace=half > n_t)).astype(np.int64),
    )


def pairwise_loss(e_batch: mx.array, ppmi_sub: mx.array) -> mx.array:
    B = e_batch.shape[0]
    sq = mx.sum(e_batch * e_batch, axis=-1)
    dist_sq = mx.maximum(sq[:, None] + sq[None, :] - 2.0 * (e_batch @ e_batch.T), 0.0)
    return mx.sum(ppmi_sub * dist_sq) / float(B * B)


def _ce(logits: mx.array, y: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(logits, y, reduction="none"))


def train_acdne(embed: EmbeddingModule, classifier: NodeClassifier, discriminator: DomainDiscriminator, *, X_s: mx.array, N_s: mx.array, A_s_ppmi: mx.array, y_s: mx.array, X_t: mx.array, N_t: mx.array, A_t_ppmi: mx.array, n_iters: int, batch_size: int, mu_0: float, p_pair: float, weight_decay: float, momentum: float, seed: int, verbose: bool) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    opt_enc = optim.SGD(learning_rate=mu_0, momentum=momentum, weight_decay=weight_decay)
    opt_cls = optim.SGD(learning_rate=mu_0, momentum=momentum, weight_decay=weight_decay)
    opt_disc = optim.SGD(learning_rate=mu_0, momentum=momentum, weight_decay=weight_decay)
    history = {"loss_y": [], "loss_p": [], "loss_d": [], "loss_total": [], "domain_acc": [], "lr": [], "grl_lambda": []}
    log_every = max(1, n_iters // 10)
    n_s, n_t = int(X_s.shape[0]), int(X_t.shape[0])

    for it in range(n_iters):
        progress = it / max(1, n_iters - 1)
        lr = lr_at(progress, mu_0)
        lam = grl_lambda_at(progress)
        opt_enc.learning_rate = lr
        opt_cls.learning_rate = lr
        opt_disc.learning_rate = lr
        idx_s, idx_t = sample_minibatch(rng, n_s, n_t, batch_size)
        s_idx = mx.array(idx_s, dtype=mx.int32)
        t_idx = mx.array(idx_t, dtype=mx.int32)
        d_labels = mx.concatenate([mx.zeros((len(idx_s),), dtype=mx.int32), mx.ones((len(idx_t),), dtype=mx.int32)], axis=0)

        def encoder_loss_fn(model: EmbeddingModule) -> mx.array:
            e_s = model(X_s[s_idx], N_s[s_idx])
            e_t = model(X_t[t_idx], N_t[t_idx])
            loss_y = _ce(classifier(e_s), y_s[s_idx])
            loss_p = pairwise_loss(e_s, A_s_ppmi[s_idx][:, s_idx]) + pairwise_loss(e_t, A_t_ppmi[t_idx][:, t_idx])
            # Gradient-reversal equivalent: encoder minimizes -lambda * L_d.
            loss_d = _ce(discriminator(mx.concatenate([e_s, e_t], axis=0)), d_labels)
            loss = loss_y + p_pair * loss_p - lam * loss_d
            return loss

        def classifier_loss_fn(model: NodeClassifier) -> mx.array:
            e_s = mx.stop_gradient(embed(X_s[s_idx], N_s[s_idx]))
            return _ce(model(e_s), y_s[s_idx])

        def discriminator_loss_fn(model: DomainDiscriminator) -> mx.array:
            e_s = mx.stop_gradient(embed(X_s[s_idx], N_s[s_idx]))
            e_t = mx.stop_gradient(embed(X_t[t_idx], N_t[t_idx]))
            return _ce(model(mx.concatenate([e_s, e_t], axis=0)), d_labels)

        enc_loss, enc_grads = nn.value_and_grad(embed, encoder_loss_fn)(embed)
        cls_loss, cls_grads = nn.value_and_grad(classifier, classifier_loss_fn)(classifier)
        disc_loss, disc_grads = nn.value_and_grad(discriminator, discriminator_loss_fn)(discriminator)
        opt_enc.update(embed, enc_grads)
        opt_cls.update(classifier, cls_grads)
        opt_disc.update(discriminator, disc_grads)
        mx.eval(embed.parameters(), classifier.parameters(), discriminator.parameters(), opt_enc.state, opt_cls.state, opt_disc.state)

        e_s = embed(X_s[s_idx], N_s[s_idx])
        e_t = embed(X_t[t_idx], N_t[t_idx])
        loss_y = _ce(classifier(e_s), y_s[s_idx])
        loss_p = pairwise_loss(e_s, A_s_ppmi[s_idx][:, s_idx]) + pairwise_loss(e_t, A_t_ppmi[t_idx][:, t_idx])
        logits_d = discriminator(mx.concatenate([e_s, e_t], axis=0))
        loss_d = _ce(logits_d, d_labels)
        domain_acc = float(mx.mean((mx.argmax(logits_d, axis=-1) == d_labels).astype(mx.float32)))
        loss_total = float(loss_y + p_pair * loss_p + loss_d)
        history["loss_y"].append(float(loss_y))
        history["loss_p"].append(float(loss_p))
        history["loss_d"].append(float(loss_d))
        history["loss_total"].append(loss_total)
        history["domain_acc"].append(domain_acc)
        history["lr"].append(float(lr))
        history["grl_lambda"].append(float(lam))
        if verbose and (it < 3 or it % log_every == 0 or it == n_iters - 1):
            print(f"  iter {it + 1:>5d}/{n_iters}  L_y={float(loss_y):.4f}  L_p={float(loss_p):.4f}  L_d={float(loss_d):.4f}  d_acc={domain_acc:.3f}  λ={lam:.3f}  lr={lr:.4f}")
    return history
