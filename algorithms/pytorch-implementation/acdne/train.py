"""Joint training loop for ACDNE (paper §3, Algorithm 1, Eqs. 7/10/11/12).

Reference: Shen, Dai, Chung, Lu, Choi — "Adversarial Deep Network Embedding
for Cross-network Node Classification", AAAI 2020.

Implements Algorithm 1 (paper §3, "Joint Training"):
    while not converged:
      for each minibatch B (b/2 source + b/2 target):
          - run shared EmbeddingModule on the source half
          - run shared EmbeddingModule on the target half
          - L_y = cross-entropy on labelled source (Eq. 7)
          - L_p = within-batch pairwise constraint (Eq. 5) on each side
          - L_d = binary cross-entropy from discriminator (Eq. 10);
                  GRL flips its gradient flowing into the encoder
          - one SGD step on (encoder ∪ classifier ∪ discriminator)
          - decay LR per μ_p = μ_0 / (1+10p)^0.75       (paper §Impl)
          - ramp GRL λ per λ = 2/(1+e^{-10p}) - 1       (paper §Impl)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from layers import DomainDiscriminator, EmbeddingModule, NodeClassifier


# --- Schedules (paper §Implementation Details) ---------------------------------

def lr_at(progress: float, mu_0: float) -> float:
    """μ_p = μ_0 / (1 + 10p)^0.75; p in [0, 1]."""
    p = float(min(max(progress, 0.0), 1.0))
    return mu_0 / (1.0 + 10.0 * p) ** 0.75


def grl_lambda_at(progress: float) -> float:
    """λ = 2 / (1 + exp(-10p)) - 1; p in [0, 1]; ramps 0 → ~1."""
    p = float(min(max(progress, 0.0), 1.0))
    # exp(-10) ≈ 4.5e-5 — well within float32 range, but clamp the input
    # anyway to keep the schedule defensive against caller drift.
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# --- Mini-batch sampling (Algorithm 1, line 2) ---------------------------------

def sample_minibatch(
    rng: np.random.Generator,
    n_s: int,
    n_t: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-source / half-target sampling without replacement.

    Returns sorted int64 index arrays. If a side is smaller than the
    requested half-batch, samples with replacement on that side
    (which is a no-op for the smoke test sizes but keeps the loop
    robust on tiny synthetic graphs).
    """
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2, got {batch_size}")
    half = batch_size // 2
    replace_s = half > n_s
    replace_t = half > n_t
    idx_s = np.sort(rng.choice(n_s, size=half, replace=replace_s)).astype(np.int64)
    idx_t = np.sort(rng.choice(n_t, size=half, replace=replace_t)).astype(np.int64)
    return idx_s, idx_t


# --- Pairwise constraint loss (Eq. 5) ------------------------------------------

def pairwise_loss(e_batch: torch.Tensor, ppmi_sub: torch.Tensor) -> torch.Tensor:
    """Within-batch piece of paper Eq. 5.

        L_p^{batch} = (1 / B^2) sum_{i, j in B} a_ij ||e_i - e_j||^2

    where ``ppmi_sub`` is the precomputed (B, B) sub-block of the full
    PPMI matrix indexed by the batch nodes. The diagonal is zero by
    construction (the full matrix is built that way), so self-pairs
    contribute nothing.
    """
    if e_batch.dim() != 2:
        raise ValueError(f"e_batch must be 2-D, got shape {tuple(e_batch.shape)}")
    if ppmi_sub.shape != (e_batch.shape[0], e_batch.shape[0]):
        raise ValueError(
            f"ppmi_sub shape {tuple(ppmi_sub.shape)} != ({e_batch.shape[0]},)*2"
        )
    B = e_batch.shape[0]
    # Pairwise squared distance via the |a-b|^2 = |a|^2 + |b|^2 - 2 a·b expansion.
    sq = (e_batch * e_batch).sum(dim=-1)                       # (B,)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (e_batch @ e_batch.t())
    dist_sq = dist_sq.clamp(min=0.0)                           # numerical floor
    return (ppmi_sub * dist_sq).sum() / float(B * B)


# --- Joint training loop (Algorithm 1) -----------------------------------------

def train_acdne(
    embed: EmbeddingModule,
    classifier: NodeClassifier,
    discriminator: DomainDiscriminator,
    *,
    X_s: torch.Tensor,
    N_s: torch.Tensor,
    A_s_ppmi: torch.Tensor,
    y_s: torch.Tensor,
    X_t: torch.Tensor,
    N_t: torch.Tensor,
    A_t_ppmi: torch.Tensor,
    n_iters: int,
    batch_size: int,
    mu_0: float,
    p_pair: float,
    weight_decay: float,
    momentum: float,
    seed: int,
    verbose: bool,
) -> dict[str, list[float]]:
    """Run Algorithm 1 end-to-end and return a per-iteration history.

    A single SGD optimiser updates encoder, classifier, AND discriminator;
    the GRL ensures that gradients flowing into the encoder from L_d are
    sign-flipped while the discriminator's own parameters get the natural
    gradient. This matches paper Eq. 12 exactly.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    device = X_s.device
    rng = np.random.default_rng(seed)

    params = (
        list(embed.parameters())
        + list(classifier.parameters())
        + list(discriminator.parameters())
    )
    opt = torch.optim.SGD(
        params, lr=mu_0, momentum=momentum, weight_decay=weight_decay
    )

    history: dict[str, list[float]] = {
        "loss_y": [],
        "loss_p": [],
        "loss_d": [],
        "loss_total": [],
        "domain_acc": [],
        "lr": [],
        "grl_lambda": [],
    }

    log_every = max(1, n_iters // 10)
    n_s, n_t = int(X_s.shape[0]), int(X_t.shape[0])

    for it in range(n_iters):
        progress = it / max(1, n_iters - 1)
        lr = lr_at(progress, mu_0)
        lam = grl_lambda_at(progress)
        for pg in opt.param_groups:
            pg["lr"] = lr
        discriminator.set_lambda(lam)

        idx_s, idx_t = sample_minibatch(rng, n_s, n_t, batch_size)
        idx_s_t = torch.from_numpy(idx_s).to(device)
        idx_t_t = torch.from_numpy(idx_t).to(device)

        embed.train(); classifier.train(); discriminator.train()

        e_s = embed(X_s[idx_s_t], N_s[idx_s_t])
        e_t = embed(X_t[idx_t_t], N_t[idx_t_t])

        # L_y on labelled source (Eq. 7).
        logits_y = classifier(e_s)
        loss_y = F.cross_entropy(logits_y, y_s[idx_s_t])

        # L_p on within-batch pairs of each side (Eq. 5).
        ppmi_s_sub = A_s_ppmi[idx_s_t][:, idx_s_t]
        ppmi_t_sub = A_t_ppmi[idx_t_t][:, idx_t_t]
        loss_p = pairwise_loss(e_s, ppmi_s_sub) + pairwise_loss(e_t, ppmi_t_sub)

        # L_d on the concatenated batch (Eq. 10). 0 = source, 1 = target.
        e_all = torch.cat([e_s, e_t], dim=0)
        d_labels = torch.cat([
            torch.zeros(e_s.shape[0], dtype=torch.long, device=device),
            torch.ones(e_t.shape[0], dtype=torch.long, device=device),
        ])
        logits_d = discriminator(e_all)
        loss_d = F.cross_entropy(logits_d, d_labels)

        # Eq. 11: combined objective. The GRL is wired so that the
        # encoder receives -lambda * dL_d/de while the discriminator
        # receives +dL_d/dtheta_d — one SGD step does the right thing
        # for all three parameter sets.
        loss = loss_y + p_pair * loss_p + loss_d

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            domain_acc = (logits_d.argmax(dim=-1) == d_labels).float().mean().item()

        history["loss_y"].append(float(loss_y.item()))
        history["loss_p"].append(float(loss_p.item()))
        history["loss_d"].append(float(loss_d.item()))
        history["loss_total"].append(float(loss.item()))
        history["domain_acc"].append(float(domain_acc))
        history["lr"].append(float(lr))
        history["grl_lambda"].append(float(lam))

        if verbose and (it < 3 or it % log_every == 0 or it == n_iters - 1):
            print(
                f"  iter {it + 1:>5d}/{n_iters}  "
                f"L_y={loss_y.item():.4f}  "
                f"L_p={loss_p.item():.4f}  "
                f"L_d={loss_d.item():.4f}  "
                f"d_acc={domain_acc:.3f}  "
                f"λ={lam:.3f}  lr={lr:.4f}"
            )

    return history
