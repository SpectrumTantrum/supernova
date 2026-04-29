"""ACDNE neural-network components (paper §3, Eqs. 1-10).

Reference: Shen, Dai, Chung, Lu, Choi — "Adversarial Deep Network Embedding
for Cross-network Node Classification", AAAI 2020.

Components:
    - FeatureExtractor   : 2-hidden-layer MLP for FE1 (Eq. 1) and FE2 (Eq. 2).
    - EmbeddingModule    : FE1 + FE2 + concat fusion (Eq. 4); shared across
                           source and target (paper §3, "shared trainable
                           parameters between G^s & G^t").
    - GradientReversal   : Ganin & Lempitsky 2015 GRL with a mutable lambda;
                           inserted before the discriminator per Eq. 11.
    - NodeClassifier     : final softmax classifier (Eq. 6, multi-class).
    - DomainDiscriminator: 2-hidden-layer MLP outputting a 2-way logit
                           (source vs target) per Eq. 9.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Feature extractors (paper Eqs. 1-2) ---------------------------------------

class FeatureExtractor(nn.Module):
    """Two-hidden-layer MLP with ReLU between layers and at the output.

    Implements paper Eq. 1 (FE1, attribute encoder) when fed x_i, and
    Eq. 2 (FE2, neighbour encoder) when fed n_i. Both share this same
    architectural shape (paper §3.1, "the number of hidden layers l_f and
    the dimensionality of each k-th hidden layer f(k) are set as the
    same for FE1 and FE2") but are instantiated as independent modules
    with separate W_{f_1}, W_{f_2} weights.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))


class EmbeddingModule(nn.Module):
    """Deep network embedding module — FE1 + FE2 + concat fusion (Eq. 4).

    Paper §3, Fig. 1: the SAME `EmbeddingModule` is invoked on source and
    target nodes (shared parameters); domain alignment is then enforced
    by the adversarial discriminator rather than by separate embedding
    networks.

    Returns e_i in R^embed_dim per Eq. 4:
        e_i = ReLU([h_{f_1}(x_i), h_{f_2}(n_i)] W_c + b_c)
    """

    def __init__(self, in_dim: int, hidden_dim: int, fe_out_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.fe1 = FeatureExtractor(in_dim, hidden_dim, fe_out_dim)
        self.fe2 = FeatureExtractor(in_dim, hidden_dim, fe_out_dim)
        self.concat = nn.Linear(2 * fe_out_dim, embed_dim)

    def forward(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        if x.shape != n.shape:
            raise ValueError(
                f"x and n must have the same shape; got {tuple(x.shape)} vs {tuple(n.shape)}"
            )
        h1 = self.fe1(x)                                # (B, fe_out_dim)
        h2 = self.fe2(n)                                # (B, fe_out_dim)
        return F.relu(self.concat(torch.cat([h1, h2], dim=-1)))


# --- Gradient Reversal Layer (Ganin & Lempitsky 2015) --------------------------

class GradientReversalFn(torch.autograd.Function):
    """Forward identity; backward multiplies the upstream gradient by -lambda.

    Paper Eq. 11 / Eq. 12: this is the mechanism by which the encoder
    receives the *negated* discriminator gradient, turning the inner
    max_{theta_d} into a single SGD step.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Stateful wrapper around `GradientReversalFn`.

    `lambda_` is a plain Python float updated in-place by the trainer
    each iteration (DANN ramp `2/(1+exp(-10p)) - 1` per paper §Implementation
    Details). Nothing is registered as a Parameter or Buffer — `lambda_`
    is a non-learnable schedule, not part of the model state.
    """

    def __init__(self, lambda_: float = 0.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambda_)


# --- Heads (paper Eqs. 6, 9) ---------------------------------------------------

class NodeClassifier(nn.Module):
    """Single linear layer over the embedding (Eq. 6, multi-class softmax).

    Cross-entropy / softmax is left to `F.cross_entropy` in the trainer
    for numerical stability — this module returns raw logits.
    """

    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc(e)


class DomainDiscriminator(nn.Module):
    """Two-hidden-layer MLP predicting source (0) vs target (1) — paper Eq. 9.

    The forward pass starts with the GRL: gradients flowing back to the
    embedding module are sign-flipped, while the discriminator's own
    parameters receive the natural gradient and so are trained to
    *maximise* domain-classification accuracy.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.grl = GradientReversal()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

    def set_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = float(lambda_)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.grl(e)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.out(z)
