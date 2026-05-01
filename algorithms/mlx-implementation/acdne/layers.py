"""MLX ACDNE neural-network components (paper §3, Eqs. 1-10)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.fc2(nn.relu(self.fc1(x))))


class EmbeddingModule(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, fe_out_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.fe1 = FeatureExtractor(in_dim, hidden_dim, fe_out_dim)
        self.fe2 = FeatureExtractor(in_dim, hidden_dim, fe_out_dim)
        self.concat = nn.Linear(2 * fe_out_dim, embed_dim)

    def __call__(self, x: mx.array, n: mx.array) -> mx.array:
        h1 = self.fe1(x)
        h2 = self.fe2(n)
        return nn.relu(self.concat(mx.concatenate([h1, h2], axis=-1)))


class NodeClassifier(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def __call__(self, e: mx.array) -> mx.array:
        return self.fc(e)


class DomainDiscriminator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

    def __call__(self, e: mx.array) -> mx.array:
        z = nn.relu(self.fc1(e))
        z = nn.relu(self.fc2(z))
        return self.out(z)
