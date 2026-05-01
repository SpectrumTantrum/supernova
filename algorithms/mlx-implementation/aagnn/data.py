"""Data structures and synthetic-data generator for AAGNN.

Reference: Zhou et al., "Subtractive Aggregation for Attributed Network
Anomaly Detection", CIKM 2021, §2 (problem definition) and §4 (datasets &
anomaly-injection protocol, citing Ding et al. 2019 / Song et al. 2007).

Defines:
    - AttributedNetwork (frozen record of X, edges, optional labels)
    - degree, k_hop_neighbors (graph helpers used by the §3.1 aggregator)
    - SyntheticAttributedNetwork (SBM graph + structural-clique and
      contextual-feature-swap anomaly injection that drives example.py)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# --- Paper §4 / common defaults -------------------------------------------------

DEFAULT_FEATURE_DIM = 32
DEFAULT_N_COMMUNITIES = 5
DEFAULT_NODES_PER_COMMUNITY = 60   # 300 nodes total


# --- Records --------------------------------------------------------------------

@dataclass(frozen=True)
class AttributedNetwork:
    """Paper §2 Def. 1: G = (V, E, X) with optional ground-truth anomaly labels.

    Edges are stored as a (E, 2) int64 array of undirected pairs with i < j;
    duplicates are forbidden. `labels` is populated for synthetic / labelled
    data only — users supplying their own real graph may pass None.
    """

    X: np.ndarray              # (n, f) float32
    edges: np.ndarray          # (E, 2) int64, i < j
    n: int
    f: int
    labels: np.ndarray | None = None   # (n,) int64, 0 normal / 1 anomaly

    def __post_init__(self) -> None:
        if self.X.shape != (self.n, self.f):
            raise ValueError(
                f"X shape {self.X.shape} does not match (n={self.n}, f={self.f})"
            )
        if self.X.dtype != np.float32:
            raise ValueError(f"X must be float32, got {self.X.dtype}")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be (E, 2), got {self.edges.shape}")
        if self.edges.dtype != np.int64:
            raise ValueError(f"edges must be int64, got {self.edges.dtype}")
        if self.edges.size:
            lo, hi = int(self.edges.min()), int(self.edges.max())
            if lo < 0 or hi >= self.n:
                raise ValueError(f"edge indices out of range [0, {self.n}): [{lo}, {hi}]")
            if (self.edges[:, 0] >= self.edges[:, 1]).any():
                raise ValueError("edges must satisfy i < j (no self-loops, undirected canonical form)")
            if np.unique(self.edges, axis=0).shape[0] != self.edges.shape[0]:
                raise ValueError("duplicate edges are forbidden")
        if self.labels is not None:
            if self.labels.shape != (self.n,):
                raise ValueError(f"labels shape {self.labels.shape} != ({self.n},)")
            if self.labels.dtype != np.int64:
                raise ValueError(f"labels must be int64, got {self.labels.dtype}")


# --- Graph helpers (paper §3.1) -------------------------------------------------

def degree(edges: np.ndarray, n: int) -> np.ndarray:
    """Node degrees as int64 array of shape (n,)."""
    deg = np.zeros(n, dtype=np.int64)
    if edges.size:
        # Each undirected edge (i, j) with i < j contributes 1 to both endpoints.
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    return deg


def _adjacency_lists(edges: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges.tolist():
        adj[i].append(j)
        adj[j].append(i)
    return adj


def k_hop_neighbors(edges: np.ndarray, n: int, k: int = 1) -> list[list[int]]:
    """For each node, return the list of nodes reachable in 1..k hops (excluding self).

    Paper §3.1 uses N_i^k for the k-hop neighbourhood. For k=1 this is the
    standard 1-hop adjacency. For k>1 we BFS truncated at depth k. Self-loops
    are excluded from the result.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    adj = _adjacency_lists(edges, n)
    out: list[list[int]] = []
    for src in range(n):
        seen: set[int] = {src}
        frontier: list[int] = [src]
        for _ in range(k):
            next_frontier: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
            if not frontier:
                break
        seen.discard(src)
        out.append(sorted(seen))
    return out


# --- Synthetic generator (paper §4 protocol) ------------------------------------

@dataclass
class SyntheticAttributedNetwork:
    """Stochastic-block-model graph with planted structural + contextual anomalies.

    Anomaly-injection follows the protocol cited in §4 (Ding et al. 2019 /
    Song et al. 2007):
        - Structural: pick nodes, fully connect them as cliques.
        - Contextual: for each chosen node, swap its feature with the most
          dissimilar node among `contextual_swap_topk` random candidates.
    """

    n_communities: int = DEFAULT_N_COMMUNITIES
    nodes_per_community: int = DEFAULT_NODES_PER_COMMUNITY
    feat_dim: int = DEFAULT_FEATURE_DIM
    p_in: float = 0.10           # SBM intra-community edge probability
    p_out: float = 0.005         # SBM inter-community edge probability
    n_structural_anomalies: int = 15      # number of nodes to clique-inject
    structural_clique_size: int = 6       # size of each injected clique
    n_contextual_anomalies: int = 15      # number of feature-swap anomalies
    contextual_swap_topk: int = 50        # paper §4: candidate set size
    feature_noise_std: float = 0.5        # within-community feature spread
    seed: int = 0

    def generate(self) -> AttributedNetwork:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        n = self.n_communities * self.nodes_per_community
        f = self.feat_dim

        # 1. Community assignment + features --------------------------------
        community = np.repeat(np.arange(self.n_communities), self.nodes_per_community)
        centroids = np_rng.standard_normal(size=(self.n_communities, f))
        noise = np_rng.normal(0.0, self.feature_noise_std, size=(n, f))
        X = (centroids[community] + noise).astype(np.float32)

        # 2. SBM edges ------------------------------------------------------
        edge_set: set[tuple[int, int]] = set()
        # Vectorised draws per (community-pair) block keep this tractable for
        # 300 nodes; for much larger n you'd want a Bernoulli sampler with
        # geometric-skip tricks, but that's out of scope here.
        for i in range(n):
            ci = community[i]
            # Sample j > i in one shot using uniform draws.
            js = np.arange(i + 1, n)
            if js.size == 0:
                continue
            same = community[js] == ci
            probs = np.where(same, self.p_in, self.p_out)
            draws = np_rng.random(size=js.size)
            for j in js[draws < probs].tolist():
                edge_set.add((i, j))

        # 3. Structural anomalies: cliques ----------------------------------
        all_nodes = list(range(n))
        rng.shuffle(all_nodes)
        struct_nodes = all_nodes[: self.n_structural_anomalies]
        remaining_pool = all_nodes[self.n_structural_anomalies:]

        for start in range(0, len(struct_nodes), self.structural_clique_size):
            # Trailing slice may be shorter than clique_size when n_structural_anomalies
            # is not a multiple of structural_clique_size; the partial clique is intentional.
            clique = struct_nodes[start:start + self.structural_clique_size]
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx + 1, len(clique)):
                    a, b = clique[a_idx], clique[b_idx]
                    if a > b:
                        a, b = b, a
                    edge_set.add((a, b))

        # 4. Contextual anomalies: feature swap -----------------------------
        # Must be disjoint from the structural set.
        rng.shuffle(remaining_pool)
        ctx_nodes = remaining_pool[: self.n_contextual_anomalies]

        # Use a separate snapshot so swaps don't influence each other's
        # "farthest" lookup mid-stream.
        X_snapshot = X.copy()
        for v in ctx_nodes:
            candidates = rng.sample(
                [u for u in range(n) if u != v],
                k=min(self.contextual_swap_topk, n - 1),
            )
            cand_arr = np.asarray(candidates, dtype=np.int64)
            diffs = X_snapshot[cand_arr] - X_snapshot[v]
            dists = np.einsum("ij,ij->i", diffs, diffs)   # squared L2 is enough
            farthest = int(cand_arr[int(dists.argmax())])
            X[v] = X_snapshot[farthest]

        # 5. Labels ---------------------------------------------------------
        labels = np.zeros(n, dtype=np.int64)
        labels[np.asarray(struct_nodes, dtype=np.int64)] = 1
        labels[np.asarray(ctx_nodes, dtype=np.int64)] = 1

        # 6. Edges to canonical (E, 2) int64 array --------------------------
        if edge_set:
            edges = np.asarray(sorted(edge_set), dtype=np.int64)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)

        return AttributedNetwork(X=X, edges=edges, n=n, f=f, labels=labels)
