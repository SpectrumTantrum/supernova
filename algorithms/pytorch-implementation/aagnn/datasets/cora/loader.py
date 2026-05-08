"""Real-data adapter: Cora citation network → AAGNN ``AttributedNetwork``.

``load_cora_clean()`` parses the LINQS ``cora.content`` / ``cora.cites`` files
into the (X, edges, n, f) tuple expected by the parent ``data.py``.

``load_cora_with_anomalies(seed)`` then applies the paper §4 anomaly-injection
protocol — structural cliques + contextual feature swap — and returns an
``AttributedNetwork``. The injection logic mirrors ``data.py`` lines 175-209
(``SyntheticAttributedNetwork.generate``); anomaly counts are bumped from the
synthetic 15+15 to 30+30 since Cora is ~9× larger than the synthetic SBM.
"""

from __future__ import annotations

import pathlib
import random
import sys

import numpy as np

# Parent module shim — the AAGNN sources live two directories up
# (aagnn/datasets/cora/loader.py → aagnn/data.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import AttributedNetwork  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
RAW_DIR = HERE / "raw" / "cora"
CONTENT_FILE = RAW_DIR / "cora.content"
CITES_FILE = RAW_DIR / "cora.cites"

CORA_N = 2708
CORA_F = 1433

# Anomaly-injection knobs — paper §4 protocol, scaled to Cora's size.
N_STRUCTURAL_ANOMALIES = 30
STRUCTURAL_CLIQUE_SIZE = 6
N_CONTEXTUAL_ANOMALIES = 30
CONTEXTUAL_SWAP_TOPK = 50


def _require_raw() -> None:
    if not (CONTENT_FILE.exists() and CITES_FILE.exists()):
        raise FileNotFoundError(
            f"Cora raw files not found under {RAW_DIR}. "
            f"Run `python fetch.py` first."
        )


def load_cora_clean() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Parse the raw Cora dump into (X, edges, n, f).

    - ``X`` is float32 (n, 1433); the original 0/1 ints are cast.
    - ``edges`` is int64 (E, 2) with i < j, undirected, deduplicated, no self-loops.
    - Class labels from ``cora.content`` are intentionally discarded — AAGNN's
      ``AttributedNetwork.labels`` is binary anomaly labels, not 7-way classes.
    """
    _require_raw()

    paper_ids: list[int] = []
    feature_rows: list[np.ndarray] = []
    with CONTENT_FILE.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            # Expected: paper_id (1) + 1433 binary features + class label (1).
            if len(parts) != 1 + CORA_F + 1:
                raise ValueError(
                    f"Unexpected column count {len(parts)} in cora.content; "
                    f"expected {1 + CORA_F + 1}."
                )
            paper_ids.append(int(parts[0]))
            feature_rows.append(np.asarray(parts[1:1 + CORA_F], dtype=np.float32))

    n = len(paper_ids)
    if n != CORA_N:
        raise ValueError(f"Expected {CORA_N} papers, got {n}.")

    X = np.stack(feature_rows, axis=0)  # (n, f) float32
    if X.shape != (CORA_N, CORA_F):
        raise ValueError(f"Cora feature matrix shape {X.shape}, expected {(CORA_N, CORA_F)}.")

    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    edge_set: set[tuple[int, int]] = set()
    skipped = 0
    with CITES_FILE.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            a_pid, b_pid = int(parts[0]), int(parts[1])
            a = id_to_idx.get(a_pid)
            b = id_to_idx.get(b_pid)
            if a is None or b is None:
                # Some Cora distributions reference IDs that aren't in .content.
                skipped += 1
                continue
            if a == b:
                continue  # self-loop
            lo, hi = (a, b) if a < b else (b, a)
            edge_set.add((lo, hi))

    if skipped:
        print(f"  [load_cora_clean] skipped {skipped} cite rows referencing unknown paper_ids")

    edges = (
        np.asarray(sorted(edge_set), dtype=np.int64)
        if edge_set else np.zeros((0, 2), dtype=np.int64)
    )
    return X, edges, CORA_N, CORA_F


def inject_anomalies(
    X: np.ndarray,
    edges: np.ndarray,
    n: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the AAGNN paper §4 injection protocol.

    Mirrors ``SyntheticAttributedNetwork.generate`` lines 175-209: structural
    cliques + contextual most-distant feature swap. ``random.Random(seed)``
    drives the choices (matching the parent's RNG mix exactly).

    Returns (X_injected, edges_with_cliques, labels) — X is a fresh copy with
    swapped rows, edges is a new (E', 2) int64 array, labels is (n,) int64.
    """
    rng = random.Random(seed)

    # Re-hydrate the existing edge set so cliques merge cleanly.
    edge_set: set[tuple[int, int]] = {(int(a), int(b)) for a, b in edges.tolist()}

    # 1. Structural anomalies: cliques.
    all_nodes = list(range(n))
    rng.shuffle(all_nodes)
    struct_nodes = all_nodes[:N_STRUCTURAL_ANOMALIES]
    remaining_pool = all_nodes[N_STRUCTURAL_ANOMALIES:]

    for start in range(0, len(struct_nodes), STRUCTURAL_CLIQUE_SIZE):
        clique = struct_nodes[start:start + STRUCTURAL_CLIQUE_SIZE]
        for a_idx in range(len(clique)):
            for b_idx in range(a_idx + 1, len(clique)):
                a, b = clique[a_idx], clique[b_idx]
                if a > b:
                    a, b = b, a
                edge_set.add((a, b))

    # 2. Contextual anomalies: feature swap with the most-distant of TOPK candidates.
    rng.shuffle(remaining_pool)
    ctx_nodes = remaining_pool[:N_CONTEXTUAL_ANOMALIES]

    # X_out is the mutable buffer; X is treated as a read-only snapshot so
    # swaps don't influence each other's "farthest" lookup mid-stream
    # (matches SyntheticAttributedNetwork.generate's X_snapshot semantics).
    X_out = X.copy()
    for v in ctx_nodes:
        candidates = rng.sample(
            [u for u in range(n) if u != v],
            k=min(CONTEXTUAL_SWAP_TOPK, n - 1),
        )
        cand_arr = np.asarray(candidates, dtype=np.int64)
        diffs = X[cand_arr] - X[v]
        dists = np.einsum("ij,ij->i", diffs, diffs)
        farthest = int(cand_arr[int(dists.argmax())])
        X_out[v] = X[farthest]

    # 3. Labels.
    labels = np.zeros(n, dtype=np.int64)
    labels[np.asarray(struct_nodes, dtype=np.int64)] = 1
    labels[np.asarray(ctx_nodes, dtype=np.int64)] = 1

    edges_out = (
        np.asarray(sorted(edge_set), dtype=np.int64)
        if edge_set else np.zeros((0, 2), dtype=np.int64)
    )
    return X_out, edges_out, labels


def load_cora_with_anomalies(seed: int = 0) -> AttributedNetwork:
    """Cora + paper-protocol injection, packaged as an ``AttributedNetwork``."""
    X, edges, n, f = load_cora_clean()
    X_inj, edges_inj, labels = inject_anomalies(X, edges, n, seed=seed)
    return AttributedNetwork(X=X_inj, edges=edges_inj, n=n, f=f, labels=labels)


if __name__ == "__main__":
    net = load_cora_with_anomalies(seed=0)
    print(f"n={net.n}  f={net.f}  edges={len(net.edges)}  anomalies={int(net.labels.sum())}")
