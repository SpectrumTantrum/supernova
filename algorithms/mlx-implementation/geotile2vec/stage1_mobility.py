"""Stage 1: mobility-event Skip-Gram + triplet metric learning + averaging.

Reference: Luo et al., "Geo-Tile2Vec", ACM TSAS 2023, §3.2.
"""

from __future__ import annotations

import random
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data import NUM_POI_CATEGORIES, NUM_TIME_BUCKETS, MobilityEvent, TileId


class MobilityEventModel(nn.Module):
    """Skip-Gram-with-negative-sampling over O mobility events (paper §3.2.1)."""

    def __init__(
        self,
        n_events: int,
        n_categories: int = NUM_POI_CATEGORIES,
        n_time_buckets: int = NUM_TIME_BUCKETS,
        d_event: int = 300,
        d_class: int = 64,
        d_time: int = 36,
    ) -> None:
        super().__init__()
        if n_events <= 0:
            raise ValueError(f"n_events must be positive, got {n_events}")
        self.n_events = n_events
        self.d_event = d_event
        self.d_unified = d_event + d_class + d_time
        self.event_emb = mx.random.uniform(-0.5 / d_event, 0.5 / d_event, (n_events, d_event))
        self.class_emb = mx.random.uniform(-0.5 / d_class, 0.5 / d_class, (n_categories, d_class))
        self.time_emb = mx.random.uniform(-0.5 / d_time, 0.5 / d_time, (n_time_buckets, d_time))
        self.context_emb = mx.zeros((n_events, self.d_unified))

    def unified(self, ev_idx: mx.array, cls_idx: mx.array, time_idx: mx.array) -> mx.array:
        return mx.concatenate(
            [self.event_emb[ev_idx], self.class_emb[cls_idx], self.time_emb[time_idx]],
            axis=-1,
        )

    def __call__(
        self,
        ev_idx: mx.array,
        cls_idx: mx.array,
        time_idx: mx.array,
        pos_idx: mx.array,
        neg_idx: mx.array,
    ) -> mx.array:
        u = self.unified(ev_idx, cls_idx, time_idx)
        v_pos = self.context_emb[pos_idx]
        v_neg = self.context_emb[neg_idx]
        pos_score = mx.sum(u * v_pos, axis=-1)
        neg_score = mx.sum(u[:, None, :] * v_neg, axis=-1)
        loss = -mx.mean(nn.log_sigmoid(pos_score)) - mx.mean(mx.sum(nn.log_sigmoid(-neg_score), axis=-1))
        return loss


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def train_skipgram(
    model: MobilityEventModel,
    pairs: list[tuple[int, int]],
    o_events: list[MobilityEvent],
    *,
    n_negatives: int = 5,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 5e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """Train Skip-Gram with uniform negative sampling."""
    del device
    if not pairs:
        raise ValueError("No co-occurrence pairs were generated. Check time_threshold_min and that D-events share categories.")
    if n_negatives <= 0:
        raise ValueError(f"n_negatives must be positive, got {n_negatives}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    pair_arr = np.asarray(pairs, dtype=np.int32)
    cls = np.asarray([e.poi.category_idx for e in o_events], dtype=np.int32)
    tim = np.asarray([e.hour_bucket for e in o_events], dtype=np.int32)
    rng = np.random.default_rng(0)
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m: MobilityEventModel, a: mx.array, ac: mx.array, at: mx.array, b: mx.array, neg: mx.array) -> mx.array:
        return m(a, ac, at, b, neg)

    grad_fn = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for ep in range(epochs):
        running, n_batches = 0.0, 0
        for batch in _batch_indices(len(pair_arr), batch_size, rng):
            anchors = pair_arr[batch, 0]
            positives = pair_arr[batch, 1]
            negatives = rng.integers(0, model.n_events, size=(len(batch), n_negatives), dtype=np.int32)
            loss, grads = grad_fn(
                model,
                mx.array(anchors),
                mx.array(cls[anchors]),
                mx.array(tim[anchors]),
                mx.array(positives),
                mx.array(negatives),
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            running += float(loss)
            n_batches += 1
        ep_loss = running / max(1, n_batches)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage1/skipgram] epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")
    return losses


def _semi_hard_triplet_loss(anchor: mx.array, positive: mx.array, negatives: mx.array, margin: float) -> mx.array:
    d_ap = mx.sqrt(mx.sum((anchor - positive) ** 2, axis=-1) + 1e-12)
    d_an = mx.sqrt(mx.sum((anchor[:, None, :] - negatives) ** 2, axis=-1) + 1e-12)
    semi_mask = (d_an > d_ap[:, None]) & (d_an < (d_ap[:, None] + margin))
    inf = mx.full(d_an.shape, mx.inf)
    semi_d_an = mx.min(mx.where(semi_mask, d_an, inf), axis=-1)
    fallback_d_an = mx.min(d_an, axis=-1)
    chosen_d_an = mx.where(mx.isfinite(semi_d_an), semi_d_an, fallback_d_an)
    return mx.mean(nn.relu(d_ap - chosen_d_an + margin))


def _build_tile_to_event_indices(o_events: list[MobilityEvent]) -> dict[TileId, list[int]]:
    out: dict[TileId, list[int]] = defaultdict(list)
    for ev in o_events:
        out[ev.tile].append(ev.event_idx)
    return out


def train_triplet_metric(
    model: MobilityEventModel,
    o_events: list[MobilityEvent],
    *,
    margin: float = 1.0,
    epochs: int = 5,
    steps_per_epoch: int = 200,
    batch_size: int = 256,
    n_negatives: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """Triplet loss on the 300-dim event embedding table only."""
    del device
    rng = random.Random(seed)
    tile_to_events = _build_tile_to_event_indices(o_events)
    eligible_tiles = [t for t, e in tile_to_events.items() if len(e) >= 2]
    all_indices = [e for evs in tile_to_events.values() for e in evs]
    if not eligible_tiles:
        raise ValueError("Need at least one tile with >=2 O-events for triplet learning.")
    event_sets = {t: set(evs) for t, evs in tile_to_events.items()}
    negative_candidates = {
        t: [idx for idx in all_indices if idx not in event_sets[t]]
        for t in eligible_tiles
    }
    empty_negative_tiles = [t for t, candidates in negative_candidates.items() if not candidates]
    if empty_negative_tiles:
        raise ValueError(
            "Need at least two tiles with O-events for triplet learning; "
            f"{len(empty_negative_tiles)} eligible tile(s) have no outside-tile negatives."
        )

    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m: MobilityEventModel, a_idx: mx.array, p_idx: mx.array, n_idx: mx.array) -> mx.array:
        return _semi_hard_triplet_loss(m.event_emb[a_idx], m.event_emb[p_idx], m.event_emb[n_idx], margin)

    grad_fn = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for ep in range(epochs):
        running, n = 0.0, 0
        for _ in range(steps_per_epoch):
            anchor_idx, pos_idx, neg_idx = [], [], []
            for _ in range(batch_size):
                t = rng.choice(eligible_tiles)
                a, p = rng.sample(tile_to_events[t], 2)
                negs = rng.choices(negative_candidates[t], k=n_negatives)
                anchor_idx.append(a)
                pos_idx.append(p)
                neg_idx.append(negs)
            loss, grads = grad_fn(
                model,
                mx.array(np.asarray(anchor_idx, dtype=np.int32)),
                mx.array(np.asarray(pos_idx, dtype=np.int32)),
                mx.array(np.asarray(neg_idx, dtype=np.int32)),
            )
            # Keep auxiliary Skip-Gram tables fixed for metric learning.
            for name in ("class_emb", "time_emb", "context_emb"):
                if name in grads:
                    grads[name] = mx.zeros_like(getattr(model, name))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            running += float(loss)
            n += 1
        ep_loss = running / max(1, n)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage1/triplet]  epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")
    return losses


def average_to_tiles(
    model: MobilityEventModel,
    o_events: list[MobilityEvent],
) -> tuple[mx.array, list[TileId]]:
    """Compute frequency-weighted tile embeddings per paper §3.2.3, Eq. (5)."""
    by_tile: dict[TileId, list[MobilityEvent]] = defaultdict(list)
    for ev in o_events:
        by_tile[ev.tile].append(ev)

    tile_order = sorted(by_tile.keys())
    emb = np.asarray(model.event_emb)
    V = np.zeros((len(tile_order), model.d_event), dtype=np.float32)
    for k, tile in enumerate(tile_order):
        evs = by_tile[tile]
        mi = len(evs)
        cat_counts: dict[int, int] = defaultdict(int)
        for ev in evs:
            cat_counts[ev.poi.category_idx] += 1
        acc = np.zeros((model.d_event,), dtype=np.float32)
        for ev in evs:
            f_ij = cat_counts[ev.poi.category_idx] / mi
            acc += f_ij * emb[ev.event_idx]
        V[k] = acc / mi
    return mx.array(V), tile_order
