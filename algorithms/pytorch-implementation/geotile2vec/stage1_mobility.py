"""Stage 1: mobility-event Skip-Gram + triplet metric learning + averaging.

Reference: Luo et al., "Geo-Tile2Vec", ACM TSAS 2023, §3.2.

Pipeline:
    1. MobilityEventModel  — Skip-Gram with negative sampling over the unified
       embedding [event_emb(d=300); poi_class_emb; time_emb].
    2. train_triplet_metric — fine-tunes ONLY the 300-dim event_emb so that
       events from the same geo-tile cluster together (semi-hard mining).
    3. average_to_tiles — frequency-weighted average per Eq. (5), producing
       the preliminary tile embedding matrix V_stage1.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import (
    NUM_POI_CATEGORIES,
    NUM_TIME_BUCKETS,
    MobilityEvent,
    TileId,
)


# --- Skip-Gram (Mobility Event Model) -------------------------------------------

class MobilityEventModel(nn.Module):
    """Skip-Gram-with-Negative-Sampling over O mobility events (paper §3.2.1).

    The "vocabulary" is the set of O mobility events. For each event the
    *unified input embedding* is the concatenation
        [event_emb(d_event) ; poi_class_emb(d_class) ; time_emb(d_time)],
    only `event_emb` flows downstream into stage-1 metric learning and the
    tile-averaging step.
    """

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
        self.d_event = d_event
        self.d_unified = d_event + d_class + d_time

        self.event_emb = nn.Embedding(n_events, d_event)
        self.class_emb = nn.Embedding(n_categories, d_class)
        self.time_emb = nn.Embedding(n_time_buckets, d_time)

        # Output ("context") embedding table — separate from inputs, classical
        # word2vec convention. One vector per event, sized to match the unified
        # input dim so we can score with a plain dot product.
        self.context_emb = nn.Embedding(n_events, self.d_unified)

        # Initialise so dot-product scores start near 0.
        nn.init.uniform_(self.event_emb.weight, -0.5 / d_event, 0.5 / d_event)
        nn.init.uniform_(self.class_emb.weight, -0.5 / d_class, 0.5 / d_class)
        nn.init.uniform_(self.time_emb.weight, -0.5 / d_time, 0.5 / d_time)
        nn.init.zeros_(self.context_emb.weight)

    def unified(self, ev_idx, cls_idx, time_idx):
        return torch.cat(
            [self.event_emb(ev_idx), self.class_emb(cls_idx), self.time_emb(time_idx)],
            dim=-1,
        )

    def forward(self, anchor, pos_idx, neg_idx):
        """SGNS loss.

        anchor: tuple(ev_idx, cls_idx, time_idx), each shape (B,)
        pos_idx: (B,) context event index
        neg_idx: (B, K) negative event indices
        """
        u = self.unified(*anchor)                      # (B, d_unified)
        v_pos = self.context_emb(pos_idx)              # (B, d_unified)
        v_neg = self.context_emb(neg_idx)              # (B, K, d_unified)

        pos_score = (u * v_pos).sum(dim=-1)            # (B,)
        neg_score = torch.einsum("bd,bkd->bk", u, v_neg)

        loss = -F.logsigmoid(pos_score).mean() \
               - F.logsigmoid(-neg_score).sum(dim=-1).mean()
        return loss


# --- Skip-Gram dataset / training -----------------------------------------------

class _SkipGramDataset(Dataset):
    def __init__(self, pairs: list[tuple[int, int]], events: list[MobilityEvent]):
        self.pairs = pairs
        self.cls = np.array([e.poi.category_idx for e in events], dtype=np.int64)
        self.tim = np.array([e.hour_bucket for e in events], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        return a, int(self.cls[a]), int(self.tim[a]), b


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
    """Train the Skip-Gram with uniform negative sampling.

    Returns one mean loss per epoch.
    """
    if not pairs:
        raise ValueError("No co-occurrence pairs were generated. Check time_threshold_min and that D-events share categories.")
    model.to(device).train()
    n_events = model.event_emb.num_embeddings
    ds = _SkipGramDataset(pairs, o_events)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for ep in range(epochs):
        running, n_batches = 0.0, 0
        for a, ac, at, b in loader:
            a = a.to(device); ac = ac.to(device); at = at.to(device); b = b.to(device)
            neg = torch.randint(0, n_events, (a.shape[0], n_negatives), device=device)
            opt.zero_grad()
            loss = model((a, ac, at), b, neg)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        ep_loss = running / max(1, n_batches)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage1/skipgram] epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")
    return losses


# --- Triplet metric learning on event embeddings (paper §3.2.2) ------------------

def _semi_hard_triplet_loss(anchor, positive, negatives, margin: float):
    """FaceNet-style semi-hard triplet loss.

    anchor, positive: (B, D)
    negatives:        (B, K, D)
    """
    d_ap = (anchor - positive).norm(dim=-1)                     # (B,)
    d_an = (anchor.unsqueeze(1) - negatives).norm(dim=-1)       # (B, K)

    # Semi-hard: d_ap < d_an < d_ap + margin
    semi_mask = (d_an > d_ap.unsqueeze(1)) & (d_an < d_ap.unsqueeze(1) + margin)
    inf = torch.full_like(d_an, float("inf"))
    semi_d_an, _ = torch.where(semi_mask, d_an, inf).min(dim=-1)
    has_semi = torch.isfinite(semi_d_an)

    # Fallback: if no semi-hard exists for this anchor, use the hardest
    # (smallest) negative — keeps the gradient flowing.
    fallback_d_an, _ = d_an.min(dim=-1)
    chosen_d_an = torch.where(has_semi, semi_d_an, fallback_d_an)

    return F.relu(d_ap - chosen_d_an + margin).mean()


@dataclass
class _TripletBatch:
    anchor: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor


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
    """Triplet loss `L_e-e` on the 300-dim `event_emb` only.

    Other parameters of `model` are frozen for this stage so the auxiliary
    class/time embeddings don't move.
    """
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

    # Freeze everything except event_emb.
    for p in model.parameters():
        p.requires_grad_(False)
    model.event_emb.weight.requires_grad_(True)
    opt = torch.optim.Adam([model.event_emb.weight], lr=lr)
    model.to(device).train()

    losses: list[float] = []
    for ep in range(epochs):
        running, n = 0.0, 0
        for _ in range(steps_per_epoch):
            anchor_idx, pos_idx, neg_idx = [], [], []
            for _ in range(batch_size):
                t = rng.choice(eligible_tiles)
                a, p = rng.sample(tile_to_events[t], 2)
                # Negatives: sample from any other tile.
                negs = rng.choices(negative_candidates[t], k=n_negatives)
                anchor_idx.append(a); pos_idx.append(p); neg_idx.append(negs)

            anchor_idx = torch.tensor(anchor_idx, device=device)
            pos_idx = torch.tensor(pos_idx, device=device)
            neg_idx = torch.tensor(neg_idx, device=device)

            opt.zero_grad()
            a_emb = model.event_emb(anchor_idx)
            p_emb = model.event_emb(pos_idx)
            n_emb = model.event_emb(neg_idx)
            loss = _semi_hard_triplet_loss(a_emb, p_emb, n_emb, margin)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        ep_loss = running / max(1, n)
        losses.append(ep_loss)
        if verbose:
            print(f"  [stage1/triplet]  epoch {ep + 1}/{epochs}  loss={ep_loss:.4f}")

    # Re-enable grads so downstream code can keep training if it wants to.
    for p in model.parameters():
        p.requires_grad_(True)
    return losses


# --- Frequency-weighted averaging (paper §3.2.3, Eq. 5) -------------------------

def average_to_tiles(
    model: MobilityEventModel,
    o_events: list[MobilityEvent],
) -> tuple[torch.Tensor, list[TileId]]:
    """Compute v_i = (1/M_i) * sum_j(f_ij · e_emb_j) per tile.

    `f_ij` = (count of POI-category-of-event-j in tile i) / M_i, so events of
    the dominant category in a tile get the most weight.

    Returns (V, tile_order) where V[k] is the embedding of tile_order[k].
    """
    by_tile: dict[TileId, list[MobilityEvent]] = defaultdict(list)
    for ev in o_events:
        by_tile[ev.tile].append(ev)

    tile_order = sorted(by_tile.keys())
    d = model.d_event
    device = model.event_emb.weight.device
    V = torch.zeros(len(tile_order), d, device=device)

    with torch.no_grad():
        emb = model.event_emb.weight                  # (n_events, d)
        for k, tile in enumerate(tile_order):
            evs = by_tile[tile]
            Mi = len(evs)
            cat_counts: dict[int, int] = defaultdict(int)
            for ev in evs:
                cat_counts[ev.poi.category_idx] += 1
            acc = torch.zeros(d, device=device)
            for ev in evs:
                f_ij = cat_counts[ev.poi.category_idx] / Mi
                acc = acc + f_ij * emb[ev.event_idx]
            V[k] = acc / Mi
    return V, tile_order
