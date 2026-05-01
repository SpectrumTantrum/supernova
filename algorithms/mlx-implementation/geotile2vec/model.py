"""GeoTile2Vec orchestrator: ties Stage 1 and Stage 2 together.

Reference: Luo et al., "GeoTile2Vec", ACM TSAS 2023.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

from data import (
    DEFAULT_POI_SNAP_METERS,
    DEFAULT_TILE_LEVEL,
    DEFAULT_TIME_THRESHOLD_MIN,
    POI,
    StreetViewShot,
    TileId,
    Trajectory,
    build_mobility_events,
    build_skipgram_pairs,
)
from stage1_mobility import MobilityEventModel, average_to_tiles, train_skipgram, train_triplet_metric
from stage2_streetview import (
    Places365PretrainedResNet18,
    concat_per_shot,
    extract_image_features,
    fit_and_apply_pca,
    train_stage2_triplet,
)


@dataclass
class GeoTile2VecConfig:
    tile_level: int = DEFAULT_TILE_LEVEL
    poi_snap_meters: float = DEFAULT_POI_SNAP_METERS
    time_threshold_min: int = DEFAULT_TIME_THRESHOLD_MIN
    d_event: int = 300
    d_class: int = 64
    d_time: int = 36
    skipgram_epochs: int = 5
    skipgram_batch_size: int = 512
    skipgram_negatives: int = 5
    skipgram_lr: float = 5e-3
    triplet1_epochs: int = 5
    triplet1_steps: int = 200
    triplet1_batch_size: int = 256
    triplet1_negatives: int = 16
    triplet1_lr: float = 1e-3
    margin1: float = 1.0
    pca_components: int = 128
    triplet2_epochs: int = 5
    triplet2_steps: int = 200
    triplet2_batch_size: int = 128
    triplet2_negatives: int = 16
    triplet2_lr: float = 1e-3
    margin2: float = 2.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True
    places365_weights_path: Optional[str] = None
    use_synthetic_stage2_features: bool = True


@dataclass
class TrainingHistory:
    skipgram_losses: list[float] = field(default_factory=list)
    triplet1_losses: list[float] = field(default_factory=list)
    triplet2_losses: list[float] = field(default_factory=list)


class GeoTile2Vec:
    """End-to-end GeoTile2Vec trainer."""

    def __init__(self, config: GeoTile2VecConfig | None = None) -> None:
        self.config = config or GeoTile2VecConfig()
        self.history = TrainingHistory()
        self._mobility_model: MobilityEventModel | None = None
        self._V: mx.array | None = None
        self._tile_order: list[TileId] | None = None

    def fit(
        self,
        pois: list[POI],
        trajectories: list[Trajectory],
        shots: list[StreetViewShot] | None = None,
        precomputed_image_features: np.ndarray | None = None,
    ) -> "GeoTile2Vec":
        cfg = self.config
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)

        if cfg.verbose:
            print("[GeoTile2Vec] Stage 1: building mobility events...")
        o_events, d_events = build_mobility_events(
            trajectories,
            pois,
            snap_meters=cfg.poi_snap_meters,
            tile_level=cfg.tile_level,
        )
        if not o_events:
            raise ValueError("No mobility events were built — check POI proximity to trajectories.")
        if cfg.verbose:
            print(f"  built {len(o_events)} O / {len(d_events)} D events")

        pairs = build_skipgram_pairs(o_events, d_events, time_threshold_min=cfg.time_threshold_min)
        if cfg.verbose:
            print(f"  generated {len(pairs)} Skip-Gram co-occurrence pairs")

        mobility_model = MobilityEventModel(
            n_events=len(o_events),
            d_event=cfg.d_event,
            d_class=cfg.d_class,
            d_time=cfg.d_time,
        )
        self.history.skipgram_losses = train_skipgram(
            mobility_model,
            pairs,
            o_events,
            n_negatives=cfg.skipgram_negatives,
            epochs=cfg.skipgram_epochs,
            batch_size=cfg.skipgram_batch_size,
            lr=cfg.skipgram_lr,
            device=cfg.device,
            verbose=cfg.verbose,
        )
        self.history.triplet1_losses = train_triplet_metric(
            mobility_model,
            o_events,
            margin=cfg.margin1,
            epochs=cfg.triplet1_epochs,
            steps_per_epoch=cfg.triplet1_steps,
            batch_size=cfg.triplet1_batch_size,
            n_negatives=cfg.triplet1_negatives,
            lr=cfg.triplet1_lr,
            seed=cfg.seed,
            device=cfg.device,
            verbose=cfg.verbose,
        )

        V, tile_order = average_to_tiles(mobility_model, o_events)
        self._mobility_model = mobility_model
        self._V = V
        self._tile_order = tile_order
        if cfg.verbose:
            print(f"  preliminary tile embeddings V_stage1: {tuple(V.shape)}")

        if shots:
            if cfg.verbose:
                print("[GeoTile2Vec] Stage 2: preparing street-view features...")
            extractor_or_features: Places365PretrainedResNet18 | np.ndarray
            if precomputed_image_features is not None:
                extractor_or_features = precomputed_image_features
            else:
                extractor_or_features = Places365PretrainedResNet18(
                    weights_path=cfg.places365_weights_path,
                    device=cfg.device,
                    allow_synthetic=cfg.use_synthetic_stage2_features,
                )
            per_image = extract_image_features(extractor_or_features, shots)
            per_image_pca = fit_and_apply_pca(per_image, n_components=cfg.pca_components)
            shot_feats = concat_per_shot(per_image_pca, len(shots))
            shot_tiles = [s.tile for s in shots]
            if cfg.verbose:
                print(f"  shot embeddings: {shot_feats.shape}")
            V_final, losses = train_stage2_triplet(
                V,
                tile_order,
                shot_feats,
                shot_tiles,
                margin=cfg.margin2,
                epochs=cfg.triplet2_epochs,
                steps_per_epoch=cfg.triplet2_steps,
                batch_size=cfg.triplet2_batch_size,
                n_negatives=cfg.triplet2_negatives,
                lr=cfg.triplet2_lr,
                seed=cfg.seed,
                device=cfg.device,
                verbose=cfg.verbose,
            )
            self._V = V_final
            self.history.triplet2_losses = losses
        elif cfg.verbose:
            print("[GeoTile2Vec] Stage 2 skipped (no street-view shots provided).")
        return self

    def embeddings(self) -> tuple[mx.array, list[TileId]]:
        if self._V is None or self._tile_order is None:
            raise RuntimeError("Call fit() first.")
        return self._V, self._tile_order

    def embedding_for(self, tile: TileId) -> mx.array:
        V, order = self.embeddings()
        return V[order.index(tile)]

    def save(self, path: str) -> None:
        if self._V is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez(
            path,
            V=np.asarray(self._V),
            tile_order=np.asarray(self._tile_order, dtype=object),
            config=np.asarray(self.config.__dict__, dtype=object),
            history=np.asarray(self.history.__dict__, dtype=object),
            allow_pickle=True,
        )

    @classmethod
    def load(cls, path: str) -> "GeoTile2Vec":
        ckpt = np.load(path, allow_pickle=True)
        obj = cls(GeoTile2VecConfig(**ckpt["config"].item()))
        obj._V = mx.array(ckpt["V"].astype(np.float32))
        obj._tile_order = list(ckpt["tile_order"])
        obj.history = TrainingHistory(**ckpt["history"].item())
        return obj
