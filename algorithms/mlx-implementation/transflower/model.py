"""MLX TransFlower orchestrator for paper §3.1 and §3.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data import (
    D_FEATURE,
    Flow,
    Region,
    build_flow_counts,
    build_flow_proportions,
    prepare_region_tensors,
)
from flow_predictor import FlowPredictor, common_part_of_commuters, flow_cross_entropy
from geo_encoder import GeoSpatialEncoder


@dataclass
class TransFlowerConfig:
    # Encoder dims (paper §3.1: d_geo = d_loc = 256, d_model = 512).
    d_feature: int = D_FEATURE
    d_geo: int = 128
    d_loc: int = 128

    # RLE (paper §3.1.2).
    rle_variant: str = "rle"
    n_scales: int = 16
    lambda_min: float = 1.0
    lambda_max: float = 20013.0

    # Flow predictor (paper §3.2).
    n_transformer_layers: int = 2
    n_heads: int = 8
    dim_ff: int = 256
    dropout: float = 0.1

    # Optimisation (paper §4.1.3).
    epochs: int = 30
    batch_origins: int = 32
    lr: float = 1e-4
    momentum: float = 0.9
    patience: int = 20

    seed: int = 1234
    device: str = "cpu"
    verbose: bool = True


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_cpc: list[float] = field(default_factory=list)


class TransFlowerNet(nn.Module):
    """Joint module so MLX optimizers update encoder and predictor together."""

    def __init__(self, encoder: GeoSpatialEncoder, predictor: FlowPredictor) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, x_o: mx.array, x_d: mx.array, r: mx.array, rl: mx.array) -> mx.array:
        return self.predictor(self.encoder(x_o, x_d, r, rl))


class TransFlower:
    """End-to-end MLX TransFlower trainer/predictor."""

    def __init__(self, config: TransFlowerConfig | None = None) -> None:
        self.config = config or TransFlowerConfig()
        self.history = TrainingHistory()
        self._encoder: GeoSpatialEncoder | None = None
        self._predictor: FlowPredictor | None = None
        self._net: TransFlowerNet | None = None
        self._n_regions: int | None = None

    def fit(
        self,
        regions: list[Region],
        train_flows: list[Flow],
        val_flows: Optional[list[Flow]] = None,
    ) -> "TransFlower":
        cfg = self.config
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)

        n_regions = len(regions)
        self._n_regions = n_regions
        feats, dist_m, rl_m = prepare_region_tensors(regions, device=cfg.device)
        region_id_to_idx = self._region_id_to_idx(regions)
        train_F = build_flow_counts(train_flows, n_regions, device=cfg.device, region_id_to_idx=region_id_to_idx)
        train_P = build_flow_proportions(train_F)
        train_origins = np.nonzero(np.array(mx.sum(train_F, axis=-1) > 0))[0].astype(np.int64).tolist()
        if not train_origins:
            raise ValueError("No origins in training set have any outflow.")

        val_F = None
        if val_flows:
            val_F = build_flow_counts(val_flows, n_regions, device=cfg.device, region_id_to_idx=region_id_to_idx)

        self._build_modules()
        assert self._net is not None
        opt = optim.RMSprop(learning_rate=cfg.lr, alpha=cfg.momentum)

        best_val_cpc = -float("inf")
        best_state: dict | None = None
        patience_left = cfg.patience

        def loss_fn(model: TransFlowerNet, x_o: mx.array, x_d: mx.array, r: mx.array, rl: mx.array, target: mx.array) -> mx.array:
            return flow_cross_entropy(model(x_o, x_d, r, rl), target)

        grad_fn = nn.value_and_grad(self._net, loss_fn)
        for ep in range(cfg.epochs):
            self._net.train(True)
            order = list(train_origins)
            np.random.shuffle(order)
            losses: list[float] = []
            for s in range(0, len(order), cfg.batch_origins):
                batch = order[s : s + cfg.batch_origins]
                idx = mx.array(batch, dtype=mx.int32)
                batch_size = len(batch)

                x_o = mx.broadcast_to(mx.expand_dims(feats[idx], 1), (batch_size, n_regions, cfg.d_feature))
                x_d = mx.broadcast_to(mx.expand_dims(feats, 0), (batch_size, n_regions, cfg.d_feature))
                r = dist_m[idx]
                rl = rl_m[idx]
                target = train_P[idx]

                loss, grads = grad_fn(self._net, x_o, x_d, r, rl, target)
                opt.update(self._net, grads)
                mx.eval(self._net.parameters(), opt.state)
                losses.append(float(loss))

            ep_loss = float(np.mean(losses)) if losses else float("nan")
            self.history.train_loss.append(ep_loss)

            if val_F is not None:
                val_cpc = self._eval_cpc(feats, dist_m, rl_m, val_F)
                self.history.val_cpc.append(val_cpc)
                if cfg.verbose:
                    print(f"  [transflower-mlx] epoch {ep + 1}/{cfg.epochs}  loss={ep_loss:.4f}  val_cpc={val_cpc:.4f}")
                if val_cpc > best_val_cpc + 1e-6:
                    best_val_cpc = val_cpc
                    best_state = self._snapshot_params()
                    patience_left = cfg.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if cfg.verbose:
                            print(f"  [transflower-mlx] early stop @ epoch {ep + 1} (best val_cpc={best_val_cpc:.4f})")
                        break
            elif cfg.verbose:
                print(f"  [transflower-mlx] epoch {ep + 1}/{cfg.epochs}  loss={ep_loss:.4f}")

        if best_state is not None:
            self._net.update(best_state)
            mx.eval(self._net.parameters())
        return self

    def predict_distributions(self, regions: list[Region]) -> mx.array:
        """Predict P_{i,j} for every origin/destination pair. Returns (N, N)."""
        cfg = self.config
        self._require_fitted()
        assert self._net is not None
        n_regions = len(regions)
        feats, dist_m, rl_m = prepare_region_tensors(regions, device=cfg.device)
        self._net.train(False)

        rows: list[mx.array] = []
        bs = max(1, cfg.batch_origins)
        for s in range(0, n_regions, bs):
            batch = list(range(s, min(s + bs, n_regions)))
            idx = mx.array(batch, dtype=mx.int32)
            batch_size = len(batch)
            x_o = mx.broadcast_to(mx.expand_dims(feats[idx], 1), (batch_size, n_regions, cfg.d_feature))
            x_d = mx.broadcast_to(mx.expand_dims(feats, 0), (batch_size, n_regions, cfg.d_feature))
            rows.append(self._net(x_o, x_d, dist_m[idx], rl_m[idx]))
        return mx.concatenate(rows, axis=0)

    def cpc(self, regions: list[Region], flows: list[Flow]) -> float:
        """CPC (paper Eq. 4) between predicted and observed flows."""
        n_regions = len(regions)
        F_true = build_flow_counts(
            flows,
            n_regions,
            device=self.config.device,
            region_id_to_idx=self._region_id_to_idx(regions),
        )
        P = self.predict_distributions(regions)
        outflow = mx.expand_dims(mx.sum(F_true, axis=-1), -1)
        return common_part_of_commuters(P * outflow, F_true)

    def _build_modules(self) -> None:
        cfg = self.config
        d_model = cfg.d_geo + cfg.d_loc
        self._encoder = GeoSpatialEncoder(
            d_feature=cfg.d_feature,
            d_geo=cfg.d_geo,
            d_loc=cfg.d_loc,
            rle_variant=cfg.rle_variant,
            n_scales=cfg.n_scales,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
            dropout=cfg.dropout,
        )
        self._predictor = FlowPredictor(
            d_model=d_model,
            n_layers=cfg.n_transformer_layers,
            n_heads=cfg.n_heads,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
        )
        self._net = TransFlowerNet(self._encoder, self._predictor)

    @staticmethod
    def _region_id_to_idx(regions: list[Region]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for idx, region in enumerate(regions):
            if region.region_id in mapping:
                raise ValueError(f"duplicate region_id {region.region_id!r}")
            mapping[region.region_id] = idx
        return mapping

    def _require_fitted(self) -> None:
        if self._net is None or self._encoder is None or self._predictor is None:
            raise RuntimeError("Call fit() first.")

    def _eval_cpc(self, feats: mx.array, dist_m: mx.array, rl_m: mx.array, F_eval: mx.array) -> float:
        cfg = self.config
        self._require_fitted()
        assert self._net is not None
        self._net.train(False)
        n_regions = F_eval.shape[0]
        outflow = mx.expand_dims(mx.sum(F_eval, axis=-1), -1)
        rows: list[mx.array] = []
        bs = max(1, cfg.batch_origins)
        for s in range(0, n_regions, bs):
            batch = list(range(s, min(s + bs, n_regions)))
            idx = mx.array(batch, dtype=mx.int32)
            batch_size = len(batch)
            x_o = mx.broadcast_to(mx.expand_dims(feats[idx], 1), (batch_size, n_regions, cfg.d_feature))
            x_d = mx.broadcast_to(mx.expand_dims(feats, 0), (batch_size, n_regions, cfg.d_feature))
            rows.append(self._net(x_o, x_d, dist_m[idx], rl_m[idx]))
        P = mx.concatenate(rows, axis=0)
        return common_part_of_commuters(P * outflow, F_eval)

    def _snapshot_params(self) -> dict:
        self._require_fitted()
        assert self._net is not None
        return _tree_copy_mx(self._net.parameters())

    def save(self, path: str) -> None:
        self._require_fitted()
        assert self._net is not None
        np.savez(
            path,
            config=np.array([self.config.__dict__], dtype=object),
            history=np.array([self.history.__dict__], dtype=object),
            n_regions=np.array(self._n_regions),
            params=np.array([_tree_to_numpy(self._net.parameters())], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "TransFlower":
        ckpt = np.load(path, allow_pickle=True)
        cfg = dict(ckpt["config"][0])
        cfg["device"] = "cpu"
        obj = cls(TransFlowerConfig(**cfg))
        obj._n_regions = int(ckpt["n_regions"])
        obj._build_modules()
        assert obj._net is not None
        obj._net.update(_tree_to_mx(ckpt["params"][0]))
        mx.eval(obj._net.parameters())
        obj.history = TrainingHistory(**ckpt["history"][0])
        return obj


def _tree_copy_mx(tree):
    if isinstance(tree, dict):
        return {k: _tree_copy_mx(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_copy_mx(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_copy_mx(v) for v in tree)
    return mx.array(np.array(tree))


def _tree_to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_to_numpy(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_to_numpy(v) for v in tree)
    return np.array(tree)


def _tree_to_mx(tree):
    if isinstance(tree, dict):
        return {k: _tree_to_mx(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_to_mx(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_to_mx(v) for v in tree)
    return mx.array(tree)
