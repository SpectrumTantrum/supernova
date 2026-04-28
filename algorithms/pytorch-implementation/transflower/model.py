"""TransFlower orchestrator: ties §3.1 (Geo-Spatial Encoder) and §3.2 (Flow Predictor) together.

Reference: Luo et al., "TransFlower" arXiv:2402.15398v1 (2024).

Typical usage:
    cfg = TransFlowerConfig(lambda_max=meta["lambda_max_m"])
    model = TransFlower(cfg).fit(regions, train_flows, val_flows)
    P = model.predict_distributions(regions)            # (N, N)
    cpc = model.cpc(regions, eval_flows)                # eq. 4
    model.save("./transflower.pt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

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
    d_geo: int = 128                  # halved for the synthetic smoke test
    d_loc: int = 128

    # RLE (paper §3.1.2)
    rle_variant: str = "rle"          # 'rle' | 'rle_prime'
    n_scales: int = 16
    lambda_min: float = 1.0
    lambda_max: float = 20013.0       # default = California study-area diameter

    # Flow predictor (paper §3.2)
    n_transformer_layers: int = 2
    n_heads: int = 8
    dim_ff: int = 256
    dropout: float = 0.1

    # Optimisation (paper §4.1.3)
    epochs: int = 30
    batch_origins: int = 32
    lr: float = 1e-4
    momentum: float = 0.9             # RMSprop
    patience: int = 20

    seed: int = 1234
    device: str = "cpu"
    verbose: bool = True


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_cpc: list[float] = field(default_factory=list)


class TransFlower:
    """End-to-end TransFlower trainer/predictor."""

    def __init__(self, config: TransFlowerConfig | None = None) -> None:
        self.config = config or TransFlowerConfig()
        self.history = TrainingHistory()
        self._encoder: GeoSpatialEncoder | None = None
        self._predictor: FlowPredictor | None = None
        self._n_regions: int | None = None

    # -------------------------------------------------------------------- training

    def fit(
        self,
        regions: list[Region],
        train_flows: list[Flow],
        val_flows: Optional[list[Flow]] = None,
    ) -> "TransFlower":
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        N = len(regions)
        self._n_regions = N

        feats, dist_m, rl_m = prepare_region_tensors(regions, device=cfg.device)
        train_F = build_flow_counts(train_flows, N, device=cfg.device)
        train_P = build_flow_proportions(train_F)
        train_origins = torch.nonzero(train_F.sum(dim=-1) > 0, as_tuple=False).flatten().tolist()
        if not train_origins:
            raise ValueError("No origins in training set have any outflow.")

        val_F = None
        if val_flows:
            val_F = build_flow_counts(val_flows, N, device=cfg.device)

        self._build_modules()
        opt = torch.optim.RMSprop(
            list(self._encoder.parameters()) + list(self._predictor.parameters()),
            lr=cfg.lr, momentum=cfg.momentum,
        )

        best_val_cpc = -float("inf")
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        patience_left = cfg.patience

        for ep in range(cfg.epochs):
            self._encoder.train(True)
            self._predictor.train(True)
            order = list(train_origins)
            np.random.shuffle(order)
            losses: list[float] = []
            for s in range(0, len(order), cfg.batch_origins):
                batch = order[s : s + cfg.batch_origins]
                idx = torch.tensor(batch, dtype=torch.long, device=cfg.device)
                B = idx.shape[0]

                # x_o: (B, N, D)  (origin features broadcast across destinations)
                x_o = feats[idx].unsqueeze(1).expand(B, N, -1)
                # x_d: (B, N, D)  (destination features = full feature table)
                x_d = feats.unsqueeze(0).expand(B, N, -1)
                r = dist_m[idx]                # (B, N)
                rl = rl_m[idx]                 # (B, N, 2)

                opt.zero_grad()
                e = self._encoder(x_o, x_d, r, rl)
                pred = self._predictor(e)      # (B, N) — softmax over N destinations
                loss = flow_cross_entropy(pred, train_P[idx])
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            ep_loss = float(np.mean(losses)) if losses else float("nan")
            self.history.train_loss.append(ep_loss)

            if val_F is not None:
                val_cpc = self._eval_cpc(feats, dist_m, rl_m, val_F)
                self.history.val_cpc.append(val_cpc)
                if cfg.verbose:
                    print(f"  [transflower] epoch {ep + 1}/{cfg.epochs}  loss={ep_loss:.4f}  val_cpc={val_cpc:.4f}")
                if val_cpc > best_val_cpc + 1e-6:
                    best_val_cpc = val_cpc
                    best_state = {
                        "enc": {k: v.detach().clone() for k, v in self._encoder.state_dict().items()},
                        "pred": {k: v.detach().clone() for k, v in self._predictor.state_dict().items()},
                    }
                    patience_left = cfg.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if cfg.verbose:
                            print(f"  [transflower] early stop @ epoch {ep + 1} "
                                  f"(best val_cpc={best_val_cpc:.4f})")
                        break
            elif cfg.verbose:
                print(f"  [transflower] epoch {ep + 1}/{cfg.epochs}  loss={ep_loss:.4f}")

        if best_state is not None:
            self._encoder.load_state_dict(best_state["enc"])
            self._predictor.load_state_dict(best_state["pred"])

        return self

    # -------------------------------------------------------------------- inference

    @torch.no_grad()
    def predict_distributions(self, regions: list[Region]) -> torch.Tensor:
        """Predict P_{i,j} for every (origin, destination) pair. Returns (N, N)."""
        cfg = self.config
        self._require_fitted()
        N = len(regions)
        feats, dist_m, rl_m = prepare_region_tensors(regions, device=cfg.device)
        self._encoder.train(False)
        self._predictor.train(False)

        rows: list[torch.Tensor] = []
        bs = max(1, cfg.batch_origins)
        all_idx = torch.arange(N, device=cfg.device)
        for s in range(0, N, bs):
            idx = all_idx[s : s + bs]
            B = idx.shape[0]
            x_o = feats[idx].unsqueeze(1).expand(B, N, -1)
            x_d = feats.unsqueeze(0).expand(B, N, -1)
            r = dist_m[idx]
            rl = rl_m[idx]
            e = self._encoder(x_o, x_d, r, rl)
            rows.append(self._predictor(e))
        return torch.cat(rows, dim=0)              # (N, N)

    @torch.no_grad()
    def cpc(self, regions: list[Region], flows: list[Flow]) -> float:
        """CPC (paper Eq. 4) between predicted and observed flows.

        Predicted counts = P_{i,j} · O_i where O_i = sum_j f_{ij}.
        """
        N = len(regions)
        F_true = build_flow_counts(flows, N, device=self.config.device)
        P = self.predict_distributions(regions)
        outflow = F_true.sum(dim=-1, keepdim=True)
        F_pred = P * outflow
        return common_part_of_commuters(F_pred, F_true)

    # -------------------------------------------------------------------- internals

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
        ).to(cfg.device)
        self._predictor = FlowPredictor(
            d_model=d_model,
            n_layers=cfg.n_transformer_layers,
            n_heads=cfg.n_heads,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
        ).to(cfg.device)

    def _require_fitted(self) -> None:
        if self._encoder is None or self._predictor is None:
            raise RuntimeError("Call fit() first.")

    @torch.no_grad()
    def _eval_cpc(
        self,
        feats: torch.Tensor,
        dist_m: torch.Tensor,
        rl_m: torch.Tensor,
        F_eval: torch.Tensor,
    ) -> float:
        cfg = self.config
        self._encoder.train(False)
        self._predictor.train(False)
        N = F_eval.shape[0]
        outflow = F_eval.sum(dim=-1, keepdim=True)
        rows: list[torch.Tensor] = []
        all_idx = torch.arange(N, device=cfg.device)
        bs = max(1, cfg.batch_origins)
        for s in range(0, N, bs):
            idx = all_idx[s : s + bs]
            B = idx.shape[0]
            x_o = feats[idx].unsqueeze(1).expand(B, N, -1)
            x_d = feats.unsqueeze(0).expand(B, N, -1)
            r = dist_m[idx]
            rl = rl_m[idx]
            e = self._encoder(x_o, x_d, r, rl)
            rows.append(self._predictor(e))
        P = torch.cat(rows, dim=0)
        F_pred = P * outflow
        return common_part_of_commuters(F_pred, F_eval)

    # -------------------------------------------------------------------- I/O

    def save(self, path: str) -> None:
        self._require_fitted()
        torch.save(
            {
                "config": self.config.__dict__,
                "encoder": self._encoder.state_dict(),
                "predictor": self._predictor.state_dict(),
                "history": self.history.__dict__,
                "n_regions": self._n_regions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TransFlower":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(TransFlowerConfig(**ckpt["config"]))
        obj._n_regions = ckpt["n_regions"]
        obj._build_modules()
        obj._encoder.load_state_dict(ckpt["encoder"])
        obj._predictor.load_state_dict(ckpt["predictor"])
        obj.history = TrainingHistory(**ckpt["history"])
        return obj
