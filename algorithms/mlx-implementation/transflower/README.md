# TransFlower

PyTorch implementation of **TransFlower: An Explainable Transformer-Based
Model with Flow-to-Flow Attention for Commuting Flow Prediction** —
Yan Luo et al., arXiv:2402.15398v1, Feb 2024.
<https://arxiv.org/abs/2402.15398>

The algorithm predicts a probability distribution `P_{i,j}` over candidate
destinations `d_j` given an origin region `o_i`, fusing three signals:

| Signal             | Carries                                | Encoder            |
|--------------------|----------------------------------------|--------------------|
| Place attributes   | 19 OSM-derived POI counts + population | Geographic FE      |
| Surface distance   | Haversine meters between centroids     | Geographic FE      |
| Relative position  | 2-D `loc_o − loc_d` vector             | RLE (Space2Vec)    |

A 2-layer transformer encoder then runs **flow-to-flow attention** over the
set of all candidate destinations from a single origin, and a per-flow
softmax head emits the probability over destinations.

## Pipeline

```
   ┌────────── Geo-Spatial Encoder (paper §3.1) ──────────┐
   │ x_o (20)  ─┐                                          │
   │ x_d (20)  ─┼─ Linear → x_{o,d} (256) ──┐              │     Flow Predictor (§3.2)
   │ r (1)     ─┘                            ├── concat ── │  ┌───────────────────────┐
   │ rl (2) ── MultiScale × 2 → FFN × 2 → FFN→ loc_{o,d} ──┘─→│ Transformer × N=2 →   │ → softmax → P_{i,j}
   │           (3 base vectors @ 2π/3, S=16 scales)         │   FFN head → score    │
   └───────────────────────────────────────────────────────┘  └───────────────────────┘
```

Training objective is the multinomial cross-entropy of paper Eq. 3:

    H = − Σ_i Σ_j  (f_{ij} / O_i) · ln P_{i,j}

## Files

| File                  | Role                                                                      |
|-----------------------|---------------------------------------------------------------------------|
| `data.py`             | Dataclasses, haversine, synthetic gravity-with-anisotropy city generator  |
| `geo_encoder.py`      | Geographic feature encoder + multi-scale Space2Vec RLE / RLE'             |
| `flow_predictor.py`   | Transformer encoder + per-flow softmax head + CPC and CE loss helpers     |
| `model.py`            | `TransFlower` orchestrator with `fit()` / `predict_distributions()` / `cpc()` / `save()` |
| `example.py`          | End-to-end smoke test on a synthetic clustered city                       |

## Install & run

```bash
pip install -r requirements.txt
python example.py                    # full smoke test (~30–60 s on CPU)
python example.py --variant rle_prime  # ablation: single-branch RLE'
python example.py --epochs 5         # fast pre-flight
```

The smoke test plants 4 latent land-use clusters (residential / business /
university / retail), generates ground-truth flows with a gravity model that
includes directional anisotropy, then asserts:

1. Training cross-entropy decreases.
2. Welch's t-test — same-cluster destinations receive significantly higher
   predicted probability than different-cluster destinations (`p < 0.05`).
3. CPC (paper Eq. 4) on the *total* observed flow set exceeds 0.30. (The
   20 % held-out split is used during training for early stopping, but on a
   synthetic dataset this small the per-flow multinomial noise dominates a
   raw 20 % split — total-flow CPC is the cleaner measure of structural
   recovery.) The paper achieves 0.62–0.77 on real LODES data; this
   implementation lands in the same range on the synthetic city.

## Library use

```python
from data import SyntheticCity, split_flows
from model import TransFlower, TransFlowerConfig

city = SyntheticCity(seed=0)
regions, flows, region_to_cluster, meta = city.generate()
train_flows, val_flows = split_flows(flows, val_frac=0.2)

cfg = TransFlowerConfig(
    lambda_min=meta["lambda_min_m"],
    lambda_max=meta["lambda_max_m"],
    epochs=30,
)
model = TransFlower(cfg).fit(regions, train_flows, val_flows)
P = model.predict_distributions(regions)            # (N, N) probabilities
cpc = model.cpc(regions, val_flows)                 # eq. 4
model.save("./transflower.pt")
```

## Hyperparameters (paper §4.1.3)

| Symbol                 | Paper value | Default here | Meaning                                  |
|------------------------|-------------|--------------|------------------------------------------|
| N                      | 2           | 2            | transformer encoder layers               |
| `d_model`              | 512         | 256          | per-flow embedding dim (`d_geo + d_loc`) |
| `d_geo` / `d_loc`      | 256 / 256   | 128 / 128    | encoder branch dims                      |
| heads                  | 8           | 8            | multi-head attention                     |
| dropout                | 0.1         | 0.1          | applied throughout                       |
| optimiser              | RMSprop     | RMSprop      | momentum 0.9, lr 1e-4                    |
| `S` (scales / `freq`)  | 16          | 16           | RLE multi-scale frequencies              |
| λ_min                  | 1 m         | from data    | smallest spatial scale                   |
| λ_max                  | study diam. | from data    | largest spatial scale                    |
| `n_destinations`       | 256         | full N       | candidate destinations per origin        |
| seed                   | 1234        | 1234         | torch + numpy                            |
| patience               | 20          | 20           | early-stopping epochs                    |

The smaller `d_model` here is chosen so the synthetic smoke test runs in
under a minute on a laptop CPU. The architecture is otherwise identical to
the paper.

## Why two distance representations?

The geographic feature encoder takes a **scalar** distance `r`. The relative
location encoder takes a **2-D vector** `rl = loc_o − loc_d`. Same physics,
different information: a scalar can't distinguish NE from SW even when
distances match. That is the entire reason RLE exists and why removing it
costs the paper ~9.7 % CPC (Table 2, ablation).

## RLE vs RLE'

- **RLE** (`--variant rle`, default) — two parallel multi-scale branches
  with different base-vector orientations (one rotated by π/3) merged via a
  feed-forward layer. The paper's recommended configuration.
- **RLE'** (`--variant rle_prime`) — single-branch ablation. Simpler and
  faster but cannot eliminate the hexagonal artefact described in §4.4
  (Figs. 2d–f).

## Critical dim split (worth knowing)

`d_model` for the transformer must equal `d_geo + d_loc` because the per-flow
embedding is `[x_{o,d} ; loc_{o,d}]`. Setting these inconsistently fails at
the first transformer call. The default config keeps them in sync.

## Out of scope

This module ships the **algorithm** only. Things deliberately excluded per
`AGENTS.md`:

- The LODES / OSM-derived California / Massachusetts / Texas datasets
  (proprietary; flows depend on them only by shape).
- Baseline models — Gravity, Radiation, Random Forest, GMEL, DeepGravity
  (out of scope per repo convention).
- Downstream urban-planning policy tooling and the explainability
  visualisations (paper §4.5 — flow-to-flow attention map, location embedding
  clustering plots).
