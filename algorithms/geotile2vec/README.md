# Geo-Tile2Vec

PyTorch implementation of **Geo-Tile2Vec: A Multi-Modal and Multi-Stage
Embedding Framework for Urban Analytics** — Yan Luo et al., *ACM Trans.
Spatial Algorithms Syst.* 9(2) Article 10, April 2023.
<https://doi.org/10.1145/3571741>

The algorithm learns a `d=300` dimensional embedding per **geo-tile** (a
fixed-size square of the Earth, here level-18 Mercator tiles ≈ 120 m × 120 m)
by fusing three modalities:

| Modality        | Carries                              | Stage |
|-----------------|--------------------------------------|-------|
| POI data        | static intra-tile activity classes   | 1     |
| Trajectory data | inter-tile human movement            | 1     |
| Street view     | static physical-environment cues     | 2     |

It trains in **two sequential stages** because joint training causes
convergence-rate mismatch between the mobility branch and the street-view
branch — the paper's ablation shows multi-stage beats simultaneous training
by 10–20 % F1 (Fig. 6).

## Pipeline

```
                Stage 1                                        Stage 2
   ┌─────────────────────────────────┐               ┌────────────────────────┐
   │ POI + trajectory                │               │ Street view (4-dir)    │
   │   → Mobility events (50 m snap) │               │   → Places365 ResNet18 │
   │   → Skip-Gram with neg. samples │               │   → PCA 512 → 128      │
   │   → Triplet loss (semi-hard)    │               │   → concat 4 → 512     │
   │   → freq-weighted average       │  v_i (300d) → │   → Triplet loss + W   │
   └─────────────────────────────────┘               │   → V_final (300d)     │
                                                     └────────────────────────┘
```

## Files

| File                    | Role                                                              |
|-------------------------|-------------------------------------------------------------------|
| `data.py`               | Dataclasses, lat/lon → tile id, mobility-event builder, synthetic generator |
| `stage1_mobility.py`    | Skip-Gram model + triplet metric learning + freq-weighted averaging |
| `stage2_streetview.py`  | Places365 ResNet-18 + IncrementalPCA + Stage-2 triplet trainer    |
| `model.py`              | `GeoTile2Vec` orchestrator with `fit()` / `embeddings()` / `save()` |
| `example.py`            | End-to-end smoke test on a synthetic clustered city               |

## Install & run

```bash
pip install -r requirements.txt
python example.py            # full pipeline (auto-downloads Places365, ~45 MB)
python example.py --no-sv    # Stage 1 only — no internet needed
```

The smoke test plants 4 latent land-use clusters
(residential / commercial / scenic / educational), trains both stages, then
runs **Welch's t-test** on cosine similarities of same-cluster vs.
different-cluster tile pairs. It exits 0 only if same-cluster tiles are
significantly closer than different-cluster tiles (`p < 0.05`, gap > 0).

## Library use

```python
from data import POI, Trajectory, StreetViewShot
from model import GeoTile2Vec, GeoTile2VecConfig

cfg = GeoTile2VecConfig(d_event=300, margin1=1.0, margin2=2.0)
model = GeoTile2Vec(cfg).fit(my_pois, my_trajectories, my_shots)
V, tile_order = model.embeddings()       # V.shape == (n_tiles, 300)
model.save("./geotile2vec.pt")
```

`shots` is optional — pass `None` (or omit) to run **Stage 1 only**, useful
when no street-view imagery is available. `POI`, `Trajectory`, and
`StreetViewShot` are simple dataclasses defined in `data.py`.

## Hyperparameters (paper §4.1)

| Symbol | Value | Meaning                                                |
|--------|-------|--------------------------------------------------------|
| `d`    | 300   | tile-embedding dimension                               |
| level  | 18    | geo-tile zoom level (~120 m square)                    |
| `T`    | 15 min| time threshold for Skip-Gram co-occurrence pairs       |
| —      | 50 m  | trajectory-O/D ↔ POI snap distance                     |
| `m`    | 1     | Stage-1 triplet margin                                 |
| `m'`   | 2     | Stage-2 triplet margin                                 |
| —      | 16    | POI categories (Table A.1)                             |
| —      | 24    | hourly time buckets                                    |
| —      | 128   | per-image PCA components                               |

## Critical dim split (worth knowing)

The Skip-Gram input is `[event_emb(300) ; poi_class_emb(64) ; time_emb(36)]`.
Only **`event_emb` (300-dim)** flows downstream into the triplet loss and
the tile-averaging step (Eq. 5); the class/time pieces exist solely to make
the Skip-Gram class-aware and time-aware. Get this split wrong and the
final tile vector will have the wrong dimensionality.

## Out of scope

This module ships the **algorithm** only. Things deliberately excluded:

- The Beijing / Nanjing / Nanchang datasets (proprietary).
- The 17 baseline models in §4.2.
- The downstream XGBoost classifiers (POI category, land use, restaurant
  price, firm count). They take `V` as `n_tiles × 300` features and
  ground-truth labels per tile — wire them up with stock `xgboost.XGBClassifier`.
