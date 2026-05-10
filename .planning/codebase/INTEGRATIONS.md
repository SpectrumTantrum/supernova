# External Integrations

**Analysis Date:** 2026-05-10

## APIs & External Services

The repo makes **no live API calls at runtime** — there is no SDK for Stripe / AWS / OpenAI / Supabase / etc. The only outbound HTTP traffic is one-time anonymous dataset / model-weight downloads via `requests` (dataset adapters) or `torch.hub` (Places365 backbone). All transfers are GET-only over HTTPS (LODES uses HTTPS as well; Places365 mirror uses HTTP).

**Outbound HTTP clients in use:**
- `requests>=2.31` — declared in every dataset adapter `requirements.txt`. Used in `algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py`, `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/fetch.py`, and `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/fetch.py`. Each fetcher does a HEAD pre-flight before streaming the GET, with chunked writes to a `*.part` file then atomic rename.
- `torch.hub.load_state_dict_from_url` — invoked twice in the repo, both for the same Places365 ResNet-18 checkpoint:
  - `algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py:82`
  - `algorithms/mlx-implementation/geotile2vec/places365_backbone.py:219` (lazy-imported torch, build-time only)

## Pretrained Model Backbones

### Places365 ResNet-18 (Geo-Tile2Vec only)

**Upstream:** CSAILVision / `places365` reference checkpoint.
- URL: `http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar`
- Size: ~45 MB (the `Stage 1 / Stage 2` README at `algorithms/pytorch-implementation/geotile2vec/README.md:51` quotes this).
- License / origin: CSAIL Places2 model release (research use; see `http://places2.csail.mit.edu`).

**PyTorch path** (`algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py`):
- Class: `Places365PretrainedResNet18` (line 49). Wraps `torchvision.models.resnet18(num_classes=365)` and replaces `backbone.fc` with `nn.Identity()` to expose the 512-dim global-average-pool features (line 63).
- Weight loading: `torch.hub.load_state_dict_from_url(PLACES365_RESNET18_URL, map_location="cpu", weights_only=False)` (line 82). Strips the DataParallel `module.` prefix before `load_state_dict(strict=True)` (line 97).
- Default cache path on download failure: `~/.cache/torch/hub/checkpoints/resnet18_places365.pth.tar`. The `RuntimeError` raised on failure (line 90) names this exact path so users can side-load.
- Preprocessing: `transforms.Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(IMAGENET_MEAN, IMAGENET_STD)` — ImageNet means/stds because Places365 was trained with the same preprocessing.

**MLX path** (`algorithms/mlx-implementation/geotile2vec/places365_backbone.py`):
- Re-implements ResNet-18 natively in MLX (`MLXResNet18`, line 86) so runtime stays torch-free.
- Lazy `import torch` inside `_download_and_convert` (line 208) only when the MLX cache is missing. Downloads via the same `torch.hub.load_state_dict_from_url` call (line 219), then translates each PyTorch state-dict key via `_map_pt_key` (line 155) and permutes Conv2d weights from PyTorch NCHW `(out, in, kH, kW)` → MLX NHWC `(out, kH, kW, in)` (line 200).
- Cache path: `~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz` (line 45, `_default_cache_path`). Stored as a `np.savez` archive of flat `mlx-key → float32 ndarray` pairs.
- Offline-fallback: if neither the cache nor torch is available, `RuntimeError` at line 210 spells out the URL **and** the cache path so a user on a torch-equipped machine can pre-build the `.npz` and copy it across.

## Dataset Sources

There are **three committed real-data adapters** plus one in-progress directory. Each adapter sits at `algorithms/<framework>/<algo>/datasets/<source>/` with `fetch.py` (downloader, idempotent), `loader.py` (parser → algorithm dataclass), `run_example.py` (smoke test on real data), `requirements.txt`, and `raw/` (gitignored cache). The `--small` CLI flag is honoured across all fetchers for cross-adapter parity.

Datasets the prompt mentioned that are **not** integrated in this repo: Citi Bike, OSMnx-style OpenStreetMap downloads, NYC LODES, NYC street-view imagery. The Geo-Tile2Vec README references Beijing / Nanjing / Nanchang as proprietary; TransFlower's README cites OSM-derived POI counts as paper context but the actual TransFlower adapter consumes **NAICS supersector job counts** from LEHD WAC, not OSM. No Citi Bike loader exists.

### Cora — AAGNN

`algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py`

- **What:** 2,708 ML-paper citation network with 5,429 directed citations and a 1,433-dim binary bag-of-words feature per node.
- **License/source:** UC Santa Cruz LINQS group, research-use. Canonical reference: Sen et al., "Collective Classification in Network Data", *AI Magazine* 29(3), 2008.
- **Mirrors (tried in order, first 200 wins):**
  1. `https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz` (primary)
  2. `https://github.com/pyg-team/pyg-datasets/raw/master/cora.tgz` (community)
  3. `https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.x` (Planetoid pickle — last resort, different layout, loader will not consume it)
- **Download flow:** HEAD pre-flight (treats 200 and 405 as OK) → stream GET to `raw/cora.tgz.part` → atomic rename → `tarfile.extractall(filter="data")` into `raw/cora/`. User-Agent is set to `"Mozilla/5.0 (supernova/aagnn cora fetcher)"` to dodge LINQS's bare-`requests` filter.
- **Adapter:** `loader.py` parses LINQS `cora.content` + `cora.cites` into a `(X: float32, edges: int64, n=2708, f=1433)` tuple and discards the original 7-class taxonomy (AAGNN's `AttributedNetwork.labels` is *binary anomaly*, not multi-class).
- **Anomaly-injection protocol** (`algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:113`, `inject_anomalies`):
  - Mirrors the AAGNN paper §4 protocol implemented in the synthetic generator (`algorithms/pytorch-implementation/aagnn/data.py:175-209`).
  - `N_STRUCTURAL_ANOMALIES = 30`, `STRUCTURAL_CLIQUE_SIZE = 6`, `N_CONTEXTUAL_ANOMALIES = 30`, `CONTEXTUAL_SWAP_TOPK = 50`.
  - Counts are **doubled vs the synthetic 15+15** to scale with Cora's ~9× larger node count.
  - Step 1 — structural: shuffle nodes, take 30, partition into 5 cliques of 6, add all clique edges to the existing graph.
  - Step 2 — contextual: from the remaining nodes, take 30; for each, sample `min(50, n-1)` candidate rows and replace the target row's features with the **most-distant** candidate (squared-L2 over the original `X` snapshot, so swaps don't influence each other mid-stream).
  - Step 3 — labels: union of structural + contextual nodes flagged 1, all others 0.
  - RNG: `random.Random(seed)` with `seed=0` default — matches the parent generator's RNG mix.
- **Smoke-test contract** (`run_example.py`): Welch's t-test `p<0.05` with positive anomaly-vs-normal score gap is the gating condition; ROC-AUC ≥ 0.65 is reported as a soft floor (the 0.75 synthetic-SBM floor is too tight for Cora's high-dim sparse features).

### DBLPv7 → ACMv9 — ACDNE

`algorithms/pytorch-implementation/acdne/datasets/dblp_acm/fetch.py`

- **What:** Two paper-citation networks sharing a 6,775-token bag-of-words vocabulary and a 5-class research-area taxonomy (DB / DM / AI / CV / IR). Used as a source/target pair for cross-network node classification.
- **License/source:** Distributed alongside Shen et al., "Adversarial Deep Network Embedding for Cross-network Node Classification", *AAAI 2020*. Upstream repo has no `LICENSE`; the adapter docstring (`fetch.py:13-16`) marks it research-only.
- **URLs (raw GitHub):**
  - `https://raw.githubusercontent.com/shenxiaocam/ACDNE/master/ACDNE_codes/input/dblpv7.mat` (~559 KB)
  - `https://raw.githubusercontent.com/shenxiaocam/ACDNE/master/ACDNE_codes/input/acmv9.mat` (~1.1 MB)
- **Format:** MATLAB v5 `.mat` files (loadmat-compatible, no h5py needed). Each file holds `attrb` (n × w uint8 BoW), `group` (n × c uint8 one/multi-hot labels), `network` (n × n sparse symmetric adjacency).
- **Download flow:** `requests.head` for the upstream `Content-Length`, skip if local file matches expected bytes, otherwise stream GET to `*.part` + atomic rename. No User-Agent override (raw.githubusercontent.com accepts the default `python-requests/...` UA).
- **Adapter:** `loader.py` (`load_dblp_acmv9`, line 104) returns a `CrossNetwork` (defined in `algorithms/pytorch-implementation/acdne/data.py`). The multi-label rows (~1.5% of DBLP, ~6.2% of ACM) are collapsed to single-label via `argmax` (line 89) — the same convention used by the paper authors' upstream training code. All-zero label rows raise rather than silently mapping to class 0 (line 85). Adjacency is symmetrised via `scipy.sparse.triu(coo, k=1)` to enforce `i < j` and drop self-loops.
- **Anomaly injection:** N/A — ACDNE is a cross-network classifier, not an anomaly detector.
- **Smoke-test contract** (`run_example.py:127`): all four conditions must hold — source classification loss `L_y` decreases, final domain-discriminator accuracy lands within `|d_acc − 0.5| ≤ 0.15` (i.e., the GRL succeeded in confusing the discriminator), Micro-F1 on ACMv9 ≥ floor (0.55 full / 0.50 smoke), and Micro-F1 strictly beats the source's majority-class baseline.

### LODES8 Massachusetts — TransFlower

`algorithms/pytorch-implementation/transflower/datasets/lodes_ma/fetch.py`

- **What:** Three CSV.GZ files from the U.S. Census Bureau LEHD program at `https://lehd.ces.census.gov/data/lodes/LODES8/ma/`:
  - **OD main** — `ma_od_main_JT00_<year>.csv.gz`: block-to-block primary-job commute counts (`S000` column) for all MA workers whose home and workplace are both inside MA. ~50 MB.
  - **WAC** — `ma_wac_S000_JT00_<year>.csv.gz`: per-block job counts split by NAICS supersector (CNS01–CNS20).
  - **Crosswalk** — `ma_xwalk.csv.gz`: block-level metadata; the loader uses `tabblk2020`, `trct`, `blklatdd`, `blklondd` (centroid lat/lon and parent tract GEOID).
- **License:** U.S. federal public domain (17 U.S.C. § 105). Vintages tried newest-first: `(2021, 2020, 2019)`, configurable via `--year`. The xwalk has no year suffix.
- **Download flow:** HEAD pre-flight on every URL, fail-fast if any 404 before any download starts (defensive because the OD CSV is ~50 MB). Streams to `*.part` with a megabyte-level progress bar (`fetch.py:88-91`).
- **`--small`:** skips the OD CSV, leaves WAC + xwalk (still ~30 MB total — `fetch.py:25`).
- **Adapter:** `loader.py` aggregates block-level WAC + crosswalk to **Census Tract** (paper §4.1.3 operates at tract resolution; tract GEOID = first 11 chars of the 15-digit block GEOID). Per-tract `place_features` are the 19 NAICS supersector columns `CNS02..CNS20` — `CNS01` agriculture is dropped because it's mostly zero statewide (`loader.py:43`). `population` per tract is the total workplace jobs `C000`, used as the mass term for outflow scaling. OD flows are grouped by `(h_tract, w_tract)`, summed over `S000`, and self-flows are dropped by default.
- **Subsetting:** `load_regions(county_prefix="25025")` filters to Suffolk County, MA — the `run_example.py` smoke test runs on this subset (line 36) so it fits in memory and trains in a few minutes.
- **Anomaly injection:** N/A — TransFlower predicts commute-flow distributions, not anomalies.
- **Smoke-test contract** (`run_example.py:117`): cross-entropy decreases, `predict_distributions` returns a row-stochastic `(N, N)` matrix (rows sum to 1 within `1e-4`), and CPC ≥ 0.20 is reported as a soft target on the held-out 20% flow split.

### Amazon Photo — MHGL (in-progress, not yet committed)

`algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/`

- Only `raw/amazon_electronics_photo.npz` (a co-purchase product graph from the standard Shchur et al. GNN-Benchmark distribution) and a stale `__pycache__/loader.cpython-313.pyc` are on disk. **No `fetch.py`, `loader.py`, `run_example.py`, or `requirements.txt`** has been committed. `git status` confirms the directory is untracked.
- Treat as a planned adapter: when committed, expect `loader.py` to map the Amazon Photo classes onto MHGL's seen/unseen anomaly partition described in `algorithms/pytorch-implementation/mhgl/data.py:158-187` (rare-class communities act as anomalies).

## Anomaly-Injection Protocols (synthetic generators)

These are paper-protocol injection routines that ship inside each algorithm's `data.py`, not external integrations — but the prompt asked for them, and the AAGNN Cora adapter directly mirrors AAGNN's synthetic protocol.

- **AAGNN** (`algorithms/pytorch-implementation/aagnn/data.py:175-209`, `SyntheticAttributedNetwork.generate`):
  - Synthetic SBM (300 nodes, 5 communities) + 15 structural-clique anomalies + 15 contextual feature-swap anomalies. Same logic the Cora adapter scales 2× to inject 30+30 anomalies. Citation: Ding et al. 2019 / Song et al. 2007 protocol.
- **MHGL** (`algorithms/pytorch-implementation/mhgl/data.py:158-187`, `SyntheticAttributedNetwork`):
  - Distinct from AAGNN: MHGL distinguishes **seen** anomalies (`ANOM_SEEN = 1`, labelled at training time) from **unseen** anomalies (`ANOM_UNSEEN = 2`, never appears in `V_train`). Two rare-class SBM communities (defaults: 30 seen + 30 unseen nodes) each get their own feature centroid; both share the normal communities' edge probabilities, so anomalies are anomalous *by feature divergence and small size*, not by structural perturbation. Train/test split per paper §2: `q` labelled seen anomalies + `p%` labelled normals enter `V_train`; `V_test` is everything else.
- **Geo-Tile2Vec, ACDNE, TransFlower:** no anomaly injection — these are not anomaly-detection algorithms.

## Data Storage

**Databases:** None — all datasets are flat files on disk under `algorithms/<framework>/<algo>/datasets/<source>/raw/` (gitignored — covered by the repo `.gitignore` patterns plus the implicit convention that adapters write to `raw/`).

**File Storage:** Local filesystem only. Two cache locations outside the repo:
- `~/.cache/torch/hub/checkpoints/` — torch.hub default for the PyTorch Geo-Tile2Vec backbone.
- `~/.cache/supernova/mlx-places365/` — adapter-defined cache for the MLX Geo-Tile2Vec converted weights.

**Caching:** The `requests` fetchers are idempotent — they skip the download when the on-disk file already matches the upstream `Content-Length` (DBLP/ACM) or simply exists and is non-empty (Cora, LODES). Re-running `fetch.py` is a no-op once cached.

## Authentication & Identity

None. Every download is anonymous. No auth headers, API keys, OAuth flows, or token files are read or written. No `.env` / `secrets/` directory exists in the repo.

## Monitoring & Observability

**Error Tracking:** None.
**Logs:** `print()` to stdout/stderr inside fetchers, loaders, and training loops. No `logging` module configuration anywhere. Training progress is reported via per-epoch `print(...)` lines in each algorithm's `train.py`/`fit()` orchestrator.

## CI/CD & Deployment

**Hosting:** Not applicable — research portfolio, not a deployed service.
**CI Pipeline:** None. No `.github/workflows/`, `.gitlab-ci.yml`, `.circleci/config.yml`, or comparable file is present.
**Release process:** Manual `gh release create vX.Y.0 --generate-notes` per `AGENTS.md` step 7.

## Environment Configuration

**Required env vars:** None.
**Secrets location:** None — there are no secrets to store.

## Webhooks & Callbacks

**Incoming:** None.
**Outgoing:** None.

---

*Integration audit: 2026-05-10*
