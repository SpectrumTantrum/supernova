# Technology Stack

**Analysis Date:** 2026-05-10

## Languages

**Primary:**
- Python 3.12+ — every algorithm under `algorithms/`. Floor inferred from `algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:104` (`tarfile.extractall(..., filter="data")` requires Python 3.12+) and broad use of `from __future__ import annotations` + PEP 604 `int | None` style. Local interpreter when audited: Python 3.13.9. No `.python-version` is committed.

**Secondary:**
- None. The repo is pure Python; no Rust / C / shell-script extensions ship.

## Runtime

**Environment:**
- CPython, CPU-first. PyTorch algorithms accept `--device cuda`/`--device mps` via the orchestrator configs (e.g., `algorithms/pytorch-implementation/acdne/model.py` `ACDNEConfig(device="cpu")`); MLX algorithms run on Apple Silicon GPUs via the unified-memory `mlx.core` array.

**Package Manager:**
- `pip` against per-algorithm `requirements.txt`. No lockfiles (`requirements*.lock`, `poetry.lock`, `uv.lock`) anywhere in the tree.
- No top-level `pyproject.toml`, `setup.py`, `setup.cfg`, or `tox.ini`. Each algorithm folder is intentionally an isolated install target.

## Frameworks

**Core (PyTorch lane):**
- `torch>=2.0` — every `algorithms/pytorch-implementation/*/requirements.txt`. Used for tensors, `nn.Module`, autograd, and (for Geo-Tile2Vec) `torch.hub` weight downloads.
- `torchvision>=0.15` — Geo-Tile2Vec only (`algorithms/pytorch-implementation/geotile2vec/requirements.txt`); supplies `torchvision.models.resnet18` + `torchvision.transforms` for the Places365 backbone.

**Core (MLX lane):**
- `mlx>=0.31` — every `algorithms/mlx-implementation/*/requirements.txt`. Apple-Silicon-native array library used as a torch substitute. The MLX Geo-Tile2Vec re-implements ResNet-18 in `algorithms/mlx-implementation/geotile2vec/places365_backbone.py` using `mlx.nn.Conv2d` / `nn.BatchNorm` / `nn.MaxPool2d` to keep runtime torch-free; torch is **lazy-imported once** inside `_download_and_convert` (line ~208) only when the cached `.npz` weights are missing.

**Numerics & ML utilities (shared by both lanes):**
- `numpy>=1.24` — universal. PyTorch lane: `algorithms/pytorch-implementation/{aagnn,acdne,geotile2vec,mhgl,transflower}/requirements.txt`. MLX lane: same five.
- `scipy>=1.11` — universal across all ten algorithm folders. Used for `scipy.sparse` (ACDNE adjacency parsing in `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/loader.py:27`), `scipy.io.loadmat` (same loader, line 28), and `scipy.stats.ttest_ind` for smoke-test contracts.
- `scikit-learn>=1.3` — present in all algorithm `requirements.txt` **except** `algorithms/pytorch-implementation/transflower/requirements.txt` (TransFlower's algorithm code uses pure torch + numpy; the `f1_score` / `roc_auc_score` calls live only in adapter `run_example.py` files, which inherit sklearn from the parent algorithm install when present). Geo-Tile2Vec uses `sklearn.decomposition.IncrementalPCA` in `algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py:28`.

**Imaging:**
- `pillow>=10.0` — Geo-Tile2Vec only, both lanes (`algorithms/pytorch-implementation/geotile2vec/requirements.txt`, `algorithms/mlx-implementation/geotile2vec/requirements.txt`). Required for street-view image preprocessing; the MLX path imports it lazily inside `imagenet_preprocess` (`algorithms/mlx-implementation/geotile2vec/places365_backbone.py:119`).

**Testing:**
- No `pytest` / `unittest` framework in any `requirements.txt`. The smoke-test contract is encoded in each algorithm's `example.py` (and, for real-data adapters, `datasets/<name>/run_example.py`) which exits non-zero unless a measurable property holds — Welch's t-test on cluster similarity (Geo-Tile2Vec, AAGNN), Micro-F1 floor + balanced domain accuracy (ACDNE: `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/run_example.py:127`), CPC threshold + row-sum check (TransFlower: `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/run_example.py:117`), or ROC-AUC floor (AAGNN Cora: `algorithms/pytorch-implementation/aagnn/datasets/cora/run_example.py:35`).

**Build/Dev:**
- Ruff — `0.12.0` cache present at `.ruff_cache/0.12.0/`. **No committed `pyproject.toml` / `ruff.toml` / `.ruff.toml`**, so ruff runs against its built-in defaults and is invoked ad-hoc by the developer (not pinned in any requirements file or pre-commit hook).
- No formatter (`black`, `prettier`, `isort`) and no linter (`mypy`, `pyright`) configured.
- No CI config: `.github/workflows/`, `.gitlab-ci.yml`, `.circleci/` are all absent.
- No pre-commit framework (`.pre-commit-config.yaml` absent).

## Per-algorithm dependency variations

Top-level (pinned in `requirements.txt`):

| Algorithm | torch / mlx | numpy | scipy | scikit-learn | torchvision | pillow |
|-----------|-------------|-------|-------|--------------|-------------|--------|
| `algorithms/pytorch-implementation/aagnn/`        | torch>=2.0 | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/pytorch-implementation/acdne/`        | torch>=2.0 | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/pytorch-implementation/geotile2vec/`  | torch>=2.0 | 1.24+ | 1.11+ | 1.3+ | 0.15+ | 10.0+ |
| `algorithms/pytorch-implementation/mhgl/`         | torch>=2.0 | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/pytorch-implementation/transflower/`  | torch>=2.0 | 1.24+ | 1.11+ | **omitted** | — | — |
| `algorithms/mlx-implementation/aagnn/`            | mlx>=0.31  | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/mlx-implementation/acdne/`            | mlx>=0.31  | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/mlx-implementation/geotile2vec/`      | mlx>=0.31  | 1.24+ | 1.11+ | 1.3+ | — | 10.0+ |
| `algorithms/mlx-implementation/mhgl/`             | mlx>=0.31  | 1.24+ | 1.11+ | 1.3+ | — | — |
| `algorithms/mlx-implementation/transflower/`      | mlx>=0.31  | 1.24+ | 1.11+ | 1.3+ | — | — |

Notes:
- **TransFlower PyTorch** uniquely omits `scikit-learn` from its top-level `requirements.txt`. The MLX TransFlower retains it.
- **Geo-Tile2Vec** is the only algorithm needing the imaging stack (`torchvision`/`pillow`), in both lanes.
- The MLX TransFlower equivalent of `torch.optim.RMSprop` and `torch.nn.functional.cross_entropy` is implemented against `mlx.optimizers` and `mlx.nn.losses`; no extra deps required beyond `mlx>=0.31`.

Real-data adapters add their own pin set (see also `INTEGRATIONS.md`):

| Adapter | Path | Extra deps |
|---------|------|------------|
| Cora (AAGNN)            | `algorithms/pytorch-implementation/aagnn/datasets/cora/requirements.txt`            | `requests>=2.31` |
| DBLPv7→ACMv9 (ACDNE)    | `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/requirements.txt`        | `requests>=2.31` (file's own header notes torch/numpy/scipy/scikit-learn must come from the parent algorithm install) |
| LODES8 MA (TransFlower) | `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/requirements.txt`  | `pandas>=2.0`, `requests>=2.31` |
| Amazon Photo (MHGL)     | `algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/`                     | adapter not yet committed — only `raw/amazon_electronics_photo.npz` and a stale `__pycache__/loader.cpython-313.pyc` are on disk; directory is untracked per `git status` |

## Configuration

**Environment:**
- No `.env` files, no environment-variable consumption anywhere in `algorithms/`. Algorithms are configured exclusively via the per-algorithm `<Algo>Config` dataclass (e.g., `AAGNNConfig` in `algorithms/pytorch-implementation/aagnn/model.py`, `ACDNEConfig` in `algorithms/pytorch-implementation/acdne/model.py`, `TransFlowerConfig` in `algorithms/pytorch-implementation/transflower/model.py`) plus CLI flags on `example.py` / `run_example.py`.

**Build:**
- No build configuration files. Each algorithm directory is a flat module (no `src/`, no packaging metadata). `python example.py` from inside the algorithm folder is the only build/run step.

## Platform Requirements

**Development:**
- Python ≥ 3.12.
- For the MLX lane: Apple Silicon hardware (M-series) — `mlx>=0.31` does not support x86-64 macOS or Linux GPUs.
- For the PyTorch lane: any CPU/CUDA/MPS-capable host that satisfies `torch>=2.0`.
- ~50 MB free for the Places365 ResNet-18 checkpoint cache (Geo-Tile2Vec) at `~/.cache/torch/hub/checkpoints/resnet18_places365.pth.tar` and/or `~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz`.
- Up to ~50 MB for the LODES8 MA OD CSV at `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/raw/`.

**Production:**
- Not applicable — there is no deployment target. Each algorithm is a research/portfolio implementation distributed as source. Releases are tagged via `gh release create vX.Y.0 --generate-notes` per `AGENTS.md`.

---

*Stack analysis: 2026-05-10*
