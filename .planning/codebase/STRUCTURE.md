# Codebase Structure

**Analysis Date:** 2026-05-10

## Directory Layout

```
supernova/
├── AGENTS.md                                  # Canonical agent instructions (single source of truth)
├── CLAUDE.md -> AGENTS.md                     # Symlink for Claude Code
├── GEMINI.md -> AGENTS.md                     # Symlink for Gemini CLI
├── CHANGELOG.md                               # Repo-level release notes
├── .gitignore                                 # Excludes .research/, research/, __pycache__, .claude/, .venv/, etc.
├── .planning/                                 # GSD planning workspace (this directory)
│   └── codebase/                              # Codebase maps (ARCHITECTURE.md, STRUCTURE.md, …)
├── .claude/                                   # Claude Code workspace (gitignored)
├── .ruff_cache/                               # Ruff cache (gitignored)
├── research/                                  # Paywalled paper PDFs (gitignored — see note below)
└── algorithms/
    ├── pytorch-implementation/                # PyTorch reference implementations
    │   ├── geotile2vec/                       # Luo et al., ACM TSAS 2023
    │   ├── transflower/                       # Luo et al., arXiv:2402.15398v1, 2024
    │   ├── aagnn/                             # Zhou et al., CIKM '21
    │   ├── mhgl/                              # Zhou et al., SIAM SDM 2022
    │   └── acdne/                             # Shen et al., AAAI 2020
    └── mlx-implementation/                    # MLX mirror implementations (Apple-silicon)
        ├── geotile2vec/
        ├── transflower/
        ├── aagnn/
        ├── mhgl/
        └── acdne/
```

### Per-algorithm folder layout (uniform across all 10 implementations)

```
algorithms/<framework>/<name>/
├── README.md                                  # Paper ref, theory recap, run instructions
├── requirements.txt                           # Algorithm-local dependencies
├── data.py                                    # Records, helpers, Synthetic* generator
├── <algo>_<stage>.py  (×1–N)                  # One module per pipeline stage in the paper
├── model.py                                   # <Algo>Config dataclass + <Algo> orchestrator
├── example.py                                 # Synthetic-data smoke test (CLI entry, sys.exit 0/1)
└── datasets/                  (PyTorch only, optional)
    └── <dataset-name>/
        ├── fetch.py                           # Idempotent multi-mirror downloader
        ├── loader.py                          # raw → parent dataclass adapter
        ├── run_example.py                     # Real-data analogue of example.py
        ├── requirements.txt                   # Adapter-local dependencies (e.g. requests)
        └── raw/                               # Cached download (gitignored archives)
```

## Directory Purposes

**`/` (repo root):**
- Purpose: Hosts agent docs, the `algorithms/` portfolio, and gitignored scratch dirs.
- Contains: `AGENTS.md`, `CLAUDE.md` and `GEMINI.md` (symlinks to `AGENTS.md`), `CHANGELOG.md`, `.gitignore`.
- Key files: `AGENTS.md` (the canonical convention doc).

**`algorithms/`:**
- Purpose: Container for the algorithm portfolio.
- Contains: Two framework subfolders (`pytorch-implementation/`, `mlx-implementation/`), each with one folder per algorithm.
- Key files: none — purely organisational.

**`algorithms/pytorch-implementation/`:**
- Purpose: PyTorch reference implementations (the default backend per `AGENTS.md`).
- Contains: `geotile2vec/`, `transflower/`, `aagnn/`, `mhgl/`, `acdne/`.

**`algorithms/mlx-implementation/`:**
- Purpose: Hand-ported MLX siblings of the PyTorch algorithms for Apple-silicon.
- Contains: `geotile2vec/`, `transflower/`, `aagnn/`, `mhgl/`, `acdne/`.

**`algorithms/<framework>/<name>/`:**
- Purpose: A single paper, end-to-end, independently runnable.
- Contains: 5–8 source files (see uniform layout above), a README, a `requirements.txt`, optionally a `datasets/` subdir.
- Key files: `model.py` (the orchestrator), `example.py` (the smoke test).

**`algorithms/pytorch-implementation/<name>/datasets/<dataset>/`:**
- Purpose: Real-data adapter for the parent algorithm (opt-in; not all algorithms have one).
- Contains: `fetch.py`, `loader.py`, `run_example.py`, `requirements.txt`, `raw/` cache.
- Key files: `loader.py` exposes a function returning the parent's record type (e.g. `AttributedNetwork`).

**`research/` (and the AGENTS.md-canonical `.research/`):**
- Purpose: Paywalled paper PDFs for reference while implementing.
- Contains: One PDF per algorithm (e.g. `aagnn-subtractive-aggregation-network-anomaly-cikm-2021.pdf`, `geo-tile2vec-multimodal-urban-analytics-tsas-2023.pdf`, `transflower-explainable-transformer-commuting-flow-arxiv-2024.pdf`, `mhgl-unseen-anomaly-detection-multi-hypersphere-learning-sdm-2022.pdf`, `acdne-adversarial-deep-network-embedding-aaai-2020.pdf`, plus exploratory `decl-…`).
- Note: `AGENTS.md` and `.gitignore` reference this as `.research/`; the working directory is `research/`. Both names are gitignored — never commit PDFs.

**`.planning/`:**
- Purpose: GSD (Get-Stuff-Done) planning workspace for AI agents.
- Contains: `codebase/` (this map), and any future phase plans / execution logs.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

**`.claude/`:**
- Purpose: Claude Code session state (worktrees, scratch). Gitignored.
- Generated: Yes, by Claude Code.
- Committed: No.

**`.ruff_cache/`:**
- Purpose: Ruff lint cache. Gitignored.
- Generated: Yes.
- Committed: No.

## Key File Locations

**Repo manifest:**
- `AGENTS.md`: Canonical instructions, layout convention, algorithm registry, contribution checklist.

**Per-algorithm entry points (PyTorch):**
- `algorithms/pytorch-implementation/geotile2vec/example.py`: Geo-Tile2Vec synthetic city smoke test.
- `algorithms/pytorch-implementation/transflower/example.py`: TransFlower flow-prediction smoke test.
- `algorithms/pytorch-implementation/aagnn/example.py`: AAGNN anomaly-detection smoke test.
- `algorithms/pytorch-implementation/mhgl/example.py`: MHGL multi-hypersphere anomaly smoke test.
- `algorithms/pytorch-implementation/acdne/example.py`: ACDNE cross-network classification smoke test.

**Per-algorithm entry points (MLX):**
- `algorithms/mlx-implementation/geotile2vec/example.py`
- `algorithms/mlx-implementation/transflower/example.py`
- `algorithms/mlx-implementation/aagnn/example.py`
- `algorithms/mlx-implementation/mhgl/example.py`
- `algorithms/mlx-implementation/acdne/example.py`

**Per-algorithm orchestrators (PyTorch):**
- `algorithms/pytorch-implementation/geotile2vec/model.py` — `GeoTile2Vec`, `GeoTile2VecConfig`.
- `algorithms/pytorch-implementation/transflower/model.py` — `TransFlower`, `TransFlowerConfig`.
- `algorithms/pytorch-implementation/aagnn/model.py` — `AAGNN`, `AAGNNConfig`.
- `algorithms/pytorch-implementation/mhgl/model.py` — `MHGL`, `MHGLConfig`.
- `algorithms/pytorch-implementation/acdne/model.py` — `ACDNE`, `ACDNEConfig`.

**Per-algorithm stage modules (PyTorch — illustrating the `<algo>_<stage>.py` and topical-name patterns):**
- Geo-Tile2Vec: `algorithms/pytorch-implementation/geotile2vec/stage1_mobility.py`, `stage2_streetview.py`.
- TransFlower: `algorithms/pytorch-implementation/transflower/geo_encoder.py`, `flow_predictor.py`.
- AAGNN: `algorithms/pytorch-implementation/aagnn/layer.py`, `train.py`.
- MHGL: `algorithms/pytorch-implementation/mhgl/gcn.py`, `pde.py`, `train.py`.
- ACDNE: `algorithms/pytorch-implementation/acdne/layers.py`, `train.py`.

**MLX-only addition:**
- `algorithms/mlx-implementation/geotile2vec/places365_backbone.py` — Native MLX ResNet-18 + Places365 weight loader (the PyTorch sibling uses `torch.hub` instead).

**Real-data adapters (current set):**
- `algorithms/pytorch-implementation/aagnn/datasets/cora/` — Cora citation network (LINQS), 2,708 nodes / 1,433 features, with paper §4 anomaly injection.
- `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/` — DBLPv7 → ACMv9 cross-network adapter.
- `algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/` — Amazon Photo (raw cache currently present; loader/fetch may be in-progress).
- `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/` — LODES Massachusetts commuting flows.

**Documentation per algorithm:**
- `algorithms/<framework>/<name>/README.md` — paper citation, pipeline diagram, run instructions.

**Repo-level docs:**
- `AGENTS.md` (canonical convention).
- `CHANGELOG.md` (release notes).

## Naming Conventions

**Repo-level files:**
- `AGENTS.md` is canonical; any tool-specific entry point is a symlink (`CLAUDE.md`, `GEMINI.md`).
- Add a new agent by `ln -s AGENTS.md <NEW_FILE>.md` rather than copying content.

**Algorithm folder names:**
- Lowercase, no hyphens between words inside the algorithm name (`geotile2vec`, `transflower`, `aagnn`, `mhgl`, `acdne`). The framework parent is hyphenated (`pytorch-implementation`, `mlx-implementation`).

**Source file names inside an algorithm folder:**
- `data.py` — records, helpers, `Synthetic*` generator. (Always present.)
- `model.py` — `<Algo>Config` dataclass and `<Algo>` orchestrator class. (Always present.)
- `example.py` — runnable synthetic smoke test. (Always present.)
- `<algo>_<stage>.py` — one file per paper-defined stage when stages are explicitly numbered (Geo-Tile2Vec uses `stage1_mobility.py`, `stage2_streetview.py`).
- Topical names (`layer.py`, `layers.py`, `gcn.py`, `pde.py`, `train.py`, `flow_predictor.py`, `geo_encoder.py`) — used when paper structure suggests components rather than numbered stages.
- `README.md`, `requirements.txt` — always present.

**Real-data adapter file names (inside `datasets/<dataset>/`):**
- `fetch.py` — multi-mirror, idempotent download script.
- `loader.py` — parses `raw/` contents and returns the parent algorithm's dataclass.
- `run_example.py` — real-data analogue of the parent `example.py`.
- `requirements.txt` — adapter-local pip pins (typically just `requests` plus parsing libraries).
- `raw/` — cache directory for downloaded archives + extracted files.

**Dataset folder names (inside `datasets/`):**
- Lowercase with underscores (`cora`, `dblp_acm`, `amazon_photo`, `lodes_ma`).

**Class & dataclass naming:**
- Orchestrator class: PascalCase matching the paper's algorithm name (`GeoTile2Vec`, `AAGNN`, `MHGL`, `ACDNE`, `TransFlower`).
- Config dataclass: orchestrator name + `Config` (`GeoTile2VecConfig`, `AAGNNConfig`, …).
- History dataclass (when present): `TrainingHistory`.
- Record dataclasses: `@dataclass(frozen=True)` PascalCase (`POI`, `Trajectory`, `MobilityEvent`, `Region`, `Flow`, `AttributedNetwork`, `CrossNetwork`).
- Synthetic generator: `Synthetic*` PascalCase (`SyntheticCity`, `SyntheticAttributedNetwork`, `SyntheticCrossNetwork`).

## Where to Add New Code

**A new algorithm:**
- Primary code: `algorithms/pytorch-implementation/<name>/` with the uniform layout (`README.md`, `requirements.txt`, `data.py`, `<algo>_<stage>.py`, `model.py`, `example.py`).
- Smoke test: `algorithms/pytorch-implementation/<name>/example.py` (must exit 0/1 on a measurable property).
- After verifying with `python example.py`, add a row to the **Existing algorithms** table in `AGENTS.md` and tag with `gh release create vX.Y.0 --generate-notes`.
- Drop the paper PDF in `research/` (do NOT commit).

**An MLX port of an existing PyTorch algorithm:**
- New folder: `algorithms/mlx-implementation/<name>/` mirroring the PyTorch sibling file-for-file.
- Substitute `mlx.core` / `mlx.nn` for `torch` / `torch.nn`; keep the `<Algo>Config` field set identical so user code can swap backends by changing the import path.

**A new pipeline stage inside an existing algorithm:**
- New file at the algorithm root: `algorithms/<framework>/<name>/<algo>_<stage>.py` (or a topical name like `train.py`).
- Wire it into `model.py`'s `fit()` between the existing stages.
- Do NOT create a `src/` or nested package — flat layout is mandatory.

**A new real-data adapter for an existing algorithm:**
- New folder: `algorithms/pytorch-implementation/<name>/datasets/<dataset>/`.
- Follow the established triplet: `fetch.py` (mirrors + idempotent download), `loader.py` (uses `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))` to import the parent `data.py`), `run_example.py` (uses the same shim for `model.py`).
- Reference templates: `algorithms/pytorch-implementation/aagnn/datasets/cora/`, `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/`, `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/`.

**A shared utility used by two algorithms:**
- Do not extract it. Duplicate it inside each algorithm's `data.py`. (See `haversine_meters` repeated across `geotile2vec/data.py:64` and `transflower/data.py`.) Cross-algorithm imports break the independent-runnability invariant.

**Tests:**
- Do not create a `tests/` directory. Tighten the assertion at the bottom of `example.py` (or `datasets/<dataset>/run_example.py` for real-data) so the gating `sys.exit` reflects the new requirement.

**Pretrained weight loaders:**
- PyTorch path: use `torch.hub.load_state_dict_from_url` with an offline-fallback `RuntimeError` naming the expected cache path.
- MLX path: hand-convert the PyTorch checkpoint to an `.npz` cache (template: `algorithms/mlx-implementation/geotile2vec/places365_backbone.py`), and raise `RuntimeError` with the URL + expected cache path if both the cache and torch are missing.

## Special Directories

**`research/` (paper PDFs):**
- Purpose: Paywalled source papers used while implementing. Both `research/` and the AGENTS.md-canonical `.research/` are gitignored.
- Generated: No (manually populated).
- Committed: No — listed in `.gitignore`.

**`.planning/`:**
- Purpose: GSD workspace; codebase maps and phase plans live here.
- Generated: Yes (by GSD commands).
- Committed: Generally yes for planning artefacts; check the active `.gitignore` policy.

**`.claude/`:**
- Purpose: Claude Code session state and worktrees.
- Generated: Yes, by Claude Code.
- Committed: No (gitignored).

**`.ruff_cache/`, `__pycache__/`, `.venv/`, `.pytest_cache/`, `.mypy_cache/`:**
- Purpose: Tool caches.
- Generated: Yes.
- Committed: No (gitignored).

**`raw/` (inside `datasets/<dataset>/`):**
- Purpose: Cache for downloaded dataset archives + extracted files.
- Generated: Yes, by `fetch.py`.
- Committed: Archives should not be committed. Treat as ephemeral; `fetch.py` is idempotent.

---

*Structure analysis: 2026-05-10*
