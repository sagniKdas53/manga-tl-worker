# ML worker

This directory contains the Python machine learning worker service for the Manga Translation Platform. The worker processes computationally heavy and AI tasks asynchronously, coordinating with the Rust backend via a Valkey/Redis task queue.

---

## Architecture and duties

The worker runs a loop to consume tasks from Valkey/Redis and coordinates with MinIO S3 for downloading raw images and uploading processed layers and masks.

Primary responsibilities:

1. **Layout analysis and OCR**: Runs local OCR (PaddleOCR on amd64, RapidOCR/ONNX Runtime on Linux ARM64) for text detection/recognition and a YOLO bubble segmentation model for speech bubble coordinates and polygons.
2. **Spatial OCR region merging**: Groups individual text lines into logical speech bubbles before panel mapping. Configurable via `OCR_MERGE_THRESHOLD`.
3. **AI translation pass**: Translates text using:
   - **VLM vision-language pass**: Contextual visual dialogue mapping via OpenRouter or NVIDIA NIM APIs.
   - **LLM text pass**: Translation via configured providers in `config/providers.json`.
   - **Fallbacks**: Standard translations via DeepL.
4. **Typesetting and canvas fitting**: Calculates offscreen canvas typography bounds, wrapping words, and rendering translated text within bubble constraints.

---

## Project structure

```txt
worker/
├── app.py                   # Main entry point (starts HTTP health server and worker loop)
├── Dockerfile               # Production container image configuration
├── requirements.txt         # Core Python dependencies
├── tests/                   # Test suite
└── src/worker/              # Core application package
    ├── config.py            # Environment configurations and defaults
    ├── model_manager.py     # OCR model loaders and caching managers
    ├── health_server.py     # FastAPI health check endpoint server
    ├── handlers/            # Queue task handlers (OCR, Translation, Render, etc.)
    ├── services/            # Client interfaces (MinIO, Valkey/Redis, Translation APIs)
    └── utils/               # Image manipulation, geometry calculations, and helpers
```

---

## Pre-built image

Published to the GitHub Container Registry on every merge to `main`:

```bash
docker pull ghcr.io/sagnikdas53/manga-tl-worker:latest
```

The parent stack `docker-compose.yml` references this image, so `docker compose up -d` pulls rather than builds locally.

| Tag | Points at | Use it for |
| --- | --- | --- |
| `latest` | current `main` | Deployments. Followed by Watchtower. |
| `main` | current `main` | Alias of `latest`. |
| `1.4.0` | that release | Pinning to an exact version. |
| `sha-a1b2c3d` | one commit | Rollback to a specific build. Kept for 7 days. |

Version tags are generated from Conventional Commits on `main`.

> **Linux ARM64 POC.** The worker now selects PaddleOCR/PaddlePaddle on amd64 and RapidOCR
> with ONNX Runtime on Linux ARM64. `LOCAL_OCR_BACKEND=auto` is the default: it preserves
> PaddleOCR on x86 deployments and selects RapidOCR on aarch64. Set
> `RAPIDOCR_MODEL_ROOT=/home/worker/.cache/rapidocr` only when overriding the default cache.
>
> The ARM path uses PP-OCRv6 medium models for Japanese, Chinese, and English, and PP-OCRv5
> mobile recognition for Korean. The dedicated
> [ARM64 POC workflow](.github/workflows/ci-arm64-poc.yml) builds and smoke-tests the image
> under aarch64 emulation. The production publish workflow also emits amd64 and arm64
> manifests; benchmark the ARM path on representative manga pages before making it the only
> deployment target.

---

## Setup and local development

### 1. System prerequisites

Ensure you have Python 3.13 installed and system dependencies required by OpenCV.

On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender-dev
```

You also need [`uv`](https://docs.astral.sh/uv/) — it is not bundled with Python and every step below invokes it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment and dependencies

Per the repository standard, all worker development uses `uv` and the root virtual environment:

```bash
# From repository root
uv venv --python 3.13 .venv
uv pip install -r worker/requirements.txt --python ./.venv/bin/python
```

`LOCAL_OCR_BACKEND` picks the local OCR engine — default `auto` (PaddleOCR on amd64,
RapidOCR on Linux ARM64). Forcing the non-default engine requires its package installed
(`uv pip install rapidocr`, or `paddleocr paddlepaddle`); the worker rejects an override
whose package is missing with a clear error.

### 3. Run the worker

Start the HTTP health server and task listener:

```bash
# From worker/ directory using root venv. PYTHONPATH=src puts the `worker`
# package on the path (installing requirements.txt does not install this project).
# Dev only: the worker also exits on startup unless WORKER_API_SECRET is set or
# the unauthenticated opt-out is enabled (AUDIT-S3).
PYTHONPATH=src ALLOW_UNAUTHENTICATED_WORKER_API=true ../.venv/bin/python app.py
```

The health check endpoint is available at `http://localhost:8000/health`.

---

## Running tests

Run the test suite with pytest from the worker directory:

```bash
../.venv/bin/python -m pytest -q
```

---

## Linting and formatting

Lint and format with `ruff` via the root virtualenv:

```bash
../.venv/bin/python -m ruff check --fix . && ../.venv/bin/python -m ruff format .
../.venv/bin/python -m pyright .
```

See [COMMANDS.md](COMMANDS.md) for individual commands.
