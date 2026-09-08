# ML Worker

This directory contains the Python-based Machine Learning (ML) Worker service for the Manga Translation Platform. The worker processes computationally heavy and AI-driven tasks asynchronously, coordinating with the Spring Boot backend via a Valkey/Redis task queue.

---

## Architecture & core duties

The worker runs a loop to consume tasks from Valkey/Redis and coordinates with MinIO S3 for downloading raw images and uploading processed layers and masks.

Its primary responsibilities include:

1. **Layout Analysis & OCR**: Runs local OCR (PaddleOCR for text detection/recognition and a YOLO bubble segmentation model for speech bubble coordinates and polygons).
2. **Spatial OCR Region Merging**: Groups individual text lines into logical speech bubbles before panel mapping. Configurable via `OCR_MERGE_THRESHOLD` vertical/horizontal proximity algorithm multiplier.
3. **AI Translation Pass**: Translates text using:
   - **VLM Vision-Language pass**: Contextual visual-dialogue mapping (NVIDIA NIM APIs like `nvidia/nemotron-nano-12b-v2-vl` or `microsoft/phi-4-multimodal-instruct`).
   - **LLM Text pass**: Translation via `google/gemma-3n-e4b-it` / `google/gemma-3n-e2b-it`.
   - **Fallbacks**: Standard translations via DeepL/Google Translate.
4. **Typesetting & Canvas Fitting**: Calculates offscreen canvas typography bounds, wrapping words (or characters if necessary), and rendering the translated text within the bubble constraints.

---

## Project structure

```txt
worker/
├── app.py                   # Main entry point (starts HTTP health server and worker loop)
├── Dockerfile               # Production container image configuration
├── requirements.txt         # Core Python dependencies
├── run_tests.py             # Validation test runner
├── linting.md               # Linting and formatting instructions
├── tests/                   # Test suite for merging and translation validation
└── worker/                  # Core application package
    ├── config.py            # Environment configurations & defaults
    ├── model_manager.py     # OCR model loaders & caching managers
    ├── health_server.py     # FastAPI/BaseHTTP health check endpoint server
    ├── handlers/            # Queue task handlers (OCR, Translation, Render, etc.)
    ├── services/            # Client interfaces (MinIO, Valkey/Redis, Translation APIs)
    └── utils/               # Image manipulation, geometry calculations, and helpers
```

---

## Pre-built image

Published to the GitHub Container Registry on every merge to `main`, and public — no
`docker login` is needed to pull it.

```bash
docker pull ghcr.io/sagnikdas53/manga-tl-worker:latest
```

The parent stack's `docker-compose.yml` already references this image, so `docker compose up -d`
there pulls rather than builds.

| Tag | Points at | Use it for |
| --- | --- | --- |
| `latest` | current `main` | Deployments. This is the tag Watchtower follows. |
| `main`, `master` | current `main` | Aliases of `latest`. Both exist so that `:master` — this repo's default branch is `main`, but yt-diff's is `master` — does not fail. |
| `1.4.0` | that release | Pinning to an exact version. Note there is **no** leading `v`; the git tag is `v1.4.0` but `docker/metadata-action` strips it. |
| `1.4` / `1` | newest 1.4.x / 1.x | Auto-updating within a minor or major line. |
| `sha-a1b2c3d` | one commit | Rollback to a specific build. Kept for 7 days. |

Version tags are cut automatically from [Conventional Commits](https://www.conventionalcommits.org/):
a `feat:` on `main` bumps the minor, a `fix:` bumps the patch, and `BREAKING CHANGE:` bumps the
major. A merge with no conventional prefix does not cut a release.

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

## Setup & local development

### 1. Prerequisites

Ensure you have Python 3.10+ installed and system dependencies required by OpenCV.

On Linux:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender-dev
```

### 2. Installation

Create and activate a virtual environment, then install dependencies:

```bash
# From workspace root
python -m venv .venv
source .venv/bin/activate

# Install requirements
cd worker
pip install -r requirements.txt

# Optional backend override:
# LOCAL_OCR_BACKEND=auto   # PaddleOCR on amd64, RapidOCR on ARM64 (default)
# LOCAL_OCR_BACKEND=rapidocr
```

### 3. Run the worker

Start the HTTP health server and task listener:

```bash
python app.py
```

By default, the health check endpoint will be available at `http://localhost:8000/health`.

---

## Running tests

A test runner is provided to verify spatial OCR merging and translation validation logic:

```bash
python run_tests.py
```

---

## Linting & formatting

Lint and format with `ruff` (it replaces Flake8, Black and isort). See [COMMANDS.md](COMMANDS.md)
for the exact invocations, and [../docs/guides/quality_gate.md](../docs/guides/quality_gate.md) for
the full gate.
