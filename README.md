# ML Worker (unified-workers)

This directory contains the Python-based Machine Learning (ML) Worker service for the Manga Translation Platform. The worker processes computationally heavy and AI-driven tasks asynchronously, coordinating with the Spring Boot backend via a Valkey/Redis task queue.

---

## 🏗️ Architecture & Core Duties

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

## 📂 Project Structure

```txt
unified-workers/
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

## 🚀 Setup & Local Development

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
cd unified-workers
pip install -r requirements.txt
```

### 3. Run the Worker

Start the HTTP health server and task listener:

```bash
python app.py
```

By default, the health check endpoint will be available at `http://localhost:8000/health`.

---

## 🧪 Running Tests

A test runner is provided to verify spatial OCR merging and translation validation logic:

```bash
python run_tests.py
```

---

## 🧼 Linting & Formatting

See [linting.md](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/linting.md) for details on code style guidelines using `ruff`, `black`, and `flake8`.
