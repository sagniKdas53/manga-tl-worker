# Output-quality workstream

Branch: `feat/output-quality`. The parent app's `docs/output-quality-implementation-tracker.md` owns the milestone/task tracker. Keep this worker usable as a standalone checkout; commit/push it before updating the parent submodule pointer.

The target is new processing: early SFX intent, independent fragment ownership, glyph masks/restoration artifacts and a shared browser renderer for final typography. ARM64 remains a POC, not a release gate. Existing corpus outputs predate later fixes; reproduce against the current worker before changing algorithms.

## Model-cache preparation

The new CLI does not start the worker server or publish provider configuration to Redis:

```bash
PYTHONPATH=src ../.venv/bin/python -m worker.seed_models --help
# In the built image, after supplying the pinned YOLO artifact and cache volumes:
python -m worker.seed_models --languages ja ko zh
```

Set `YOLO_MODEL_PATH` explicitly for host use. Its file must match `YOLO_PINNED_CHECKSUM`; the CLI rejects missing/mismatched files before initializing ONNX Runtime. OCR readers use the existing catalog/language routing and may download weights when caches are empty. A `None` reader fails the command. `--skip-local-ocr` or `DISABLE_LOCAL_OCR=true` verifies only YOLO. The parent `scripts/dev_setup.py` prepares the dev Compose mounts and private credentials; see its `docs/dev-box-setup.md`.

No OCR/grouping/render quality algorithm was changed as part of this setup work. Cache warmup is not evidence that the output-quality gates pass.
