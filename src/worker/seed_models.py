"""Model cache preparation CLI for worker container volumes.

This module deliberately imports ML dependencies only after argument parsing.  It is safe to use
``python -m worker.seed_models --help`` in a minimal environment, and it never downloads the YOLO
artifact or contacts the worker API. OCR libraries may download missing weights.
"""

import argparse
import hashlib
import os
from collections.abc import Sequence


class SeedModelsError(RuntimeError):
    """An expected model seeding failure that should be shown without a traceback."""


def _disabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SeedModelsError(f"cannot read YOLO model at {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_yolo() -> None:
    # Keep config and ONNX Runtime imports lazy so --help has no ML import requirements.
    from worker import config

    path = config.YOLO_MODEL_PATH
    if not path or not os.path.isfile(path):
        raise SeedModelsError(f"YOLO model is missing at {path or '<unset>'}")

    actual = _sha256(path)
    if actual != config.YOLO_PINNED_CHECKSUM:
        raise SeedModelsError(f"YOLO checksum mismatch at {path}: expected {config.YOLO_PINNED_CHECKSUM}, got {actual}")

    from worker.services.bubble_detector import get_ort_session

    if get_ort_session() is None:
        raise SeedModelsError("YOLO ONNX Runtime session initialized to None")


def seed_models(languages: Sequence[str], *, skip_local_ocr: bool = False) -> list[tuple[str, str]]:
    """Verify YOLO and initialize one local OCR reader for each requested language."""
    _verify_yolo()
    if skip_local_ocr or _disabled(os.environ.get("DISABLE_LOCAL_OCR")):
        return []

    from worker.model_manager import get_local_ocr_backend, model_manager

    try:
        backend = get_local_ocr_backend()
    except Exception as exc:
        raise SeedModelsError(f"local OCR backend could not be selected: {exc}") from exc

    seeded: list[tuple[str, str]] = []
    for language in languages:
        language = language.strip().lower()
        if not language:
            raise SeedModelsError("language names must not be empty")
        try:
            reader = (
                model_manager.get_rapid_ocr_reader(language)
                if backend == "rapidocr"
                else model_manager.get_paddle_ocr_reader(language)
            )
        except Exception as exc:
            raise SeedModelsError(f"{backend} OCR initialization failed for {language}: {exc}") from exc
        if reader is None:
            raise SeedModelsError(f"{backend} OCR initialization returned None for {language}")
        seeded.append((language, backend))
    return seeded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and warm worker ML model caches.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ja", "ko", "zh"],
        metavar="LANG",
        help="source languages to warm (default: ja ko zh)",
    )
    parser.add_argument(
        "--skip-local-ocr",
        action="store_true",
        help="verify only the pinned YOLO model and skip local OCR model initialization",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seeded = seed_models(args.languages, skip_local_ocr=args.skip_local_ocr)
    except SeedModelsError as exc:
        print(f"model seeding failed: {exc}")
        return 1
    if args.skip_local_ocr or _disabled(os.environ.get("DISABLE_LOCAL_OCR")):
        print("model seeding succeeded: YOLO verified; local OCR skipped")
    else:
        details = ", ".join(f"{language}={backend}" for language, backend in seeded)
        print(f"model seeding succeeded: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
