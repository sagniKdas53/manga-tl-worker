"""Model caching and manager logic for OCR libraries."""

import gc
import importlib.util
import logging
import os
import platform
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Configure PaddleOCR environment variables
try:
    os.environ.setdefault("PADDLEX_OFFLINE_MODE", "0")
    os.environ.setdefault("PADDLE_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
except Exception as err_env:  # pylint: disable=broad-except
    logger.error(f"[Unified Worker] Failed to set PaddleOCR environment: {err_env}")


def get_local_ocr_backend() -> str:
    """Return the configured local OCR backend, selecting the ARM-safe default automatically.

    PaddleOCR remains the default on x86 Linux so existing deployments keep their current
    behaviour. Linux ARM64 uses RapidOCR with ONNX Runtime because the PaddlePaddle wheel
    required by this worker is not published for that architecture.
    """
    requested = os.environ.get("LOCAL_OCR_BACKEND", "auto").strip().lower()
    if requested == "auto":
        machine = platform.machine().strip().lower()
        # Match the requirements.txt markers: rapidocr ships only for Linux aarch64/arm64.
        # macOS Apple Silicon and Windows ARM64 also report "arm64" but get PaddleOCR.
        is_linux_arm = sys.platform.startswith("linux") and machine in {"aarch64", "arm64"}
        return "rapidocr" if is_linux_arm else "paddle"
    if requested in {"paddle", "rapidocr"}:
        # requirements.txt installs paddleocr/paddlepaddle everywhere except Linux
        # aarch64/arm64, and rapidocr only there. Reject an override whose package is
        # absent now, with a clear message, rather than at the first OCR job.
        package = "paddleocr" if requested == "paddle" else "rapidocr"
        if importlib.util.find_spec(package) is None:
            raise ValueError(
                f"LOCAL_OCR_BACKEND={requested} but the '{package}' package is not installed. "
                "requirements.txt installs PaddleOCR everywhere except Linux aarch64/arm64, and "
                "RapidOCR only there; install the missing package to use the override elsewhere."
            )
        return requested
    raise ValueError(f"Unsupported LOCAL_OCR_BACKEND={requested!r}; expected 'auto', 'paddle', or 'rapidocr'.")


def _rapidocr_route(source_language: str | None) -> tuple[str, str, str, str]:
    """Return RapidOCR version/model/language settings for one source language.

    PP-OCRv6 provides the Japanese/Chinese/English path used by this application. PP-OCRv6
    has no Korean recognition model, so Korean follows the PP-OCRv5 mobile route.
    """
    language = (source_language or "ja").strip().lower()
    if language == "ko":
        return ("PP-OCRv5", "mobile", "ch", "korean")

    rec_language = {
        "ja": "japan",
        "jp": "japan",
        "zh": "ch",
        "zh-cn": "ch",
        "zh-tw": "chinese_cht",
        "en": "en",
    }.get(language, "en")
    model_type = os.environ.get("RAPIDOCR_MODEL_TYPE", "medium").strip().lower() or "medium"
    # PP-OCRv6 publishes one multilingual detector (multi_PP-OCRv6_det_*), while the
    # recognition model selects the requested script. PP-OCRv5's Korean route uses its
    # ch_PP-OCRv5 detector instead.
    return ("PP-OCRv6", model_type, "multi", rec_language)


LANG_TO_PADDLE: dict = {
    "ja": "japan",
    "zh": "chinese_cht",  # Traditional Chinese
    "zh-tw": "chinese_cht",
    "zh-cn": "ch",  # Simplified Chinese
    "ko": "korean",
    "en": "en",
}


def resolve_local_ocr_model(source_language: str, model_id: str | None = None):
    """Resolve the det+rec pair to load for a language, honouring an explicit choice when it fits.

    Imported lazily: provider_config pulls in the whole loader chain, and model_manager is imported
    at worker start-up before any of that is needed.
    """
    from worker.provider_config import get_config_loader  # pylint: disable=import-outside-toplevel

    return get_config_loader().get_local_ocr_catalog().resolve(model_id, source_language)


LANG_TO_EASY: dict = {
    "ja": "ja",
    "zh": "ch_tra",
    "zh-tw": "ch_tra",
    "zh-cn": "ch_sim",
    "ko": "ko",
    "en": "en",
}


class ModelManager:
    """Manager class to cache and evict machine learning OCR model instances."""

    paddle_ocr_available = True
    rapidocr_available = True

    def __init__(self):
        # Cached reader instances
        self.paddle_readers = {}

        # Access timestamps
        self.paddle_last_used = {}

        # RapidOCR engines are separate from PaddleOCR readers so both backends can be
        # exercised in one process during migration and benchmark runs.
        self.rapid_readers = {}
        self.rapid_last_used = {}

        self.lock = threading.Lock()

    def get_paddle_ocr_reader(self, source_language: str, model_id: str | None = None):
        """Return a cached PaddleOCR reader for *source_language* (ISO 639-1 code).

        *model_id* names a choice from the local OCR catalog; when it cannot read the language (or
        is omitted) the catalog picks one that can. Readers are cached per resolved det/rec pair
        rather than per language, so Japanese on PP-OCRv6 and Korean on PP-OCRv5 coexist instead of
        evicting each other.
        """
        if not ModelManager.paddle_ocr_available:
            return None

        resolved = resolve_local_ocr_model(source_language, model_id)
        if resolved is None:
            logger.error(f"[Unified Worker] No local OCR model can read language '{source_language}'.")
            return None

        cache_key = resolved.cache_key

        with self.lock:
            if cache_key not in self.paddle_readers or self.paddle_readers[cache_key] is None:
                try:
                    ocr_device = os.environ.get("PADDLEOCR_DEVICE", "cpu").strip().lower()

                    logger.info(
                        f"[Unified Worker] Initializing PaddleOCR "
                        f"(Det: {resolved.det}, Rec: {resolved.rec}, Device: {ocr_device}, "
                        f"lang='{resolved.language}', model='{resolved.model_id}')..."
                    )
                    from paddleocr import (  # type: ignore
                        PaddleOCR as _PaddleOCR,
                    )  # pylint: disable=import-outside-toplevel

                    # `lang` is deliberately not passed: PaddleOCR ignores it whenever explicit model
                    # names are given (and warns), so the recognition model alone decides which
                    # script can be read. Passing it here only ever produced a false sense that
                    # lang='korean' was doing something while PP-OCRv6_medium_rec transcribed noise.
                    self.paddle_readers[cache_key] = _PaddleOCR(
                        device=ocr_device,
                        text_detection_model_name=resolved.det,
                        text_recognition_model_name=resolved.rec,
                        use_textline_orientation=False,
                        use_doc_unwarping=False,
                        use_doc_orientation_classify=False,
                        enable_mkldnn=False,
                    )
                    logger.info(
                        f"[Unified Worker] PaddleOCR reader ready for lang='{resolved.language}' "
                        f"({resolved.det} + {resolved.rec})."
                    )
                except Exception as err_init_paddle:  # pylint: disable=broad-except
                    logger.error(
                        f"[Unified Worker] Failed to initialize PaddleOCR for "
                        f"lang='{resolved.language}': {err_init_paddle}"
                    )
                    self.paddle_readers[cache_key] = None
                    ModelManager.paddle_ocr_available = False

            if self.paddle_readers.get(cache_key) is not None:
                self.paddle_last_used[cache_key] = time.time()

            return self.paddle_readers.get(cache_key)

    def get_paddle_ocr_detector(self, source_language: str, model_id: str | None = None):
        """Return a cached PaddleOCR reader in detection-only mode (rec=False) for *source_language*.

        Detection is script-agnostic — it finds text boxes, it does not read them — so which pair the
        catalog picks matters far less here than in :meth:`get_paddle_ocr_reader`. It still goes
        through the same resolution so the detector matches the family the page would be read with.
        """
        if not ModelManager.paddle_ocr_available:
            return None

        resolved = resolve_local_ocr_model(source_language, model_id)
        if resolved is None:
            logger.error(f"[Unified Worker] No local OCR detector available for language '{source_language}'.")
            return None

        cache_key = f"{resolved.model_id}:{resolved.det}:det"

        with self.lock:
            if cache_key not in self.paddle_readers or self.paddle_readers[cache_key] is None:
                try:
                    ocr_device = os.environ.get("PADDLEOCR_DEVICE", "cpu").strip().lower()

                    logger.info(
                        f"[Unified Worker] Initializing PaddleOCR Detector "
                        f"(Det: {resolved.det}, Device: {ocr_device}, lang='{resolved.language}')..."
                    )
                    from paddleocr import (  # type: ignore
                        PaddleOCR as _PaddleOCR,
                    )  # pylint: disable=import-outside-toplevel

                    self.paddle_readers[cache_key] = _PaddleOCR(
                        device=ocr_device,
                        text_detection_model_name=resolved.det,
                        use_textline_orientation=False,
                        use_doc_unwarping=False,
                        use_doc_orientation_classify=False,
                        enable_mkldnn=False,
                    )
                    logger.info(f"[Unified Worker] PaddleOCR detector ready for lang='{resolved.language}'.")
                except Exception as err_init_paddle:  # pylint: disable=broad-except
                    logger.error(
                        f"[Unified Worker] Failed to initialize PaddleOCR Detector "
                        f"for lang='{resolved.language}': {err_init_paddle}"
                    )
                    self.paddle_readers[cache_key] = None

            if self.paddle_readers.get(cache_key) is not None:
                self.paddle_last_used[cache_key] = time.time()

            return self.paddle_readers.get(cache_key)

    def get_rapid_ocr_reader(self, source_language: str, use_rec: bool = True):
        """Return a cached RapidOCR/ONNX Runtime reader for one source language.

        The same engine is used for full local OCR and detection-only candidate generation for
        cloud VLM OCR. Model files are downloaded into the worker user's writable cache.
        """
        if not ModelManager.rapidocr_available:
            return None

        version, model_type_name, det_language, rec_language = _rapidocr_route(source_language)
        cache_key = f"{version}:{model_type_name}:{rec_language}:{'ocr' if use_rec else 'det'}"

        with self.lock:
            if cache_key not in self.rapid_readers or self.rapid_readers[cache_key] is None:
                try:
                    from rapidocr import RapidOCR as _RapidOCR  # type: ignore
                    from rapidocr.utils.typings import ModelType, OCRVersion  # type: ignore[reportMissingImports]

                    model_type = ModelType(model_type_name)
                    ocr_version = OCRVersion(version)
                    model_root = os.environ.get(
                        "RAPIDOCR_MODEL_ROOT",
                        os.path.join(os.environ.get("HOME", "/tmp"), ".cache", "rapidocr"),
                    )
                    os.makedirs(model_root, exist_ok=True)

                    logger.info(
                        f"[Unified Worker] Initializing RapidOCR/ONNX Runtime "
                        f"(Det: {ocr_version.value}/{model_type.value}, "
                        f"Rec: {rec_language}, use_rec={use_rec}, lang='{source_language}')..."
                    )
                    self.rapid_readers[cache_key] = _RapidOCR(
                        params={
                            "Global.model_root_dir": model_root,
                            # The worker already downsizes the page before OCR. Avoid a second
                            # page resize while retaining RapidOCR's vertical padding.
                            "Global.use_preprocess_img": False,
                            "Global.use_rec": use_rec,
                            "Global.use_cls": False,
                            # Keep low-confidence candidates available to the worker's existing
                            # filtering and merge logic.
                            "Global.text_score": 0.0,
                            "Det.lang_type": det_language,
                            "Det.model_type": model_type,
                            "Det.ocr_version": ocr_version,
                            "Rec.lang_type": rec_language,
                            "Rec.model_type": model_type,
                            "Rec.ocr_version": ocr_version,
                        }
                    )
                    logger.info(
                        f"[Unified Worker] RapidOCR reader ready for "
                        f"{ocr_version.value}/{model_type.value} ({rec_language}, use_rec={use_rec})."
                    )
                except Exception as err_init_rapidocr:  # pylint: disable=broad-except
                    logger.error(
                        f"[Unified Worker] Failed to initialize RapidOCR for "
                        f"lang='{source_language}': {err_init_rapidocr}"
                    )
                    self.rapid_readers[cache_key] = None
                    ModelManager.rapidocr_available = False

            if self.rapid_readers.get(cache_key) is not None:
                self.rapid_last_used[cache_key] = time.time()

            return self.rapid_readers.get(cache_key)

    def get_rapid_ocr_model_identifier(self, source_language: str, use_rec: bool = True) -> str:
        """Return a stable provenance label for the RapidOCR model selected for a page."""
        version, model_type, _det_language, rec_language = _rapidocr_route(source_language)
        role = rec_language if use_rec else "detector"
        return f"RapidOCR({version}/{model_type}, {role})"

    def unload_expired_models(self, ttl_seconds: float):
        """Unload models that have been idle for longer than *ttl_seconds*."""
        now = time.time()

        with self.lock:
            # Check PaddleOCR readers. Keys identify a det/rec pair, not a language — one language
            # can have several loaded (reader and detector, or two model families).
            for cache_key in list(self.paddle_readers.keys()):
                reader = self.paddle_readers[cache_key]
                if reader is not None:
                    last_used = self.paddle_last_used.get(cache_key, 0.0)
                    if now - last_used > ttl_seconds:
                        logger.info(
                            f"[Model Manager] Unloading PaddleOCR ({cache_key}) "
                            f"due to inactivity (idle for {now - last_used:.1f}s)."
                        )
                        self.paddle_readers[cache_key] = None
                        gc.collect()

            # Check RapidOCR engines.
            for cache_key in list(self.rapid_readers.keys()):
                reader = self.rapid_readers[cache_key]
                if reader is not None:
                    last_used = self.rapid_last_used.get(cache_key, 0.0)
                    if now - last_used > ttl_seconds:
                        logger.info(
                            f"[Model Manager] Unloading RapidOCR ({cache_key}) "
                            f"due to inactivity (idle for {now - last_used:.1f}s)."
                        )
                        self.rapid_readers[cache_key] = None
                        gc.collect()

    def get_loaded_models_status(self, ttl_seconds: float):
        """Return the list of currently loaded models and their eviction timers."""
        now = time.time()
        loaded = []

        with self.lock:
            # PaddleOCR readers
            for cache_key, reader in self.paddle_readers.items():
                if reader is not None:
                    last_used = self.paddle_last_used.get(cache_key, 0.0)
                    remaining = max(0.0, ttl_seconds - (now - last_used))
                    loaded.append(f"PaddleOCR:{cache_key} (unloads in {int(remaining)}s)")

            for cache_key, reader in self.rapid_readers.items():
                if reader is not None:
                    last_used = self.rapid_last_used.get(cache_key, 0.0)
                    remaining = max(0.0, ttl_seconds - (now - last_used))
                    loaded.append(f"RapidOCR:{cache_key} (unloads in {int(remaining)}s)")

            return loaded


# Shared global instance
model_manager = ModelManager()
