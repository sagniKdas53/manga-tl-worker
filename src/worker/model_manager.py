"""Model caching and manager logic for OCR libraries."""

import gc
import logging
import os
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

    def __init__(self):
        # Cached reader instances
        self.paddle_readers = {}

        # Access timestamps
        self.paddle_last_used = {}

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

            return loaded


# Shared global instance
model_manager = ModelManager()
