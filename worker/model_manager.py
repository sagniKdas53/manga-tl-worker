"""Model caching and manager logic for OCR libraries."""

import os
import gc
import time
import threading

# Configure PaddleOCR environment variables
try:
    os.environ.setdefault("PADDLEX_OFFLINE_MODE", "0")
    os.environ.setdefault("PADDLE_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
except Exception as err_env:  # pylint: disable=broad-except
    print(
        f"[Unified Worker] Failed to set PaddleOCR environment: {err_env}", flush=True
    )


LANG_TO_PADDLE: dict = {
    "ja": "japan",
    "zh": "chinese_cht",  # Traditional Chinese
    "zh-tw": "chinese_cht",
    "zh-cn": "ch",  # Simplified Chinese
    "ko": "korean",
    "en": "en",
}

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

    def get_paddle_ocr_reader(self, source_language: str):
        """Return a cached PaddleOCR reader for *source_language* (ISO 639-1 code)."""
        if not ModelManager.paddle_ocr_available:
            return None

        paddle_lang = LANG_TO_PADDLE.get((source_language or "ja").lower(), "japan")

        with self.lock:
            if (
                paddle_lang not in self.paddle_readers
                or self.paddle_readers[paddle_lang] is None
            ):
                try:
                    det_model = os.environ.get(
                        "PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
                    ).strip()
                    rec_model = os.environ.get(
                        "PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec"
                    ).strip()
                    ocr_device = (
                        os.environ.get("PADDLEOCR_DEVICE", "cpu").strip().lower()
                    )

                    print(
                        f"[Unified Worker] Initializing PaddleOCR "
                        f"(Det: {det_model}, Rec: {rec_model}, Device: {ocr_device}, lang='{paddle_lang}')...",
                        flush=True,
                    )
                    from paddleocr import (
                        PaddleOCR as _PaddleOCR,
                    )  # pylint: disable=import-outside-toplevel

                    self.paddle_readers[paddle_lang] = _PaddleOCR(
                        lang=paddle_lang,
                        device=ocr_device,
                        text_detection_model_name=det_model,
                        text_recognition_model_name=rec_model,
                        use_textline_orientation=False,
                        use_doc_unwarping=False,
                        use_doc_orientation_classify=False,
                        enable_mkldnn=False,
                    )
                    print(
                        f"[Unified Worker] PaddleOCR reader ready for lang='{paddle_lang}'.",
                        flush=True,
                    )
                except Exception as err_init_paddle:  # pylint: disable=broad-except
                    print(
                        f"[Unified Worker] Failed to initialize PaddleOCR "
                        f"for lang='{paddle_lang}': {err_init_paddle}",
                        flush=True,
                    )
                    self.paddle_readers[paddle_lang] = None
                    ModelManager.paddle_ocr_available = False

            if self.paddle_readers.get(paddle_lang) is not None:
                self.paddle_last_used[paddle_lang] = time.time()

            return self.paddle_readers.get(paddle_lang)

    def get_paddle_ocr_detector(self, source_language: str):
        """Return a cached PaddleOCR reader in detection-only mode (rec=False) for *source_language*."""
        if not ModelManager.paddle_ocr_available:
            return None

        paddle_lang = LANG_TO_PADDLE.get((source_language or "ja").lower(), "japan")
        cache_key = f"{paddle_lang}_det"

        with self.lock:
            if (
                cache_key not in self.paddle_readers
                or self.paddle_readers[cache_key] is None
            ):
                try:
                    det_model = os.environ.get(
                        "PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
                    ).strip()
                    ocr_device = (
                        os.environ.get("PADDLEOCR_DEVICE", "cpu").strip().lower()
                    )

                    print(
                        f"[Unified Worker] Initializing PaddleOCR Detector "
                        f"(Det: {det_model}, Device: {ocr_device}, lang='{paddle_lang}')...",
                        flush=True,
                    )
                    from paddleocr import (
                        PaddleOCR as _PaddleOCR,
                    )  # pylint: disable=import-outside-toplevel

                    self.paddle_readers[cache_key] = _PaddleOCR(
                        lang=paddle_lang,
                        device=ocr_device,
                        text_detection_model_name=det_model,
                        use_textline_orientation=False,
                        use_doc_unwarping=False,
                        use_doc_orientation_classify=False,
                        enable_mkldnn=False,
                    )
                    print(
                        f"[Unified Worker] PaddleOCR detector ready for lang='{paddle_lang}'.",
                        flush=True,
                    )
                except Exception as err_init_paddle:  # pylint: disable=broad-except
                    print(
                        f"[Unified Worker] Failed to initialize PaddleOCR Detector "
                        f"for lang='{paddle_lang}': {err_init_paddle}",
                        flush=True,
                    )
                    self.paddle_readers[cache_key] = None

            if self.paddle_readers.get(cache_key) is not None:
                self.paddle_last_used[cache_key] = time.time()

            return self.paddle_readers.get(cache_key)

    def unload_expired_models(self, ttl_seconds: float):
        """Unload models that have been idle for longer than *ttl_seconds*."""
        now = time.time()

        with self.lock:
            # Check PaddleOCR readers
            for paddle_lang in list(self.paddle_readers.keys()):
                reader = self.paddle_readers[paddle_lang]
                if reader is not None:
                    last_used = self.paddle_last_used.get(paddle_lang, 0.0)
                    if now - last_used > ttl_seconds:
                        print(
                            f"[Model Manager] Unloading PaddleOCR ({paddle_lang}) "
                            f"due to inactivity (idle for {now - last_used:.1f}s).",
                            flush=True,
                        )
                        self.paddle_readers[paddle_lang] = None
                        gc.collect()

    def get_loaded_models_status(self, ttl_seconds: float):
        """Return the list of currently loaded models and their eviction timers."""
        now = time.time()
        loaded = []

        with self.lock:
            # PaddleOCR readers
            for paddle_lang, reader in self.paddle_readers.items():
                if reader is not None:
                    last_used = self.paddle_last_used.get(paddle_lang, 0.0)
                    remaining = max(0.0, ttl_seconds - (now - last_used))
                    loaded.append(
                        f"PaddleOCR:{paddle_lang} (unloads in {int(remaining)}s)"
                    )

            return loaded


# Shared global instance
model_manager = ModelManager()
