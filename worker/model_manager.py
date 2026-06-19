import os
import gc
import time
from manga_ocr import MangaOcr

# Configure PaddleOCR import and environment variables
PaddleOCR = None
try:
    print("[Unified Worker] Importing PaddleOCR...", flush=True)
    os.environ["PADDLEX_OFFLINE_MODE"] = "1"
    os.environ["PADDLE_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

    from paddleocr import PaddleOCR as _PaddleOCR

    PaddleOCR = _PaddleOCR
    print(
        "[Unified Worker] PaddleOCR imported successfully (readers will be initialized on first use per language).",
        flush=True,
    )
except Exception as e:
    print(f"[Unified Worker] Failed to import PaddleOCR: {e}", flush=True)


LANG_TO_PADDLE: dict = {
    "ja": "japan",
    "zh": "chinese_cht",  # Traditional Chinese
    "zh-tw": "chinese_cht",
    "zh-cn": "ch",  # Simplified Chinese
    "ko": "korean",
    "en": "en",
}


import threading


class ModelManager:
    def __init__(self):
        # Cached reader instances
        self.paddle_readers = {}
        self.easy_reader = None
        self.manga_reader = None

        # Access timestamps
        self.paddle_last_used = {}
        self.easy_last_used = 0.0
        self.manga_last_used = 0.0

        self.lock = threading.Lock()

    def get_paddle_ocr_reader(self, source_language: str):
        """Return a cached PaddleOCR reader for *source_language* (ISO 639-1 code)."""
        if PaddleOCR is None:
            return None

        paddle_lang = LANG_TO_PADDLE.get((source_language or "ja").lower(), "japan")

        with self.lock:
            if (
                paddle_lang not in self.paddle_readers
                or self.paddle_readers[paddle_lang] is None
            ):
                try:
                    print(
                        f"[Unified Worker] Initializing PaddleOCR (PP-OCRv5 Mobile, lang='{paddle_lang}')...",
                        flush=True,
                    )
                    self.paddle_readers[paddle_lang] = PaddleOCR(
                        lang=paddle_lang,
                        device="cpu",
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name="PP-OCRv5_mobile_rec",
                        use_textline_orientation=False,
                        use_doc_unwarping=False,
                        use_doc_orientation_classify=False,
                        enable_mkldnn=False,
                    )
                    print(
                        f"[Unified Worker] PaddleOCR reader ready for lang='{paddle_lang}'.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[Unified Worker] Failed to initialize PaddleOCR for lang='{paddle_lang}': {e}",
                        flush=True,
                    )
                    self.paddle_readers[paddle_lang] = None

            if self.paddle_readers.get(paddle_lang) is not None:
                self.paddle_last_used[paddle_lang] = time.time()

            return self.paddle_readers.get(paddle_lang)

    def get_easy_ocr_reader(self):
        """Return a cached EasyOCR reader."""
        with self.lock:
            if self.easy_reader is None:
                try:
                    print("[Unified Worker] Importing EasyOCR...", flush=True)
                    import easyocr

                    print(
                        "[Unified Worker] Initializing EasyOCR Reader (ja, en)...",
                        flush=True,
                    )
                    self.easy_reader = easyocr.Reader(["ja", "en"], gpu=False)
                except Exception as e:
                    print(
                        f"[Unified Worker] Failed to initialize EasyOCR: {e}",
                        flush=True,
                    )
                    self.easy_reader = None

            if self.easy_reader is not None:
                self.easy_last_used = time.time()

            return self.easy_reader

    def get_manga_ocr_reader(self):
        """Return a cached MangaOCR reader."""
        with self.lock:
            if self.manga_reader is None:
                try:
                    print(
                        "[Unified Worker] Initializing MangaOCR Reader...", flush=True
                    )
                    force_cpu = os.environ.get(
                        "MANGA_OCR_FORCE_CPU", "true"
                    ).lower() in (
                        "true",
                        "1",
                        "t",
                    )
                    use_local = os.environ.get(
                        "MANGA_OCR_USE_LOCAL", "false"
                    ).lower() in (
                        "true",
                        "1",
                        "t",
                    )
                    model_path = os.environ.get(
                        "MANGA_OCR_MODEL_PATH", "kha-white/manga-ocr-base"
                    )

                    pretrained_path = "kha-white/manga-ocr-base"
                    if use_local:
                        resolved_path = model_path
                        if not os.path.exists(
                            os.path.join(resolved_path, "config.json")
                        ):
                            hub_dir = os.path.join(
                                model_path,
                                "hub/models--kha-white--manga-ocr-base/snapshots",
                            )
                            if os.path.exists(hub_dir):
                                snapshots = [
                                    os.path.join(hub_dir, d)
                                    for d in os.listdir(hub_dir)
                                    if os.path.isdir(os.path.join(hub_dir, d))
                                ]
                                if snapshots:
                                    for s in snapshots:
                                        if os.path.exists(
                                            os.path.join(s, "config.json")
                                        ):
                                            resolved_path = s
                                            break
                        pretrained_path = resolved_path
                        print(
                            f"[Unified Worker] Using local cached MangaOCR model resolved to: {pretrained_path}",
                            flush=True,
                        )

                    if force_cpu:
                        print(
                            f"[Unified Worker] Forcing CPU for MangaOCR (model={pretrained_path})...",
                            flush=True,
                        )
                        self.manga_reader = MangaOcr(
                            pretrained_model_name_or_path=pretrained_path,
                            force_cpu=True,
                        )
                    else:
                        try:
                            self.manga_reader = MangaOcr(
                                pretrained_model_name_or_path=pretrained_path
                            )
                        except Exception as init_err:
                            print(
                                f"[Unified Worker] Failed to initialize MangaOCR with default settings: {init_err}. Retrying with force_cpu=True...",
                                flush=True,
                            )
                            self.manga_reader = MangaOcr(
                                pretrained_model_name_or_path=pretrained_path,
                                force_cpu=True,
                            )
                except Exception as e:
                    print(
                        f"[Unified Worker] Failed to initialize MangaOCR: {e}.",
                        flush=True,
                    )
                    self.manga_reader = None

            if self.manga_reader is not None:
                self.manga_last_used = time.time()

            return self.manga_reader

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
                            f"[Model Manager] Unloading PaddleOCR ({paddle_lang}) due to inactivity (idle for {now - last_used:.1f}s).",
                            flush=True,
                        )
                        self.paddle_readers[paddle_lang] = None
                        gc.collect()

            # Check EasyOCR reader
            if self.easy_reader is not None:
                if now - self.easy_last_used > ttl_seconds:
                    print(
                        f"[Model Manager] Unloading EasyOCR due to inactivity (idle for {now - self.easy_last_used:.1f}s).",
                        flush=True,
                    )
                    self.easy_reader = None
                    gc.collect()

            # Check MangaOCR reader
            if self.manga_reader is not None:
                if now - self.manga_last_used > ttl_seconds:
                    print(
                        f"[Model Manager] Unloading MangaOCR due to inactivity (idle for {now - self.manga_last_used:.1f}s).",
                        flush=True,
                    )
                    self.manga_reader = None
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

            # EasyOCR
            if self.easy_reader is not None:
                remaining = max(0.0, ttl_seconds - (now - self.easy_last_used))
                loaded.append(f"EasyOCR (unloads in {int(remaining)}s)")

            # MangaOCR
            if self.manga_reader is not None:
                remaining = max(0.0, ttl_seconds - (now - self.manga_last_used))
                loaded.append(f"MangaOCR (unloads in {int(remaining)}s)")

            return loaded


# Shared global instance
model_manager = ModelManager()
