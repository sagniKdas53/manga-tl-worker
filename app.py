"""Worker entrypoint — launches the FastAPI application via uvicorn."""

import glob
import logging
import os
import time

logger = logging.getLogger(__name__)


def seed_models():
    """Verify and seed the required ML models on startup."""
    logger.info("[Worker] Seeding models...")

    from worker.services.bubble_detector import get_ort_session

    try:
        logger.info("[Worker] Verifying YOLO bubble detector model...")
        get_ort_session()
        logger.info("[Worker] YOLO bubble detector model verified successfully.")
    except Exception as e:
        logger.error(f"[Worker] Critical Error: YOLO model verification failed: {e}")
        raise e

    disable_local_ocr = os.environ.get("DISABLE_LOCAL_OCR", "").strip().lower() in ("true", "1", "yes")
    if not disable_local_ocr:
        try:
            from worker.model_manager import model_manager

            logger.info("[Worker] Seeding PaddleOCR default Japanese models...")
            model_manager.get_paddle_ocr_reader("ja")
            logger.info("[Worker] PaddleOCR default Japanese models seeded successfully.")
        except Exception as e:
            logger.error(f"[Worker] Critical Error: PaddleOCR seeding failed: {e}")
            raise e


def cleanup_audit_cache():
    from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR

    if ENABLE_QA_AUDIT_CACHE:
        logger.info("[Worker] Cleaning up old QA audit cache files...")
        try:
            now = time.time()
            max_age = 24 * 3600
            if os.path.exists(QA_AUDIT_CACHE_DIR):
                files = glob.glob(os.path.join(QA_AUDIT_CACHE_DIR, "*.jpg"))
                # AUDIT-Q3: this was a `sum(1 for f in files if ... and not os.remove(f))` — the
                # deletion happened as a side effect inside the generator, relying on os.remove
                # returning None. Beyond being unreadable, one raising unlink aborted the whole
                # sweep: the remaining files were never considered and the count never printed.
                # Per-file, so a file that vanishes underneath us costs one file, not the sweep.
                count = 0
                for f in files:
                    try:
                        if os.path.isfile(f) and (now - os.path.getmtime(f)) > max_age:
                            os.remove(f)
                            count += 1
                    except OSError as e:
                        logger.error(f"[Worker] Could not remove QA audit cache file {f}: {e}")
                logger.info(f"[Worker] Cleaned up {count} old files in {QA_AUDIT_CACHE_DIR}.")
        except Exception as e:
            logger.error(f"[Worker] Error cleaning up QA audit cache: {e}")


if __name__ == "__main__":
    import uvicorn

    from worker.config import HEALTH_PORT

    uvicorn.run("worker.main:app", host="0.0.0.0", port=HEALTH_PORT)
