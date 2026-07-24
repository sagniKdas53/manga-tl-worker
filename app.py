"""Worker entrypoint — launches the FastAPI application via uvicorn."""

import glob
import os
import time


def seed_models():
    """Verify and seed the required ML models on startup."""
    print("[Worker] Seeding models...", flush=True)

    from worker.services.bubble_detector import get_ort_session

    try:
        print("[Worker] Verifying YOLO bubble detector model...", flush=True)
        get_ort_session()
        print("[Worker] YOLO bubble detector model verified successfully.", flush=True)
    except Exception as e:
        print(f"[Worker] Critical Error: YOLO model verification failed: {e}", flush=True)
        raise e

    disable_local_ocr = os.environ.get("DISABLE_LOCAL_OCR", "").strip().lower() in ("true", "1", "yes")
    if not disable_local_ocr:
        try:
            from worker.model_manager import model_manager

            print("[Worker] Seeding PaddleOCR default Japanese models...", flush=True)
            model_manager.get_paddle_ocr_reader("ja")
            print("[Worker] PaddleOCR default Japanese models seeded successfully.", flush=True)
        except Exception as e:
            print(f"[Worker] Critical Error: PaddleOCR seeding failed: {e}", flush=True)
            raise e


def cleanup_audit_cache():
    from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR

    if ENABLE_QA_AUDIT_CACHE:
        print("[Worker] Cleaning up old QA audit cache files...", flush=True)
        try:
            now = time.time()
            max_age = 24 * 3600
            if os.path.exists(QA_AUDIT_CACHE_DIR):
                files = glob.glob(os.path.join(QA_AUDIT_CACHE_DIR, "*.jpg"))
                count = sum(
                    1 for f in files if os.path.isfile(f) and (now - os.path.getmtime(f)) > max_age and not os.remove(f)
                )
                print(f"[Worker] Cleaned up {count} old files in {QA_AUDIT_CACHE_DIR}.", flush=True)
        except Exception as e:
            print(f"[Worker] Error cleaning up QA audit cache: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn

    from worker.config import HEALTH_PORT

    uvicorn.run("worker.main:app", host="0.0.0.0", port=HEALTH_PORT)
