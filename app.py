"""Unified workers daemon loop entrypoint."""

import os
import time

from worker.config import HEALTH_PORT, MODEL_TTL
from worker.health_server import set_seeding_complete, start_health_server
from worker.model_manager import model_manager


def seed_models():
    """Verify and seed the required ML models on startup."""
    print("[Unified Worker] Seeding models...", flush=True)

    # 1. Verify YOLO model is present and load ONNX Session
    from worker.services.bubble_detector import get_ort_session

    try:
        print("[Unified Worker] Verifying YOLO bubble detector model...", flush=True)
        get_ort_session()
        print(
            "[Unified Worker] YOLO bubble detector model verified successfully.",
            flush=True,
        )
    except Exception as e:
        print(
            f"[Unified Worker] Critical Error: YOLO model verification failed: {e}",
            flush=True,
        )
        raise e

    # 2. Pre-initialize/download PaddleOCR models if local OCR is enabled
    disable_local_ocr = os.environ.get("DISABLE_LOCAL_OCR", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if not disable_local_ocr:
        try:
            print(
                "[Unified Worker] Seeding PaddleOCR default Japanese models...",
                flush=True,
            )
            model_manager.get_paddle_ocr_reader("ja")
            print(
                "[Unified Worker] PaddleOCR default Japanese models seeded successfully.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Unified Worker] Critical Error: PaddleOCR seeding failed: {e}",
                flush=True,
            )
            raise e



def cleanup_audit_cache():
    import glob
    import os
    import time
    from worker.config import QA_AUDIT_CACHE_DIR, ENABLE_QA_AUDIT_CACHE

    if ENABLE_QA_AUDIT_CACHE:
        print("[Unified Worker] Cleaning up old QA audit cache files...", flush=True)
        try:
            now = time.time()
            max_age = 24 * 3600  # 24 hours
            if os.path.exists(QA_AUDIT_CACHE_DIR):
                # QA audit saves as .jpg
                files = glob.glob(os.path.join(QA_AUDIT_CACHE_DIR, "*.jpg"))
                count = 0
                for f in files:
                    if os.path.isfile(f):
                        if (now - os.path.getmtime(f)) > max_age:
                            os.remove(f)
                            count += 1
                print(f"[Unified Worker] Cleaned up {count} old files in {QA_AUDIT_CACHE_DIR}.", flush=True)
        except Exception as e:
            print(f"[Unified Worker] Error cleaning up QA audit cache: {e}", flush=True)

def main():  # pylint: disable=too-many-locals
    """Main daemon loop running worker processes and dispatching Redis tasks."""
    start_time = time.time()

    # Start the daemon HTTP health check server
    start_health_server(HEALTH_PORT)

    # Cleanup old audit cache if enabled
    cleanup_audit_cache()

    # Seed models on startup
    try:
        seed_models()
        set_seeding_complete(True)
    except Exception as e:
        print(f"[Unified Worker] Seeding failed, exiting. Error: {e}", flush=True)
        import sys

        sys.exit(1)

    print(
        f"[Unified Worker] Running in HTTP-Push mode. Listening on port {HEALTH_PORT} for ML tasks.",
        flush=True,
    )

    last_status_time = 0.0
    status_interval = 300.0  # 5 minutes in seconds

    while True:
        try:
            now = time.time()

            # Periodically unload expired models (TTL checks)
            model_manager.unload_expired_models(MODEL_TTL)

            # Periodically print general status (uptime, loaded models, queue states)
            if now - last_status_time >= status_interval:
                uptime_seconds = now - start_time
                hours, remainder = divmod(int(uptime_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours}h {minutes}m {seconds}s"

                # Fetch currently loaded models
                loaded = model_manager.get_loaded_models_status(MODEL_TTL)
                loaded_str = ", ".join(loaded) if loaded else "None"

                print(
                    f"[Unified Worker Status] Uptime: {uptime_str} | Loaded Models: {loaded_str}",
                    flush=True,
                )
                last_status_time = now

            time.sleep(5)

        except Exception as err_main:  # pylint: disable=broad-except
            print(f"[Unified Worker] Error in main loop: {err_main}", flush=True)
            import traceback

            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
