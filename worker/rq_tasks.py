import traceback
import requests
from worker.config import CALLBACK_URL, BACKEND_HEADERS
from worker.handlers import (
    process_panel_detection,
    process_ocr,
    process_layout,
    process_translation,
    process_region_redo,
    process_stub,
    process_render,
    process_qa,
)


def check_stale_job(queue_name, job_data):
    if queue_name in ("queue:translation", "queue:ocr", "queue:qa", "queue:layout"):
        image_id = job_data.get("imageId")
        if not image_id:
            return False
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        try:
            res = requests.get(backend_url, headers=BACKEND_HEADERS)
            if res.status_code == 200:
                # If image exists we can proceed. Future logic for specific cancellation can go here.
                return False
            elif res.status_code == 404:
                print(
                    f"[RQ Task] Image {image_id} not found, aborting job.", flush=True
                )
                return True
        except Exception:
            pass
    return False


def process_job_rq(queue_name, job_data):
    try:
        if check_stale_job(queue_name, job_data):
            return

        if queue_name == "queue:panel-detection":
            process_panel_detection(job_data)
        elif queue_name == "queue:ocr":
            process_ocr(job_data)
        elif queue_name == "queue:layout":
            process_layout(job_data)
        elif queue_name == "queue:translation":
            process_translation(job_data)
        elif queue_name == "queue:region-redo":
            process_region_redo(job_data)
        elif queue_name == "queue:render":
            process_render(job_data)
        elif queue_name == "queue:qa":
            process_qa(job_data)
    except Exception as e:
        print(f"[RQ Worker] Error processing job from {queue_name}: {e}", flush=True)
        traceback.print_exc()
        raise e  # Triggers RQ retry
