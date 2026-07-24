import time
import traceback

import requests

from worker.config import BACKEND_HEADERS, CALLBACK_URL
from worker.handlers import (
    process_layout,
    process_ocr,
    process_panel_detection,
    process_qa,
    process_qa_re_ocr,
    process_region_redo,
    process_render,
    process_translation,
)


def check_stale_job(queue_name, job_data):
    image_bound_queues = {
        "queue:panel-detection",
        "queue:ocr",
        "queue:layout",
        "queue:translation",
        "queue:render",
        "queue:qa",
        "queue:qa-re-ocr",
        "queue:region-redo-ocr",
        "queue:region-redo-tl",
    }
    if queue_name in image_bound_queues:
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


def update_job_status(job_id, status, error=None, attempt=None):
    if not job_id:
        return
    try:
        url = CALLBACK_URL.replace("/jobs/callback", f"/jobs/{job_id}/status")
        payload = {"status": status}
        if error:
            payload["error"] = str(error)
        if attempt is not None:
            payload["attempt"] = str(attempt)
        requests.patch(url, json=payload, headers=BACKEND_HEADERS, timeout=5)
    except Exception as e:
        print(f"[RQ Worker] Failed to update job status to {status}: {e}", flush=True)


def process_job_rq(queue_name, job_data):
    job_id = job_data.get("jobId")
    try:
        if check_stale_job(queue_name, job_data):
            update_job_status(job_id, "FAILED", "Stale job")
            return

        if job_id:
            try:
                url = CALLBACK_URL.replace("/jobs/callback", f"/jobs/{job_id}")
                res = requests.get(url, headers=BACKEND_HEADERS, timeout=5)
                if res.status_code == 404:
                    print(
                        f"[RQ Worker] Job {job_id} was deleted/cancelled, skipping.",
                        flush=True,
                    )
                    return
                elif res.status_code == 200:
                    job_status = res.json().get("status")
                    if job_status != "PENDING":
                        print(
                            f"[RQ Worker] Job {job_id} is {job_status} (not PENDING), skipping processing.",
                            flush=True,
                        )
                        return
            except Exception as e:
                print(
                    f"[RQ Worker] Failed to check job status from backend: {e}",
                    flush=True,
                )

        update_job_status(job_id, "PROCESSING")

        if queue_name == "queue:panel-detection":
            process_panel_detection(job_data)
        elif queue_name == "queue:ocr":
            process_ocr(job_data)
        elif queue_name == "queue:layout":
            process_layout(job_data)
        elif queue_name == "queue:translation":
            process_translation(job_data)
        elif queue_name in (
            "queue:region-redo-ocr",
            "queue:region-redo-tl",
            "queue:region-redo",
        ):
            process_region_redo(job_data)
        elif queue_name == "queue:render":
            process_render(job_data)
        elif queue_name == "queue:qa":
            process_qa(job_data)
        elif queue_name == "queue:qa-re-ocr":
            process_qa_re_ocr(job_data)

        update_job_status(job_id, "COMPLETED")
    except Exception as e:
        print(f"[RQ Worker] Error processing job from {queue_name}: {e}", flush=True)
        traceback.print_exc()

        attempt = int(job_data.get("attempt", 1))
        max_attempts = int(job_data.get("maxAttempts", 3))

        if attempt < max_attempts:
            print(
                f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. Retrying in 2 seconds...",
                flush=True,
            )
            time.sleep(2)
            job_data["attempt"] = attempt + 1
            update_job_status(job_id, "PENDING", str(e), attempt + 1)
        else:
            print(
                f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. Max attempts reached.",
                flush=True,
            )
            update_job_status(job_id, "FAILED", str(e), attempt)
