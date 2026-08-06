import traceback

import requests
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

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
            # AUDIT-W7: HEAD, not GET — all we read is the status code, and the GET handler builds
            # a presigned URL plus every panel, region and layer for the image before we throw it
            # away. And a timeout, which this call alone was missing: without one a wedged backend
            # holds a worker slot open indefinitely.
            res = requests.head(backend_url, headers=BACKEND_HEADERS, timeout=5)
            if res.status_code == 200:
                # If image exists we can proceed. Future logic for specific cancellation can go here.
                return False
            elif res.status_code == 404:
                print(f"[RQ Task] Image {image_id} not found, aborting job.", flush=True)
                return True
        except Exception:
            pass
    return False


class StatusUpdateFailed(Exception):
    """A job-status PATCH that failed in a way another attempt could fix."""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(StatusUpdateFailed),
    reraise=True,
)
def _patch_job_status(url, payload):
    """PATCH the job status, retrying anything that looks transient.

    AUDIT-P6: this used to be a single unguarded call whose exception was printed and dropped.
    Nothing else tells the backend a job finished — the results callbacks write results, not
    status, and only the empty-OCR branch sets COMPLETED itself — so a PATCH lost to one socket
    timeout leaves the row PROCESSING until the stale sweeper requeues it ten minutes later and
    the whole stage runs again, on top of results that already landed. Four attempts cost at most
    about 27s of a worker slot; the duplicate OCR or translation pass they prevent costs minutes.
    """
    try:
        res = requests.patch(url, json=payload, headers=BACKEND_HEADERS, timeout=5)
    except requests.exceptions.RequestException as e:
        raise StatusUpdateFailed(f"transport error: {e}") from e

    # requests does not raise on an error status, so without these checks a 500 loses the update
    # exactly as silently as the swallowed timeout did — and without even an exception to print.
    if res.status_code == 404:
        # The row is gone: deleted or cancelled while the job ran. Nothing to update, and no
        # number of retries will bring it back.
        print("[RQ Worker] Job status PATCH returned 404 — job no longer exists.", flush=True)
        return
    if res.status_code >= 500 or res.status_code in (408, 429):
        raise StatusUpdateFailed(f"backend returned {res.status_code}")
    if res.status_code >= 400:
        # A rejected payload does not heal on retry; say so once rather than spending the budget.
        print(
            f"[RQ Worker] Job status PATCH rejected with {res.status_code}: {res.text}",
            flush=True,
        )


def update_job_status(job_id, status, error=None, attempt=None):
    if not job_id:
        return
    url = CALLBACK_URL.replace("/jobs/callback", f"/jobs/{job_id}/status")
    payload = {"status": status}
    if error:
        payload["error"] = str(error)
    if attempt is not None:
        payload["attempt"] = str(attempt)
    try:
        _patch_job_status(url, payload)
    except Exception as e:
        print(
            f"[RQ Worker] Failed to update job {job_id} status to {status} after retries: {e} — "
            f"the backend will hold it PROCESSING until the stale sweeper requeues it",
            flush=True,
        )


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
                f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. "
                f"Marking as PENDING for retry by backend.",
                flush=True,
            )
            update_job_status(job_id, "PENDING", str(e), attempt + 1)
        else:
            print(
                f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. Max attempts reached.",
                flush=True,
            )
            update_job_status(job_id, "FAILED", str(e), attempt)
