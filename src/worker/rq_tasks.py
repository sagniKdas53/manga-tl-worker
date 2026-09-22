import contextvars
import logging
import re
import threading
import time
from datetime import datetime, timezone

import requests
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

from worker.config import (
    CALLBACK_URL,
    backend_headers,
    reset_stage,
    reset_trace_id,
    set_stage,
    set_trace_id,
)
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
from worker.job_attempt import AttemptExpired, bind_attempt, current_attempt
from worker.utils.rate_limit import reset_job_costs

logger = logging.getLogger(__name__)


def check_stale_job(queue_name, job_data):
    image_bound_queues = {
        "queue:panel-detection",
        "queue:ocr",
        "queue:cleanup",
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
            res = requests.head(backend_url, headers=backend_headers(), timeout=5)
            if res.status_code == 200:
                # If image exists we can proceed. Future logic for specific cancellation can go here.
                return False
            elif res.status_code == 404:
                logger.error(f"[RQ Task] Image {image_id} not found, aborting job.")
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
        res = requests.patch(url, json=payload, headers=backend_headers(), timeout=5)
    except requests.exceptions.RequestException as e:
        raise StatusUpdateFailed(f"transport error: {e}") from e

    # requests does not raise on an error status, so without these checks a 500 loses the update
    # exactly as silently as the swallowed timeout did — and without even an exception to print.
    if res.status_code == 404:
        # The row is gone: deleted or cancelled while the job ran. Nothing to update, and no
        # number of retries will bring it back.
        logger.info("[RQ Worker] Job status PATCH returned 404 — job no longer exists.")
        return False
    if res.status_code >= 500 or res.status_code in (408, 429):
        raise StatusUpdateFailed(f"backend returned {res.status_code}")
    if res.status_code >= 400:
        # A rejected payload does not heal on retry; say so once rather than spending the budget.
        logger.error(f"[RQ Worker] Job status PATCH rejected with {res.status_code}: {res.text}")
        return False
    return 200 <= res.status_code < 300


def update_job_status(job_id, status, error=None, attempt=None, *, heartbeat=False):
    if not job_id:
        return False
    url = CALLBACK_URL.replace("/jobs/callback", f"/jobs/{job_id}/status")
    payload = {"status": status}
    if error:
        payload["error"] = str(error)
    if attempt is not None:
        payload["attempt"] = str(attempt)
    if heartbeat:
        payload["heartbeat"] = "true"
        authority = current_attempt.get()
        if authority is not None:
            payload["progress"] = str(authority.progress)
    try:
        return _patch_job_status(url, payload)
    except Exception as e:
        logger.error(
            f"[RQ Worker] Failed to update job {job_id} status to {status} after retries: {e} — "
            f"the backend will hold it PROCESSING until the stale sweeper requeues it"
        )
        return False


class JobHeartbeat:
    """Renew only while the finite attempt is alive, independently of blocked inference."""

    interval = 30.0

    def __init__(self, job_id):
        self.job_id = job_id
        self.stop = threading.Event()
        context = contextvars.copy_context()
        self.thread = threading.Thread(target=context.run, args=(self.run,), daemon=True)

    def run(self):
        while not self.stop.wait(self.interval):
            authority = current_attempt.get()
            if authority is None:
                return
            try:
                authority.check()
                if not update_job_status(self.job_id, "PROCESSING", heartbeat=True):
                    authority.revoked.set()
                    return
            except AttemptExpired:
                authority.revoked.set()
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)


def _seconds_since(iso_timestamp) -> float | None:
    if not isinstance(iso_timestamp, str) or not iso_timestamp:
        return None
    # The backend (Rust chrono) writes nanosecond fractions; fromisoformat takes at most six digits.
    normalised = re.sub(r"(\.\d{6})\d+", r"\1", iso_timestamp.replace("Z", "+00:00"))
    try:
        created = datetime.fromisoformat(normalised)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)  # noqa: UP017
    now = datetime.now(timezone.utc)  # noqa: UP017
    return max(0.0, (now - created).total_seconds())


def process_job_rq(queue_name, job_data):
    job_id = job_data.get("jobId")
    authority_token = bind_attempt(job_data)
    heartbeat = None
    # Every job the worker runs comes through here, which makes this the one place the pipeline's
    # trace id needs binding. The backend has been sending it in the payload as "traceId" all along;
    # from here it lands on every log line this job produces (via the formatter's %(trace)s) and on
    # every backend call it makes (via backend_headers), so one page's six stages share a single
    # greppable string across both containers.
    trace_token = set_trace_id(job_data.get("traceId"))
    # The queue name is the stage, so every job labels its own spending with no mapping table and
    # no handler signature change: "queue:region-redo-ocr" -> "region-redo-ocr". Bound beside the
    # trace id because this is the one place every job passes through, and chunk workers inherit it
    # through the copy_context().run submissions already in place for the cost list.
    stage_token = set_stage(queue_name.removeprefix("queue:"))
    # Same argument as the trace id, and the same one place to do it: cost records accumulate per
    # job, so the list is bound here rather than reset inside each handler. Handler-level resets
    # against a shared global meant a job starting mid-flight discarded another job's costs.
    reset_job_costs()
    stage = queue_name.removeprefix("queue:")
    attempt = int(job_data.get("attempt", 1))
    max_attempts = int(job_data.get("maxAttempts", 3))
    started = time.perf_counter()
    # Time since the backend enqueued this job (createdAt). On attempt 1 that is pure queue wait;
    # on a re-dispatched attempt it also contains the earlier attempt(s) -- either way it is the
    # part of a page's wall time that this worker did not spend working on it.
    queued_s = _seconds_since(job_data.get("createdAt"))
    logger.info(
        f"[RQ Worker] Job {job_id} ({stage}) started, attempt {attempt}/{max_attempts}"
        + (f", {queued_s:.0f}s since enqueue" if queued_s is not None else "")
    )
    try:
        if check_stale_job(queue_name, job_data):
            update_job_status(job_id, "FAILED", "Stale job")
            return

        if job_id:
            try:
                url = CALLBACK_URL.replace("/jobs/callback", f"/jobs/{job_id}")
                res = requests.get(url, headers=backend_headers(), timeout=5)
                if res.status_code == 404:
                    logger.warning(f"[RQ Worker] Job {job_id} was deleted/cancelled, skipping.")
                    return
                elif res.status_code == 200:
                    job_status = res.json().get("status")
                    if job_status != "PENDING":
                        logger.warning(f"[RQ Worker] Job {job_id} is {job_status} (not PENDING), skipping processing.")
                        return
            except Exception as e:
                logger.error(f"[RQ Worker] Failed to check job status from backend: {e}")
                return

        if not update_job_status(job_id, "PROCESSING"):
            return
        heartbeat = JobHeartbeat(job_id)
        heartbeat.thread.start()

        if queue_name == "queue:panel-detection":
            process_panel_detection(job_data)
        elif queue_name == "queue:ocr":
            process_ocr(job_data)
        elif queue_name == "queue:cleanup":
            from worker.handlers.cleanup import process_cleanup

            process_cleanup(job_data)
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

        if not update_job_status(job_id, "COMPLETED"):
            logger.warning("[RQ Worker] Job %s completion was not accepted", job_id)
            return
        logger.info(f"[RQ Worker] Job {job_id} ({stage}) completed in {time.perf_counter() - started:.1f}s")
    except Exception as e:
        # logger.exception attaches the traceback to the log record, so it goes through the same
        # handler as everything else and carries the trace id. traceback.print_exc() wrote straight
        # to stderr: unlevelled, uncorrelated, and invisible to any level setting.
        logger.exception(
            f"[RQ Worker] Error processing job from {queue_name} after {time.perf_counter() - started:.1f}s"
        )

        if attempt < max_attempts:
            logger.error(
                f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. "
                f"Marking as PENDING for retry by backend."
            )
            update_job_status(job_id, "PENDING", str(e), attempt + 1)
        else:
            logger.error(f"[RQ Worker] Job {job_id} failed on attempt {attempt}/{max_attempts}. Max attempts reached.")
            update_job_status(job_id, "FAILED", str(e), attempt)
    finally:
        if heartbeat is not None:
            heartbeat.close()
        current_attempt.reset(authority_token)
        # Jobs run concurrently on reused threads; an id left bound would label the next unrelated
        # job's output with this one's pipeline.
        reset_trace_id(trace_token)
        reset_stage(stage_token)
