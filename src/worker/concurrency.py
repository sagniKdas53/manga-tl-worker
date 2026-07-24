"""Concurrency state management and job slot allocation for the worker."""

import os
import platform
import threading
import time

from worker.rq_tasks import process_job_rq

START_TIME = time.time()
SEEDING_COMPLETE = False

ACTIVE_JOBS = 0
ACTIVE_HEAVY_JOBS = 0
ACTIVE_LIGHT_JOBS = 0
ACTIVE_JOBS_LOCK = threading.Lock()


def _parse_env_int(key: str, default_val: int) -> int:
    val = os.environ.get(key, "").strip()
    if not val:
        return default_val
    try:
        return int(val)
    except ValueError:
        return default_val


MAX_CONCURRENT_JOBS = _parse_env_int("CONCURRENT_JOBS", _parse_env_int("CONCURRENT_WORKERS", 2))
MAX_HEAVY_SLOTS = _parse_env_int("MAX_HEAVY_SLOTS", 1)
MAX_LIGHT_SLOTS = _parse_env_int("MAX_LIGHT_SLOTS", MAX_CONCURRENT_JOBS - MAX_HEAVY_SLOTS)
REUSE_IDLE_SLOTS = os.environ.get("REUSE_IDLE_SLOTS", "true").strip().lower() == "true"
WORKER_API_SECRET = os.environ.get("WORKER_API_SECRET", "").strip()
WORKER_API_SECRET_FILE = os.environ.get("WORKER_API_SECRET_FILE", "").strip()

HEAVY_QUEUES = {
    "queue:panel-detection",
    "queue:ocr",
    "queue:qa-re-ocr",
    "queue:region-redo-ocr",
}

LIGHT_QUEUES = {
    "queue:layout",
    "queue:translation",
    "queue:render",
    "queue:qa",
    "queue:region-redo-tl",
}

if WORKER_API_SECRET_FILE and os.path.exists(WORKER_API_SECRET_FILE):
    try:
        with open(WORKER_API_SECRET_FILE) as f:
            WORKER_API_SECRET = f.read().strip()
    except Exception as e:
        print(f"[Worker Concurrency] Failed to read WORKER_API_SECRET_FILE: {e}", flush=True)

WORKER_ID = os.environ.get("WORKER_ID", platform.node())


def set_seeding_complete(complete: bool):
    global SEEDING_COMPLETE
    SEEDING_COMPLETE = complete


def run_job_async(queue_name: str, job_data: dict):
    global ACTIVE_JOBS, ACTIVE_HEAVY_JOBS, ACTIVE_LIGHT_JOBS
    try:
        process_job_rq(queue_name, job_data)
    except Exception as e:
        print(f"[Worker Concurrency] Async job execution failed: {e}", flush=True)
    finally:
        with ACTIVE_JOBS_LOCK:
            if queue_name in HEAVY_QUEUES:
                ACTIVE_HEAVY_JOBS = max(0, ACTIVE_HEAVY_JOBS - 1)
            else:
                ACTIVE_LIGHT_JOBS = max(0, ACTIVE_LIGHT_JOBS - 1)
            ACTIVE_JOBS = ACTIVE_HEAVY_JOBS + ACTIVE_LIGHT_JOBS
