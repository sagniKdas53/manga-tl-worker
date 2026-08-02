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


def resolve_slot_config(concurrent: int, heavy: int, light: int) -> tuple[int, int, int, list[str]]:
    """Clamp the slot triple into a configuration that can actually dispatch work.

    The arithmetic behind ``MAX_LIGHT_SLOTS``'s default is subtractive, so plausible settings used
    to produce zero or negative light slots — ``CONCURRENT_JOBS=1`` with the default
    ``MAX_HEAVY_SLOTS=1`` yields 0, and ``MAX_HEAVY_SLOTS=3`` with ``CONCURRENT_JOBS=2`` yields -1.
    Nothing validated the result, so the worker answered every light queue with a permanent 429 and
    the pipeline deadlocked with no diagnostic (AUDIT-W6).

    Each tier keeps at least one slot. A tier total above ``concurrent`` is left alone: the global
    cap is a deliberate ceiling on simultaneous work, and the per-tier numbers are reservations
    within it, not an additional budget.

    Returns the clamped ``(concurrent, heavy, light)`` plus warnings describing every adjustment.
    """
    warnings: list[str] = []

    if concurrent < 1:
        warnings.append(f"CONCURRENT_JOBS={concurrent} is below 1; using 1.")
        concurrent = 1
    if heavy < 1:
        warnings.append(f"MAX_HEAVY_SLOTS={heavy} is below 1; using 1. Heavy queues would never be served.")
        heavy = 1
    if light < 1:
        warnings.append(
            f"MAX_LIGHT_SLOTS={light} is below 1; using 1. Light queues would have answered every "
            "dispatch with 429 and stalled the pipeline."
        )
        light = 1
    if heavy + light > concurrent:
        warnings.append(
            f"MAX_HEAVY_SLOTS({heavy}) + MAX_LIGHT_SLOTS({light}) exceeds CONCURRENT_JOBS({concurrent}); "
            "the global cap wins, so the tiers will contend rather than both running full."
        )

    return concurrent, heavy, light, warnings


# Raised from 2/1/1 on 2026-08-02 (AUDIT-W10). The first fully-drained run measured 90.8% of total
# job lifetime as queue wait, essentially all of it in the light tier: four light stages whose
# per-job cost spans three orders of magnitude (layout 0.2s, qa 53.8s) shared a single slot, so a
# layout job waited a 591s median to do 0.2s of work. The light tier was 94.7 s/page against the
# heavy tier's 23.4 s/page — 4x slower — while the heavy slot sat idle 95.9% of the time and worker
# CPU averaged 22.5%. Four light slots brings the two tiers level. Light work is network-bound LLM
# calls, so the extra concurrency costs little CPU.
MAX_CONCURRENT_JOBS = _parse_env_int("CONCURRENT_JOBS", _parse_env_int("CONCURRENT_WORKERS", 5))
MAX_HEAVY_SLOTS = _parse_env_int("MAX_HEAVY_SLOTS", 1)
MAX_LIGHT_SLOTS = _parse_env_int("MAX_LIGHT_SLOTS", MAX_CONCURRENT_JOBS - MAX_HEAVY_SLOTS)

MAX_CONCURRENT_JOBS, MAX_HEAVY_SLOTS, MAX_LIGHT_SLOTS, _SLOT_WARNINGS = resolve_slot_config(
    MAX_CONCURRENT_JOBS, MAX_HEAVY_SLOTS, MAX_LIGHT_SLOTS
)
for _warning in _SLOT_WARNINGS:
    print(f"[Worker Concurrency] {_warning}", flush=True)

REUSE_IDLE_SLOTS = os.environ.get("REUSE_IDLE_SLOTS", "true").strip().lower() == "true"
WORKER_API_SECRET = os.environ.get("WORKER_API_SECRET", "").strip()
WORKER_API_SECRET_FILE = os.environ.get("WORKER_API_SECRET_FILE", "").strip()
# Deliberate, explicitly-named opt-out for running the worker without authentication locally.
# The point of AUDIT-S3 is that a *missing* credential must not be what disables the check.
ALLOW_UNAUTHENTICATED_API = os.environ.get("ALLOW_UNAUTHENTICATED_WORKER_API", "").strip().lower() == "true"

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

if WORKER_API_SECRET_FILE:
    if not os.path.exists(WORKER_API_SECRET_FILE):
        print(
            f"[Worker Concurrency] WORKER_API_SECRET_FILE points at {WORKER_API_SECRET_FILE}, which does not "
            "exist. Check that the secret is mounted into the container.",
            flush=True,
        )
    else:
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
