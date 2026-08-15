"""FastAPI application replacing the legacy BaseHTTPRequestHandler health server."""

import asyncio
import contextlib
import logging
import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

import worker.concurrency as conc
from worker.config import HEALTH_PORT, MODEL_TTL
from worker.model_manager import model_manager
from worker.schemas import JobSubmitRequest

logger = logging.getLogger(__name__)


def _require_api_secret():
    """Refuse to start unauthenticated (AUDIT-S3).

    ``/api/v1/jobs/submit`` mutates the pipeline, so a worker with no configured secret is a
    remote job-submission endpoint open to anything that can reach it. Exiting here — rather than
    serving traffic and hoping the per-request check catches it — makes a broken secret mount an
    obvious crash-loop instead of a silent hole.
    """
    import sys

    if conc.ALLOW_UNAUTHENTICATED_API:
        logger.info(
            "[Worker] ALLOW_UNAUTHENTICATED_WORKER_API=true — every endpoint, including "
            "/api/v1/jobs/submit, is public. Development only."
        )
        return
    if not conc.WORKER_API_SECRET:
        logger.info(
            "[Worker] FATAL: WORKER_API_SECRET is not set. Mount the secret (WORKER_API_SECRET_FILE) "
            "or set ALLOW_UNAUTHENTICATED_WORKER_API=true if you really want an open worker."
        )
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Replaces app.py's manual while-True loop."""
    _require_api_secret()

    # Cleanup old audit cache
    from app import cleanup_audit_cache, seed_models

    cleanup_audit_cache()

    # Seed ML models
    try:
        seed_models()
        conc.set_seeding_complete(True)
    except Exception as e:
        import sys

        logger.error(f"[Worker] Seeding failed, exiting. Error: {e}")
        sys.exit(1)

    logger.info(f"[Worker] Running in HTTP-Push mode. Listening on port {HEALTH_PORT} for ML tasks.")

    # Background maintenance task (model eviction + status logging)
    maintenance_task = asyncio.create_task(_periodic_maintenance())

    yield  # App is running

    # Shutdown
    maintenance_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await maintenance_task


async def _periodic_maintenance():
    """Periodically unload expired models and log status. Replaces app.py's while True loop."""
    import time as _time

    last_status_time = 0.0
    status_interval = 300.0  # 5 minutes

    while True:
        try:
            now = _time.time()
            model_manager.unload_expired_models(MODEL_TTL)

            if now - last_status_time >= status_interval:
                uptime_seconds = now - conc.START_TIME
                hours, remainder = divmod(int(uptime_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                loaded = model_manager.get_loaded_models_status(MODEL_TTL)
                loaded_str = ", ".join(loaded) if loaded else "None"
                logger.info(f"[Worker Status] Uptime: {hours}h {minutes}m {seconds}s | Loaded Models: {loaded_str}")
                last_status_time = now

        except Exception as e:
            logger.error(f"[Worker] Error in maintenance loop: {e}")

        await asyncio.sleep(5)


app = FastAPI(lifespan=lifespan)


def verify_auth(worker_api_secret: str | None = Header(None, alias="WORKER_API_SECRET")):
    """Dependency for endpoints requiring auth.

    Fails closed (AUDIT-S3). The previous form was ``if conc.WORKER_API_SECRET and ...``, so an
    empty secret — which is exactly what an absent or unreadable secret file produces — skipped the
    comparison and opened every endpoint, ``/api/v1/jobs/submit`` included. Absence of a credential
    now denies unless ``ALLOW_UNAUTHENTICATED_WORKER_API=true`` says so deliberately.
    """
    if conc.ALLOW_UNAUTHENTICATED_API:
        return
    if not conc.WORKER_API_SECRET or not secrets.compare_digest(worker_api_secret or "", conc.WORKER_API_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
@app.get("/ping")
async def health():
    """Health check endpoint. Returns 503 while seeding, 200/503 based on Redis status."""
    import time as _time

    if not conc.SEEDING_COMPLETE:
        return JSONResponse(
            status_code=503,
            content={"status": "seeding", "uptime_seconds": int(_time.time() - conc.START_TIME)},
        )

    from worker.config import redis_client

    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
    except Exception:
        redis_status = "disconnected"

    uptime_seconds = _time.time() - conc.START_TIME
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    loaded_models = model_manager.get_loaded_models_status(MODEL_TTL)

    status_code = 200 if redis_status == "connected" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if redis_status == "connected" else "unhealthy",
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "uptime_seconds": int(uptime_seconds),
            "redis": redis_status,
            "loaded_models": loaded_models,
        },
    )


@app.get("/capabilities", dependencies=[Depends(verify_auth)])
async def capabilities():
    """Returns worker capabilities and current load."""
    return {
        "worker_id": conc.WORKER_ID,
        "supported_tasks": [
            "queue:panel-detection",
            "queue:ocr",
            "queue:layout",
            "queue:translation",
            "queue:render",
            "queue:qa",
            "queue:qa-re-ocr",
            "queue:region-redo-ocr",
            "queue:region-redo-tl",
        ],
        "max_concurrent_jobs": conc.MAX_CONCURRENT_JOBS,
        "max_heavy_slots": conc.MAX_HEAVY_SLOTS,
        "max_light_slots": conc.MAX_LIGHT_SLOTS,
        "reuse_idle_slots": conc.REUSE_IDLE_SLOTS,
        "active_jobs": conc.ACTIVE_JOBS,
        "active_heavy_jobs": conc.ACTIVE_HEAVY_JOBS,
        "active_light_jobs": conc.ACTIVE_LIGHT_JOBS,
        "overflow_light_jobs": max(0, conc.ACTIVE_LIGHT_JOBS - conc.MAX_LIGHT_SLOTS),
    }


@app.post("/api/v1/jobs/submit", status_code=202, dependencies=[Depends(verify_auth)])
async def submit_job(req: JobSubmitRequest):
    """Accept a job for async processing."""
    queue_name = req.queue_name
    job_data = req.job_data.model_dump(exclude_unset=True)
    is_heavy = queue_name in conc.HEAVY_QUEUES

    with conc.ACTIVE_JOBS_LOCK:
        if conc.ACTIVE_JOBS >= conc.MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Too Many Requests: Global concurrency limit reached")

        if is_heavy:
            if conc.ACTIVE_HEAVY_JOBS >= conc.MAX_HEAVY_SLOTS:
                raise HTTPException(status_code=429, detail="Too Many Requests: Heavy job slot occupied")
            conc.ACTIVE_HEAVY_JOBS += 1
        else:
            if conc.ACTIVE_LIGHT_JOBS >= conc.MAX_LIGHT_SLOTS and not (
                conc.REUSE_IDLE_SLOTS and conc.ACTIVE_JOBS < conc.MAX_CONCURRENT_JOBS
            ):
                raise HTTPException(status_code=429, detail="Too Many Requests: Light job slot occupied")
            conc.ACTIVE_LIGHT_JOBS += 1

        conc.ACTIVE_JOBS = conc.ACTIVE_HEAVY_JOBS + conc.ACTIVE_LIGHT_JOBS

    # Spawn job in background thread
    t = threading.Thread(target=conc.run_job_async, args=(queue_name, job_data), daemon=True)
    t.start()

    return {"status": "accepted"}
