"""Unified workers daemon loop entrypoint."""

import os
import time
import json
import traceback
import subprocess
import redis
from rq import Queue, Retry

from worker.config import redis_client, MODEL_TTL, HEALTH_PORT
from worker.health_server import start_health_server
from worker.model_manager import model_manager
from worker.rq_tasks import process_job_rq


def main():  # pylint: disable=too-many-locals
    """Main daemon loop running worker processes and dispatching Redis tasks."""
    start_time = time.time()

    # Start the daemon HTTP health check server
    start_health_server(HEALTH_PORT)

    queues = [
        "queue:panel-detection",
        "queue:ocr",
        "queue:layout",
        "queue:translation",
        "queue:render",
        "queue:qa",
        "queue:region-redo",
    ]

    concurrent_workers = int(os.environ.get("CONCURRENT_WORKERS", "4"))
    print(
        f"[Unified Worker] Listening to Redis queues: {queues}. "
        f"Dispatching to RQ with {concurrent_workers} concurrent workers...",
        flush=True,
    )

    # Start RQ workers in the background
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_url = f"redis://{redis_host}:{redis_port}/0"
    worker_procs = []
    for _ in range(concurrent_workers):
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            ["rq", "worker", "manga_tasks", "--url", redis_url]
        )
        worker_procs.append(proc)

    rq_queue = Queue("manga_tasks", connection=redis_client)

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

                # Fetch Redis queue lengths
                try:
                    queue_lengths = [f"{q}: {redis_client.llen(q)}" for q in queues]
                    states_str = ", ".join(queue_lengths)
                except redis.RedisError as redis_err:
                    states_str = f"Error fetching queue states ({redis_err})"

                print(
                    f"[Unified Worker Status] Uptime: {uptime_str} | "
                    f"Loaded Models: {loaded_str} | "
                    f"Raw Queues: {states_str}",
                    flush=True,
                )
                last_status_time = now

            # Listen for new jobs on the queue
            job_tuple = redis_client.blpop(queues, timeout=5)
            if isinstance(job_tuple, (list, tuple)) and len(job_tuple) == 2:
                # pylint: disable=unsubscriptable-object
                queue_bytes = job_tuple[0]
                job_json = job_tuple[1]
                queue_name = queue_bytes.decode("utf-8")
                job_data = json.loads(job_json)

                # Dispatch to RQ with exponential backoff
                rq_queue.enqueue(
                    process_job_rq,
                    queue_name,
                    job_data,
                    retry=Retry(max=3, interval=[10, 30, 60]),
                    job_timeout=600,
                )
        except Exception as err_main:  # pylint: disable=broad-except
            print(f"[Unified Worker] Error in main loop: {err_main}", flush=True)
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()
