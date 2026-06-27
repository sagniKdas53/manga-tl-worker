import os
import time
import json
import traceback
import subprocess
from rq import Queue, Retry

from worker.config import redis_client, MODEL_TTL, HEALTH_PORT
from worker.health_server import start_health_server
from worker.model_manager import model_manager
from worker.rq_tasks import process_job_rq


def main():
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
        f"[Unified Worker] Listening to Redis queues: {queues}. Dispatching to RQ with {concurrent_workers} concurrent workers...",
        flush=True,
    )

    # Start RQ workers in the background
    redis_url = f"redis://{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', 6379)}/0"
    worker_procs = []
    for _ in range(concurrent_workers):
        proc = subprocess.Popen(["rq", "worker", "manga_tasks", "--url", redis_url])
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
                except Exception as redis_err:
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
            if job_tuple:
                queue_bytes, job_json = job_tuple
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
        except Exception as e:
            print(f"[Unified Worker] Error in main loop: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()
