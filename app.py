import os
import time
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

from worker.config import redis_client, MODEL_TTL, HEALTH_PORT
from worker.health_server import start_health_server
from worker.model_manager import model_manager
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


def process_job(queue_name, job_data):
    try:
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
        print(
            f"[Unified Worker] Error processing job from {queue_name}: {e}", flush=True
        )
        traceback.print_exc()


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
        f"[Unified Worker] Listening to Redis queues: {queues} with {concurrent_workers} concurrent threads...",
        flush=True,
    )

    pool = ThreadPoolExecutor(max_workers=concurrent_workers)

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
                    f"Queues: {states_str}",
                    flush=True,
                )
                last_status_time = now

            # Listen for new jobs on the queue
            job_tuple = redis_client.blpop(queues, timeout=5)
            if job_tuple:
                queue_bytes, job_json = job_tuple
                queue_name = queue_bytes.decode("utf-8")
                job_data = json.loads(job_json)

                pool.submit(process_job, queue_name, job_data)
        except Exception as e:
            print(f"[Unified Worker] Error in main loop: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()
