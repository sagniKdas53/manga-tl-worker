import json
import time
import threading
import os
import platform
from http.server import BaseHTTPRequestHandler
from worker.config import redis_client, MODEL_TTL
from worker.model_manager import model_manager

# Import process_job_rq for executing tasks
from worker.rq_tasks import process_job_rq

# Global reference to start time
START_TIME = time.time()

# Seeding completion status
SEEDING_COMPLETE = False

# Worker State
ACTIVE_JOBS = 0
ACTIVE_HEAVY_JOBS = 0
ACTIVE_LIGHT_JOBS = 0
ACTIVE_JOBS_LOCK = threading.Lock()
MAX_CONCURRENT_JOBS = int(os.environ.get("CONCURRENT_JOBS", os.environ.get("CONCURRENT_WORKERS", "2")))
WORKER_API_SECRET = os.environ.get("WORKER_API_SECRET", "").strip()
WORKER_API_SECRET_FILE = os.environ.get("WORKER_API_SECRET_FILE", "").strip()

HEAVY_QUEUES = {
    "queue:panel-detection",
    "queue:ocr",
    "queue:qa-re-ocr",
    "queue:region-redo-ocr",
    "queue:region-redo",  # Legacy unified queue name included for backward compatibility
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
        with open(WORKER_API_SECRET_FILE, "r") as f:
            WORKER_API_SECRET = f.read().strip()
    except Exception as e:
        print(f"[Health Server] Failed to read WORKER_API_SECRET_FILE: {e}", flush=True)

WORKER_ID = os.environ.get("WORKER_ID", platform.node())


def set_seeding_complete(complete: bool):
    global SEEDING_COMPLETE
    SEEDING_COMPLETE = complete


def _run_job_async(queue_name, job_data):
    global ACTIVE_JOBS, ACTIVE_HEAVY_JOBS, ACTIVE_LIGHT_JOBS
    try:
        process_job_rq(queue_name, job_data)
    except Exception as e:
        print(f"[Health Server] Async job execution failed: {e}", flush=True)
    finally:
        with ACTIVE_JOBS_LOCK:
            if queue_name in HEAVY_QUEUES:
                ACTIVE_HEAVY_JOBS = max(0, ACTIVE_HEAVY_JOBS - 1)
            else:
                ACTIVE_LIGHT_JOBS = max(0, ACTIVE_LIGHT_JOBS - 1)
            ACTIVE_JOBS = ACTIVE_HEAVY_JOBS + ACTIVE_LIGHT_JOBS


class HealthCheckHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logs to keep stdout clean
        pass

    def check_auth(self):
        if not WORKER_API_SECRET:
            return True
        secret_header = self.headers.get("WORKER_API_SECRET")
        if secret_header != WORKER_API_SECRET:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized")
            return False
        return True

    def do_GET(self):
        if self.path in ("/health", "/ping"):
            global SEEDING_COMPLETE
            if not SEEDING_COMPLETE:
                response_data = {
                    "status": "seeding",
                    "uptime_seconds": int(time.time() - START_TIME),
                }
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                return

            # Check Redis connection
            redis_status = "connected"
            try:
                if redis_client.ping():
                    redis_status = "connected"
                else:
                    redis_status = "disconnected"
            except Exception:
                redis_status = "disconnected"

            uptime_seconds = time.time() - START_TIME
            hours, remainder = divmod(int(uptime_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"

            loaded_models = model_manager.get_loaded_models_status(MODEL_TTL)

            response_data = {
                "status": "healthy" if redis_status == "connected" else "unhealthy",
                "uptime": uptime_str,
                "uptime_seconds": int(uptime_seconds),
                "redis": redis_status,
                "loaded_models": loaded_models,
            }

            self.send_response(200 if redis_status == "connected" else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        elif self.path == "/capabilities":
            if not self.check_auth():
                return

            global ACTIVE_JOBS
            with ACTIVE_JOBS_LOCK:
                current_active = ACTIVE_JOBS

            response_data = {
                "worker_id": WORKER_ID,
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
                "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                "active_jobs": current_active,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/api/v1/jobs/submit":
            if not self.check_auth():
                return

            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode("utf-8"))

                queue_name = payload.get("queue_name")
                job_data = payload.get("job_data")

                if not queue_name or not job_data:
                    raise ValueError("Missing queue_name or job_data")

                global ACTIVE_JOBS, ACTIVE_HEAVY_JOBS, ACTIVE_LIGHT_JOBS
                with ACTIVE_JOBS_LOCK:
                    # Check legacy global limit first (mainly for patched tests)
                    if ACTIVE_JOBS >= MAX_CONCURRENT_JOBS:
                        self.send_response(429)
                        self.end_headers()
                        self.wfile.write(b"Too Many Requests: Global concurrency limit reached")
                        return

                    # Check slot-specific limits
                    is_heavy = queue_name in HEAVY_QUEUES
                    if is_heavy:
                        if ACTIVE_HEAVY_JOBS >= 1:
                            self.send_response(429)
                            self.end_headers()
                            self.wfile.write(b"Too Many Requests: Heavy job slot occupied")
                            return
                        ACTIVE_HEAVY_JOBS += 1
                    else:
                        if ACTIVE_LIGHT_JOBS >= 1:
                            self.send_response(429)
                            self.end_headers()
                            self.wfile.write(b"Too Many Requests: Light job slot occupied")
                            return
                        ACTIVE_LIGHT_JOBS += 1

                    ACTIVE_JOBS = ACTIVE_HEAVY_JOBS + ACTIVE_LIGHT_JOBS

                try:
                    self.send_response(202)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "accepted"}).encode("utf-8"))
                    self.wfile.flush()

                    # Start job in background
                    t = threading.Thread(
                        target=_run_job_async, args=(queue_name, job_data), daemon=True
                    )
                    t.start()
                except Exception as start_err:
                    with ACTIVE_JOBS_LOCK:
                        if is_heavy:
                            ACTIVE_HEAVY_JOBS = max(0, ACTIVE_HEAVY_JOBS - 1)
                        else:
                            ACTIVE_LIGHT_JOBS = max(0, ACTIVE_LIGHT_JOBS - 1)
                        ACTIVE_JOBS = ACTIVE_HEAVY_JOBS + ACTIVE_LIGHT_JOBS
                    raise start_err

            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Bad Request: {e}".encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Server Error: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def start_health_server(port: int):
    """Start the health check HTTP server on a daemon thread."""

    def run_server():
        try:
            from http.server import ThreadingHTTPServer

            server = ThreadingHTTPServer(("0.0.0.0", port), HealthCheckHandler)
            print(f"[Health Server] Running on port {port}...", flush=True)
            server.serve_forever()
        except Exception as e:
            print(f"[Health Server] Failed to start: {e}", flush=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
