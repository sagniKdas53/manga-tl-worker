import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from worker.config import redis_client, MODEL_TTL
from worker.model_manager import model_manager

# Global reference to start time
START_TIME = time.time()

# Seeding completion status
SEEDING_COMPLETE = False


def set_seeding_complete(complete: bool):
    global SEEDING_COMPLETE
    SEEDING_COMPLETE = complete


class HealthCheckHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logs to keep stdout clean
        pass

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
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def start_health_server(port: int):
    """Start the health check HTTP server on a daemon thread."""

    def run_server():
        try:
            server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            print(f"[Health Server] Running on port {port}...", flush=True)
            server.serve_forever()
        except Exception as e:
            print(f"[Health Server] Failed to start: {e}", flush=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
