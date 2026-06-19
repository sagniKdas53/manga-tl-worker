import os
import logging
import redis
from minio import Minio

# Configure structured logging
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


def trace(self, message, *args, **kws):
    if self.isEnabledFor(logging.TRACE):
        self._log(logging.TRACE, message, args, **kws)


logging.Logger.trace = trace

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
level = logging.TRACE if LOG_LEVEL == "TRACE" else getattr(logging, LOG_LEVEL)
logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("translation")

# Connection Configs
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

# Callback & Auth Configs
CALLBACK_URL = os.environ.get(
    "BACKEND_CALLBACK_URL", "http://localhost:8080/api/internal/jobs/callback"
)
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")
BACKEND_HEADERS = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}

# Service Settings
RATE_LIMIT = os.environ.get("RATE_LIMIT", "").strip()
MODEL_TTL = int(os.environ.get("MODEL_TTL", "3600"))  # Default: 1 hour in seconds
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8000"))

# Clients
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    socket_timeout=15,
    socket_connect_timeout=5,
    socket_keepalive=True,
    retry_on_timeout=True,
)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# YOLO Speech Bubble Segmentation Configs
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "")
if not YOLO_MODEL_PATH:
    local_path = "/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface/models/yolo11n_bubble.onnx"
    docker_path = "/root/.cache/huggingface/models/yolo11n_bubble.onnx"
    if os.path.exists(local_path):
        YOLO_MODEL_PATH = local_path
    else:
        YOLO_MODEL_PATH = docker_path

YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", "1280"))
YOLO_MASK_EROSION = int(os.environ.get("YOLO_MASK_EROSION", "5"))
YOLO_PINNED_CHECKSUM = (
    "f081f02a40601e3a1d4f5bf4e1a5a1a84340a0e52212d170e3bc5b679df97dcf"
)
YOLO_FALLBACK_MODE = os.environ.get("YOLO_FALLBACK_MODE", "opencv").lower()
