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

# Suppress LiteLLM debug logs to avoid massive base64 payload prints
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

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
YOLO_MASK_EROSION = int(os.environ.get("YOLO_MASK_EROSION", "3"))
YOLO_PINNED_CHECKSUM = (
    "f081f02a40601e3a1d4f5bf4e1a5a1a84340a0e52212d170e3bc5b679df97dcf"
)
YOLO_FALLBACK_MODE = os.environ.get("YOLO_FALLBACK_MODE", "opencv").lower()

# Model Configuration
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "").lower().strip()
API_KEY = os.environ.get("API_KEY", "").strip()
PREFERRED_LLM_MODEL = os.environ.get("PREFERRED_LLM_MODEL", "").strip()
PREFERRED_VLM_MODEL = os.environ.get("PREFERRED_VLM_MODEL", "").strip()
LOCAL_LLM_PROVIDER = os.environ.get("LOCAL_LLM_PROVIDER", "").strip()
LOCAL_LLM_ENDPOINT = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "").strip()
LOCAL_VLM_MODEL = os.environ.get("LOCAL_VLM_MODEL", "").strip()

# QA Configuration
# Modes: "none" = skip QA, "llm" = text-only LLM review, "vlm" = full vision review, "auto" = auto-detect based on capabilities.
QA_MODE = os.environ.get("QA_MODE", "auto").lower().strip()

# QA Mode Auto-Detection Logic:
# Decides between "vlm", "llm", or "none" dynamically at startup based on configured models and key states.
# Respects the DISABLE_LOCAL_LLM configuration (ignoring local LLM/VLM models if disabled).
if QA_MODE == "auto":
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in ("true", "1", "yes")
    effective_local_vlm = "" if disable_local else LOCAL_VLM_MODEL
    effective_local_llm = "" if disable_local else LOCAL_LLM_MODEL

    # Detect VLM capability (Cloud VLM or effective local VLM)
    has_vlm = bool(os.environ.get("QA_VLM_MODEL", "").strip() or PREFERRED_VLM_MODEL or effective_local_vlm)
    
    # Detect LLM capability (Cloud provider or effective local LLM)
    if has_vlm:
        QA_MODE = "vlm"
    elif MODEL_PROVIDER or effective_local_llm:
        QA_MODE = "llm"
    else:
        QA_MODE = "none"

# Render cache
RENDER_CACHE_DIR = os.environ.get("RENDER_CACHE_DIR", "/app/rendered_cache")

