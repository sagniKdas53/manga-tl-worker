"""Configuration parameters and initialization for the unified workers."""

import logging
import os

import redis
from minio import Minio

# Configure structured logging
TRACE_LEVEL_NUM = 5
logging.TRACE = TRACE_LEVEL_NUM  # type: ignore
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def trace(self, message, *args, **kws):
    """Log a message with TRACE level."""
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)  # type: ignore


logging.Logger.trace = trace  # type: ignore

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
level = TRACE_LEVEL_NUM if LOG_LEVEL == "TRACE" else getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress noisy third-party loggers that flood output at DEBUG level
for _noisy_logger in ("PIL", "PIL.PngImagePlugin"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger("translation")


def _is_sensitive(path: str) -> bool:
    sensitive_patterns = [".ssh", ".aws", ".env"]
    return any(pattern in path for pattern in sensitive_patterns)


def _load_docker_secrets():
    """Load secrets into os.environ.

    PRECEDENCE & RESOLUTION ORDER:
    1. Initial container environment variables (populated from .env via docker-compose).
    2. DOCKER_SECRETS_JSON (secrets/api_keys.json): Overwrites matching keys in os.environ.
       If a key is present in secrets/api_keys.json, its value takes precedence over .env.
       If a key is omitted or empty in secrets/api_keys.json, the value from .env is preserved.
    3. Individual *_FILE secrets: Overwrite keys only if they were not already loaded from JSON.
    """
    import json

    loaded_from_json = set()
    secrets_json = os.environ.get("DOCKER_SECRETS_JSON")
    if secrets_json:
        resolved_json_path = os.path.realpath(secrets_json)
        if not _is_sensitive(resolved_json_path) and os.path.exists(resolved_json_path):
            try:
                with open(resolved_json_path) as f:
                    secrets = json.load(f)
                    for k, v in secrets.items():
                        # Override container environment variables with values from secrets/api_keys.json
                        os.environ[k] = str(v)
                        loaded_from_json.add(k)
            except Exception as e:
                logging.error(f"Failed to load DOCKER_SECRETS_JSON: {e}")

    for k, v in list(os.environ.items()):
        if k.endswith("_FILE") and v:
            real_key = k[:-5]
            if real_key not in loaded_from_json:
                resolved_v = os.path.realpath(v)
                if not _is_sensitive(resolved_v) and os.path.exists(resolved_v):
                    try:
                        with open(resolved_v) as f:
                            os.environ[real_key] = f.read().strip()
                    except Exception as e:
                        logging.error(f"Failed to read secret file for {k}: {e}")


_load_docker_secrets()


# Connection Configs
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
RENDER_CACHE_DIR = os.environ.get("RENDER_CACHE_DIR", "/app/data/rendered_cache")

ENABLE_QA_AUDIT_CACHE = os.environ.get("ENABLE_QA_AUDIT_CACHE", "false").lower() in (
    "true",
    "1",
    "yes",
)
QA_AUDIT_CACHE_DIR = os.environ.get("QA_AUDIT_CACHE_DIR", "/app/data/qa_audit")

# Callback & Auth Configs
CALLBACK_URL = os.environ.get("BACKEND_CALLBACK_URL", "http://localhost:8080/api/internal/jobs/callback")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")
BACKEND_HEADERS = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}

# Service Settings
RATE_LIMIT = os.environ.get("RATE_LIMIT", "").strip()
# Default: 1 hour in seconds
MODEL_TTL = int(os.environ.get("MODEL_TTL", "3600"))
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8000"))

# Clients
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    socket_timeout=15,
    socket_connect_timeout=5,
    socket_keepalive=True,
    # Valkey 8 supports RESP2; redis-py 8 otherwise enables Redis-only
    # maintenance notifications through RESP3 during connection setup.
    protocol=2,
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
    LOCAL_PATH = (
        "/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface/models/yolo11n_bubble.onnx"
    )
    # AUDIT-D2: the container no longer runs as root, so the cache lives under the
    # worker user's home rather than /root.
    DOCKER_PATH = "/home/worker/.cache/huggingface/models/yolo11n_bubble.onnx"
    YOLO_MODEL_PATH = LOCAL_PATH if os.path.exists(LOCAL_PATH) else DOCKER_PATH

YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", "1280"))
YOLO_MASK_EROSION = int(os.environ.get("YOLO_MASK_EROSION", "3"))
YOLO_PINNED_CHECKSUM = "c9208cb610aa35b8f8dc7ef0890182322992a43399a853093ad5d04a3764af4f"
YOLO_FALLBACK_MODE = os.environ.get("YOLO_FALLBACK_MODE", "opencv").lower()

# When YOLO is active but matched no bubble to a text fragment, try the OpenCV contour search on
# that fragment before giving up and using the raw text bbox as the "bubble".
#
# Off by default, because measurement says it recovers almost nothing. The original number behind
# this flag -- "a usable containing shape for ~48% of unmatched fragments, median 2.6x wider" --
# counted results that were the contour search's own crop window. Free-floating text sits on the
# page background, the background is what the threshold finds, and that blob has no boundary inside
# the crop, so its bounding box comes back as the crop: text bbox + 2 * pad, which is where a
# "median 2.6x wider" for a 40px pad on a ~50px-wide column comes from. Re-measured over 300 such
# regions with that case excluded, acceptance is 1 in 300 (0.3%), not 48%.
#
# The guard now lives in contour_bubble_for_unmatched, so turning this back on cannot reintroduce
# window geometry -- but a 0.3% hit rate does not pay for a contour search per unmatched fragment.
BUBBLE_CONTOUR_FALLBACK = os.environ.get("BUBBLE_CONTOUR_FALLBACK", "false").lower() in ("1", "true", "yes", "on")

# Guards on an accepted contour. Growth and page-fraction caps come from the original sweep; both
# were near-inert on their own, since a crop window passes them by construction.
BUBBLE_CONTOUR_MAX_GROWTH = float(os.environ.get("BUBBLE_CONTOUR_MAX_GROWTH", "5.0"))
BUBBLE_CONTOUR_MAX_PAGE_FRACTION = float(os.environ.get("BUBBLE_CONTOUR_MAX_PAGE_FRACTION", "0.35"))

# --- Fragment grouping -----------------------------------------------------------------------
#
# How OCR line fragments are grouped into regions. All four values were measured over seven
# hand-annotated pages and validated over all forty corpus pages; see
# docs/region_waist_probe_2026-08-09.md and corpus/runs/2026-08-09/region-grouping/.
#
# Proximity budget, as a multiple of the estimated character size: "join two fragments if the
# white space between them is under this many characters wide". 0.35 replaces a hardcoded 2.0 on
# the in-bubble path -- at 2.0 two touching balloons inside one YOLO blob are always joined, which
# fuses two speakers into one translation unit and one flat fill. Over the annotated pages this
# alone takes mergers 17 -> 5.
OCR_MERGE_THRESHOLD = float(os.environ.get("OCR_MERGE_THRESHOLD", "0.35"))

# Clearance veto. Two fragments inside one pinched balloon mask are kept apart when the path
# between them squeezes within this many characters of the outline -- the geometric waist where
# two balloons touch. Distance alone cannot separate those cases: within-balloon gaps (0.3-1
# character) and cross-balloon gaps (1-2) overlap. Set to 0 to disable.
OCR_WAIST_GATE = float(os.environ.get("OCR_WAIST_GATE", "1.0"))

# ...but only inside masks below this solidity (area / convex hull area). A convex mask has no
# waist to find and measuring one produces noise: below 0.90 the veto was exact on 8/8 annotated
# bubbles, above 0.95 it was worse than distance.
OCR_WAIST_MAX_SOLIDITY = float(os.environ.get("OCR_WAIST_MAX_SOLIDITY", "0.90"))

# Where text orientation comes from. "vote" derives it from the fragments' own aspect ratios;
# "reading_direction" reads it off the binding direction, which is BUG-6 -- binding direction is
# which way pages turn, not which way text runs, so every horizontally-set Japanese page got
# vertical geometry and its whole page collapsed into one or two regions (sample23: 2 regions
# where there are 17).
OCR_ORIENTATION = os.environ.get("OCR_ORIENTATION", "vote").strip().lower()


def is_usable_model(model):
    """A model id counts as usable only if it is a real, non-sentinel value."""
    if not model or not isinstance(model, str):
        return False
    m = model.strip()
    if not m:
        return False
    return m.lower() != "n/a" and m not in ("default", "inherit") and "[ORPHANED]" not in m


# Model Configuration
class ModelConfig:
    def __init__(
        self,
        provider_env: str,
        llm_env: str = "",
        vlm_env: str = "",
        llm_list_env: str = "",
        vlm_list_env: str = "",
    ):
        self.provider = os.environ.get(provider_env, "").lower().strip()
        self.llm_model = os.environ.get(llm_env, "").strip()
        self.vlm_model = os.environ.get(vlm_env, "").strip()

        llm_list_raw = os.environ.get(llm_list_env, "").strip() if llm_list_env else ""
        self.llm_model_list = [x.strip() for x in llm_list_raw.split(",") if x.strip()] if llm_list_raw else []

        vlm_list_raw = os.environ.get(vlm_list_env, "").strip() if vlm_list_env else ""
        self.vlm_model_list = [x.strip() for x in vlm_list_raw.split(",") if x.strip()] if vlm_list_raw else []

    def resolve_key(self, provider: str | None = None) -> str:
        prov = (provider or self.provider or "").lower().strip()
        if not prov:
            return ""
        from worker.provider_config import get_config_loader

        loader = get_config_loader()
        if prov in loader.providers and loader.providers[prov].api_key:
            return loader.providers[prov].api_key

        env_var_map = {
            "openrouter": ["OPENROUTER_API_KEY", "API_KEY"],
            "gemini": ["GEMINI_API_KEY", "API_KEY"],
            "nvidia": ["NVIDIA_API_KEY", "API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "API_KEY"],
            "openai": ["OPENAI_API_KEY", "API_KEY"],
            "neurometric": ["NEUROMETRIC_API_KEY", "API_KEY"],
            "cloudflare": ["CLOUDFLARE_API_TOKEN", "API_KEY"],
        }
        candidates = env_var_map.get(prov, ["API_KEY"])
        for var in candidates:
            val = os.environ.get(var, "").strip()
            if val:
                return val
        return ""


OCR_CONFIG = ModelConfig(
    provider_env="OCR_MODEL_PROVIDER",
    vlm_env="OCR_VLM_MODEL",
    vlm_list_env="OCR_VLM_MODEL_LIST",
)

TL_CONFIG = ModelConfig(
    provider_env="TL_MODEL_PROVIDER",
    llm_env="TL_LLM_MODEL",
    llm_list_env="TL_LLM_MODEL_LIST",
)

QA_CONFIG = ModelConfig(
    provider_env="QA_MODEL_PROVIDER",
    llm_env="QA_LLM_MODEL",
    vlm_env="QA_VLM_MODEL",
    llm_list_env="QA_LLM_MODEL_LIST",
    vlm_list_env="QA_VLM_MODEL_LIST",
)

LOCAL_LLM_PROVIDER = os.environ.get("LOCAL_LLM_PROVIDER", "").strip()
LOCAL_LLM_ENDPOINT = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "").strip()
LOCAL_VLM_MODEL = os.environ.get("LOCAL_VLM_MODEL", "").strip()

# QA Configuration
# Modes: "none" = skip QA, "llm" = text-only LLM review,
# "vlm" = full vision review, "auto" = auto-detect based on capabilities.
QA_MODE = os.environ.get("QA_MODE", "auto").lower().strip()

# QA Mode Auto-Detection Logic:
# Decides between "vlm", "llm", or "none" dynamically at startup
# based on configured models and key states.
# Respects the DISABLE_LOCAL_LLM configuration (ignoring local LLM/VLM models if disabled).
if QA_MODE == "auto":
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    effective_local_vlm = "" if disable_local else LOCAL_VLM_MODEL
    effective_local_llm = "" if disable_local else LOCAL_LLM_MODEL

    # Detect VLM capability (usable Cloud VLM or effective local VLM)
    has_vlm = (
        is_usable_model(QA_CONFIG.vlm_model)
        or is_usable_model(OCR_CONFIG.vlm_model)
        or is_usable_model(effective_local_vlm)
    )

    # Detect LLM capability (usable Cloud LLM with provider or effective local LLM)
    has_llm = is_usable_model(QA_CONFIG.llm_model) or is_usable_model(effective_local_llm)

    if has_vlm:
        QA_MODE = "vlm"
    elif (QA_CONFIG.provider and has_llm) or is_usable_model(effective_local_llm):
        QA_MODE = "llm"
    else:
        QA_MODE = "none"

# Publish provider config map to Redis on startup
try:
    from worker.provider_config import get_config_loader

    get_config_loader().publish_config_to_redis(redis_client)
except Exception as _e:
    logger.warning(f"Could not publish provider config to Redis at startup: {_e}")


# Validate and fetch openrouter costs on startup
if OCR_CONFIG.provider == "openrouter" or TL_CONFIG.provider == "openrouter" or QA_CONFIG.provider == "openrouter":
    from worker.utils.rate_limit import update_model_costs

    models_to_check = []
    if TL_CONFIG.llm_model:
        models_to_check.append(TL_CONFIG.llm_model)
    if OCR_CONFIG.vlm_model:
        models_to_check.append(OCR_CONFIG.vlm_model)
    if QA_CONFIG.llm_model:
        models_to_check.append(QA_CONFIG.llm_model)
    if QA_CONFIG.vlm_model:
        models_to_check.append(QA_CONFIG.vlm_model)

    if models_to_check:
        try:
            update_model_costs(list(set(models_to_check)))
        except ValueError as e:
            logger.critical(f"Startup failed: {e}")
            import sys

            sys.exit(1)
