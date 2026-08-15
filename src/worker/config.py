"""Configuration parameters and initialization for the unified workers."""

import contextvars
import json
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

# The pipeline's trace id for whatever job this thread is currently running, or "" between jobs.
#
# The backend has minted one id per pipeline for some time (JobCoordinatorService, keyed in Redis
# under pipeline:trace:<imageId>) and ships it in every job payload as "traceId" — the worker simply
# never read it. Setting it here, from the one place every job enters (process_job_rq), is what lets
# a single page's six stages be grepped out of both containers' logs with one string.
#
# A ContextVar rather than a global because jobs run concurrently (CONCURRENT_JOBS defaults to 5);
# each worker thread gets its own value with no locking.
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace_id(trace_id):
    """Bind a pipeline trace id to the current job context. Returns the reset token."""
    return _trace_id.set(str(trace_id) if trace_id else "")


def reset_trace_id(token):
    """Unbind, restoring whatever was bound before. Pair with the token from set_trace_id."""
    try:
        _trace_id.reset(token)
    except ValueError:
        # The token belongs to another context (the job ran on a different thread than it started
        # on). Clearing outright is the safe fallback: an unset id is correct-but-empty, a stale one
        # mislabels the next job.
        _trace_id.set("")


def get_trace_id() -> str:
    return _trace_id.get()


class _TraceIdFilter(logging.Filter):
    """Injects ``trace`` into every record so the formatter can print it unconditionally.

    Shortened to 8 characters to match the backend's log pattern: a full UUID on every line wraps in
    Dozzle and buys nothing at this cardinality. Records emitted outside a job (startup, the health
    endpoint, model loading) get dashes, which keeps the columns aligned.
    """

    def filter(self, record):
        tid = _trace_id.get()
        record.trace = tid[:8] if tid else "--------"
        return True


class _HealthProbeFilter(logging.Filter):
    """Drops uvicorn's access line for the container healthcheck.

    That probe runs every 5 seconds and produced 733 lines in a 65-minute window — more lines than
    the worker logged at WARNING and ERROR combined, all of them saying 200. A *failing* probe is
    still visible: it shows up as the container going unhealthy in `docker ps`.
    """

    def filter(self, record):
        return "/health" not in record.getMessage()


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
level = TRACE_LEVEL_NUM if LOG_LEVEL == "TRACE" else getattr(logging, LOG_LEVEL, logging.INFO)

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(trace)s] %(message)s"))
_handler.addFilter(_TraceIdFilter())
logging.basicConfig(level=level, handlers=[_handler], force=True)

# Suppress noisy third-party loggers that flood output at DEBUG level
for _noisy_logger in ("PIL", "PIL.PngImagePlugin"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
# urllib3 logs a line per connection and a line per request at DEBUG. With six stages per page each
# making several backend calls, that was a large share of the worker's DEBUG output and none of it
# says anything the handlers' own logging does not.
logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_HealthProbeFilter())
# Routes warnings.warn() through logging instead of straight to stderr, so Paddle's and PIL's
# UserWarnings arrive levelled and carrying a trace id like everything else. What remains unlevelled
# after this is native output from Paddle's C++ layer, which Python cannot intercept.
logging.captureWarnings(True)

logger = logging.getLogger("translation")

# Cap on a single logged payload, in characters. 0 disables truncation.
#
# The QA handler dumps its full region metadata and the model's full response at DEBUG, which is
# genuinely what you want when the thing under investigation is a prompt — but at 19 KB on one line
# it makes every *other* DEBUG line unreadable, and running the worker at DEBUG is the normal
# configuration here. Truncating by default keeps DEBUG usable; set LOG_PAYLOAD_MAX_CHARS=0 for the
# sessions where the whole blob is the point.
LOG_PAYLOAD_MAX_CHARS = int(os.environ.get("LOG_PAYLOAD_MAX_CHARS", "2000"))


def log_payload(value, indent=2):
    """Render ``value`` for a log line, JSON-encoding it and truncating to a readable length."""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, indent=indent)
        except (TypeError, ValueError):
            value = repr(value)
    if LOG_PAYLOAD_MAX_CHARS and len(value) > LOG_PAYLOAD_MAX_CHARS:
        omitted = len(value) - LOG_PAYLOAD_MAX_CHARS
        return (
            f"{value[:LOG_PAYLOAD_MAX_CHARS]}\n"
            f"… [{omitted} more chars; set LOG_PAYLOAD_MAX_CHARS=0 for the full payload]"
        )
    return value


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


def backend_headers():
    """Auth headers for a backend call, plus the current pipeline trace id.

    The backend's TraceIdFilter reads ``X-Trace-Id`` and binds it for the life of the request, so
    the six callbacks and status PATCHes a page generates land in the backend's log under the same
    id the worker is logging — which is the whole point of carrying it. Falls back to the bare auth
    headers outside a job context, where there is no trace to send.

    Prefer this over the ``BACKEND_HEADERS`` dict for any request made while handling a job.
    """
    if not _trace_id.get():
        return BACKEND_HEADERS
    return {**BACKEND_HEADERS, "X-Trace-Id": _trace_id.get()}


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

# D1/D3 (docs/render_quality_gap_2026-08-05.md): a flat median-colour fill only disguises a region
# that is actually close to flat. Above this per-channel median absolute deviation the sampled
# area has real detail (line art, screentone, a picture frame) and painting one colour over it is
# a paint bucket, not erasure -- worst on free-floating text, which has no balloon interior to be
# flat in the first place. MAD rather than stddev because a handful of anti-aliased text-edge
# pixels in the sample would otherwise spike a plain stddev on an otherwise-flat region; taken
# per-channel and maxed rather than over the flattened array, because a saturated solid colour
# (e.g. pure blue) has large B-vs-R separation despite being spatially perfectly flat. A starting
# guess, not yet measured against an annotated set the way WAIST_MAX_SOLIDITY was.
BACKGROUND_FILL_MAX_SPREAD = float(os.environ.get("BACKGROUND_FILL_MAX_SPREAD", "20.0"))

# R1 (docs/issues.md): a balloon has to contain the text it is the balloon for.
#
# Fragments are assigned to whichever YOLO mask they overlap most, and "most" was the only test --
# any overlap at all won, and the winner's geometry was then accepted as the region's balloon. YOLO
# is a single-class segmenter and fires on the white *stroke* drawn around unenclosed lettering,
# which is a text-shaped blob sitting exactly on the text. sample10's 待って is the case: the
# accepted "balloon" was 0.60x the area of its own text, so a white glyph-shaped slab was painted
# onto a yellow burst and "WAIT!" was set in the sliver left over.
#
# **Measured over all 40 corpus pages, 2026-08-13, and the first version of this test was wrong.**
# It asked only "does the balloon cover its text", at 0.75. Coverage turns out to have no threshold
# in it: over 351 elements the values run smoothly from 0.005 to 1.0 with no gap, verified-correct
# rejections sit anywhere from 0.005 to 0.749, and probable false positives interleave with them
# throughout. At 0.75 it flagged 19 elements, of which roughly half read as ordinary dialogue in a
# real balloon whose merged text box simply overhangs the outline.
#
# What does separate them is that there are two distinct failures, and coverage only sees part of
# each:
#
# - **a stroke mistaken for a balloon.** YOLO fires on the white outline drawn around unenclosed
#   lettering, which is a text-shaped blob sitting exactly on the text, so the "balloon" ends up
#   entirely *inside* its own text box. A container cannot be contained by its contents. Over the
#   corpus this flags 5 of 239 real contours at 0.95 and every one is correct: sample10's 待って and
#   its dark-panel twin, page 19's `WAARU (CLANG)` and `IMAJI (MENTAL IMAGE)`, and page 20's
#   vertical column on bare white panel. The next value down is 0.936 and is genuine dialogue.
# - **a balloon assigned to text it barely touches.** Assignment is by greatest overlap with no
#   floor, so a region can be handed a balloon that overlaps it by half a percent. Below 0.25 the
#   corpus holds 5 elements — `综`, `写`, `*`, `Meya Me (GULP GULP)`, `So… (sound of conclusion)` —
#   all junk or sfx, and the next value up is 0.476.
#
# Together they flag ~10 of 239 (4%) against the first version's 19, and every one has been looked
# at. Only apply these to a *detected* contour: a 4-point polygon is the raw OCR rectangle standing
# in for a balloon, and it is engulfed by its own text by construction.
BUBBLE_MIN_TEXT_COVERAGE = float(os.environ.get("BUBBLE_MIN_TEXT_COVERAGE", "0.25"))
BUBBLE_MAX_SELF_CONTAINMENT = float(os.environ.get("BUBBLE_MAX_SELF_CONTAINMENT", "0.95"))

# R2 (docs/issues.md): what to do for a region that has no flat colour to match.
#
# Returning None meant "draw nothing", which put English on top of unerased Japanese -- the worst
# of the available outcomes, and the one thing no reference output ever does. mangatranslator.ai
# does not erase sample10's yellow blanket lettering better than we can; it declines to erase it
# and draws a new flat balloon over it instead. This is that: cover the source text with a
# synthesized balloon in the region's dominant colour.
#
# Dominant, not median: the median of a shaded yellow blanket is a muddy mid-tone, while the
# quantised mode is the yellow a reader would name. Bin width 16 over each channel.
COVER_FILL_ENABLED = os.environ.get("COVER_FILL_ENABLED", "true").lower() in ("1", "true", "yes", "on")
COVER_FILL_QUANT = int(os.environ.get("COVER_FILL_QUANT", "16"))
# Padding around the source text's extent, as a fraction of the shorter side of that extent.
COVER_FILL_PAD_FRACTION = float(os.environ.get("COVER_FILL_PAD_FRACTION", "0.18"))
# How far outside the text box to sample for that dominant colour. Sampling *inside* the box
# samples the lettering: unenclosed manga text carries a thick white stroke so it reads against
# artwork, and on sample10's yellow blanket that stroke is the most common colour in the box --
# which made "the region's dominant colour" come back white. Wider than the drawn margin, so the
# sample is background rather than the edge of the shape about to be painted.
COVER_FILL_RING_FRACTION = float(os.environ.get("COVER_FILL_RING_FRACTION", "0.35"))

# Minimum WCAG contrast between lettering and the backdrop under it before the renderer overrides
# the text colour. 3.0 is the large-text threshold, and lettering sized to fill a balloon is large
# text. Only reached because R2 introduced backdrops sampled from artwork: a text colour is chosen
# without reference to the fill, so a balloon covering a dark panel arrived as black on near-black.
CONTRAST_FLOOR = float(os.environ.get("CONTRAST_FLOOR", "3.0"))

# R3 (docs/issues.md): sound effects and unreadable regions are not dialogue.
#
# The references leave sfx in the artwork untouched. We typeset them, and when the recogniser
# misreads one the translator turns the garbage into a confident sentence and we paint a slab on
# the artwork to hold it -- sample10's misread `cu3gichi` became "Deadline countdown activated!".
TYPESET_SFX = os.environ.get("TYPESET_SFX", "false").lower() in ("1", "true", "yes", "on")
# A region with no balloon *and* a recogniser this unsure of itself is not a line of dialogue.
# Both halves are required: unenclosed lettering read confidently is real dialogue (sample10's
# yellow blanket), and a low score inside a balloon is still a line somebody said.
JUNK_REGION_MIN_CONFIDENCE = float(os.environ.get("JUNK_REGION_MIN_CONFIDENCE", "0.55"))

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
