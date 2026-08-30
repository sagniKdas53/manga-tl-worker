import contextvars
import json
import os
import threading
import time

import requests

from worker.config import logger, redis_client

RATE_LIMIT_LOCK = threading.Lock()
PROVIDER_LAST_REQUEST_TIME = {}


def enforce_rate_limit(provider: str | None = None, provider_rpm: float | None = None):
    global PROVIDER_LAST_REQUEST_TIME
    rpm = None

    if provider and provider_rpm and provider_rpm > 0:
        rpm = float(provider_rpm)
    else:
        # AUDIT-W2: unset means unlimited, and unset is now what ships. A provider added to
        # providers.json without `rateLimits` must not silently inherit a throttle from an env var
        # that was aimed at something else. Every provider that wants a limit carries its own.
        rate_limit_env = os.environ.get("RATE_LIMIT", "").strip()
        if rate_limit_env:
            try:
                if "/" in rate_limit_env:
                    parts = rate_limit_env.split("/")
                    val = float(parts[0])
                    unit = parts[1].lower().strip()
                    rpm = val * 60.0 if unit in ("s", "sec", "second", "seconds") else val
                else:
                    rpm = float(rate_limit_env)
            except Exception as e:
                # AUDIT-Q3: [RateLimit], not [Translation] — enforce_rate_limit is shared by the
                # translation, OCR and QA paths, so the old prefix sent anyone grepping the worker
                # log for an OCR or QA stall looking at the wrong stage.
                logger.error(f"[RateLimit] Error parsing RATE_LIMIT env: {e}")

    if rpm and rpm > 0:
        min_delay = 60.0 / rpm
        sleep_time = 0.0
        lock_key = provider if provider else "global"

        with RATE_LIMIT_LOCK:
            now = time.time()
            last_time = PROVIDER_LAST_REQUEST_TIME.get(lock_key, 0.0)
            elapsed = now - last_time
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                PROVIDER_LAST_REQUEST_TIME[lock_key] = now + sleep_time
            else:
                PROVIDER_LAST_REQUEST_TIME[lock_key] = now

        if sleep_time > 0:
            logger.debug(
                f"[RateLimit] Sleeping for {sleep_time:.2f} seconds to respect {rpm} RPM limit for {lock_key}..."
            )
            time.sleep(sleep_time)


COSTS_LOCK = threading.Lock()

# Job-scoped, not process-global. Jobs run concurrently on reused threads (CONCURRENT_JOBS defaults
# to 5) and every handler used to call reset_job_costs() on entry against one shared list, so a job
# starting mid-flight wiped another job's accumulated costs and then aggregated the wrong calls.
# A ContextVar gives each job its own list, mirroring the trace id in config.py.
#
# The default is None rather than [] because a mutable default is shared by every context that never
# binds one — the exact aliasing this exists to prevent. _current_job_costs() binds a fresh list on
# first use instead.
#
# Chunked stages (OCR crops, translation retries) submit work to a ThreadPoolExecutor, and a new
# thread does NOT inherit the spawning thread's context. Those call sites submit through
# contextvars.copy_context().run so the worker thread sees the *same* list object and its appends
# stay visible to the job that owns them.
_job_costs: contextvars.ContextVar[list | None] = contextvars.ContextVar("job_costs", default=None)


def _current_job_costs() -> list:
    costs = _job_costs.get()
    if costs is None:
        costs = []
        _job_costs.set(costs)
    return costs


def reset_job_costs():
    """Bind a fresh, empty cost list to the current context.

    Called once per job at the single dispatch point (rq_tasks.process_job_rq), not per handler.
    """
    _job_costs.set([])


def format_cost(cost):
    """
    Format cost in a human-friendly format.
    e.g. $0.00, $0.0045, $0.000001, <$0.000001 ($2.30e-07)
    """
    if cost is None:
        return "N/A"
    if cost == 0.0:
        return "$0.00"
    if cost >= 0.01:
        return f"${cost:.4f}"
    if cost >= 0.0001:
        return f"${cost:.6f}"
    return f"${cost:.2e}"


def _endpoint_rates(endpoints):
    """Cheapest prompt/completion/cache-read rates across a model's serving endpoints.

    The cheapest, not the average. Requests are routed with `provider: {"sort": "price"}` (see
    llm_client._inject_routing_and_caching), so they land on the least expensive endpoint —
    averaging across all of them overstated the TL model's cost by 3.2x.
    """
    prompt_costs = []
    completion_costs = []
    cache_read_costs = []
    for ep in endpoints:
        pricing = ep.get("pricing")
        if not pricing:
            continue
        prompt_costs.append(float(pricing.get("prompt") or 0))
        completion_costs.append(float(pricing.get("completion") or 0))
        cache_read = pricing.get("input_cache_read")
        if cache_read is not None:
            cache_read_costs.append(float(cache_read))

    if not prompt_costs or not completion_costs:
        return None

    return (
        min(prompt_costs),
        min(completion_costs),
        min(cache_read_costs) if cache_read_costs else None,
    )


def _fetch_endpoints(model):
    """Return (endpoints, status_code) for a model, resolving a suffixed slug to its base.

    Raises ValueError when a `:free` slug has no free endpoint of its own. Falling through to the
    base slug there is what let three configured-but-nonexistent `:free` models be recorded as a
    successful $0 while their paid base slugs were the thing actually billable.
    """
    url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
    res = requests.get(url, timeout=10)
    endpoints = []
    if res.status_code == 200:
        endpoints = res.json().get("data", {}).get("endpoints", [])

    if endpoints or ":" not in model:
        return endpoints, res.status_code

    base_model, _, suffix = model.partition(":")
    base_url = f"https://openrouter.ai/api/v1/models/{base_model}/endpoints"
    res_fallback = requests.get(base_url, timeout=10)
    base_endpoints = []
    if res_fallback.status_code == 200:
        base_endpoints = res_fallback.json().get("data", {}).get("endpoints", [])

    if base_endpoints and suffix == "free":
        raise ValueError(
            f"Model {model} is not available on OpenRouter: no free endpoint is served, "
            f"and the base slug {base_model} is paid. Pricing it as $0.00 would hide real spend."
        )

    return base_endpoints, res_fallback.status_code


def update_model_costs(models=None, fatal=True):
    """Refresh cached per-token pricing from OpenRouter for the given models.

    Writes to Redis under `model_cost:{model}`. Raises ValueError when a model is unavailable and
    `fatal` is set; fallback models pass fatal=False so an unreachable alternate cannot stop the
    worker from booting.
    """
    if not models:
        return

    # Load existing costs (now only from Redis, local file deprecated Phase E.3)
    persisted_costs = {}
    try:
        keys = redis_client.keys("model_cost:*")
        for key in keys:
            key_text = key.decode("utf-8") if isinstance(key, bytes) else key
            model = key_text.split(":", 1)[1]
            data = redis_client.get(key)
            if data:
                # The timestamp is read as stored. It used to be overwritten with time.time() here,
                # which made the staleness check below trivially true forever — so a price fetched
                # once was frozen until Redis was flushed, and the weekly refresh never ran.
                persisted_costs[model] = json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to read from Redis: {e}")

    now = time.time()
    one_week = 7 * 24 * 3600

    for model in models:
        try:
            model_key = model.lower()

            cached_data = persisted_costs.get(model_key)
            if cached_data and (now - cached_data.get("timestamp", 0) < one_week):
                continue

            endpoints, status_code = _fetch_endpoints(model)

            if status_code == 404:
                raise ValueError(f"Model {model} is not available on OpenRouter (404 Not Found).")
            if status_code != 200:
                logger.warning(f"Failed to fetch endpoints for {model}: {status_code}")
                continue
            if not endpoints:
                raise ValueError(f"Model {model} is not available on OpenRouter (no endpoints returned).")

            rates = _endpoint_rates(endpoints)
            if not rates:
                logger.warning(f"No usable pricing in endpoints for {model}")
                continue

            prompt_rate, completion_rate, cache_read_rate = rates
            cost_data = {
                "prompt": prompt_rate,
                "completion": completion_rate,
                "timestamp": now,
            }
            if cache_read_rate is not None:
                cost_data["cache_read"] = cache_read_rate

            persisted_costs[model_key] = cost_data
            redis_client.set(f"model_cost:{model_key}", json.dumps(cost_data))
            logger.info(
                f"Updated cheapest cost for {model}: Prompt=${(prompt_rate * 1e6):.4f}/M, "
                f"Completion=${(completion_rate * 1e6):.4f}/M"
            )
        except ValueError:
            if fatal:
                raise
            logger.warning(f"Skipping pricing for fallback model {model}: not available on OpenRouter")
        except Exception as e:
            logger.error(f"Error fetching cost for {model}: {e}")


def get_job_costs():
    with COSTS_LOCK:
        return list(_current_job_costs())


def _is_free_model(model_lower: str) -> bool:
    """True only for an explicit `:free` slug.

    Substring matching used to be used here, which made any model whose name merely contained
    "free" cost nothing — and the provider-name clause could zero out every model at once.
    """
    return model_lower.endswith(":free")


def _resolve_rates(model_lower: str, provider_lower: str):
    """(prompt_rate, completion_rate, cache_read_rate) for a model, or None when unknown.

    None means "no price available" and must stay distinguishable from a genuine zero — a
    zero-valued sentinel is what let an unpriced call be reported as $0.00.
    """
    cached = None
    try:
        cached = redis_client.get(f"model_cost:{model_lower}")
    except Exception as e:
        # Never silent: a worker whose Redis link is down otherwise emits a cost report that looks
        # exactly like a healthy one. Fall through to the anchored table rather than giving up —
        # a cache outage should degrade the price, not erase it.
        logger.warning(f"[Cost] Redis lookup failed for {model_lower}, falling back: {e}")

    if cached:
        try:
            cost_data = json.loads(cached)
            return (
                float(cost_data.get("prompt", 0)),
                float(cost_data.get("completion", 0)),
                float(cost_data["cache_read"]) if cost_data.get("cache_read") is not None else None,
            )
        except Exception as e:
            logger.warning(f"[Cost] Malformed cached price for {model_lower}, falling back: {e}")

    # Anchored fallbacks for the handful of models worth hardcoding. Matching is on the slug's
    # final segment so that -0813, :batch and -lite variants — which are priced differently — stop
    # inheriting the base model's rate.
    base = model_lower.split("/")[-1].split(":")[0]
    if base == "deepseek-v4-pro":
        return (0.435 / 1_000_000.0, 0.87 / 1_000_000.0, None)
    if base == "gemini-2.5-flash":
        if provider_lower == "gemini":
            return (0.075 / 1_000_000.0, 0.30 / 1_000_000.0, None)
        return (0.30 / 1_000_000.0, 2.50 / 1_000_000.0, None)

    return None


def _price_call(model, provider, prompt_tokens, completion_tokens, cached_tokens):
    """(cost, source) for one call. cost is None when no price is known."""
    if os.environ.get("DISABLE_COST_CALCULATION", "").strip().lower() in (
        "true",
        "1",
        "yes",
    ):
        return None, "disabled"

    model_lower = (model or "").lower()
    provider_lower = (provider or "").lower()

    is_local = provider_lower in (
        "ollama",
        "lmstudio",
        "local",
        "deepl",
        "google_translate",
        "free_api",
    )
    if is_local or _is_free_model(model_lower):
        return 0.0, "free"

    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0, "free"

    rates = _resolve_rates(model_lower, provider_lower)
    if rates is None:
        return None, "unknown"

    in_rate, out_rate, cache_read_rate = rates

    # Cached prompt tokens are a subset of prompt_tokens on the OpenAI/OpenRouter shape and are
    # billed at the cache-read rate, not the full input rate. Charging them at full rate inflated
    # the input component several-fold on a pipeline that deliberately engineers cache hits.
    billable_prompt = max(prompt_tokens - cached_tokens, 0)
    cost = (billable_prompt * in_rate) + (completion_tokens * out_rate)
    if cached_tokens:
        cost += cached_tokens * (cache_read_rate if cache_read_rate is not None else in_rate)

    return cost, "estimated"


def record_llm_call(
    model,
    prompt_tokens,
    completion_tokens,
    provider=None,
    cached_tokens=0,
    cache_write_tokens=0,
    total_tokens=None,
    generation_id="",
    upstream_provider="",
    model_resolved="",
    stage="",
    duration_ms=None,
    authoritative_cost=None,
):
    """Record one LLM call against the current job and return (cost, cost_source).

    `authoritative_cost` is the provider's own figure (OpenRouter returns it as usage.cost when the
    request asks for it). It wins over any local estimate: it already accounts for which endpoint
    actually served the request and for the cache discount, neither of which a local rate table can
    know.
    """
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    cached_tokens = cached_tokens or 0
    cache_write_tokens = cache_write_tokens or 0

    if authoritative_cost is not None:
        cost, source = float(authoritative_cost), "authoritative"
    else:
        cost, source = _price_call(model, provider, prompt_tokens, completion_tokens, cached_tokens)

    cost_info = {
        "estimated_cost": cost,
        "currency": "USD",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "provider": provider or "unknown",
        # Added alongside the original six keys; every consumer reads by name, so this is additive.
        "total_tokens": total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "generation_id": generation_id or "",
        "upstream_provider": upstream_provider or "",
        "model_resolved": model_resolved or model,
        "cost_source": source,
        "stage": stage or "",
        "duration_ms": duration_ms,
    }

    with COSTS_LOCK:
        _current_job_costs().append(cost_info)

    return cost, source


def estimate_cost(model, prompt_tokens, completion_tokens, provider=None, cached_tokens=0):
    """Price one call and record it. Thin wrapper over record_llm_call for callers with no
    provider-reported cost."""
    cost, _ = record_llm_call(
        model,
        prompt_tokens,
        completion_tokens,
        provider=provider,
        cached_tokens=cached_tokens,
    )
    return cost


def build_cost_payload(costs):
    """Build the callback `cost` object from a job's recorded calls, or None when there were none.

    `estimated_cost` is omitted when any call is unpriced, so a partial total is never presented as
    a complete one. `unknown_calls` carries that state explicitly — without it, "we don't know"
    reaching a dashboard is indistinguishable from "$0.00".
    """
    if not costs:
        return None

    unknown_calls = sum(1 for c in costs if c.get("estimated_cost") is None)
    total_estimated_cost = None if unknown_calls else sum(c.get("estimated_cost") or 0.0 for c in costs)

    payload = {
        "currency": "USD",
        "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in costs),
        "completion_tokens": sum(c.get("completion_tokens") or 0 for c in costs),
        "cached_tokens": sum(c.get("cached_tokens") or 0 for c in costs),
        "unknown_calls": unknown_calls,
        "priced_calls": len(costs) - unknown_calls,
        "breakdown": costs,
    }
    if total_estimated_cost is not None:
        payload["estimated_cost"] = total_estimated_cost
    return payload
