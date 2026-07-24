import json
import os
import threading
import time

import requests

from worker.config import RENDER_CACHE_DIR, logger, redis_client

COSTS_FILE = os.environ.get("COSTS_FILE", os.path.join(RENDER_CACHE_DIR, "costs.json"))

RATE_LIMIT_LOCK = threading.Lock()
LAST_REQUEST_TIME = 0.0


def enforce_rate_limit():
    global LAST_REQUEST_TIME
    rate_limit_env = os.environ.get("RATE_LIMIT", "").strip()
    if not rate_limit_env:
        return
    try:
        # Parse formats like "60", "60/m", "60/min", "5/s", "5/sec"
        rpm = None
        if "/" in rate_limit_env:
            parts = rate_limit_env.split("/")
            val = float(parts[0])
            unit = parts[1].lower().strip()
            rpm = val * 60.0 if unit in ("s", "sec", "second", "seconds") else val
        else:
            rpm = float(rate_limit_env)

        if rpm > 0:
            min_delay = 60.0 / rpm
            sleep_time = 0.0

            with RATE_LIMIT_LOCK:
                now = time.time()
                elapsed = now - LAST_REQUEST_TIME
                if elapsed < min_delay:
                    sleep_time = min_delay - elapsed
                    LAST_REQUEST_TIME = now + sleep_time
                else:
                    LAST_REQUEST_TIME = now

            if sleep_time > 0:
                print(
                    f"[Translation] Rate limit: Sleeping for {sleep_time:.2f} seconds to respect {rate_limit_env} rate limit...",
                    flush=True,
                )
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[Translation] Error enforcing rate limit: {e}", flush=True)


COSTS_LOCK = threading.Lock()
_job_costs = []


def reset_job_costs():
    global _job_costs
    with COSTS_LOCK:
        _job_costs = []


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


def update_model_costs(models=None):
    """
    Fetch average cost per token from OpenRouter for given models.
    Saves to costs.json and Redis. Raises ValueError if a model is not available.
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
                parsed = json.loads(data)
                # Keep timestamp hack by using current time if it's cached in redis
                parsed["timestamp"] = time.time()
                persisted_costs[model] = parsed
    except Exception as e:
        logger.warning(f"Failed to read from Redis: {e}")

    now = time.time()
    one_week = 7 * 24 * 3600

    try:
        for model in models:
            try:
                model_key = model.lower()

                # For free models, initialize price to zero directly
                is_free = (
                    ":free" in model_key or "-free" in model_key or "free" in model_key
                )
                if is_free:
                    cost_data = {
                        "prompt": 0.0,
                        "completion": 0.0,
                        "prompt_per_million": 0.0,
                        "completion_per_million": 0.0,
                        "prompt_display": "$0.0000/M tokens",
                        "completion_display": "$0.0000/M tokens",
                        "timestamp": now,
                    }
                    persisted_costs[model_key] = cost_data
                    redis_client.set(
                        f"model_cost:{model_key}",
                        json.dumps({"prompt": 0.0, "completion": 0.0}),
                    )
                    logger.info(f"Initialized cost for free model {model}: $0.00")
                    continue

                cached_data = persisted_costs.get(model_key)
                if cached_data and (now - cached_data.get("timestamp", 0) < one_week):
                    # Still fresh, just push to Redis
                    redis_client.set(
                        f"model_cost:{model_key}",
                        json.dumps(
                            {
                                "prompt": cached_data["prompt"],
                                "completion": cached_data["completion"],
                            }
                        ),
                    )
                    continue

                # Need to fetch
                # Try fetching with full name first (especially for models with no paid/base version)
                url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
                res = requests.get(url, timeout=10)
                endpoints = []
                if res.status_code == 200:
                    endpoints = res.json().get("data", {}).get("endpoints", [])

                # Fall back to stripping :free suffix if no endpoints found
                if not endpoints and ":" in model:
                    base_model = model.split(":")[0]
                    url = f"https://openrouter.ai/api/v1/models/{base_model}/endpoints"
                    res_fallback = requests.get(url, timeout=10)
                    if res_fallback.status_code == 200:
                        res = res_fallback
                        endpoints = res.json().get("data", {}).get("endpoints", [])
                    else:
                        res = res_fallback

                if res.status_code == 200:
                    if not endpoints:
                        raise ValueError(
                            f"Model {model} is not available on OpenRouter (no endpoints returned)."
                        )

                    prompt_costs = []
                    completion_costs = []
                    for ep in endpoints:
                        pricing = ep.get("pricing")
                        if pricing:
                            prompt_cost = float(pricing.get("prompt") or 0)
                            comp_cost = float(pricing.get("completion") or 0)
                            prompt_costs.append(prompt_cost)
                            completion_costs.append(comp_cost)

                    if prompt_costs and completion_costs:
                        avg_prompt = sum(prompt_costs) / len(prompt_costs)
                        avg_comp = sum(completion_costs) / len(completion_costs)

                        cost_data = {
                            "prompt": avg_prompt,
                            "completion": avg_comp,
                            "prompt_per_million": avg_prompt * 1e6,
                            "completion_per_million": avg_comp * 1e6,
                            "prompt_display": f"${(avg_prompt * 1e6):.4f}/M tokens",
                            "completion_display": f"${(avg_comp * 1e6):.4f}/M tokens",
                            "timestamp": now,
                        }
                        persisted_costs[model_key] = cost_data
                        redis_client.set(
                            f"model_cost:{model_key}",
                            json.dumps({"prompt": avg_prompt, "completion": avg_comp}),
                        )
                        logger.info(
                            f"Updated average cost for {model}: Prompt=${(avg_prompt * 1e6):.4f}/M, Completion=${(avg_comp * 1e6):.4f}/M"
                        )
                elif res.status_code == 404:
                    raise ValueError(
                        f"Model {model} is not available on OpenRouter (404 Not Found)."
                    )
                else:
                    logger.warning(
                        f"Failed to fetch endpoints for {model}: {res.status_code}"
                    )
            except ValueError as ve:
                raise ve
            except Exception as e:
                logger.error(f"Error fetching cost for {model}: {e}")
    finally:
        # Saving to costs.json removed as part of Phase E.3
        pass


def get_job_costs():
    global _job_costs
    with COSTS_LOCK:
        return list(_job_costs)


def estimate_cost(model, prompt_tokens, completion_tokens, provider=None):
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    if os.environ.get("DISABLE_COST_CALCULATION", "").strip().lower() in (
        "true",
        "1",
        "yes",
    ):
        cost_info = {
            "estimated_cost": None,
            "currency": "USD",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model,
            "provider": provider or "unknown",
        }
        with COSTS_LOCK:
            _job_costs.append(cost_info)
        return None

    model_lower = (model or "").lower()
    provider_lower = (provider or "").lower()

    # Determine if local or free
    is_local = provider_lower in (
        "ollama",
        "lmstudio",
        "local",
        "deepl",
        "google_translate",
        "free_api",
    )
    is_free = (
        ":free" in model_lower
        or "-free" in model_lower
        or "free" in model_lower
        or "free" in provider_lower
    )

    if is_local or is_free:
        cost_info = {
            "estimated_cost": 0.0,
            "currency": "USD",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model,
            "provider": provider or "unknown",
        }
        with COSTS_LOCK:
            _job_costs.append(cost_info)
        return 0.0

    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0

    in_rate = 0.0
    out_rate = 0.0

    try:
        cached = redis_client.get(f"model_cost:{model_lower}")
        if cached:
            cost_data = json.loads(cached)
            in_rate = float(cost_data.get("prompt", 0))
            out_rate = float(cost_data.get("completion", 0))
        else:
            # Removed costs.json fallback as part of Phase E.3
            pass
    except Exception:
        pass

    if in_rate == 0.0 and out_rate == 0.0:
        if "deepseek-v4-pro" in model_lower:
            in_rate = 0.435 / 1_000_000.0
            out_rate = 0.87 / 1_000_000.0
        elif "gemini-2.5-flash" in model_lower:
            if provider_lower == "gemini":
                in_rate = 0.075 / 1_000_000.0
                out_rate = 0.30 / 1_000_000.0
            else:  # OpenRouter
                in_rate = 0.30 / 1_000_000.0
                out_rate = 2.50 / 1_000_000.0
        elif "claude-3-5-sonnet" in model_lower:
            in_rate = 3.0 / 1_000_000.0
            out_rate = 15.0 / 1_000_000.0
        else:
            # Pricing not available or calculatable
            cost_info = {
                "estimated_cost": None,
                "currency": "USD",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model,
                "provider": provider or "unknown",
            }
            with COSTS_LOCK:
                _job_costs.append(cost_info)
            return None

    cost = (prompt_tokens * in_rate) + (completion_tokens * out_rate)

    cost_info = {
        "estimated_cost": cost,
        "currency": "USD",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "provider": provider or "unknown",
    }

    with COSTS_LOCK:
        _job_costs.append(cost_info)

    return cost
