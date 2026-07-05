import os
import time
import json
import requests
import threading
from worker.config import redis_client, logger, RENDER_CACHE_DIR

COSTS_FILE = os.environ.get("COSTS_FILE", os.path.join(RENDER_CACHE_DIR, "costs.json"))

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
            if unit in ("s", "sec", "second", "seconds"):
                rpm = val * 60.0
            else:
                rpm = val
        else:
            rpm = float(rate_limit_env)

        if rpm > 0:
            min_delay = 60.0 / rpm
            now = time.time()
            elapsed = now - LAST_REQUEST_TIME
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                print(
                    f"[Translation] Rate limit: Sleeping for {sleep_time:.2f} seconds to respect {rate_limit_env} rate limit...",
                    flush=True,
                )
                time.sleep(sleep_time)
            LAST_REQUEST_TIME = time.time()
    except Exception as e:
        print(f"[Translation] Error enforcing rate limit: {e}", flush=True)


_local_data = threading.local()


def reset_job_costs():
    _local_data.costs = []


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

    # Load existing costs
    persisted_costs = {}
    if os.path.exists(COSTS_FILE):
        try:
            with open(COSTS_FILE, "r") as f:
                persisted_costs = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {COSTS_FILE}: {e}")

    now = time.time()
    one_week = 7 * 24 * 3600

    for model in models:
        try:
            model_key = model.lower()
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
            base_model = model.split(":")[0]  # Strip any :free suffix for API query
            url = f"https://openrouter.ai/api/v1/models/{base_model}/endpoints"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                endpoints = data.get("data", {}).get("endpoints", [])
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

    # Save persisted costs
    try:
        with open(COSTS_FILE, "w") as f:
            json.dump(persisted_costs, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write {COSTS_FILE}: {e}")


def get_job_costs():
    if not hasattr(_local_data, "costs"):
        _local_data.costs = []
    return _local_data.costs


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
        if not hasattr(_local_data, "costs"):
            _local_data.costs = []
        _local_data.costs.append(cost_info)
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
        if not hasattr(_local_data, "costs"):
            _local_data.costs = []
        _local_data.costs.append(cost_info)
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
            # Fallback to checking costs.json
            if os.path.exists(COSTS_FILE):
                with open(COSTS_FILE, "r") as f:
                    persisted = json.load(f)
                    if model_lower in persisted:
                        in_rate = float(persisted[model_lower].get("prompt", 0))
                        out_rate = float(persisted[model_lower].get("completion", 0))
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
            if not hasattr(_local_data, "costs"):
                _local_data.costs = []
            _local_data.costs.append(cost_info)
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

    if not hasattr(_local_data, "costs"):
        _local_data.costs = []
    _local_data.costs.append(cost_info)

    return cost
