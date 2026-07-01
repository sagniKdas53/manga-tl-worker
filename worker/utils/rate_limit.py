import os
import time

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


import threading

_local_data = threading.local()


def reset_job_costs():
    _local_data.costs = []


def get_job_costs():
    if not hasattr(_local_data, "costs"):
        _local_data.costs = []
    return _local_data.costs


def estimate_cost(model, prompt_tokens, completion_tokens, provider=None):
    if not prompt_tokens or not completion_tokens:
        return 0.0
    in_rate = 0.0
    out_rate = 0.0
    model_lower = (model or "").lower()

    if "deepseek-v4-pro" in model_lower:
        in_rate = 0.435 / 1_000_000
        out_rate = 0.87 / 1_000_000
    elif "gemini-2.5-flash" in model_lower:
        if provider == "gemini":
            in_rate = 0.075 / 1_000_000
            out_rate = 0.30 / 1_000_000
        else:  # OpenRouter
            in_rate = 0.30 / 1_000_000
            out_rate = 2.50 / 1_000_000
    elif "claude-3-5-sonnet" in model_lower:
        in_rate = 3.0 / 1_000_000
        out_rate = 15.0 / 1_000_000

    cost = (prompt_tokens * in_rate) + (completion_tokens * out_rate)
    
    cost_info = {
        "estimated_cost": cost,
        "currency": "USD",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "provider": provider or "unknown"
    }
    
    if not hasattr(_local_data, "costs"):
        _local_data.costs = []
    _local_data.costs.append(cost_info)
    
    return cost
