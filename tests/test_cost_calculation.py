import json
import os
from unittest.mock import MagicMock, patch

import pytest

from worker.utils.rate_limit import (
    build_cost_payload,
    estimate_cost,
    get_job_costs,
    reset_job_costs,
    update_model_costs,
)


@patch("worker.utils.rate_limit.redis_client")
def test_estimate_cost_basic(mock_redis):
    # Setup mock redis to return None (no cache)
    mock_redis.get.return_value = None
    reset_job_costs()

    # Test free / local model
    cost = estimate_cost("gemini-2.5-flash", 100, 100, provider="ollama")
    assert cost == 0.0
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["estimated_cost"] == 0.0

    # Test free model (contains :free)
    cost = estimate_cost("google/gemini-flash:free", 100, 100, provider="openrouter")
    assert cost == 0.0

    # Test fallback cost (e.g. gemini-2.5-flash on gemini provider: prompt=0.075, completion=0.30 per million)
    reset_job_costs()
    cost = estimate_cost("google/gemini-2.5-flash", 1000000, 1000000, provider="gemini")
    assert cost == (0.075 + 0.30)
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["estimated_cost"] == 0.375

    # Test fallback cost (gemini-2.5-flash on openrouter: prompt=0.30, completion=2.50 per million)
    reset_job_costs()
    cost = estimate_cost("google/gemini-2.5-flash", 1000000, 1000000, provider="openrouter")
    assert cost == (0.30 + 2.50)


@patch("worker.utils.rate_limit.redis_client")
def test_estimate_cost_bypass_flag(mock_redis):
    mock_redis.get.return_value = None
    reset_job_costs()

    with patch.dict(os.environ, {"DISABLE_COST_CALCULATION": "true"}):
        cost = estimate_cost("google/gemini-2.5-flash", 1000000, 1000000, provider="gemini")
        assert cost is None
        costs = get_job_costs()
        assert len(costs) == 1
        assert costs[0]["estimated_cost"] is None


@patch("worker.utils.rate_limit.redis_client")
def test_estimate_cost_not_available(mock_redis):
    mock_redis.get.return_value = None
    reset_job_costs()

    # Unknown model with no cache or hardcoded fallbacks
    cost = estimate_cost("unknown-model", 1000000, 1000000, provider="unknown")
    assert cost is None
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["estimated_cost"] is None


@patch("worker.utils.rate_limit.redis_client")
@patch("worker.utils.rate_limit.requests.get")
def test_update_model_costs(mock_get, mock_redis):
    mock_redis.get.return_value = None
    mock_redis.keys.return_value = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "endpoints": [
                {
                    "pricing": {
                        "prompt": "0.00000025",  # $0.25 per million
                        "completion": "0.00000150",  # $1.50 per million
                    }
                }
            ]
        }
    }
    mock_get.return_value = mock_resp

    update_model_costs(["google/gemini-3.1-flash-lite"])

    key, payload = mock_redis.set.call_args[0]
    assert key == "model_cost:google/gemini-3.1-flash-lite"
    stored = json.loads(payload)
    assert stored["prompt"] == 0.00000025
    assert stored["completion"] == 0.00000150


@patch("worker.utils.rate_limit.redis_client")
@patch("worker.utils.rate_limit.requests.get")
def test_update_model_costs_stores_the_cheapest_endpoint_not_the_average(mock_get, mock_redis):
    """Requests are routed with sort=price, so they land on the cheapest endpoint. Averaging across
    every endpoint is what overstated the translation model's cost by 3.2x."""
    mock_redis.get.return_value = None
    mock_redis.keys.return_value = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "endpoints": [
                {"pricing": {"prompt": "0.000003", "completion": "0.000006"}},
                {"pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                {"pricing": {"prompt": "0.000002", "completion": "0.000004"}},
            ]
        }
    }
    mock_get.return_value = mock_resp

    update_model_costs(["deepseek/deepseek-v4-pro"])

    stored = json.loads(mock_redis.set.call_args[0][1])
    assert stored["prompt"] == 0.000001
    assert stored["completion"] == 0.000002


@patch("worker.utils.rate_limit.redis_client")
def test_cached_prompt_tokens_are_billed_at_the_cache_read_rate(mock_redis):
    """Cache reads cost a fraction of full input. Charging them at the full prompt rate inflated
    the input component on a pipeline that deliberately engineers cache hits."""
    mock_redis.get.return_value = json.dumps({"prompt": 1e-06, "completion": 2e-06, "cache_read": 1e-07})
    reset_job_costs()

    # 1000 prompt tokens of which 900 were cache hits, plus 100 completion tokens.
    cost = estimate_cost("some/model", 1000, 100, provider="openrouter", cached_tokens=900)

    # 100 fresh prompt @ 1e-6, 900 cached @ 1e-7, 100 completion @ 2e-6
    assert cost == pytest.approx((100 * 1e-06) + (900 * 1e-07) + (100 * 2e-06))

    uncached = estimate_cost("some/model", 1000, 100, provider="openrouter")
    assert uncached is not None and cost is not None
    assert uncached > cost


def test_build_cost_payload_reports_unknown_rather_than_zero():
    """The regression this guards: OCR summed with `c.get("estimated_cost") or 0.0`, so a call with
    no known price became a confident $0.00 that reached the database and the dashboard with
    nothing marking it as unknown."""
    costs = [
        {"estimated_cost": 0.01, "prompt_tokens": 100, "completion_tokens": 10},
        {"estimated_cost": None, "prompt_tokens": 200, "completion_tokens": 20},
    ]

    payload = build_cost_payload(costs)

    assert payload is not None
    # Omitted, not zeroed — a partial total must never present as a complete one.
    assert "estimated_cost" not in payload
    assert payload["unknown_calls"] == 1
    assert payload["priced_calls"] == 1
    # Token counts are still exact, and the per-call detail survives for the database.
    assert payload["prompt_tokens"] == 300
    assert payload["completion_tokens"] == 30
    assert payload["breakdown"] == costs


def test_build_cost_payload_totals_when_every_call_is_priced():
    costs = [
        {"estimated_cost": 0.01, "prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 40},
        {"estimated_cost": 0.02, "prompt_tokens": 200, "completion_tokens": 20, "cached_tokens": 60},
    ]

    payload = build_cost_payload(costs)

    assert payload is not None
    assert payload["estimated_cost"] == pytest.approx(0.03)
    assert payload["unknown_calls"] == 0
    assert payload["priced_calls"] == 2
    assert payload["cached_tokens"] == 100


def test_build_cost_payload_is_none_when_nothing_ran():
    assert build_cost_payload([]) is None
