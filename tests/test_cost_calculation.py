import os
import json
from unittest.mock import patch, MagicMock
from worker.utils.rate_limit import (
    estimate_cost,
    update_model_costs,
    reset_job_costs,
    get_job_costs,
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
    cost = estimate_cost(
        "google/gemini-2.5-flash", 1000000, 1000000, provider="openrouter"
    )
    assert cost == (0.30 + 2.50)


@patch("worker.utils.rate_limit.redis_client")
def test_estimate_cost_bypass_flag(mock_redis):
    mock_redis.get.return_value = None
    reset_job_costs()

    with patch.dict(os.environ, {"DISABLE_COST_CALCULATION": "true"}):
        cost = estimate_cost(
            "google/gemini-2.5-flash", 1000000, 1000000, provider="gemini"
        )
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
@patch("worker.utils.rate_limit.os.path.exists")
@patch("worker.utils.rate_limit.open")
def test_update_model_costs(mock_open, mock_exists, mock_get, mock_redis):
    # Mock file system and requests
    mock_exists.return_value = False
    mock_redis.get.return_value = None

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

    with patch("worker.utils.rate_limit.COSTS_FILE", "/dummy/costs.json"):
        # We need mock_open to mock writing
        m_file = MagicMock()
        mock_open.return_value.__enter__.return_value = m_file

        update_model_costs(["google/gemini-3.1-flash-lite"])

        # Check redis set call
        mock_redis.set.assert_called_with(
            "model_cost:google/gemini-3.1-flash-lite",
            json.dumps({"prompt": 0.25, "completion": 1.50}),
        )
