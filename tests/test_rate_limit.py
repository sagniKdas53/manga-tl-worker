import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from worker.utils.rate_limit import (
    enforce_rate_limit,
    estimate_cost,
    get_job_costs,
    reset_job_costs,
    update_model_costs,
)


@patch("worker.utils.rate_limit.time")
def test_enforce_rate_limit(mock_time):
    # Test valid limit
    mock_time.time.return_value = 100
    os.environ["RATE_LIMIT"] = "60/min"

    import worker.utils.rate_limit as rlimit

    # lock_key = "global" (provider=None)
    rlimit.PROVIDER_LAST_REQUEST_TIME = {"global": 99.5}

    # 60/min = 1/s. Min delay = 1. Elapsed = 0.5. Should sleep for 0.5.
    enforce_rate_limit()
    mock_time.sleep.assert_called_with(0.5)

    # Test no-delay path (elapsed >= delay)
    rlimit.PROVIDER_LAST_REQUEST_TIME = {"default:60/min": 98.0}
    mock_time.sleep.reset_mock()
    enforce_rate_limit()
    mock_time.sleep.assert_not_called()

    del os.environ["RATE_LIMIT"]


@patch("worker.utils.rate_limit.time")
def test_unset_rate_limit_is_unlimited(mock_time, monkeypatch):
    """AUDIT-W2: with RATE_LIMIT unset nothing throttles — including a provider with no rateLimits.

    docker-compose.yml used to ship `RATE_LIMIT=${RATE_LIMIT:-10}`, so the fallback bucket was
    always populated and a provider added to providers.json without its own `rateLimits` silently
    inherited one call every 6 seconds. The compose default is now empty, which makes this the
    behaviour the deployment actually gets.
    """
    monkeypatch.delenv("RATE_LIMIT", raising=False)
    mock_time.time.return_value = 100

    import worker.utils.rate_limit as rlimit

    # A call one millisecond ago would sleep under any non-zero limit.
    rlimit.PROVIDER_LAST_REQUEST_TIME = {"global": 99.999, "no-limits-provider": 99.999}

    enforce_rate_limit()
    enforce_rate_limit(provider="no-limits-provider", provider_rpm=None)

    mock_time.sleep.assert_not_called()


@patch("worker.utils.rate_limit.time")
def test_enforce_rate_limit_does_not_log_as_translation(mock_time, caplog):
    """AUDIT-Q3: the limiter is shared by translation, OCR and QA — it must not claim to be one.

    Both messages in enforce_rate_limit were prefixed `[Translation]`, so an OCR or QA stall showed
    up in the worker log under the wrong stage.

    Reads caplog rather than capsys because these go through the logger now; the stall is a
    per-attempt event and lands at DEBUG, hence the explicit level.
    """
    mock_time.time.return_value = 100
    os.environ["RATE_LIMIT"] = "60/min"

    import worker.utils.rate_limit as rlimit

    rlimit.PROVIDER_LAST_REQUEST_TIME = {"ocr-provider": 99.5}
    with caplog.at_level(logging.DEBUG):
        enforce_rate_limit(provider="ocr-provider")
    mock_time.sleep.assert_called_with(0.5)

    out = caplog.text
    assert "Sleeping for 0.50 seconds" in out
    assert "[RateLimit]" in out
    assert "[Translation]" not in out

    del os.environ["RATE_LIMIT"]


@patch("worker.utils.rate_limit.requests")
@patch("worker.utils.rate_limit.redis_client")
def test_update_model_costs(mock_redis, mock_req, tmp_path):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "data": {"endpoints": [{"pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}
    }
    mock_req.get.return_value = mock_res

    update_model_costs(["meta-llama/llama-3-8b-instruct:free"])

    mock_redis.set.assert_called_with(
        "model_cost:meta-llama/llama-3-8b-instruct:free",
        json.dumps({"prompt": 0.0, "completion": 0.0}),
    )

    # A paid model with no available endpoint is still surfaced as an error.
    mock_res.status_code = 404
    with pytest.raises(ValueError):
        update_model_costs(["unknown/model"])


def test_job_costs():
    reset_job_costs()
    estimate_cost("model:free", 100, 50)
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["model"] == "model:free"


def test_concurrent_cost_tracking():
    from concurrent.futures import ThreadPoolExecutor

    reset_job_costs()

    def worker(idx):
        estimate_cost(f"model_{idx}", 100, 50)

    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(worker, range(num_threads))

    costs = get_job_costs()
    assert len(costs) == num_threads

    models = {c["model"] for c in costs}
    assert len(models) == num_threads
    for idx in range(num_threads):
        assert f"model_{idx}" in models


@patch("worker.utils.rate_limit.time")
def test_concurrent_rate_limiting(mock_time):
    mock_time.time.return_value = 100.0
    os.environ["RATE_LIMIT"] = "60"  # 1s delay

    import worker.utils.rate_limit as rlimit

    # lock_key = "global" (provider=None)
    rlimit.PROVIDER_LAST_REQUEST_TIME = {"global": 99.5}

    current_time = 100.0

    def mock_sleep(seconds):
        nonlocal current_time
        current_time += seconds
        mock_time.time.return_value = current_time

    mock_time.sleep.side_effect = mock_sleep

    def worker():
        enforce_rate_limit()

    from concurrent.futures import ThreadPoolExecutor

    num_threads = 3
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    assert mock_time.sleep.call_count == 3
    sleep_args = [call[0][0] for call in mock_time.sleep.call_args_list]

    # Under concurrent execution, the sleep times for each thread depend on the thread scheduling.
    # The mathematically valid sleep times are 0.5, 1.0, 1.5, 2.0, 2.5 depending on interleaving.
    for s in sleep_args:
        assert any(pytest.approx(s) == val for val in (0.5, 1.0, 1.5, 2.0, 2.5))

    # The sum of all sleep times must conform to one of the valid schedules (2.5, 3.0, 3.5, 4.0, 4.5)
    total_sleep = sum(sleep_args)
    assert any(pytest.approx(total_sleep) == val for val in (2.5, 3.0, 3.5, 4.0, 4.5))

    del os.environ["RATE_LIMIT"]
