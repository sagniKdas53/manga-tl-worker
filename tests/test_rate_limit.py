import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from worker.config import get_stage, reset_stage, set_stage
from worker.utils.rate_limit import (
    enforce_rate_limit,
    estimate_cost,
    get_job_costs,
    record_llm_call,
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
    mock_redis.keys.return_value = []

    update_model_costs(["meta-llama/llama-3-8b-instruct:free"])

    key, payload = mock_redis.set.call_args[0]
    assert key == "model_cost:meta-llama/llama-3-8b-instruct:free"
    stored = json.loads(payload)
    assert stored["prompt"] == 0.000001
    assert stored["completion"] == 0.000002

    # A paid model with no available endpoint is still surfaced as an error.
    mock_res.status_code = 404
    with pytest.raises(ValueError):
        update_model_costs(["unknown/model"])


@patch("worker.utils.rate_limit.requests")
@patch("worker.utils.rate_limit.redis_client")
def test_free_slug_without_a_free_endpoint_is_rejected(mock_redis, mock_req):
    """A `:free` slug OpenRouter does not actually serve must not be priced at $0.

    The availability check used to run *after* a free short-circuit, so a nonexistent `:free` model
    was written to Redis as a successful $0 while its paid base slug was what actually billed.
    """
    mock_redis.keys.return_value = []

    free_res = MagicMock()
    free_res.status_code = 200
    free_res.json.return_value = {"data": {"endpoints": []}}

    paid_res = MagicMock()
    paid_res.status_code = 200
    paid_res.json.return_value = {
        "data": {"endpoints": [{"pricing": {"prompt": "0.00000013", "completion": "0.00000053"}}]}
    }

    mock_req.get.side_effect = [free_res, paid_res]

    with pytest.raises(ValueError, match="no free endpoint"):
        update_model_costs(["openai/gpt-oss-20b:free"])

    mock_redis.set.assert_not_called()


@patch("worker.utils.rate_limit.requests")
@patch("worker.utils.rate_limit.redis_client")
def test_fallback_models_do_not_stop_the_worker_booting(mock_redis, mock_req):
    """Primary models are fatal on failure; fallbacks are not. Priming the fallback lists would
    otherwise turn three misconfigured `:free` slugs into a boot loop."""
    mock_redis.keys.return_value = []

    res = MagicMock()
    res.status_code = 404
    mock_req.get.return_value = res

    update_model_costs(["unknown/model"], fatal=False)
    mock_redis.set.assert_not_called()


def test_job_costs():
    reset_job_costs()
    estimate_cost("model:free", 100, 50)
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["model"] == "model:free"


def test_chunk_threads_record_against_the_owning_job():
    """Chunked stages hand work to a ThreadPoolExecutor, and a new thread does not inherit the
    spawning thread's context. The call sites submit through copy_context().run so the worker
    thread mutates the same list; without that the costs would land in a thread-local list the job
    never reads."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    reset_job_costs()

    def worker(idx):
        estimate_cost(f"model_{idx}:free", 100, 50)

    num_threads = 10
    with ThreadPoolExecutor(max_workers=4) as executor:
        # copy_context() is evaluated here, in the submitting thread — that is the whole point.
        futures = [executor.submit(contextvars.copy_context().run, worker, i) for i in range(num_threads)]
        for future in futures:
            future.result()

    costs = get_job_costs()
    assert len(costs) == num_threads

    models = {c["model"] for c in costs}
    assert len(models) == num_threads
    for idx in range(num_threads):
        assert f"model_{idx}:free" in models


def test_concurrent_jobs_do_not_see_each_others_costs():
    """The regression this replaces: costs lived in a module-global list that every handler reset on
    entry, so with CONCURRENT_JOBS=5 a job starting mid-flight wiped another job's costs and then
    aggregated the wrong calls into its own total."""
    import contextvars

    def run_job(name):
        reset_job_costs()
        estimate_cost(f"{name}:free", 100, 50)
        return [c["model"] for c in get_job_costs()]

    job_a = contextvars.copy_context().run(run_job, "job-a")
    job_b = contextvars.copy_context().run(run_job, "job-b")

    assert job_a == ["job-a:free"]
    assert job_b == ["job-b:free"]


def test_recorded_calls_carry_the_bound_stage():
    """job_costs.stage was added with the other provenance columns and stayed NULL on every row:
    the shared spending helpers are called by translation, QA and redo alike, so no caller could
    name the stage without every caller naming it. process_job_rq binds it once per job instead."""
    reset_job_costs()
    token = set_stage("translation")
    try:
        estimate_cost("model:free", 100, 50)
    finally:
        reset_stage(token)

    assert get_job_costs()[-1]["stage"] == "translation"
    # And it comes back unbound, rather than labelling whatever runs next on this thread.
    assert get_stage() == ""


def test_an_explicit_stage_still_wins():
    """The argument is not decorative: a caller that knows better than the queue name keeps its say.
    Only the hardcoded one on the redo path was removed, because there the queue knew better."""
    reset_job_costs()
    token = set_stage("qa")
    try:
        record_llm_call("model:free", 100, 50, stage="something-specific")
    finally:
        reset_stage(token)

    assert get_job_costs()[-1]["stage"] == "something-specific"


def test_chunk_threads_inherit_the_stage():
    """Same propagation the cost list depends on: a pool thread does not inherit the spawning
    thread's context, so without copy_context().run at the submit site every chunked OCR and
    translation call would record a blank stage and only the unchunked ones would be attributable."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    reset_job_costs()
    token = set_stage("ocr")
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(contextvars.copy_context().run, estimate_cost, f"model_{i}:free", 100, 50)
                for i in range(6)
            ]
            for future in futures:
                future.result()
    finally:
        reset_stage(token)

    costs = get_job_costs()
    assert len(costs) == 6
    assert {c["stage"] for c in costs} == {"ocr"}


def test_concurrent_jobs_do_not_bleed_stages():
    """An OCR job and a translation job run at the same time on different threads. A module global
    would give both whichever stage was bound last."""
    import contextvars

    def run_job(stage, model):
        reset_job_costs()
        set_stage(stage)
        estimate_cost(model, 100, 50)
        return [(c["model"], c["stage"]) for c in get_job_costs()]

    ocr = contextvars.copy_context().run(run_job, "ocr", "ocr-model:free")
    translation = contextvars.copy_context().run(run_job, "translation", "tl-model:free")

    assert ocr == [("ocr-model:free", "ocr")]
    assert translation == [("tl-model:free", "translation")]


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
