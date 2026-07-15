import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure environment variables are set for tests
os.environ["WORKER_API_SECRET"] = "test_secret"
os.environ["CONCURRENT_WORKERS"] = "2"

# Import after setting env vars
import worker.health_server as hs


@pytest.fixture
def mock_request_handler():
    handler = MagicMock()
    handler.headers = {"WORKER_API_SECRET": "test_secret"}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile.write = MagicMock()
    return handler


def test_check_auth_logic(mock_request_handler):
    mock_request_handler.headers = {}
    hs.WORKER_API_SECRET = "test_secret"

    result = hs.HealthCheckHandler.check_auth(mock_request_handler)

    assert result is False
    mock_request_handler.send_response.assert_called_with(401)


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.threading.Thread")
def test_job_submission_success(mock_thread, mock_request_handler):
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    body = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "123"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)

    mock_request_handler.send_response.assert_called_with(202)
    mock_thread.assert_called_once()


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 2)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
def test_job_submission_rate_limit(mock_request_handler):
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    body = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "123"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)

    mock_request_handler.send_response.assert_called_with(429)


@patch("worker.health_server.process_job_rq")
def test_job_execution_wrapper(mock_process_job):
    # Test that _run_job_async wrapper decrements ACTIVE_JOBS
    hs.ACTIVE_JOBS = 1

    hs._run_job_async("queue:ocr", {"id": "123"})

    mock_process_job.assert_called_once_with("queue:ocr", {"id": "123"})
    assert hs.ACTIVE_JOBS == 0


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.ACTIVE_HEAVY_JOBS", 0)
@patch("worker.health_server.ACTIVE_LIGHT_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.threading.Thread")
def test_heavy_light_concurrency_slots(mock_thread, mock_request_handler):
    # Reset helper attributes on mock request handler
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    # Reset globals in mock health_server namespace for this test scope
    hs.ACTIVE_JOBS = 0
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 0

    # Step 1: Submit Heavy Job (should succeed)
    body_heavy_1 = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "1"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body_heavy_1.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_heavy_1))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 0

    # Reset mock call history
    mock_request_handler.send_response.reset_mock()

    # Step 2: Submit another Heavy Job (should fail with 429)
    body_heavy_2 = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "2"}})
    mock_request_handler.rfile.read.return_value = body_heavy_2.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_heavy_2))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    # Check that slot counts remained unchanged
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 0

    # Reset mock call history
    mock_request_handler.send_response.reset_mock()

    # Step 3: Submit Light Job (should succeed in parallel)
    body_light_1 = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "3"}})
    mock_request_handler.rfile.read.return_value = body_light_1.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_light_1))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 1

    # Reset mock call history
    mock_request_handler.send_response.reset_mock()

    # Step 4: Submit another Light Job (should fail with 429)
    body_light_2 = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "4"}})
    mock_request_handler.rfile.read.return_value = body_light_2.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_light_2))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 1


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.ACTIVE_HEAVY_JOBS", 0)
@patch("worker.health_server.ACTIVE_LIGHT_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.REUSE_IDLE_SLOTS", False)
@patch("worker.health_server.threading.Thread")
def test_scenario_three_jobs_concurrency(mock_thread, mock_request_handler):
    # Reset mock request handler
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    # Reset globals in mock health_server namespace for this test scope
    hs.ACTIVE_JOBS = 0
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 0

    # 1. Job 1 submitted for OCR (heavy slot).
    # Since heavy slot is empty, it must succeed.
    body_job1_ocr = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "job1"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body_job1_ocr.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_job1_ocr))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 0
    assert hs.ACTIVE_JOBS == 1

    mock_request_handler.send_response.reset_mock()

    # 2. Job 2 submitted for OCR (heavy slot).
    # Since active heavy jobs is already 1, it must be rejected with 429.
    body_job2_ocr = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "job2"}})
    mock_request_handler.rfile.read.return_value = body_job2_ocr.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_job2_ocr))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 0

    mock_request_handler.send_response.reset_mock()

    # 3. Job 1 finishes OCR. Heavy slot becomes free (active heavy goes to 0).
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_JOBS = hs.ACTIVE_HEAVY_JOBS + hs.ACTIVE_LIGHT_JOBS

    # 4. Job 1 is now submitted for Translation (light slot).
    # Since light slot is empty, it must succeed.
    body_job1_tl = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "job1"}})
    mock_request_handler.rfile.read.return_value = body_job1_tl.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_job1_tl))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 0
    assert hs.ACTIVE_LIGHT_JOBS == 1
    assert hs.ACTIVE_JOBS == 1

    mock_request_handler.send_response.reset_mock()

    # 5. Now Job 2 can start OCR (heavy slot) since heavy slot is free.
    # Since active heavy jobs is 0, this must succeed.
    mock_request_handler.rfile.read.return_value = body_job2_ocr.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_job2_ocr))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 1
    assert hs.ACTIVE_JOBS == 2

    mock_request_handler.send_response.reset_mock()

    # 6. Job 2 finishes OCR. Heavy slot becomes free (active heavy goes to 0).
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_JOBS = hs.ACTIVE_HEAVY_JOBS + hs.ACTIVE_LIGHT_JOBS

    # 7. Job 2 is submitted for Translation (light slot).
    # But Job 1 is still running Translation (active light jobs = 1).
    # Since active light jobs is already 1, this must be rejected with 429.
    body_job2_tl = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "job2"}})
    mock_request_handler.rfile.read.return_value = body_job2_tl.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body_job2_tl))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_HEAVY_JOBS == 0
    assert hs.ACTIVE_LIGHT_JOBS == 1


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.ACTIVE_HEAVY_JOBS", 0)
@patch("worker.health_server.ACTIVE_LIGHT_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 3)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 2)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 1)
@patch("worker.health_server.threading.Thread")
def test_configurable_heavy_slots(mock_thread, mock_request_handler):
    """With MAX_HEAVY_SLOTS=2, two heavy jobs should be accepted; the third should get 429."""
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 0
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 0

    # Heavy job 1 — accepted
    body = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "h1"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1

    mock_request_handler.send_response.reset_mock()

    # Heavy job 2 — accepted (second heavy slot)
    body = json.dumps({"queue_name": "queue:panel-detection", "job_data": {"id": "h2"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 2

    mock_request_handler.send_response.reset_mock()

    # Heavy job 3 — rejected (both heavy slots occupied)
    body = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "h3"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_HEAVY_JOBS == 2


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.ACTIVE_HEAVY_JOBS", 0)
@patch("worker.health_server.ACTIVE_LIGHT_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 3)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 1)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 2)
@patch("worker.health_server.REUSE_IDLE_SLOTS", False)
@patch("worker.health_server.threading.Thread")
def test_configurable_light_slots(mock_thread, mock_request_handler):
    """With MAX_LIGHT_SLOTS=2, two light jobs should be accepted; the third should get 429."""
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 0
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 0

    # Light job 1 — accepted
    body = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "l1"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_LIGHT_JOBS == 1

    mock_request_handler.send_response.reset_mock()

    # Light job 2 — accepted (second light slot)
    body = json.dumps({"queue_name": "queue:layout", "job_data": {"id": "l2"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_LIGHT_JOBS == 2

    mock_request_handler.send_response.reset_mock()

    # Light job 3 — rejected (both light slots occupied)
    body = json.dumps({"queue_name": "queue:render", "job_data": {"id": "l3"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_LIGHT_JOBS == 2


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.ACTIVE_JOBS", 0)
@patch("worker.health_server.ACTIVE_HEAVY_JOBS", 0)
@patch("worker.health_server.ACTIVE_LIGHT_JOBS", 0)
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 3)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 1)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 2)
@patch("worker.health_server.threading.Thread")
def test_default_slot_allocation_concurrent_3(mock_thread, mock_request_handler):
    """With CONCURRENT_JOBS=3 default slots (1 heavy + 2 light), accept 1 heavy + 2 light, reject a 3rd light."""
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 0
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 0

    # Heavy job — accepted
    body = json.dumps({"queue_name": "queue:ocr", "job_data": {"id": "h1"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_HEAVY_JOBS == 1
    assert hs.ACTIVE_LIGHT_JOBS == 0

    mock_request_handler.send_response.reset_mock()

    # Light job 1 — accepted
    body = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "l1"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_LIGHT_JOBS == 1

    mock_request_handler.send_response.reset_mock()

    # Light job 2 — accepted (second light slot)
    body = json.dumps({"queue_name": "queue:layout", "job_data": {"id": "l2"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_LIGHT_JOBS == 2
    assert hs.ACTIVE_JOBS == 3

    mock_request_handler.send_response.reset_mock()

    # Light job 3 — rejected (all 3 concurrent slots filled)
    body = json.dumps({"queue_name": "queue:render", "job_data": {"id": "l3"}})
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)
    assert hs.ACTIVE_LIGHT_JOBS == 2
    assert hs.ACTIVE_JOBS == 3


def test_region_redo_removed_from_heavy():
    """Verify queue:region-redo is NOT in HEAVY_QUEUES or LIGHT_QUEUES after legacy removal."""
    assert "queue:region-redo" not in hs.HEAVY_QUEUES
    assert "queue:region-redo" not in hs.LIGHT_QUEUES

@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 1)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 1)
@patch("worker.health_server.REUSE_IDLE_SLOTS", True)
@patch("worker.health_server.threading.Thread")
def test_light_overflow_when_heavy_idle(mock_thread, mock_request_handler):
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 1
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 1

    body = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "light2"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(202)
    assert hs.ACTIVE_LIGHT_JOBS == 2
    assert hs.ACTIVE_JOBS == 2


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 1)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 1)
@patch("worker.health_server.REUSE_IDLE_SLOTS", True)
def test_light_overflow_blocked_at_global_limit(mock_request_handler):
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 2
    hs.ACTIVE_HEAVY_JOBS = 1
    hs.ACTIVE_LIGHT_JOBS = 1

    body = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "light3"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)


@patch("worker.health_server.WORKER_API_SECRET", "test_secret")
@patch("worker.health_server.MAX_CONCURRENT_JOBS", 2)
@patch("worker.health_server.MAX_HEAVY_SLOTS", 1)
@patch("worker.health_server.MAX_LIGHT_SLOTS", 1)
@patch("worker.health_server.REUSE_IDLE_SLOTS", False)
def test_light_overflow_disabled(mock_request_handler):
    mock_request_handler.check_auth = MagicMock(return_value=True)
    mock_request_handler.command = "POST"
    mock_request_handler.path = "/api/v1/jobs/submit"

    hs.ACTIVE_JOBS = 1
    hs.ACTIVE_HEAVY_JOBS = 0
    hs.ACTIVE_LIGHT_JOBS = 1

    body = json.dumps({"queue_name": "queue:translation", "job_data": {"id": "light2"}})
    mock_request_handler.rfile = MagicMock()
    mock_request_handler.rfile.read.return_value = body.encode("utf-8")
    mock_request_handler.headers["Content-Length"] = str(len(body))

    hs.HealthCheckHandler.do_POST(mock_request_handler)
    mock_request_handler.send_response.assert_called_with(429)

