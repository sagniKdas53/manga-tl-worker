import os
import json
from unittest.mock import patch, MagicMock

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
