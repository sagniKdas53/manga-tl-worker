"""No models/providers: exercise authority at the actual worker dispatch/request seams."""

import contextvars
from unittest.mock import Mock, patch

import pytest
import requests

from worker.config import backend_headers
from worker.job_attempt import AttemptExpired, bind_attempt, current_attempt, record_progress
from worker.rq_tasks import JobHeartbeat, process_job_rq, update_job_status


@pytest.fixture
def job():
    return {
        "jobId": "job-1",
        "imageId": "image-1",
        "attempt": 2,
        "inputGeneration": 7,
        "leaseToken": "lease-two",
        "maxRuntimeSeconds": 3600,
    }


def response(code=200, data=None):
    result = Mock(status_code=code, text="response")
    result.json.return_value = data or {"status": "PENDING"}
    return result


def test_retry_request_cannot_claim_the_next_attempt(job):
    token = bind_attempt(job)
    try:
        with patch("worker.rq_tasks.requests.patch", return_value=response()) as request:
            assert update_job_status(job["jobId"], "PENDING", "retry", 3)
        assert request.call_args.kwargs["json"]["attempt"] == "3"
        headers = request.call_args.kwargs["headers"]
        assert headers["X-Job-Attempt"] == "2"
        assert headers["X-Lease-Token"] == "lease-two"
        assert headers["X-Input-Generation"] == "7"
    finally:
        current_attempt.reset(token)
    assert "X-Job-Id" not in backend_headers()


@pytest.mark.parametrize("status", [404, 409, 403])
def test_rejected_start_does_no_inference(job, status):
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("worker.rq_tasks.requests.get", return_value=response()),
        patch("worker.rq_tasks.requests.patch", return_value=response(status)),
        patch("worker.rq_tasks.process_ocr") as inference,
    ):
        process_job_rq("queue:ocr", job)
    inference.assert_not_called()
    assert current_attempt.get() is None


def test_start_check_network_failure_fails_closed(job):
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("worker.rq_tasks.requests.get", side_effect=requests.ConnectionError()),
        patch("worker.rq_tasks.process_translation") as inference,
    ):
        process_job_rq("queue:translation", job)
    inference.assert_not_called()


def test_heartbeat_renews_through_old_ten_minute_boundary_then_stops_at_deadline(job):
    clock = [0.0]
    with patch("worker.job_attempt.time.monotonic", side_effect=lambda: clock[0]):
        token = bind_attempt(job)
        try:
            heartbeat = JobHeartbeat(job["jobId"])

            # Model waits without sleeping or running an expensive inference.
            def tick(_):
                clock[0] += 30
                return False

            heartbeat.stop = Mock()
            heartbeat.stop.wait.side_effect = tick
            with patch("worker.rq_tasks.requests.patch", return_value=response()) as request:
                heartbeat.run()
            assert request.call_count == 119
            assert clock[0] == 3600
            assert all(c.kwargs["json"]["heartbeat"] == "true" for c in request.call_args_list)
            expired = current_attempt.get()
            assert expired is not None and expired.revoked.is_set()
            with pytest.raises(AttemptExpired):
                backend_headers()
        finally:
            current_attempt.reset(token)


def test_heartbeat_rejection_revokes_callback_and_progress(job):
    token = bind_attempt(job)
    try:
        heartbeat = JobHeartbeat(job["jobId"])
        heartbeat.stop = Mock()
        heartbeat.stop.wait.return_value = False
        with patch("worker.rq_tasks.requests.patch", return_value=response(409)) as request:
            heartbeat.run()
        assert request.call_count == 1
        with pytest.raises(AttemptExpired):
            backend_headers()
        with pytest.raises(AttemptExpired):
            record_progress()
    finally:
        current_attempt.reset(token)


def test_progress_is_shared_with_heartbeat_but_not_other_jobs(job):
    token = bind_attempt(job)
    try:
        copy = contextvars.copy_context()
        copy.run(record_progress)
        record_progress()
        with patch("worker.rq_tasks.requests.patch", return_value=response()) as request:
            update_job_status(job["jobId"], "PROCESSING", heartbeat=True)
        assert request.call_args.kwargs["json"]["progress"] == "2"
        assert contextvars.Context().run(current_attempt.get) is None
    finally:
        current_attempt.reset(token)
