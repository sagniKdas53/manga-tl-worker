from unittest.mock import MagicMock, patch

import pytest
import requests

from worker.rq_tasks import process_job_rq


def _job():
    return {
        "jobId": "job-cleanup-1",
        "imageId": "image-1",
        "attempt": 1,
        "maxAttempts": 3,
    }


def _pending_response():
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "PENDING"}
    return response


@pytest.mark.parametrize("callback_decision", ["complete", "failed"])
def test_cleanup_accepted_callback_leaves_terminal_decision_to_backend(callback_decision):
    job = _job()
    job["syntheticCallbackDecision"] = callback_decision
    heartbeat = MagicMock()
    statuses = []

    def record_status(job_id, status, *args, **kwargs):
        statuses.append(status)
        return True

    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("worker.rq_tasks.requests.get", return_value=_pending_response()),
        patch("worker.rq_tasks.update_job_status", side_effect=record_status),
        patch("worker.rq_tasks.JobHeartbeat", return_value=heartbeat),
        patch("worker.handlers.cleanup.process_cleanup", return_value=None) as process_cleanup,
    ):
        process_job_rq("queue:cleanup", job)

    process_cleanup.assert_called_once_with(job)
    assert statuses == ["PROCESSING"]
    heartbeat.thread.start.assert_called_once_with()
    heartbeat.close.assert_called_once_with()


def test_cleanup_transport_exception_uses_bounded_pending_retry_and_closes_heartbeat():
    job = _job()
    heartbeat = MagicMock()
    status_calls = []

    def record_status(job_id, status, *args, **kwargs):
        status_calls.append((status, args, kwargs))
        return True

    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("worker.rq_tasks.requests.get", return_value=_pending_response()),
        patch("worker.rq_tasks.update_job_status", side_effect=record_status),
        patch("worker.rq_tasks.JobHeartbeat", return_value=heartbeat),
        patch(
            "worker.handlers.cleanup.process_cleanup",
            side_effect=requests.ConnectionError("cleanup callback unavailable"),
        ),
    ):
        process_job_rq("queue:cleanup", job)

    assert [status for status, _args, _kwargs in status_calls] == ["PROCESSING", "PENDING"]
    assert status_calls[-1][1] == ("cleanup callback unavailable", 2)
    heartbeat.thread.start.assert_called_once_with()
    heartbeat.close.assert_called_once_with()
