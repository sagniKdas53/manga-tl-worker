from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import requests

from worker.rq_tasks import (
    _patch_job_status,
    check_stale_job,
    process_job_rq,
    update_job_status,
)


def test_check_stale_job_non_image_queue():
    assert check_stale_job("queue:other", {"imageId": "123"}) is False


def test_check_stale_job_missing_image_id():
    assert check_stale_job("queue:ocr", {}) is False


def test_check_stale_job_image_exists():
    mock_res = MagicMock()
    mock_res.status_code = 200
    with patch("requests.head", return_value=mock_res):
        assert check_stale_job("queue:ocr", {"imageId": "img-1"}) is False


def test_check_stale_job_image_not_found():
    mock_res = MagicMock()
    mock_res.status_code = 404
    with patch("requests.head", return_value=mock_res):
        assert check_stale_job("queue:ocr", {"imageId": "img-1"}) is True


def test_check_stale_job_exception():
    with patch("requests.head", side_effect=Exception("Network error")):
        assert check_stale_job("queue:ocr", {"imageId": "img-1"}) is False


def test_check_stale_job_uses_head_with_a_timeout():
    """AUDIT-W7: this was a GET against the heaviest endpoint, and the only call here with no
    timeout — so a wedged backend held a worker slot open indefinitely, and every job paid for a
    presigned URL plus every panel, region and layer just to read a status code."""
    mock_res = MagicMock()
    mock_res.status_code = 200
    with patch("requests.head", return_value=mock_res) as mock_head, patch("requests.get") as mock_get:
        check_stale_job("queue:ocr", {"imageId": "img-1"})

    mock_get.assert_not_called()
    mock_head.assert_called_once()
    _, kwargs = mock_head.call_args
    assert kwargs["timeout"] == 5, "a stale-job check without a timeout can hang a worker slot"


def _ok_response():
    """A 200 the status-update path will accept.

    The response used to be an unconfigured MagicMock, which was fine while nothing read it.
    AUDIT-P6 makes the code inspect status_code, and a bare MagicMock compares as NotImplemented
    against an int — so the response has to be a real one now.
    """
    res = MagicMock()
    res.status_code = 200
    return res


@contextmanager
def _no_retry_backoff():
    """Drop tenacity's waits for the duration of a test.

    Deliberately not done by patching `time.sleep`: that patches an attribute on the shared time
    module object and silently disarms sleeps in code under test elsewhere. This replaces the
    sleep callable on this one Retrying instance and puts it back afterwards.
    """
    original = _patch_job_status.retry.sleep
    _patch_job_status.retry.sleep = lambda _: None
    try:
        yield
    finally:
        _patch_job_status.retry.sleep = original


def test_update_job_status():
    with patch("requests.patch") as mock_patch:
        mock_patch.return_value = _ok_response()
        update_job_status("job-1", "PROCESSING", error="some error", attempt=2)
        mock_patch.assert_called_once()
        _, kwargs = mock_patch.call_args
        assert kwargs["json"]["status"] == "PROCESSING"
        assert kwargs["json"]["error"] == "some error"
        assert kwargs["json"]["attempt"] == "2"


def test_update_job_status_retries_a_timeout():
    """AUDIT-P6: a COMPLETED lost to one socket timeout re-runs the whole stage.

    The backend has no other source of truth for job completion, so the row stays PROCESSING and
    the stale sweeper requeues it ten minutes later — on top of results that already landed.
    """
    timeout = requests.exceptions.Timeout("read timed out")
    with patch("requests.patch") as mock_patch, _no_retry_backoff():
        mock_patch.side_effect = [timeout, timeout, _ok_response()]
        update_job_status("job-1", "COMPLETED")
        assert mock_patch.call_count == 3, "a timed-out COMPLETED must be retried, not printed and dropped"


def test_update_job_status_retries_a_server_error():
    """AUDIT-P6, the half the entry does not mention: requests does not raise on a 500.

    The original call never looked at the response, so a 5xx lost the update with no exception at
    all — quieter than the timeout the entry describes, and identically expensive.
    """
    error = MagicMock()
    error.status_code = 503
    error.text = "upstream unavailable"
    with patch("requests.patch") as mock_patch, _no_retry_backoff():
        mock_patch.side_effect = [error, _ok_response()]
        update_job_status("job-1", "COMPLETED")
        assert mock_patch.call_count == 2, "a 5xx must be retried — requests raises nothing on it"


def test_update_job_status_gives_up_on_404():
    """A deleted or cancelled job is gone for good; spending the retry budget on it is waste."""
    gone = MagicMock()
    gone.status_code = 404
    gone.text = "not found"
    with patch("requests.patch") as mock_patch, _no_retry_backoff():
        mock_patch.return_value = gone
        update_job_status("job-1", "COMPLETED")
        assert mock_patch.call_count == 1, "a 404 job row will not come back — do not retry it"


def test_update_job_status_survives_exhausting_its_retries():
    """The caller is mid-pipeline; a backend that stays down must not take the job down with it."""
    with patch("requests.patch") as mock_patch, _no_retry_backoff():
        mock_patch.side_effect = requests.exceptions.ConnectionError("refused")
        update_job_status("job-1", "COMPLETED")
        assert mock_patch.call_count == 4, "four attempts, then give up"


def test_process_job_rq_stale():
    job_data = {"jobId": "job-1", "imageId": "img-1"}
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=True),
        patch("worker.rq_tasks.update_job_status") as mock_update,
    ):
        process_job_rq("queue:ocr", job_data)
        mock_update.assert_called_with("job-1", "FAILED", "Stale job")


def test_process_job_rq_deleted_job():
    job_data = {"jobId": "job-1"}
    mock_res = MagicMock()
    mock_res.status_code = 404
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("requests.get", return_value=mock_res),
        patch("worker.rq_tasks.update_job_status") as mock_update,
    ):
        process_job_rq("queue:ocr", job_data)
        mock_update.assert_not_called()


def test_process_job_rq_non_pending_job():
    job_data = {"jobId": "job-1"}
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"status": "COMPLETED"}
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("requests.get", return_value=mock_res),
        patch("worker.rq_tasks.update_job_status") as mock_update,
    ):
        process_job_rq("queue:ocr", job_data)
        mock_update.assert_not_called()


def test_process_job_rq_queues():
    job_data = {"jobId": "job-1"}
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"status": "PENDING"}

    queues_to_handler = [
        ("queue:panel-detection", "worker.rq_tasks.process_panel_detection"),
        ("queue:ocr", "worker.rq_tasks.process_ocr"),
        ("queue:layout", "worker.rq_tasks.process_layout"),
        ("queue:translation", "worker.rq_tasks.process_translation"),
        ("queue:region-redo-ocr", "worker.rq_tasks.process_region_redo"),
        ("queue:render", "worker.rq_tasks.process_render"),
        ("queue:qa", "worker.rq_tasks.process_qa"),
        ("queue:qa-re-ocr", "worker.rq_tasks.process_qa_re_ocr"),
    ]

    for qname, target in queues_to_handler:
        with (
            patch("worker.rq_tasks.check_stale_job", return_value=False),
            patch("requests.get", return_value=mock_res),
            patch("worker.rq_tasks.update_job_status"),
            patch(target) as mock_handler,
        ):
            process_job_rq(qname, job_data)
            mock_handler.assert_called_once_with(job_data)


def test_process_job_rq_error_retry():
    job_data = {"jobId": "job-1", "attempt": 1, "maxAttempts": 3}
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"status": "PENDING"}

    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("requests.get", return_value=mock_res),
        patch("worker.rq_tasks.process_ocr", side_effect=ValueError("Boom")),
        patch("worker.rq_tasks.update_job_status") as mock_update,
    ):
        process_job_rq("queue:ocr", job_data)
        mock_update.assert_called_with("job-1", "PENDING", "Boom", 2)


def test_process_job_rq_error_max_attempts():
    job_data = {"jobId": "job-1", "attempt": 3, "maxAttempts": 3}
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"status": "PENDING"}

    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("requests.get", return_value=mock_res),
        patch("worker.rq_tasks.process_ocr", side_effect=ValueError("Max Error")),
        patch("worker.rq_tasks.update_job_status") as mock_update,
    ):
        process_job_rq("queue:ocr", job_data)
        mock_update.assert_called_with("job-1", "FAILED", "Max Error", 3)
