from unittest.mock import MagicMock, patch

from worker.rq_tasks import check_stale_job, process_job_rq, update_job_status


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


def test_update_job_status():
    with patch("requests.patch") as mock_patch:
        update_job_status("job-1", "PROCESSING", error="some error", attempt=2)
        mock_patch.assert_called_once()
        _, kwargs = mock_patch.call_args
        assert kwargs["json"]["status"] == "PROCESSING"
        assert kwargs["json"]["error"] == "some error"
        assert kwargs["json"]["attempt"] == "2"


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
