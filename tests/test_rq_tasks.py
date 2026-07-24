from unittest.mock import MagicMock, patch

from worker.rq_tasks import process_job_rq


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
@patch("worker.rq_tasks.requests.get")
@patch("worker.rq_tasks.process_panel_detection")
def test_process_job_rq_success(mock_panel, mock_get, mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = False

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "PENDING"}
    mock_get.return_value = mock_response

    job_data = {"jobId": "job-123", "imageId": "img-456"}

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications
    mock_get.assert_called_once()
    mock_update.assert_any_call("job-123", "PROCESSING")
    mock_panel.assert_called_once_with(job_data)


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
@patch("worker.rq_tasks.requests.get")
@patch("worker.rq_tasks.process_panel_detection")
def test_process_job_rq_cancelled(mock_panel, mock_get, mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = False

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    job_data = {"jobId": "job-123", "imageId": "img-456"}

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications: Should skip execution
    mock_get.assert_called_once()
    # update_job_status("PROCESSING") should NOT be called
    for call in mock_update.call_args_list:
        assert call[0][1] != "PROCESSING"
    mock_panel.assert_not_called()


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
@patch("worker.rq_tasks.requests.get")
@patch("worker.rq_tasks.process_panel_detection")
def test_process_job_rq_paused(mock_panel, mock_get, mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = False

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "PAUSED"}
    mock_get.return_value = mock_response

    job_data = {"jobId": "job-123", "imageId": "img-456"}

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications: Should skip execution
    mock_get.assert_called_once()
    # update_job_status("PROCESSING") should NOT be called
    for call in mock_update.call_args_list:
        assert call[0][1] != "PROCESSING"
    mock_panel.assert_not_called()


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
def test_process_job_rq_stale(mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = True

    job_data = {"jobId": "job-123", "imageId": "img-456"}

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications: Should fail and return
    mock_stale.assert_called_once_with("queue:panel-detection", job_data)
    mock_update.assert_called_once_with("job-123", "FAILED", "Stale job")


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
@patch("worker.rq_tasks.requests.get")
@patch("worker.rq_tasks.process_panel_detection")
@patch("worker.rq_tasks.time.sleep")
def test_process_job_rq_retry_logic(mock_sleep, mock_panel, mock_get, mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = False

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "PENDING"}
    mock_get.return_value = mock_response

    # Force failure
    mock_panel.side_effect = Exception("Simulated failure")

    job_data = {
        "jobId": "job-123",
        "imageId": "img-456",
        "attempt": 1,
        "maxAttempts": 3,
    }

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications: Should retry
    mock_panel.assert_called_once()
    mock_update.assert_any_call("job-123", "PENDING", "Simulated failure", 2)


@patch("worker.rq_tasks.check_stale_job")
@patch("worker.rq_tasks.update_job_status")
@patch("worker.rq_tasks.requests.get")
@patch("worker.rq_tasks.process_panel_detection")
@patch("worker.rq_tasks.time.sleep")
def test_process_job_rq_max_attempts(mock_sleep, mock_panel, mock_get, mock_update, mock_stale):
    # Setup mocks
    mock_stale.return_value = False

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "PENDING"}
    mock_get.return_value = mock_response

    # Force failure
    mock_panel.side_effect = Exception("Simulated failure")

    # Already at max attempts
    job_data = {
        "jobId": "job-123",
        "imageId": "img-456",
        "attempt": 3,
        "maxAttempts": 3,
    }

    # Run the worker function
    process_job_rq("queue:panel-detection", job_data)

    # Verifications: Should NOT retry, should set FAILED
    mock_panel.assert_called_once()
    mock_sleep.assert_not_called()
    mock_update.assert_any_call("job-123", "FAILED", "Simulated failure", 3)
