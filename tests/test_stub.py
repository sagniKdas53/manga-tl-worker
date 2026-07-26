from unittest.mock import MagicMock, patch

from worker.handlers.stub import process_stub


def test_process_stub_success():
    job_data = {"imageId": "img-123"}
    mock_res = MagicMock()
    mock_res.status_code = 200

    with patch("time.sleep"), patch("requests.post", return_value=mock_res) as mock_post:
        process_stub(job_data, "test_job")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "test_job" in args[0]
        assert kwargs["json"] == {"imageId": "img-123"}


def test_process_stub_failure():
    job_data = {"imageId": "img-123"}

    with patch("time.sleep"), patch("requests.post", side_effect=Exception("Connection refused")):
        process_stub(job_data, "test_job")
