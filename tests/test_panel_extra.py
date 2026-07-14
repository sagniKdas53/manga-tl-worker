from unittest.mock import patch, MagicMock
from worker.handlers.panel import process_panel_detection


@patch("worker.handlers.panel.redis_client")
@patch("worker.handlers.panel.requests")
@patch("worker.handlers.panel.download_image")
@patch("worker.handlers.panel.detect_panels")
def test_process_panel_detection(mock_detect, mock_download, mock_requests, mock_redis):
    mock_redis.llen.return_value = 5

    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"storagePath": "path"}
    mock_requests.get.return_value = mock_res

    mock_download.return_value = b"image_data"
    mock_detect.return_value = [{"x": 10}]

    process_panel_detection(
        {
            "imageId": "img1",
            "readingDirection": "ltr",
            "pageNumber": 1,
            "chapterNumber": 2,
        }
    )

    mock_requests.post.assert_called()
    payload = mock_requests.post.call_args[1]["json"]
    assert payload["imageId"] == "img1"
    assert payload["panels"] == [{"x": 10}]

    import pytest

    # Test fetch fail
    mock_res.status_code = 404
    with pytest.raises(Exception):
        process_panel_detection({"imageId": "img1"})
    assert mock_requests.post.call_count == 1

    # Test exceptions
    mock_requests.get.side_effect = Exception("failed")
    with pytest.raises(Exception):
        process_panel_detection({"imageId": "img1"})
    assert mock_requests.post.call_count == 1
