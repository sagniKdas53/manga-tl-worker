from unittest.mock import patch, MagicMock

from worker.handlers.layout import process_layout


@patch("worker.handlers.layout.requests.get")
@patch("worker.handlers.layout.requests.post")
@patch("worker.handlers.layout.redis_client.llen")
def test_process_layout_success(mock_llen, mock_post, mock_get):
    mock_llen.return_value = 0

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "ocrRegions": [
            {
                "id": "r1",
                "bboxX": 10,
                "bboxY": 10,
                "bboxW": 100,
                "bboxH": 50,
                "text": "Hello",
            }
        ],
        "panels": [{"id": "p1", "bboxX": 0, "bboxY": 0, "bboxW": 500, "bboxH": 500}],
    }
    mock_get.return_value = mock_get_resp

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post.return_value = mock_post_resp

    job_data = {"imageId": "img123", "pageNumber": 1, "chapterNumber": 1}
    process_layout(job_data)

    mock_get.assert_called_once()
    mock_post.assert_called_once()

    # Check payload of callback
    call_args = mock_post.call_args
    assert call_args is not None
    payload = call_args.kwargs.get("json")
    assert payload is not None
    assert payload["imageId"] == "img123"
    assert len(payload["regionTypes"]) == 1
    assert payload["regionTypes"][0]["regionId"] == "r1"
    assert "conversations" in payload


@patch("worker.handlers.layout.requests.get")
def test_process_layout_api_failure(mock_get):
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 500
    mock_get.return_value = mock_get_resp

    job_data = {"imageId": "img123"}
    process_layout(job_data)

    mock_get.assert_called_once()
    # It should early return and not post anything.
