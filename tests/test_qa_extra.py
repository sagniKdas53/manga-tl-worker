from unittest.mock import MagicMock, patch

from worker.handlers.qa import (
    process_qa,
)


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_none_mode(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "hello"}]}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "none"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_unknown_mode(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": []}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "unknown"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai")
def test_process_qa_llm_mode(mock_cloud, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "hello", "translatedText": "hi"}]}
    mock_requests.get.return_value = mock_res

    mock_cloud.return_value = (
        '{"results": [{"regionId": "1", "qaStatus": "failed", "qaScore": 0.5, "qaFeedback": "bad translation"}]}'
    )

    with patch("worker.handlers.qa.QA_MODE", "llm"), patch("worker.handlers.qa.QA_CONFIG") as mock_qa:
        mock_qa.provider = "openrouter"
        mock_qa.llm_model = "gpt-4o-mini"
        mock_qa.resolve_key.return_value = "dummy"
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "failed"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_get_error(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 500
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "none"):
        process_qa({"imageId": "img1"})

    assert not mock_requests.post.called


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai")
def test_process_qa_llm_empty_regions(mock_cloud, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": []}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "llm"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert not mock_cloud.called


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
def test_process_qa_vlm_mode(mock_dl, mock_cloud_vision, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "ocrRegions": [
            {
                "id": "1",
                "text": "hello",
                "translatedText": "hi",
                "bboxX": 0,
                "bboxY": 0,
                "bboxW": 10,
                "bboxH": 10,
            }
        ]
    }
    mock_requests.get.return_value = mock_res

    import io

    from PIL import Image

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    mock_dl.return_value = valid_bytes

    mock_cloud_vision.return_value = (
        '{"results": [{"regionId": "1", "qaStatus": "passed", "qaScore": 1.0, "qaFeedback": "good"}]}'
    )

    with patch("worker.handlers.qa.QA_MODE", "vlm"), patch("worker.handlers.qa.QA_CONFIG") as mock_qa:
        mock_qa.provider = "openrouter"
        mock_qa.vision_model = "gpt-4o"
        mock_qa.resolve_key.return_value = "dummy"
        with patch("worker.handlers.qa.minio_client.get_object") as mock_minio_get:
            mock_minio_res = MagicMock()
            mock_minio_res.read.return_value = valid_bytes
            mock_minio_get.return_value = mock_minio_res
            process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "passed"
