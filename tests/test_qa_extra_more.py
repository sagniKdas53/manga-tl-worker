import json
from unittest.mock import MagicMock, patch

from worker.handlers.qa import _process_qa_llm, process_qa


def test_process_qa_none_mode():
    job_data = {"imageId": "img-1", "qaMode": "none", "ocrRegions": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = {"ocrRegions": []}
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    with (
        patch("requests.get", return_value=mock_get_res),
        patch("requests.post", return_value=mock_post_res),
        patch("worker.config.redis_client.llen", return_value=0),
    ):
        process_qa(job_data)


def test_process_qa_llm_openrouter_success():
    job_data = {
        "imageId": "img-1",
        "qaProvider": "openrouter",
        "qaLlmModel": "meta-llama/llama-3-8b-instruct:free",
        "ocrRegions": [{"id": "r1", "text": "hello"}],
    }
    mock_llm_response = json.dumps({"results": [{"regionId": "r1", "status": "passed", "qaFeedback": "good"}]})
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = {"ocrRegions": [{"id": "r1", "text": "hello"}]}
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200

    with (
        patch("worker.handlers.qa.QA_CONFIG.resolve_key", return_value="fake_key"),
        patch("worker.handlers.qa.try_cloud_ai", return_value=mock_llm_response),
        patch("requests.get", return_value=mock_get_res),
        patch("requests.post", return_value=mock_post_res),
    ):
        _process_qa_llm(job_data)
