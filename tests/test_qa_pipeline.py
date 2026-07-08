import io
import json
from unittest.mock import patch, MagicMock
from PIL import Image

from worker.handlers.qa import process_qa


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "llm")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_llm_gemini(mock_qa_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_qa_config.provider = "gemini"
    mock_qa_config.resolve_key.return_value = "fake-gemini-key"
    mock_qa_config.llm_model = "gemini-1.5-pro"

    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "confidence": 0.9,
                "translatedText": "Hello",
                "translationScore": 0.95,
                "bubbleReadingOrder": 1,
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "passed",
                    "qaScore": 0.99,
                    "qaFeedback": "Perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa({"imageId": "image-uuid-1"})

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "gemini"
    assert args[1] == "fake-gemini-key"
    assert args[2] == "gemini-1.5-pro"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["qaResults"][0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "llm")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_llm_nvidia(mock_qa_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_qa_config.provider = "nvidia"
    mock_qa_config.resolve_key.return_value = "fake-nvidia-key"
    mock_qa_config.llm_model = "google/gemma-3n-e4b-it"

    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "confidence": 0.9,
                "translatedText": "Hello",
                "translationScore": 0.95,
                "bubbleReadingOrder": 1,
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "passed",
                    "qaScore": 0.99,
                    "qaFeedback": "Perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa({"imageId": "image-uuid-1"})

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "google/gemma-3n-e4b-it"


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "vlm")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_vlm_openrouter(
    mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm
):
    mock_qa_config.provider = "openrouter"
    mock_qa_config.resolve_key.return_value = "fake-openrouter-key"
    mock_qa_config.vlm_model = "google/gemini-1.5-pro"

    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10,
                "bboxY": 20,
                "bboxW": 100,
                "bboxH": 50,
                "translatedText": "Hello",
                "bubbleReadingOrder": 1,
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    mock_minio_res = MagicMock()
    mock_minio_res.read.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value = mock_minio_res

    mock_try_cloud_vlm.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "passed",
                    "qaScore": 0.99,
                    "qaFeedback": "VLM match perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa({"imageId": "image-uuid-1"})

    mock_try_cloud_vlm.assert_called_once()
    args, kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "openrouter"
    assert args[1] == "fake-openrouter-key"
    assert args[2] == "google/gemini-1.5-pro"


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "vlm")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_vlm_nvidia(
    mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm
):
    mock_qa_config.provider = "nvidia"
    mock_qa_config.resolve_key.return_value = "fake-nvidia-key"
    mock_qa_config.vlm_model = "nvidia/nemotron-nano-12b-v2-vl"

    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10,
                "bboxY": 20,
                "bboxW": 100,
                "bboxH": 50,
                "translatedText": "Hello",
                "bubbleReadingOrder": 1,
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    mock_minio_res = MagicMock()
    mock_minio_res.read.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value = mock_minio_res

    mock_try_cloud_vlm.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "passed",
                    "qaScore": 0.99,
                    "qaFeedback": "VLM match perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa({"imageId": "image-uuid-1"})

    mock_try_cloud_vlm.assert_called_once()
    args, kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "nvidia/nemotron-nano-12b-v2-vl"


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.render.render_image_core")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_hybrid_flow(
    mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_render, mock_try_llm, mock_try_vlm
):
    mock_qa_config.provider = "gemini"
    mock_qa_config.resolve_key.return_value = "fake-key"
    mock_qa_config.llm_model = "gemini-1.5-flash"
    mock_qa_config.vlm_model = "gemini-1.5-pro"

    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10,
                "bboxY": 20,
                "bboxW": 100,
                "bboxH": 50,
                "translatedText": "Hello",
                "bubbleReadingOrder": 1,
            }
        ],
    }
    
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    # LLM QA output
    mock_try_llm.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "direct_fix",
                    "qaScore": 0.8,
                    "qaFeedback": "Needs correction.",
                    "directFix": {"correctedText": "Hi"},
                }
            ]
        }
    )

    # Mock prepare status
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Mock render
    mock_render.return_value = True

    # Mock image download & MinIO download for VLM
    mock_download.return_value = get_dummy_image_bytes()
    mock_minio_res = MagicMock()
    mock_minio_res.read.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value = mock_minio_res

    # VLM QA output
    mock_try_vlm.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "passed",
                    "qaScore": 0.95,
                    "qaFeedback": "VLM verified.",
                }
            ]
        }
    )

    process_qa({"imageId": "image-uuid-1", "qaMode": "hybrid"})

    # Verify LLM was called
    mock_try_llm.assert_called_once()
    # Verify render was called
    mock_render.assert_called_once_with("image-uuid-1")
    # Verify VLM was called
    mock_try_vlm.assert_called_once()
    
    # Verify post callbacks (one to prepare, one to final qa)
    assert mock_post.call_count == 2

