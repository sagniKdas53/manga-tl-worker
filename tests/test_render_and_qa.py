import io
import json
import os
from unittest.mock import patch, MagicMock
from PIL import Image

from worker.handlers.render import process_render
from worker.handlers.qa import process_qa

def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

@patch("worker.handlers.render.download_image")
@patch("worker.handlers.render.minio_client")
@patch("worker.handlers.render.requests.get")
@patch("worker.handlers.render.requests.post")
@patch("worker.config.QA_MODE", "vlm") # Set QA_MODE to trigger rendering
@patch("worker.config.RENDER_CACHE_DIR", "./test_rendered_cache")
def test_process_render_success(mock_post, mock_get, mock_minio, mock_download):
    # Setup mocks
    mock_download.return_value = get_dummy_image_bytes()
    
    mock_image_info = {
        "id": "image-uuid-1",
        "filename": "page1.png",
        "storagePath": "originals/page1.png",
        "layerElements": [
            {
                "text": "Hello rendered text",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "backgroundColor": "#ffffff",
                "textColor": "#000000",
                "size": 14.0,
                "fontWeight": "bold",
                "fontStyle": "normal",
                "boxShape": "rectangular",
                "font": "Comic Neue"
            }
        ]
    }
    
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res
    
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_render
    job_data = {
        "imageId": "image-uuid-1",
        "pageNumber": 1,
        "chapterNumber": 1.0
    }
    process_render(job_data)

    # Assertions
    mock_download.assert_called_once()
    mock_minio.put_object.assert_called_once()
    args, kwargs = mock_minio.put_object.call_args
    assert args[0] == "manga-library"
    assert args[1] == "rendered/image-uuid-1.png"
    assert kwargs.get("content_type") == "image/png"
    
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "render" in post_args[0]
    assert post_kwargs["json"]["imageId"] == "image-uuid-1"


@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "llm")
def test_process_qa_llm_success(mock_post, mock_get, mock_try_cloud_ai):
    # Setup mocks
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "confidence": 0.9,
                "translatedText": "Hello",
                "translationScore": 0.95,
                "bubbleReadingOrder": 1
            }
        ]
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps({
        "results": [
            {
                "regionId": "region-uuid-1",
                "qaStatus": "passed",
                "qaScore": 0.98,
                "qaFeedback": "Perfect translation."
            }
        ]
    })

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    os.environ["QA_MODEL_PROVIDER"] = "openrouter"
    os.environ["OPENROUTER_API_KEY"] = "fake-key"

    # Invoke process_qa
    job_data = {"imageId": "image-uuid-1"}
    process_qa(job_data)

    # Assertions
    mock_try_cloud_ai.assert_called_once()
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "qa" in post_args[0]
    assert post_kwargs["json"]["imageId"] == "image-uuid-1"
    qa_results = post_kwargs["json"]["qaResults"]
    assert len(qa_results) == 1
    assert qa_results[0]["regionId"] == "region-uuid-1"
    assert qa_results[0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "vlm")
def test_process_qa_vlm_cloud_success(mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm):
    # Setup mocks
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10, "bboxY": 20, "bboxW": 100, "bboxH": 50,
                "translatedText": "Hello",
                "bubbleReadingOrder": 1
            }
        ]
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    
    mock_minio_res = MagicMock()
    mock_minio_res.read.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value = mock_minio_res

    mock_try_cloud_vlm.return_value = json.dumps({
        "results": [
            {
                "regionId": "region-uuid-1",
                "qaStatus": "passed",
                "qaScore": 0.99,
                "qaFeedback": "VLM verified rendering matches text exactly."
            }
        ]
    })

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    os.environ["QA_MODEL_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = "fake-key"

    # Invoke process_qa
    job_data = {"imageId": "image-uuid-1"}
    process_qa(job_data)

    # Assertions
    mock_try_cloud_vlm.assert_called_once()
    mock_minio.get_object.assert_called_once_with("manga-library", "rendered/image-uuid-1.png")
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    qa_results = post_kwargs["json"]["qaResults"]
    assert len(qa_results) == 1
    assert qa_results[0]["regionId"] == "region-uuid-1"
    assert qa_results[0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.try_local_vlm_vision")
@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "vlm")
def test_process_qa_vlm_local_fallback(mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm, mock_try_local_vlm):
    # Setup mocks
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10, "bboxY": 20, "bboxW": 100, "bboxH": 50,
                "translatedText": "Hello",
                "bubbleReadingOrder": 1
            }
        ]
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    
    mock_minio_res = MagicMock()
    mock_minio_res.read.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value = mock_minio_res

    # Force cloud VLM to fail
    mock_try_cloud_vlm.side_effect = Exception("API quota exceeded")

    # Set up local VLM return
    mock_try_local_vlm.return_value = json.dumps({
        "results": [
            {
                "regionId": "region-uuid-1",
                "qaStatus": "direct_fix",
                "qaScore": 0.8,
                "qaFeedback": "Slight layout wrap issue.",
                "directFix": {
                    "correctedText": "Hello!",
                    "suggestedFontSize": 12.0
                }
            }
        ]
    })

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    os.environ["QA_MODEL_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = "fake-key"
    os.environ["LOCAL_VLM_MODEL"] = "qwen2.5-vl-3b-instruct"
    if "DISABLE_LOCAL_LLM" in os.environ:
        del os.environ["DISABLE_LOCAL_LLM"]

    # Invoke process_qa
    job_data = {"imageId": "image-uuid-1"}
    process_qa(job_data)

    # Assertions
    mock_try_cloud_vlm.assert_called_once()
    mock_try_local_vlm.assert_called_once()
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    qa_results = post_kwargs["json"]["qaResults"]
    assert len(qa_results) == 1
    assert qa_results[0]["regionId"] == "region-uuid-1"
    assert qa_results[0]["qaStatus"] == "direct_fix"
    assert qa_results[0]["directFix"]["correctedText"] == "Hello!"
