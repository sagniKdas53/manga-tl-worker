import io
import json
import os
from unittest.mock import MagicMock, patch

from PIL import Image

from tests.qa_binding import bound_qa_job
from worker.handlers.qa import process_qa

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CACHE_DIR = os.path.join(TEST_DIR, "test_rendered_cache")


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# The Pillow render test that lived here is gone with the Pillow path (tracker R1). The render
# handler's contract — scene in, browser renderer, callback with the immutable identity, and a
# loud failure for a job without a scene — is covered in tests/test_page_scene_renderer.py.


@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "llm")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_llm_success(mock_qa_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_qa_config.provider = "openrouter"
    mock_qa_config.resolve_key.return_value = "fake-key"
    mock_qa_config.llm_model = ""

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
                "bubbleReadingOrder": 1,
                "regionType": "speech",
                "user_override": "replace",
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
                    "qaScore": 0.98,
                    "qaFeedback": "Perfect translation.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_qa
    job_data = bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes())
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
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_vlm_cloud_success(
    mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm
):
    # AUDIT-W1: this was "gemini", which is not in config/providers.json at all — it only ever
    # resolved because QA_DEFAULT_VLM_MODELS in qa.py listed it. Defaults come from providers.json
    # now, so the provider has to be a real one.
    mock_qa_config.provider = "openrouter"
    mock_qa_config.resolve_key.return_value = "fake-key"
    mock_qa_config.vlm_model = ""

    # Setup mocks
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
                    "qaFeedback": "VLM verified rendering matches text exactly.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_qa
    job_data = bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes())
    process_qa(job_data)

    # Assertions
    mock_try_cloud_vlm.assert_called_once()
    mock_minio.get_object.assert_called_once_with("manga-library", job_data["renderArtifact"]["storagePath"])
    mock_post.assert_called_once()
    _post_args, post_kwargs = mock_post.call_args
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
@patch("worker.handlers.qa.QA_CONFIG")
@patch.dict(
    os.environ,
    {
        "LOCAL_VLM_MODEL": "qwen2.5-vl-3b-instruct",
    },
)
def test_process_qa_vlm_local_fallback(
    mock_qa_config,
    mock_post,
    mock_get,
    mock_minio,
    mock_download,
    mock_try_cloud_vlm,
    mock_try_local_vlm,
):
    # AUDIT-W1: this was "gemini", which is not in config/providers.json at all — it only ever
    # resolved because QA_DEFAULT_VLM_MODELS in qa.py listed it. Defaults come from providers.json
    # now, so the provider has to be a real one.
    mock_qa_config.provider = "openrouter"
    mock_qa_config.resolve_key.return_value = "fake-key"
    mock_qa_config.vlm_model = ""

    # Setup mocks
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

    # Force cloud VLM to fail
    mock_try_cloud_vlm.side_effect = Exception("API quota exceeded")

    # Set up local VLM return
    mock_try_local_vlm.return_value = json.dumps(
        {
            "results": [
                {
                    "regionId": "region-uuid-1",
                    "qaStatus": "direct_fix",
                    "qaScore": 0.8,
                    "qaFeedback": "Slight layout wrap issue.",
                    "directFix": {"correctedText": "Hello!", "suggestedFontSize": 12.0},
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    if "DISABLE_LOCAL_LLM" in os.environ:
        del os.environ["DISABLE_LOCAL_LLM"]

    # Invoke process_qa
    job_data = bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes())
    process_qa(job_data)

    # Assertions
    mock_try_cloud_vlm.assert_called_once()
    mock_try_local_vlm.assert_not_called()
    mock_post.assert_called_once()
    _post_args, post_kwargs = mock_post.call_args
    qa_results = post_kwargs["json"]["qaResults"]
    # This used to assert the region came back "passed": with the cloud call raising and no local
    # fallback, QA produced nothing and the handler fabricated a pass for every region. That made a
    # dead QA provider indistinguishable from a clean page. An empty verdict now tells the backend
    # QA did not run, and it records that rather than a pass.
    assert qa_results == []


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_MODE", "vlm")
def test_process_qa_vlm_empty_ocr_regions(mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm):
    # Setup mock image info with empty ocrRegions
    mock_image_info = {"id": "image-uuid-1", "ocrRegions": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_qa
    job_data = bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes())
    process_qa(job_data)

    # Assertions:
    # 1. Requests GET should be called twice (once in _process_qa_vlm, and once in _auto_pass_all)
    assert mock_get.call_count == 2

    # 2. VLM cloud model should NOT be called
    mock_try_cloud_vlm.assert_not_called()

    # 3. Image download should NOT be called
    mock_download.assert_not_called()

    # 4. Callback POST should be called with empty qaResults
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "qa" in post_args[0]
    assert post_kwargs["json"]["imageId"] == "image-uuid-1"
    assert post_kwargs["json"]["qaResults"] == []
