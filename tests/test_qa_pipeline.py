import io
import json
from unittest.mock import MagicMock, patch

from PIL import Image

from tests.qa_binding import bound_qa_job, render_result
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
                    "qaScore": 0.99,
                    "qaFeedback": "Perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa(bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes()))

    mock_try_cloud_ai.assert_called_once()
    args, _kwargs = mock_try_cloud_ai.call_args
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
                    "qaScore": 0.99,
                    "qaFeedback": "Perfect.",
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_qa(bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes()))

    mock_try_cloud_ai.assert_called_once()
    args, _kwargs = mock_try_cloud_ai.call_args
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
def test_process_qa_vlm_openrouter(mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm):
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

    process_qa(bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes()))

    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
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
def test_process_qa_vlm_nvidia(mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_try_cloud_vlm):
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

    process_qa(bound_qa_job({"imageId": "image-uuid-1"}, get_dummy_image_bytes()))

    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "nvidia/nemotron-nano-12b-v2-vl"


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.page_scene_renderer.render_page_scene")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_CONFIG")
def test_process_qa_hybrid_flow(
    mock_qa_config,
    mock_post,
    mock_get,
    mock_minio,
    mock_download,
    mock_render,
    mock_try_llm,
    mock_try_vlm,
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
                "regionType": "speech",
                "user_override": "replace",
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

    # Mock prepare status. R1: the prepare endpoint answers with the immutable render payload
    # that hybrid QA hands to the browser renderer for its VLM check.
    render_payload = {"imageId": "image-uuid-1", "pageRevision": 2, "logicalScene": {}, "renderAssetUrls": {}}
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post_res.json.return_value = render_payload
    mock_post.return_value = mock_post_res

    # Mock render
    mock_render.return_value = render_result(get_dummy_image_bytes())

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

    process_qa(bound_qa_job({"imageId": "image-uuid-1", "qaMode": "hybrid"}))

    # Verify LLM was called
    mock_try_llm.assert_called_once()
    # Verify the interim render went through the browser renderer with the prepare payload
    mock_render.assert_called_once_with({**render_payload, "jobId": "qa-job", "attempt": 1})
    # Verify VLM was called
    mock_try_vlm.assert_called_once()

    # Verify post callbacks (one to prepare, one to final qa)
    assert mock_post.call_count == 2


@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.page_scene_renderer.render_page_scene")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
@patch("worker.handlers.qa.QA_CONFIG")
def test_hybrid_incomplete_llm_pass_applies_no_fixes(
    mock_qa_config, mock_post, mock_get, mock_minio, mock_download, mock_render, mock_try_llm, mock_try_vlm
):
    """OQ-02: a truncated first pass is dropped whole; prepare receives no fixes to apply."""
    mock_qa_config.provider = "gemini"
    mock_qa_config.resolve_key.return_value = "fake-key"
    regions = [
        {"id": rid, "text": "src", "bboxX": 0, "bboxY": 0, "bboxW": 10, "bboxH": 10, "translatedText": "en"}
        for rid in ("region-a", "region-b")
    ]
    mock_get.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"ocrRegions": regions}))
    mock_try_llm.return_value = json.dumps(
        {"results": [{"regionId": "region-a", "qaStatus": "direct_fix", "directFix": {"correctedText": "x"}}]}
    )
    mock_post.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"imageId": "image-uuid-1"}))
    mock_render.return_value = render_result(get_dummy_image_bytes())
    mock_download.return_value = get_dummy_image_bytes()
    mock_minio.get_object.return_value.read.return_value = get_dummy_image_bytes()
    mock_try_vlm.return_value = json.dumps(
        {"results": [{"regionId": r, "qaStatus": "passed", "qaScore": 1} for r in ("region-a", "region-b")]}
    )

    process_qa(bound_qa_job({"imageId": "image-uuid-1", "qaMode": "hybrid"}))

    prepare_call, final_call = mock_post.call_args_list
    assert prepare_call.args[0].endswith("/qa-hybrid-prepare")
    assert prepare_call.kwargs["json"]["qaResults"] == []
    final = final_call.kwargs["json"]
    assert final["qaResponseIntegrity"]["complete"] is True
    assert sorted(final["qaTargetIds"]) == ["region-a", "region-b"]
