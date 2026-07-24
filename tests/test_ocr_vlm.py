import io
import json
import os
from unittest.mock import MagicMock, patch

from PIL import Image

from worker.handlers.ocr import process_ocr


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.try_cloud_ai_vision_batch")
@patch("worker.handlers.ocr.model_manager")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.OCR_CONFIG")
@patch.dict(
    os.environ,
    {
        "DISABLE_LOCAL_OCR": "true",
    },
)
def test_process_ocr_vlm_gemini(
    mock_ocr_config,
    mock_post,
    mock_get,
    mock_model_manager,
    mock_try_cloud_vlm,
    mock_detect_yolo,
    mock_download,
):
    mock_ocr_config.provider = "gemini"
    mock_ocr_config.resolve_key.return_value = "fake-gemini-key"
    mock_ocr_config.vlm_model = "gemini-1.5-flash"

    # Mock detector-only PaddleOCR to avoid real initialization
    mock_model_manager.get_paddle_ocr_detector.return_value = MagicMock()

    # Setup mocks
    mock_download.return_value = get_dummy_image_bytes()
    mock_detect_yolo.return_value = [
        {
            "bbox": [10, 20, 100, 80],
            "confidence": 0.95,
            "mask_polygon": [[10, 20], [110, 20], [110, 100], [10, 100]],
            "safe_rect": [15, 25, 90, 70],
        }
    ]
    mock_try_cloud_vlm.return_value = json.dumps(
        {"results": [{"id": "region_0", "text": "Hello from Gemini VLM OCR"}]}
    )

    mock_image_info = {"id": "image-uuid-1", "panels": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_ocr
    job_data = {"imageId": "image-uuid-1"}
    process_ocr(job_data)

    # Assertions
    mock_download.assert_called_once()
    mock_detect_yolo.assert_called_once()

    # Check cloud VLM called with gemini parameters
    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "gemini"
    assert args[1] == "fake-gemini-key"
    assert args[2] == "gemini-1.5-flash"
    assert len(args[3]) == 1
    assert args[3][0]["id"] == "region_0"

    # Check callback post payload
    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "ocr" in post_args[0]
    payload = post_kwargs["json"]
    assert payload["imageId"] == "image-uuid-1"
    assert len(payload["regions"]) == 1
    assert payload["regions"][0]["text"] == "Hello from Gemini VLM OCR"
    assert payload["regions"][0]["bubbleId"] == "bubble_0"


@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.try_cloud_ai_vision_batch")
@patch("worker.handlers.ocr.model_manager")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.OCR_CONFIG")
@patch.dict(
    os.environ,
    {
        "DISABLE_LOCAL_OCR": "true",
    },
)
def test_process_ocr_vlm_openrouter(
    mock_ocr_config,
    mock_post,
    mock_get,
    mock_model_manager,
    mock_try_cloud_vlm,
    mock_detect_yolo,
    mock_download,
):
    mock_ocr_config.provider = "openrouter"
    mock_ocr_config.resolve_key.return_value = "fake-openrouter-key"
    mock_ocr_config.vlm_model = "qwen/qwen3-vl-8b-instruct"

    mock_model_manager.get_paddle_ocr_detector.return_value = MagicMock()

    mock_download.return_value = get_dummy_image_bytes()
    mock_detect_yolo.return_value = [
        {
            "bbox": [10, 20, 100, 80],
            "confidence": 0.95,
            "mask_polygon": [[10, 20], [110, 20], [110, 100], [10, 100]],
            "safe_rect": [15, 25, 90, 70],
        }
    ]
    mock_try_cloud_vlm.return_value = json.dumps(
        {"results": [{"id": "region_0", "text": "Hello from OpenRouter VLM OCR"}]}
    )

    mock_image_info = {"id": "image-uuid-1", "panels": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {"imageId": "image-uuid-1"}
    process_ocr(job_data)

    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "openrouter"
    assert args[1] == "fake-openrouter-key"
    assert args[2] == "qwen/qwen3-vl-8b-instruct"
    assert len(args[3]) == 1
    assert args[3][0]["id"] == "region_0"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["regions"][0]["text"] == "Hello from OpenRouter VLM OCR"


@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.try_cloud_ai_vision_batch")
@patch("worker.handlers.ocr.model_manager")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.OCR_CONFIG")
@patch.dict(
    os.environ,
    {
        "DISABLE_LOCAL_OCR": "true",
    },
)
def test_process_ocr_vlm_nvidia(
    mock_ocr_config,
    mock_post,
    mock_get,
    mock_model_manager,
    mock_try_cloud_vlm,
    mock_detect_yolo,
    mock_download,
):
    mock_ocr_config.provider = "nvidia"
    mock_ocr_config.resolve_key.return_value = "fake-nvidia-key"
    mock_ocr_config.vlm_model = "nvidia/nemotron-nano-12b-v2-vl"

    mock_model_manager.get_paddle_ocr_detector.return_value = MagicMock()

    mock_download.return_value = get_dummy_image_bytes()
    mock_detect_yolo.return_value = [
        {
            "bbox": [10, 20, 100, 80],
            "confidence": 0.95,
            "mask_polygon": [[10, 20], [110, 20], [110, 100], [10, 100]],
            "safe_rect": [15, 25, 90, 70],
        }
    ]
    mock_try_cloud_vlm.return_value = json.dumps(
        {"results": [{"id": "region_0", "text": "Hello from Nvidia VLM OCR"}]}
    )

    mock_image_info = {"id": "image-uuid-1", "panels": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {"imageId": "image-uuid-1"}
    process_ocr(job_data)

    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "nvidia/nemotron-nano-12b-v2-vl"
    assert len(args[3]) == 1
    assert args[3][0]["id"] == "region_0"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["regions"][0]["text"] == "Hello from Nvidia VLM OCR"


@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.try_local_vlm_vision")
@patch("worker.handlers.ocr.model_manager")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.OCR_CONFIG")
@patch.dict(
    os.environ,
    {
        "DISABLE_LOCAL_OCR": "true",
        "LOCAL_VLM_MODEL": "local-vlm-model",
    },
)
def test_process_ocr_vlm_local_fallback(
    mock_ocr_config,
    mock_post,
    mock_get,
    mock_model_manager,
    mock_try_local_vlm,
    mock_detect_yolo,
    mock_download,
):
    mock_ocr_config.provider = ""
    mock_ocr_config.resolve_key.return_value = ""
    mock_ocr_config.vlm_model = ""

    mock_model_manager.get_paddle_ocr_detector.return_value = MagicMock()

    mock_download.return_value = get_dummy_image_bytes()
    mock_detect_yolo.return_value = [
        {
            "bbox": [10, 20, 100, 80],
            "confidence": 0.95,
            "mask_polygon": [[10, 20], [110, 20], [110, 100], [10, 100]],
            "safe_rect": [15, 25, 90, 70],
        }
    ]
    mock_try_local_vlm.return_value = json.dumps({"text": "Hello from Local VLM OCR"})

    mock_image_info = {"id": "image-uuid-1", "panels": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {"imageId": "image-uuid-1"}
    process_ocr(job_data)

    mock_try_local_vlm.assert_called_once()
    args, _kwargs = mock_try_local_vlm.call_args
    assert args[0] == "local-vlm-model"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["regions"][0]["text"] == "Hello from Local VLM OCR"


@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.try_cloud_ai_vision_batch")
@patch("worker.handlers.ocr.model_manager")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.OCR_CONFIG")
@patch.dict(
    os.environ,
    {
        "DISABLE_LOCAL_OCR": "true",
    },
)
def test_process_ocr_vlm_batched_multiple_regions(
    mock_ocr_config,
    mock_post,
    mock_get,
    mock_model_manager,
    mock_try_cloud_vlm,
    mock_detect_yolo,
    mock_download,
):
    mock_ocr_config.provider = "gemini"
    mock_ocr_config.resolve_key.return_value = "fake-gemini-key"
    mock_ocr_config.vlm_model = "gemini-1.5-flash"

    # Mock PaddleOCR detector to return 3 fragments (2 inside bubbles, 1 outside)
    mock_detector = MagicMock()
    mock_model_manager.get_paddle_ocr_detector.return_value = mock_detector
    # Each fragment returns: (bbox, text, confidence)
    # The detector doesn't return text in detection-only mode, but parse_paddle_ocr_results handles it.
    mock_detector.predict.return_value = [
        {
            "dt_polys": [
                [[15, 25], [70, 25], [70, 60], [15, 60]],  # inside bubble 0
                [[105, 115], [165, 115], [165, 155], [105, 155]],  # inside bubble 1
                [[10, 150], [50, 150], [50, 180], [10, 180]],  # outside (free-floating)
            ],
            "rec_texts": [],
            "rec_scores": [],
        }
    ]

    mock_download.return_value = get_dummy_image_bytes()

    # 2 speech bubbles
    mock_detect_yolo.return_value = [
        {
            "bbox": [10, 20, 80, 70],
            "confidence": 0.95,
            "mask_polygon": [[10, 20], [90, 20], [90, 90], [10, 90]],
            "safe_rect": [15, 25, 70, 60],
        },
        {
            "bbox": [100, 100, 90, 90],
            "confidence": 0.90,
            "mask_polygon": [[100, 100], [190, 100], [190, 190], [100, 190]],
            "safe_rect": [105, 105, 80, 80],
        },
    ]

    # Mock VLM response for 3 regions
    mock_try_cloud_vlm.return_value = json.dumps(
        {
            "results": [
                {"id": "region_0", "text": "Hello from Bubble 0"},
                {"id": "region_1", "text": "Hello from Bubble 1"},
                {"id": "region_2", "text": "Hello from Free Text"},
            ]
        }
    )

    mock_image_info = {"id": "image-uuid-1", "panels": []}
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_ocr
    job_data = {"imageId": "image-uuid-1"}
    process_ocr(job_data)

    # Check cloud VLM called with 3 crops
    mock_try_cloud_vlm.assert_called_once()
    args, _kwargs = mock_try_cloud_vlm.call_args
    assert len(args[3]) == 3  # 3 crops: 2 bubbles, 1 direct text
    assert args[3][0]["id"] == "region_0"
    assert args[3][1]["id"] == "region_1"
    assert args[3][2]["id"] == "region_2"

    # Check callback post payload contains all 3 regions mapped correctly
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert len(payload["regions"]) == 3

    # Map by bubbleId to check texts
    mapped_regions = {r["bubbleId"]: r["text"] for r in payload["regions"]}
    assert mapped_regions["bubble_0"] == "Hello from Bubble 0"
    assert mapped_regions["bubble_1"] == "Hello from Bubble 1"
    assert mapped_regions["direct_text_0"] == "Hello from Free Text"
