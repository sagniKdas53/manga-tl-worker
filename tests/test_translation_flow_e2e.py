import io
import json
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from worker.handlers.panel import process_panel_detection
from worker.handlers.ocr import process_ocr
from worker.handlers.layout import process_layout
from worker.handlers.translation import process_translation
from worker.handlers.render import process_render
from worker.handlers.qa import process_qa

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CACHE_DIR = os.path.join(TEST_DIR, "test_rendered_cache")


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@patch("requests.post")
@patch("requests.get")
@patch("worker.config.QA_MODE", "hybrid")
@patch("worker.config.RENDER_CACHE_DIR", TEST_CACHE_DIR)
@patch("worker.handlers.panel.download_image")
@patch("worker.handlers.ocr.downscale_for_ocr")
@patch("worker.handlers.ocr.parse_paddle_ocr_results")
@patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
@patch("worker.handlers.ocr.download_image")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.render.download_image")
@patch("worker.handlers.render.minio_client")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
@patch("worker.handlers.qa.minio_client")
@patch("worker.config.TL_CONFIG")
@patch("worker.handlers.qa.QA_CONFIG")
def test_core_translation_flow_e2e(
    mock_qa_config,
    mock_tl_config,
    mock_qa_minio,
    mock_qa_download,
    mock_try_vlm,
    mock_try_llm_qa,
    mock_render_minio,
    mock_render_download,
    mock_try_llm_tl,
    mock_ocr_download,
    mock_detect_bubbles_yolo,
    mock_get_paddle_ocr_reader,
    mock_parse_paddle_ocr_results,
    mock_downscale_for_ocr,
    mock_panel_download,
    mock_get,
    mock_post,
):
    """
    E2E integration test for the core translation pipeline components:
    Panel Detection -> OCR (Local) -> Layout -> Translation -> Rendering -> Hybrid QA
    """
    # Configure Free Model and Provider settings
    mock_tl_config.provider = "openrouter"
    mock_tl_config.resolve_key.return_value = "free-key"
    mock_tl_config.llm_model = "meta-llama/llama-3-8b-instruct:free"

    mock_qa_config.provider = "openrouter"
    mock_qa_config.resolve_key.return_value = "free-key"
    mock_qa_config.llm_model = "meta-llama/llama-3-8b-instruct:free"
    mock_qa_config.vlm_model = "meta-llama/llama-3.2-11b-vision-instruct:free"

    # Define all request.get mocks
    mock_panel_get_res = MagicMock()
    mock_panel_get_res.status_code = 200
    mock_panel_get_res.json.return_value = {
        "id": "image-uuid-1",
        "storagePath": "test/test.png",
        "width": 1000,
        "height": 1000
    }

    mock_ocr_get_res = MagicMock()
    mock_ocr_get_res.status_code = 200
    mock_ocr_get_res.json.return_value = {
        "id": "image-uuid-1",
        "storagePath": "test/test.png",
        "width": 1000,
        "height": 1000
    }

    mock_layout_get_res = MagicMock()
    mock_layout_get_res.status_code = 200
    mock_layout_get_res.json.return_value = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "bboxX": 10,
                "bboxY": 10,
                "bboxW": 90,
                "bboxH": 40,
                "bubbleReadingOrder": 1,
            }
        ],
    }

    mock_tl_get_res = MagicMock()
    mock_tl_get_res.status_code = 200
    mock_tl_get_res.json.return_value = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }

    mock_render_get_res = MagicMock()
    mock_render_get_res.status_code = 200
    mock_render_get_res.json.return_value = {
        "id": "image-uuid-1",
        "filename": "page1.png",
        "storagePath": "originals/page1.png",
        "layerElements": [
            {
                "text": "Hello",
                "x": 10.0,
                "y": 10.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "backgroundColor": "#ffffff",
                "textColor": "#000000",
                "size": 14.0,
                "fontWeight": "bold",
                "fontStyle": "normal",
                "boxShape": "rectangular",
                "font": "Comic Neue",
            }
        ],
    }

    mock_qa_get_res = MagicMock()
    mock_qa_get_res.status_code = 200
    mock_qa_get_res.json.return_value = {
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

    mock_get.side_effect = [
        mock_panel_get_res,
        mock_ocr_get_res,
        mock_layout_get_res,
        mock_tl_get_res,
        mock_render_get_res,  # for render phase
        mock_qa_get_res,      # for QA phase text check
        mock_render_get_res,  # for QA phase rendering of direct fixes
        mock_qa_get_res,      # for QA phase VLM check
    ]

    # Define all requests.post mocks
    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Mock MinIO responses
    mock_minio_read = MagicMock()
    mock_minio_read.read.return_value = get_dummy_image_bytes()
    mock_qa_minio.get_object.return_value = mock_minio_read
    mock_render_minio.get_object.return_value = mock_minio_read

    # --- 1. Panel Detection ---
    mock_panel_download.return_value = get_dummy_image_bytes()
    panel_job = {"imageId": "image-uuid-1"}
    process_panel_detection(panel_job)

    # --- 2. OCR (Local / PaddleOCR + YOLO Bubble) ---
    mock_ocr_download.return_value = b"dummy"
    mock_downscale_for_ocr.return_value = (np.full((1000, 1000, 3), 255, dtype=np.uint8), 1.0)
    mock_get_paddle_ocr_reader.return_value = MagicMock()
    mock_parse_paddle_ocr_results.return_value = [
        ([[10, 10], [100, 10], [100, 50], [10, 50]], "こんにちは", 0.99),
    ]
    mock_detect_bubbles_yolo.return_value = [
        {
            "bbox": [5, 5, 110, 60],
            "confidence": 0.9,
            "mask_polygon": [[5, 5], [115, 5], [115, 65], [5, 65]],
            "safe_rect": [10, 10, 100, 50],
        }
    ]

    ocr_job = {"imageId": "image-uuid-1", "ocrProvider": "local"}
    process_ocr(ocr_job)

    # --- 3. Layout Analysis ---
    layout_job = {"imageId": "image-uuid-1"}
    process_layout(layout_job)

    # --- 4. Translation (Free Model) ---
    mock_try_llm_tl.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.99,
                }
            ]
        }
    )

    tl_job = {"imageId": "image-uuid-1", "sourceLanguage": "ja", "targetLanguage": "en"}
    process_translation(tl_job)

    # --- 5. Rendering ---
    mock_render_download.return_value = get_dummy_image_bytes()
    render_job = {"imageId": "image-uuid-1", "pageNumber": 1, "chapterNumber": 1.0}
    process_render(render_job)

    # --- 6. Hybrid QA (LLM + VLM) ---
    mock_try_llm_qa.return_value = json.dumps(
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

    mock_qa_download.return_value = get_dummy_image_bytes()

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

    qa_job = {"imageId": "image-uuid-1", "qaMode": "hybrid"}
    process_qa(qa_job)

    # Verify LLM QA called
    mock_try_llm_qa.assert_called_once()
    # Verify VLM QA called
    mock_try_vlm.assert_called_once()
    
    # Verify both hybrid preparation and final QA results callback posts were sent
    assert mock_post.call_count == 7
    # Verify MinIO put_object was called (once during process_render, once during process_qa's hybrid direct-fix render)
    assert mock_render_minio.put_object.call_count == 2
