import pytest
from unittest.mock import patch, MagicMock

from worker.services.layout import classify_region_type, group_conversations, bubble_compare
from worker.services.ocr import perform_redo_ocr

def test_bubble_compare():
    a = {"x": 100, "y": 100}
    b = {"x": 200, "y": 100}
    # In RTL, rightmost (larger x) comes first when y is same
    assert bubble_compare(a, b, reading_direction="rtl") > 0
    # In LTR, leftmost (smaller x) comes first
    assert bubble_compare(a, b, reading_direction="ltr") < 0

def test_classify_region_type():
    panel = {"bboxX": 0, "bboxY": 0, "bboxW": 1000, "bboxH": 1000}
    # A wide region on top edge -> narration
    region = {"bboxX": 10, "bboxY": 10, "bboxW": 800, "bboxH": 100}
    assert classify_region_type(region, panel, 1000, 1000) == "narration"
    
    # A standard speech bubble in the middle
    region2 = {"bboxX": 500, "bboxY": 500, "bboxW": 200, "bboxH": 200}
    assert classify_region_type(region2, panel, 1000, 1000) == "speech"

@patch("worker.services.ocr.model_manager")
@patch("worker.services.ocr.os.environ")
def test_perform_redo_ocr_easyocr(mock_env, mock_model_manager):
    # Mock environment to skip cloud
    mock_env.get.side_effect = lambda k, d="": ""

    # Mock model manager to return EasyOCR
    mock_model_manager.get_paddle_ocr_reader.return_value = None
    mock_model_manager.get_manga_ocr_reader.return_value = None
    
    mock_easy_reader = MagicMock()
    # EasyOCR readtext returns [(bbox, text, conf)]
    mock_easy_reader.readtext.return_value = [([], "Test OCR text", 0.95)]
    mock_model_manager.get_easy_ocr_reader.return_value = mock_easy_reader

    text, conf = perform_redo_ocr(b"dummy_image_bytes", "en")
    
    assert text == "Test OCR text"
    assert conf == 0.95
    mock_easy_reader.readtext.assert_called_once_with(b"dummy_image_bytes")

@patch("worker.services.ocr.requests.post")
@patch("worker.services.ocr.os.environ")
def test_perform_redo_ocr_cloud(mock_env, mock_post):
    # Mock env for Cloud
    def env_get(k, default=""):
        if k in ("MODEL_PROVIDER", "LLM_PROVIDER"): return "openai"
        if k in ("API_KEY", "LLM_API_KEY"): return "dummy_key"
        return default
    mock_env.get.side_effect = env_get

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Cloud OCR text"}}]
    }
    mock_post.return_value = mock_response

    text, conf = perform_redo_ocr(b"dummy_image_bytes", "en")
    
    assert text == "Cloud OCR text"
    assert conf == 1.0
