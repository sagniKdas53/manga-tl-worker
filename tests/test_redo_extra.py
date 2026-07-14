from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from worker.handlers.redo import process_region_redo


@patch("worker.handlers.redo.requests")
@patch("worker.handlers.redo.download_image")
def test_process_region_redo_ocr(mock_dl, mock_req):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "storagePath": "path",
        "detectedLanguage": "ja",
        "ocrRegions": [
            {
                "id": "r1",
                "bboxX": 0,
                "bboxY": 0,
                "bboxW": 10,
                "bboxH": 10,
                "detectedLanguage": "ja",
            }
        ],
    }
    mock_req.get.return_value = mock_res

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_dl.return_value = cv2.imencode(".jpg", img)[1].tobytes()

    with patch("worker.handlers.redo.perform_redo_ocr") as mock_ocr:
        mock_ocr.return_value = ("new text", 0.99)
        with patch("worker.handlers.redo.detect_language") as mock_lang:
            mock_lang.return_value = "ja"
            process_region_redo({"imageId": "img1", "regionId": "r1", "redoType": "ocr"})

    mock_req.post.assert_called()
    payload = mock_req.post.call_args[1]["json"]
    assert payload["text"] == "new text"
    assert payload["detectedLanguage"] == "ja"


@patch("worker.handlers.redo.requests")
@patch("worker.handlers.redo.download_image")
def test_process_region_redo_translation(mock_dl, mock_req):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "storagePath": "path",
        "detectedLanguage": "ja",
        "ocrRegions": [
            {
                "id": "r1",
                "bboxX": 0,
                "bboxY": 0,
                "bboxW": 10,
                "bboxH": 10,
                "text": "hello",
                "detectedLanguage": "ja",
            }
        ],
    }
    mock_req.get.return_value = mock_res
    mock_dl.return_value = b"image_data"

    with patch("worker.handlers.redo.translate_text") as mock_tl:
        mock_tl.return_value = "world"
        process_region_redo(
            {
                "imageId": "img1",
                "regionId": "r1",
                "redoType": "translation",
                "targetLanguage": "en",
            }
        )

    mock_req.post.assert_called()
    payload = mock_req.post.call_args[1]["json"]
    assert payload["translatedText"] == "world"


@patch("worker.handlers.redo.requests")
def test_process_region_redo_not_found(mock_req):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"storagePath": "path", "ocrRegions": []}
    mock_req.get.return_value = mock_res

    process_region_redo({"imageId": "img1", "regionId": "r1", "redoType": "ocr"})

    assert not mock_req.post.called


@patch("worker.handlers.redo.requests")
def test_process_region_redo_error(mock_req):
    mock_res = MagicMock()
    mock_res.status_code = 500
    mock_req.get.return_value = mock_res

    process_region_redo({"imageId": "img1", "regionId": "r1", "redoType": "ocr"})
    assert not mock_req.post.called
