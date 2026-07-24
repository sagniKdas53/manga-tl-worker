import io
from unittest.mock import MagicMock, patch

from PIL import Image

from worker.handlers.qa_re_ocr import process_qa_re_ocr
from worker.handlers.redo import process_region_redo


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


@patch("worker.handlers.redo.download_image")
@patch("worker.handlers.redo.perform_redo_ocr")
@patch("worker.handlers.redo.requests.get")
@patch("worker.handlers.redo.requests.post")
def test_process_region_redo_ocr(mock_post, mock_get, mock_perform_ocr, mock_download):
    mock_image_info = {
        "id": "image-uuid-1",
        "storagePath": "originals/page1.png",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "bboxX": 10,
                "bboxY": 20,
                "bboxW": 100,
                "bboxH": 50,
                "detectedLanguage": "ja",
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    mock_perform_ocr.return_value = ("Redone OCR text", 0.95)

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "regionId": "region-uuid-1",
        "redoType": "ocr",
    }
    process_region_redo(job_data)

    mock_get.assert_called_once()
    mock_download.assert_called_once()

    mock_perform_ocr.assert_called_once()
    args, _kwargs = mock_perform_ocr.call_args
    assert isinstance(args[0], bytes)
    assert args[1] == "ja"

    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "region-uuid-1/callback" in post_args[0]
    payload = post_kwargs["json"]
    assert payload["text"] == "Redone OCR text"
    assert payload["confidence"] == 0.95
    assert payload["detectedLanguage"] == "en"


@patch("worker.handlers.redo.download_image")
@patch("worker.handlers.redo.translate_text")
@patch("worker.handlers.redo.requests.get")
@patch("worker.handlers.redo.requests.post")
def test_process_region_redo_translation(
    mock_post, mock_get, mock_translate, mock_download
):
    mock_image_info = {
        "id": "image-uuid-1",
        "storagePath": "originals/page1.png",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
            }
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    mock_translate.return_value = "Hello"

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "regionId": "region-uuid-1",
        "redoType": "translation",
    }
    process_region_redo(job_data)

    mock_get.assert_called_once()
    mock_translate.assert_called_once_with(
        "こんにちは",
        source_lang="ja",
        request_id=mock_translate.call_args[1]["request_id"],
    )

    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "region-uuid-1/callback" in post_args[0]
    payload = post_kwargs["json"]
    assert payload["translatedText"] == "Hello"
    assert payload["translationFailed"] is False


@patch("worker.handlers.qa_re_ocr.download_image")
@patch("worker.handlers.qa_re_ocr.perform_redo_ocr")
@patch("worker.handlers.qa_re_ocr.requests.get")
@patch("worker.handlers.qa_re_ocr.requests.post")
def test_process_qa_re_ocr(mock_post, mock_get, mock_perform_ocr, mock_download):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-1",
                "bboxX": 10,
                "bboxY": 20,
                "bboxW": 100,
                "bboxH": 50,
                "detectedLanguage": "ja",
            },
            {
                "id": "region-2",
                "bboxX": 100,
                "bboxY": 120,
                "bboxW": 80,
                "bboxH": 40,
                "detectedLanguage": "ja",
            },
        ],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_download.return_value = get_dummy_image_bytes()
    mock_perform_ocr.side_effect = [
        ("Redone OCR text 1", 0.96),
        ("Redone OCR text 2", 0.97),
    ]

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {"imageId": "image-uuid-1", "regionsToReOcr": ["region-1", "region-2"]}
    process_qa_re_ocr(job_data)

    assert mock_perform_ocr.call_count == 2
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["imageId"] == "image-uuid-1"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["regionId"] == "region-1"
    assert payload["results"][0]["text"] == "Redone OCR text 1"
    assert payload["results"][1]["regionId"] == "region-2"
    assert payload["results"][1]["text"] == "Redone OCR text 2"
