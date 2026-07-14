import io
from unittest.mock import MagicMock, patch

from PIL import Image

from worker.handlers.render import process_render


def get_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@patch("worker.handlers.render.download_image")
@patch("worker.handlers.render.minio_client")
@patch("worker.handlers.render.requests.get")
@patch("worker.handlers.render.requests.post")
@patch("worker.config.QA_MODE", "vlm")
@patch("worker.config.RENDER_CACHE_DIR", "/tmp/test_rendered_cache")
def test_render_filters_correctly(mock_post, mock_get, mock_minio, mock_download):
    # Setup mock image bytes
    mock_download.return_value = get_dummy_image_bytes()

    # Setup layerElements with various types and visibility values
    mock_image_info = {
        "id": "image-uuid-1",
        "filename": "page1.png",
        "storagePath": "originals/page1.png",
        "layerElements": [
            # 1. Translation element on visible layer -> SHOULD RENDER
            {
                "id": "el-1",
                "text": "RENDER_ME",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "layerType": "translation",
                "layerVisible": True,
                "font": "Comic Neue",
            },
            # 2. Translation element on hidden layer -> SHOULD NOT RENDER
            {
                "id": "el-2",
                "text": "SKIP_HIDDEN_LAYER",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "layerType": "translation",
                "layerVisible": False,
                "font": "Comic Neue",
            },
            # 3. OCR element on visible layer -> SHOULD NOT RENDER
            {
                "id": "el-3",
                "text": "SKIP_OCR_LAYER",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "layerType": "ocr",
                "layerVisible": True,
                "font": "Comic Neue",
            },
            # 4. Hidden element on visible layer -> SHOULD NOT RENDER
            {
                "id": "el-4",
                "text": "SKIP_HIDDEN_ELEMENT",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": False,
                "layerType": "translation",
                "layerVisible": True,
                "font": "Comic Neue",
            },
            # 5. SFX element on visible layer -> SHOULD RENDER
            {
                "id": "el-5",
                "text": "RENDER_SFX",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "layerType": "sfx",
                "layerVisible": True,
                "font": "Comic Neue",
            },
            # 6. Legacy element with no layer details (default translation / visible) -> SHOULD RENDER
            {
                "id": "el-6",
                "text": "RENDER_LEGACY",
                "x": 10.0,
                "y": 20.0,
                "maxWidth": 100,
                "maxHeight": 50,
                "visible": True,
                "font": "Comic Neue",
            },
        ],
    }

    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Patch load_font to avoid looking up system fonts and fit_text_in_box_py to capture texts
    with (
        patch("worker.handlers.render.fit_text_in_box_py") as mock_fit,
        patch("worker.handlers.render.load_font") as mock_font,
    ):
        from PIL import ImageFont

        mock_fit.return_value = {
            "fontSize": 12,
            "lines": ["test"],
            "lineCenters": [50.0],
            "overflow": False,
        }
        mock_font.return_value = ImageFont.load_default()

        # Invoke process_render
        job_data = {"imageId": "image-uuid-1", "pageNumber": 1, "chapterNumber": 1.0}
        process_render(job_data)

        # Verify which texts were fitted/rendered
        called_texts = [call.args[0] for call in mock_fit.call_args_list]

        assert "RENDER_ME" in called_texts
        assert "RENDER_SFX" in called_texts
        assert "RENDER_LEGACY" in called_texts

        assert "SKIP_HIDDEN_LAYER" not in called_texts
        assert "SKIP_OCR_LAYER" not in called_texts
        assert "SKIP_HIDDEN_ELEMENT" not in called_texts
