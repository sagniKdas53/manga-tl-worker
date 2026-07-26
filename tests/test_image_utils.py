from unittest.mock import MagicMock, patch

import numpy as np

from worker.utils.image import (
    calculate_overlap_area,
    download_image,
    downscale_for_ocr,
)


def test_downscale_for_ocr_none():
    img, scale = downscale_for_ocr(None)
    assert img is None
    assert scale == 1.0


def test_downscale_for_ocr_small():
    dummy = np.zeros((500, 500, 3), dtype=np.uint8)
    img, scale = downscale_for_ocr(dummy, max_dim=1024)
    assert img.shape == (500, 500, 3)
    assert scale == 1.0


def test_downscale_for_ocr_large():
    dummy = np.zeros((2048, 1024, 3), dtype=np.uint8)
    img, scale = downscale_for_ocr(dummy, max_dim=1024)
    assert img.shape[0] == 1024
    assert img.shape[1] == 512
    assert scale == 2.0


def test_calculate_overlap_area():
    region = {"x": 10, "y": 10, "width": 50, "height": 50}
    panel = {"bboxX": 20, "bboxY": 20, "bboxW": 50, "bboxH": 50}
    # overlap box: x=[20, 60] (w=40), y=[20, 60] (h=40) => area 1600
    area = calculate_overlap_area(region, panel)
    assert area == 1600


def test_download_image_presigned_url():
    image_info = {"presignedUrl": "http://example.com/test.png"}
    mock_resp = MagicMock()
    mock_resp.content = b"fake_png_data"

    with patch("requests.get", return_value=mock_resp) as mock_get:
        content = download_image(image_info)
        assert content == b"fake_png_data"
        mock_get.assert_called_once_with("http://example.com/test.png")


def test_download_image_minio_path():
    image_info = {"storagePath": "pages/page1.png"}
    mock_minio_resp = MagicMock()
    mock_minio_resp.read.return_value = b"minio_png_data"

    with patch("worker.config.minio_client.get_object", return_value=mock_minio_resp) as mock_get_obj:
        content = download_image(image_info)
        assert content == b"minio_png_data"
        mock_get_obj.assert_called_once_with("manga-library", "pages/page1.png")
