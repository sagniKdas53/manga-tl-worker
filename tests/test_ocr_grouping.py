import json
from unittest.mock import MagicMock, patch

import numpy as np

from worker.handlers.ocr import process_ocr


@patch("worker.handlers.ocr.redis_client")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.downscale_for_ocr")
@patch("worker.handlers.ocr.parse_paddle_ocr_results")
@patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
def test_process_ocr_yolo_preserves_grouping(
    mock_detect_bubbles_yolo,
    mock_get_paddle_ocr_reader,
    mock_parse_paddle_ocr_results,
    mock_downscale_for_ocr,
    mock_download_image,
    mock_requests_get,
    mock_requests_post,
    mock_redis,
):
    """
    Test that YOLO grouped fragments do not get blindly merged at the end of process_ocr,
    which was a regression causing all text on the page to merge into a giant convex hull.
    """
    # 1. Mock downscale_for_ocr and download_image
    mock_download_image.return_value = b"dummy"
    dummy_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    # Returns (img_decoded, ocr_upscale)
    mock_downscale_for_ocr.return_value = (dummy_img, 1.0)

    # 2. Mock PaddleOCR (returns 2 distinct text fragments that are far apart)
    mock_ocr = MagicMock()
    mock_get_paddle_ocr_reader.return_value = mock_ocr

    # Format: [ (bbox, text, confidence), ... ]
    mock_parse_paddle_ocr_results.return_value = [
        ([[10, 10], [100, 10], [100, 50], [10, 50]], "Text A", 0.99),
        ([[800, 800], [900, 800], [900, 850], [800, 850]], "Text B", 0.99),
    ]

    # Simulate YOLO finding 2 separated bubbles corresponding to the 2 fragments
    mock_detect_bubbles_yolo.return_value = [
        {
            "bbox": [0, 0, 120, 70],
            "confidence": 0.9,
            "mask_polygon": [[0, 0], [120, 0], [120, 70], [0, 70]],
            "safe_rect": [0, 0, 120, 70],
        },
        {
            "bbox": [750, 750, 200, 150],
            "confidence": 0.9,
            "mask_polygon": [[750, 750], [950, 750], [950, 900], [750, 900]],
            "safe_rect": [750, 750, 200, 150],
        },
    ]

    # Mock GET image info
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {"panels": []}
    mock_requests_get.return_value = mock_get_resp

    # 4. Mock Callback POST
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response

    # Execute OCR Processing
    job_data = {
        "imageId": "test-123",
        "imageUrl": "http://dummy",
        "sourceLanguage": "ja",
        "readingDirection": "rtl",
    }
    process_ocr(job_data)

    # 5. Assertions
    mock_requests_post.assert_called_once()
    payload = mock_requests_post.call_args.kwargs.get("json")

    assert payload is not None
    regions = payload.get("regions", [])

    # YOLO should have kept them as 2 separate regions because they are in different bubbles
    # If the bug was present, they would have been merged into 1 giant region.
    assert len(regions) == 2, "Expected 2 separate regions, but they were merged!"

    # Verify that the maskPolygon is preserved and isolated per bubble
    mask_a = json.loads(regions[0]["maskPolygon"])
    mask_b = json.loads(regions[1]["maskPolygon"])

    # Ensure they haven't been convex-hulled together (a hull would span the whole 1000x1000 image)
    for pt in mask_a:
        assert pt[0] < 500 and pt[1] < 500, "Mask A contains points from Mask B! Masking regression!"

    for pt in mask_b:
        assert pt[0] > 500 and pt[1] > 500, "Mask B contains points from Mask A! Masking regression!"


@patch("worker.handlers.ocr.redis_client")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.downscale_for_ocr")
@patch("worker.handlers.ocr.parse_paddle_ocr_results")
@patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
def test_process_ocr_different_shapes(
    mock_detect_bubbles_yolo,
    mock_get_paddle_ocr_reader,
    mock_parse_paddle_ocr_results,
    mock_downscale_for_ocr,
    mock_download_image,
    mock_requests_get,
    mock_requests_post,
    mock_redis,
):
    """
    Test that YOLO bubbles of varying shapes (square, circular, elliptical, pentagonal)
    are masked correctly and their polygons are perfectly preserved in the final payload.
    """
    mock_download_image.return_value = b"dummy"
    dummy_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    mock_downscale_for_ocr.return_value = (dummy_img, 1.0)

    mock_get_paddle_ocr_reader.return_value = MagicMock()

    # PaddleOCR detects 4 separate fragments
    mock_parse_paddle_ocr_results.return_value = [
        ([[10, 10], [50, 10], [50, 50], [10, 50]], "Square", 0.99),
        ([[110, 110], [150, 110], [150, 150], [110, 150]], "Circle", 0.99),
        ([[210, 210], [250, 210], [250, 250], [210, 250]], "Ellipse", 0.99),
        ([[310, 310], [350, 310], [350, 350], [310, 350]], "Pentagon", 0.99),
    ]

    # YOLO detects 4 shapes
    square_poly = [[0, 0], [60, 0], [60, 60], [0, 60]]
    circle_poly = [
        [130, 100],
        [150, 110],
        [160, 130],
        [150, 150],
        [130, 160],
        [110, 150],
        [100, 130],
        [110, 110],
    ]
    ellipse_poly = [
        [230, 190],
        [260, 210],
        [260, 250],
        [230, 270],
        [200, 250],
        [200, 210],
    ]
    pentagon_poly = [[330, 300], [370, 320], [360, 370], [300, 370], [290, 320]]

    mock_detect_bubbles_yolo.return_value = [
        {
            "bbox": [0, 0, 60, 60],
            "confidence": 0.9,
            "mask_polygon": square_poly,
            "safe_rect": [5, 5, 50, 50],
        },
        {
            "bbox": [100, 100, 60, 60],
            "confidence": 0.9,
            "mask_polygon": circle_poly,
            "safe_rect": [105, 105, 50, 50],
        },
        {
            "bbox": [200, 190, 60, 80],
            "confidence": 0.9,
            "mask_polygon": ellipse_poly,
            "safe_rect": [205, 195, 50, 70],
        },
        {
            "bbox": [290, 300, 80, 70],
            "confidence": 0.9,
            "mask_polygon": pentagon_poly,
            "safe_rect": [295, 305, 70, 60],
        },
    ]

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {"panels": []}
    mock_requests_get.return_value = mock_get_resp

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response

    job_data = {
        "imageId": "shapes-123",
        "imageUrl": "http://dummy",
        "sourceLanguage": "ja",
        "readingDirection": "rtl",
    }
    process_ocr(job_data)

    mock_requests_post.assert_called_once()
    payload = mock_requests_post.call_args.kwargs.get("json")
    regions = payload.get("regions", [])

    assert len(regions) == 4, "Expected exactly 4 regions"

    def poly_bbox(poly):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return (min(xs), min(ys), max(xs), max(ys))

    def bbox_match(b1, b2, tol=2):
        return all(abs(a - b) <= tol for a, b in zip(b1, b2, strict=False))

    masks_found = [json.loads(r["maskPolygon"]) for r in regions]
    bboxes_found = [poly_bbox(m) for m in masks_found]

    def check_bbox_present(target):
        return any(bbox_match(target, b) for b in bboxes_found)

    assert check_bbox_present(poly_bbox(square_poly)), "Square mask was not preserved"
    assert check_bbox_present(poly_bbox(circle_poly)), "Circular mask was not preserved"
    assert check_bbox_present(poly_bbox(ellipse_poly)), "Elliptical mask was not preserved"
    assert check_bbox_present(poly_bbox(pentagon_poly)), "Pentagonal mask was not preserved"
