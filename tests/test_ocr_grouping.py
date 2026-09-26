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


@patch("worker.handlers.ocr.redis_client")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.downscale_for_ocr")
@patch("worker.handlers.ocr.parse_paddle_ocr_results")
@patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
def test_process_ocr_failed_local_split_never_grants_the_fused_mask(
    mock_detect_bubbles_yolo,
    mock_get_paddle_ocr_reader,
    mock_parse_paddle_ocr_results,
    mock_downscale_for_ocr,
    mock_download_image,
    mock_requests_get,
    mock_requests_post,
    mock_redis,
):
    mock_download_image.return_value = b"dummy"
    mock_downscale_for_ocr.return_value = (np.full((1000, 1000, 3), 255, dtype=np.uint8), 1.0)
    mock_get_paddle_ocr_reader.return_value = MagicMock()
    mock_parse_paddle_ocr_results.return_value = [
        ([[100, 100], [160, 100], [160, 130], [100, 130]], "top", 0.99),
        ([[800, 800], [860, 800], [860, 830], [800, 830]], "bottom", 0.99),
    ]
    fused_polygon = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    mock_detect_bubbles_yolo.return_value = [
        {
            "bbox": [0, 0, 1000, 1000],
            "confidence": 0.9,
            "mask_polygon": fused_polygon,
            "safe_rect": [0, 0, 1000, 1000],
        }
    ]
    image_info = MagicMock()
    image_info.status_code = 200
    image_info.json.return_value = {"panels": []}
    mock_requests_get.return_value = image_info
    response = MagicMock()
    response.status_code = 200
    mock_requests_post.return_value = response

    with patch("worker.handlers.ocr.get_split_polygon", return_value=None):
        process_ocr(
            {
                "imageId": "split-failure",
                "imageUrl": "http://dummy",
                "sourceLanguage": "ja",
                "readingDirection": "rtl",
            }
        )

    payload = mock_requests_post.call_args.kwargs["json"]
    regions = payload["regions"]
    assert len(regions) == 2
    assert all(region["maskPolygon"] is None for region in regions)
    assert all(region["backgroundColor"] is None for region in regions)
    assert all(
        region["ownershipProvenance"]["containerResolution"] == "review-local-split-failed" for region in regions
    )
    assert all(region["ownershipProvenance"]["sourceQuad"] != fused_polygon for region in regions)


@patch("worker.handlers.ocr.redis_client")
@patch("worker.handlers.ocr.requests.post")
@patch("worker.handlers.ocr.requests.get")
@patch("worker.handlers.ocr.download_image")
@patch("worker.handlers.ocr.downscale_for_ocr")
@patch("worker.handlers.ocr.parse_paddle_ocr_results")
@patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader")
@patch("worker.handlers.ocr.detect_bubbles_yolo")
def test_process_ocr_persists_assigned_live_owner_without_capture_mode(
    mock_detect_bubbles_yolo,
    mock_get_paddle_ocr_reader,
    mock_parse_paddle_ocr_results,
    mock_downscale_for_ocr,
    mock_download_image,
    mock_requests_get,
    mock_requests_post,
    mock_redis,
):
    mock_download_image.return_value = b"dummy"
    mock_downscale_for_ocr.return_value = (np.full((200, 200, 3), 255, dtype=np.uint8), 1.0)
    mock_get_paddle_ocr_reader.return_value = MagicMock()
    mock_parse_paddle_ocr_results.return_value = [
        ([[20, 20], [100, 20], [100, 40], [20, 40]], "first", 0.99),
        ([[20, 45], [100, 45], [100, 65], [20, 65]], "second", 0.99),
    ]
    polygon = [[0, 0], [180, 0], [180, 180], [0, 180]]
    mock_detect_bubbles_yolo.return_value = [
        {
            "bbox": [0, 0, 180, 180],
            "confidence": 0.9,
            "mask_polygon": polygon,
            "safe_rect": [0, 0, 180, 180],
        }
    ]
    image_info = MagicMock()
    image_info.status_code = 200
    image_info.json.return_value = {"panels": []}
    mock_requests_get.return_value = image_info
    response = MagicMock()
    response.status_code = 200
    mock_requests_post.return_value = response

    process_ocr(
        {
            "imageId": "assigned-live-owner",
            "imageUrl": "http://dummy",
            "sourceLanguage": "ja",
            "readingDirection": "ltr",
        }
    )

    regions = mock_requests_post.call_args.kwargs["json"]["regions"]
    assert len(regions) == 1
    fragments = regions[0]["ownershipProvenance"]["fragments"]
    decisions = [fragment["provenance"]["ownerDecision"] for fragment in fragments]
    assert all(decision["state"] == "assigned" for decision in decisions)
    assert decisions[0]["owner_id"] == decisions[1]["owner_id"]
