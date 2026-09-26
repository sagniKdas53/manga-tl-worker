from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.services.cleanup_review import assess_empty_mask


@pytest.mark.parametrize(
    "quad,text,expected,reason,size",
    [
        ([[32, 32], [112, 32], [112, 72], [32, 72]], "Hello!", "Hello", "agrees", "small-text candidate"),
        ([[32, 32], [112, 32], [112, 112], [32, 112]], "福", "R", "disagrees", "not a small-text candidate"),
        ([[32, 32], [112, 32], [112, 112], [32, 112]], "短か！！", "短か！", "agrees", "not a small-text candidate"),
    ],
)
def test_review_checks_size_and_ocr_agreement_without_changing_pixels(quad, text, expected, reason, size):
    image = np.full((200, 200, 3), 210, np.uint8)
    before = image.copy()
    reader = MagicMock()
    with (
        patch("worker.services.cleanup_review.get_local_ocr_backend", return_value="paddle"),
        patch("worker.services.cleanup_review.model_manager.get_paddle_ocr_reader", return_value=reader),
        patch("worker.services.cleanup_review.parse_paddle_ocr_results", return_value=[(quad, text, 0.8)]),
    ):
        diagnostics = assess_empty_mask(image, (32, 32, 100, 100), expected_text=expected)
    assert size in diagnostics[0]
    assert reason in diagnostics[1]
    np.testing.assert_array_equal(image, before)
    reader.predict.assert_called_once()


def test_review_ocr_unavailable_remains_uncertain():
    with patch("worker.services.cleanup_review.get_local_ocr_backend", side_effect=RuntimeError("missing")):
        diagnostics = assess_empty_mask(np.zeros((30, 30, 3), np.uint8), (0, 0, 20, 20))
    assert "unavailable" in diagnostics[0]


def test_review_ignores_text_outside_region():
    with (
        patch("worker.services.cleanup_review.get_local_ocr_backend", return_value="rapidocr"),
        patch("worker.services.cleanup_review.model_manager.get_rapid_ocr_reader"),
        patch(
            "worker.services.cleanup_review.parse_rapid_ocr_results",
            return_value=[
                ([[0, 0], [10, 0], [10, 10], [0, 10]], "neighbor", 0.9),
            ],
        ),
    ):
        diagnostics = assess_empty_mask(np.zeros((100, 100, 3), np.uint8), (40, 40, 20, 20))
    assert "no text confirmed inside" in diagnostics[0]
