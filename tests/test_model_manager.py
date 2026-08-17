import sys
import time
from unittest.mock import MagicMock

import pytest

from worker.model_manager import ModelManager


@pytest.fixture(autouse=True)
def _reset_paddle_availability():
    """`paddle_ocr_available` is class-level state, and a failed init latches it to False.

    Without this reset the first test that exercises an initialisation failure silently disables
    PaddleOCR for every test defined after it, which shows up as unrelated tests getting None back.
    """
    ModelManager.paddle_ocr_available = True
    yield
    ModelManager.paddle_ocr_available = True


def test_get_paddle_ocr_reader():
    mm = ModelManager()

    mock_paddle = MagicMock()
    mock_paddle.return_value = "paddle_instance"
    sys.modules["paddleocr"] = MagicMock(PaddleOCR=mock_paddle)

    try:
        mock_paddle.return_value = "paddle_instance"

        # Test valid language. Readers are keyed by the resolved det/rec pair, not the language, so
        # one language can hold several entries (reader + detector, or two model families).
        reader = mm.get_paddle_ocr_reader("ja")
        assert reader == "paddle_instance"
        assert "PP-OCRv6:PP-OCRv6_medium_det:PP-OCRv6_medium_rec" in mm.paddle_readers

        # Test cached
        reader2 = mm.get_paddle_ocr_reader("ja")
        assert reader2 == "paddle_instance"
        assert mock_paddle.call_count == 1

        # English resolves to the same PP-OCRv6 pair as Japanese, so it reuses the loaded reader.
        # Keying by language instead loaded the identical model a second time.
        reader_en = mm.get_paddle_ocr_reader("en")
        assert reader_en == "paddle_instance"
        assert mock_paddle.call_count == 1

        # Korean needs a different recognition model, so it does load a second reader.
        assert mm.get_paddle_ocr_reader("ko") == "paddle_instance"
        assert mock_paddle.call_count == 2

        # Test error initialization
        mm.paddle_readers.clear()
        mock_paddle.side_effect = Exception("failed init")
        reader_err = mm.get_paddle_ocr_reader("fr")
        assert reader_err is None
    finally:
        del sys.modules["paddleocr"]


def test_korean_loads_a_recognition_model_that_can_read_hangul():
    """PP-OCRv6 ships no Korean recognition model, so asking for Korean must not load one.

    This is the regression that made Korean chapters come back as CJK/Latin noise: the worker pinned
    PP-OCRv6_medium_rec for every language, and PaddleOCR silently ignores `lang` once explicit model
    names are passed.
    """
    mm = ModelManager()
    mock_paddle = MagicMock(return_value="paddle_instance")
    sys.modules["paddleocr"] = MagicMock(PaddleOCR=mock_paddle)

    try:
        assert mm.get_paddle_ocr_reader("ko") == "paddle_instance"
        kwargs = mock_paddle.call_args.kwargs
        assert kwargs["text_recognition_model_name"] == "korean_PP-OCRv5_mobile_rec"
        assert kwargs["text_detection_model_name"] == "PP-OCRv5_mobile_det"
        # `lang` must not be passed alongside explicit model names — PaddleOCR ignores it and warns.
        assert "lang" not in kwargs
    finally:
        del sys.modules["paddleocr"]


def test_explicit_v6_choice_is_overridden_for_korean():
    """An explicit PP-OCRv6 selection is honoured for Japanese but routed away for Korean."""
    mm = ModelManager()
    mock_paddle = MagicMock(return_value="paddle_instance")
    sys.modules["paddleocr"] = MagicMock(PaddleOCR=mock_paddle)

    try:
        mm.get_paddle_ocr_reader("ja", "PP-OCRv6")
        assert mock_paddle.call_args.kwargs["text_recognition_model_name"] == "PP-OCRv6_medium_rec"

        mm.get_paddle_ocr_reader("ko", "PP-OCRv6")
        assert mock_paddle.call_args.kwargs["text_recognition_model_name"] == "korean_PP-OCRv5_mobile_rec"
    finally:
        del sys.modules["paddleocr"]


def test_japanese_and_korean_readers_coexist():
    """Caching per det/rec pair keeps both loaded; keying by language would evict one per page."""
    mm = ModelManager()
    mock_paddle = MagicMock(return_value="paddle_instance")
    sys.modules["paddleocr"] = MagicMock(PaddleOCR=mock_paddle)

    try:
        mm.get_paddle_ocr_reader("ja")
        mm.get_paddle_ocr_reader("ko")
        assert len(mm.paddle_readers) == 2
        # Re-requesting either must hit the cache rather than re-initialising.
        mm.get_paddle_ocr_reader("ja")
        mm.get_paddle_ocr_reader("ko")
        assert mock_paddle.call_count == 2
    finally:
        del sys.modules["paddleocr"]


def test_get_paddle_ocr_unavailable():
    mm = ModelManager()
    ModelManager.paddle_ocr_available = False
    assert mm.get_paddle_ocr_reader("ja") is None
    ModelManager.paddle_ocr_available = True


def test_unload_expired_models():
    mm = ModelManager()
    mm.paddle_readers["japan"] = "paddle_instance"
    mm.paddle_last_used["japan"] = time.time() - 100

    mm.unload_expired_models(50)
    assert mm.paddle_readers["japan"] is None


def test_get_loaded_models_status():
    mm = ModelManager()
    mm.paddle_readers["japan"] = "paddle_instance"
    mm.paddle_last_used["japan"] = time.time() - 10

    status = mm.get_loaded_models_status(50)
    assert len(status) == 1
    assert "PaddleOCR:japan" in status[0]
