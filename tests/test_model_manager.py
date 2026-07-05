import time
from unittest.mock import MagicMock
from worker.model_manager import ModelManager

import sys


def test_get_paddle_ocr_reader():
    mm = ModelManager()

    mock_paddle = MagicMock()
    mock_paddle.return_value = "paddle_instance"
    sys.modules["paddleocr"] = MagicMock(PaddleOCR=mock_paddle)

    try:
        mock_paddle.return_value = "paddle_instance"

        # Test valid language
        reader = mm.get_paddle_ocr_reader("ja")
        assert reader == "paddle_instance"
        assert "japan" in mm.paddle_readers

        # Test cached
        reader2 = mm.get_paddle_ocr_reader("ja")
        assert reader2 == "paddle_instance"
        assert mock_paddle.call_count == 1

        # Test invalid/fallback language
        reader_en = mm.get_paddle_ocr_reader("en")
        assert reader_en == "paddle_instance"
        assert mock_paddle.call_count == 2

        # Test error initialization
        mm.paddle_readers.clear()
        mock_paddle.side_effect = Exception("failed init")
        reader_err = mm.get_paddle_ocr_reader("fr")
        assert reader_err is None
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
