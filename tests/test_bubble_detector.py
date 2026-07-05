import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from worker.services.bubble_detector import (
    get_sha256,
    get_ort_session,
    letterbox,
    detect_bubbles_yolo,
)


@patch("worker.services.bubble_detector.os.path.exists")
def test_get_sha256_not_exists(mock_exists):
    mock_exists.return_value = False
    assert get_sha256("dummy") is None


@patch("worker.services.bubble_detector.os.path.exists")
@patch("builtins.open")
def test_get_sha256_exists(mock_open, mock_exists):
    mock_exists.return_value = True
    mock_file = MagicMock()
    mock_file.read.side_effect = [b"chunk1", b"chunk2", b""]
    mock_open.return_value.__enter__.return_value = mock_file

    sha = get_sha256("dummy")
    assert sha is not None
    assert len(sha) == 64  # length of sha256 hex string


@patch("worker.services.bubble_detector.os.path.exists")
def test_get_ort_session_no_model(mock_exists):
    mock_exists.return_value = False

    # Needs to reset global _ort_session for this test to be accurate if other tests run before
    import worker.services.bubble_detector as bd

    bd._ort_session = None

    with pytest.raises(FileNotFoundError):
        get_ort_session()


@patch("worker.services.bubble_detector.os.path.exists")
@patch("worker.services.bubble_detector.get_sha256")
@patch.dict("sys.modules", {"onnxruntime": MagicMock()})
def test_get_ort_session_success(mock_sha, mock_exists):
    import sys

    mock_ort = sys.modules["onnxruntime"]
    mock_exists.return_value = True
    mock_sha.return_value = "dummy_hash"
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    import worker.services.bubble_detector as bd

    bd._ort_session = None

    session = get_ort_session()
    assert session == mock_session
    assert bd._ort_session == mock_session


def test_letterbox():
    # create a dummy image
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    padded, r, (dw, dh), new_unpad = letterbox(img, new_shape=(300, 300))
    assert padded.shape[:2] == (300, 300)


def test_detect_bubbles_yolo_no_image():
    assert detect_bubbles_yolo(None) == []


@patch("worker.services.bubble_detector.get_ort_session")
def test_detect_bubbles_yolo_no_session(mock_get_session):
    mock_get_session.return_value = None
    with pytest.raises(RuntimeError):
        detect_bubbles_yolo(np.zeros((100, 100, 3), dtype=np.uint8))


@patch("worker.services.bubble_detector.get_ort_session")
def test_detect_bubbles_yolo_empty_predictions(mock_get_session):
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input_name")]

    # Mocking inference to return empty or low confidence predictions
    # preds: [1, 37, 33600]
    preds = np.zeros((1, 37, 100), dtype=np.float32)
    proto = np.zeros((1, 32, 320, 320), dtype=np.float32)
    mock_session.run.return_value = [preds, proto]

    mock_get_session.return_value = mock_session

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_bubbles_yolo(img)
    assert result == []


@patch("worker.services.bubble_detector.get_ort_session")
@patch("worker.services.bubble_detector.cv2.dnn.NMSBoxes")
def test_detect_bubbles_yolo_with_predictions(mock_nms, mock_get_session):
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input_name")]

    # Valid prediction with confidence > threshold (e.g. 0.5)
    preds = np.zeros((1, 37, 10), dtype=np.float32)
    # Set one prediction to have high confidence
    preds[0, 4, 0] = 0.9  # score
    preds[0, 0:4, 0] = [50, 50, 20, 20]  # cx, cy, w, h
    preds[0, 5:, 0] = np.random.rand(32)  # coefficients

    proto = np.random.rand(1, 32, 320, 320).astype(np.float32)
    mock_session.run.return_value = [preds, proto]

    mock_get_session.return_value = mock_session
    mock_nms.return_value = [[0]]

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_bubbles_yolo(img)

    assert len(result) == 1
    assert "bbox" in result[0]
    assert "confidence" in result[0]
    assert "mask_polygon" in result[0]
    assert "safe_rect" in result[0]
