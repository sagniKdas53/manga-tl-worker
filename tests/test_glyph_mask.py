import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.services.glyph_mask import (
    CTD_SIZE_MULTIPLE,
    _pad_to_multiple,
    _sha256,
    get_ctd_session,
    segment_crop,
    threshold_mask,
)


@patch("worker.services.glyph_mask.os.path.exists")
def test_sha256_not_exists(mock_exists):
    mock_exists.return_value = False
    assert _sha256("dummy") is None


@patch("worker.services.glyph_mask.os.path.exists")
def test_get_ctd_session_no_model(mock_exists):
    mock_exists.return_value = False

    import worker.services.glyph_mask as gm

    gm._ort_session = None

    with pytest.raises(FileNotFoundError):
        get_ctd_session()


@patch("worker.services.glyph_mask.os.path.exists")
@patch("worker.services.glyph_mask._sha256")
@patch.dict("sys.modules", {"onnxruntime": MagicMock()})
def test_get_ctd_session_success(mock_sha, mock_exists):
    import sys

    mock_ort = sys.modules["onnxruntime"]
    mock_exists.return_value = True
    mock_sha.return_value = "dummy_hash"
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    import worker.services.glyph_mask as gm

    gm._ort_session = None

    session = get_ctd_session()
    assert session == mock_session
    assert gm._ort_session == mock_session


def test_pad_to_multiple_already_aligned():
    crop = np.zeros((128, 256, 3), dtype=np.uint8)
    padded, orig_h, orig_w = _pad_to_multiple(crop, CTD_SIZE_MULTIPLE)
    assert padded.shape == (128, 256, 3)
    assert (orig_h, orig_w) == (128, 256)


def test_pad_to_multiple_pads_without_stretching():
    crop = np.full((100, 50, 3), 255, dtype=np.uint8)
    padded, orig_h, orig_w = _pad_to_multiple(crop, CTD_SIZE_MULTIPLE)
    assert padded.shape[0] % CTD_SIZE_MULTIPLE == 0
    assert padded.shape[1] % CTD_SIZE_MULTIPLE == 0
    assert (orig_h, orig_w) == (100, 50)
    # Original content is preserved verbatim in the top-left corner, not resized.
    assert np.array_equal(padded[:100, :50], crop)
    # The padding itself is zero, not a stretch/repeat of the source content.
    assert padded[100:, :].sum() == 0
    assert padded[:, 50:].sum() == 0


def test_segment_crop_rejects_non_hwc3():
    with pytest.raises(ValueError):
        segment_crop(np.zeros((10, 10), dtype=np.uint8))


def test_segment_crop_rejects_degenerate_crop():
    session = MagicMock()
    with pytest.raises(ValueError):
        segment_crop(np.zeros((0, 10, 3), dtype=np.uint8), session=session)


def test_segment_crop_uses_injected_session_and_matches_input_shape():
    session = MagicMock()
    # A 100x50 crop pads to 128x64 (next multiple of 64); the mocked session must be fed
    # that padded shape and segment_crop must crop the output back down to 100x50.
    session.run.return_value = [np.zeros((1, 1, 128, 64), dtype=np.float32)]
    crop = np.zeros((100, 50, 3), dtype=np.uint8)

    prob = segment_crop(crop, session=session)

    assert prob.shape == (100, 50)
    assert prob.dtype == np.float32
    fed_tensor = session.run.call_args[0][1]["images"]
    assert fed_tensor.shape == (1, 3, 128, 64)


def test_segment_crop_applies_sigmoid_when_output_is_logits():
    session = MagicMock()
    # Values outside [0, 1] must be sigmoid-squashed, matching CTD's own post-processing.
    session.run.return_value = [np.array([[[[10.0, -10.0], [0.0, 5.0]]]], dtype=np.float32)]
    crop = np.zeros((2, 2, 3), dtype=np.uint8)

    prob = segment_crop(crop, session=session)

    assert prob.shape == (2, 2)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert prob[0, 0] > 0.9  # sigmoid(10) ~= 1
    assert prob[0, 1] < 0.1  # sigmoid(-10) ~= 0


def test_threshold_mask():
    prob = np.array([0.1, 0.29, 0.3, 0.9], dtype=np.float32)
    mask = threshold_mask(prob, 0.3)
    assert mask.tolist() == [False, False, True, True]


CTD_MODEL_PATH_ON_DISK = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bootstrap", "ctd_seg_dyn.onnx")


@pytest.mark.skipif(
    not os.path.exists(CTD_MODEL_PATH_ON_DISK),
    reason="ctd_seg_dyn.onnx not seeded at data/bootstrap/ on this host",
)
def test_segment_crop_real_model_produces_sane_probability_map():
    """End-to-end sanity check against the real pinned CTD model (not a fixture crop from the
    21-page validation set -- just confirms the model loads and the (H, W)/[0,1] contract
    holds for a real-shaped, non-64-multiple crop)."""
    import worker.services.glyph_mask as gm

    gm._ort_session = None
    with (
        patch("worker.config.CTD_MODEL_PATH", os.path.abspath(CTD_MODEL_PATH_ON_DISK)),
        patch("worker.services.glyph_mask.CTD_MODEL_PATH", os.path.abspath(CTD_MODEL_PATH_ON_DISK)),
    ):
        crop = np.random.default_rng(0).integers(0, 255, size=(233, 417, 3), dtype=np.uint8)
        prob = segment_crop(crop)

    assert prob.shape == (233, 417)
    assert prob.dtype == np.float32
    assert prob.min() >= 0.0 and prob.max() <= 1.0
