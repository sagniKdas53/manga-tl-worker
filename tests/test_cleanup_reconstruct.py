import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.services.cleanup_reconstruct import (
    CleanupConfig,
    CleanupResult,
    _crop_with_context,
    _dilate,
    _encode_mask_png,
    _encode_patch_png,
    _gate_to_region_footprint,
    _reconstruct_aot,
    _residual_ink_pct,
    _sha256,
    get_aot_session,
    reconstruct_region,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_crop_with_context_clamps_to_image_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # x=10,y=10 - pad_px=64 would go negative on the left/top -> clamps there; the right/bottom
    # edge (10+20+64=94) stays inside the 100x100 image, so the crop is not the whole page.
    crop, x0, y0 = _crop_with_context(img, x=10, y=10, width=20, height=20, pad_px=64)
    assert crop is not None
    assert (x0, y0) == (0, 0)
    assert crop.shape[:2] == (94, 94)


def test_crop_with_context_degenerate_region_returns_none():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, _x0, _y0 = _crop_with_context(img, x=0, y=0, width=0, height=0, pad_px=0)
    assert crop is None


def test_gate_to_region_footprint_zeroes_outside_bbox():
    mask = np.ones((50, 50), dtype=bool)
    # Region bbox occupies rows/cols [20, 30) in crop-local coordinates (crop_x0=crop_y0=0).
    gated = _gate_to_region_footprint(mask, crop_x0=0, crop_y0=0, x=20, y=20, width=10, height=10, margin_px=0)
    assert gated[:20, :].sum() == 0
    assert gated[30:, :].sum() == 0
    assert gated[20:30, 20:30].all()


def test_dilate_grows_mask():
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    dilated = _dilate(mask, px=2)
    assert dilated.sum() > 1
    assert dilated[10, 10]


def test_dilate_noop_for_zero_px():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert np.array_equal(_dilate(mask, px=0), mask)


# ---------------------------------------------------------------------------
# Asset encoding
# ---------------------------------------------------------------------------


def test_encode_mask_png_roundtrip():
    import cv2

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    png_bytes = _encode_mask_png(mask)
    decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (10, 10, 4)
    assert (decoded[2:5, 2:5, 3] == 255).all()
    assert (decoded[0, 0] == 0).all()


def test_encode_patch_png_zeroes_outside_mask():
    import cv2

    reconstructed = np.full((10, 10, 3), 128, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    png_bytes = _encode_patch_png(reconstructed, mask)
    decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (10, 10, 4)
    assert (decoded[2:5, 2:5, 3] == 255).all()
    assert (decoded[2:5, 2:5, :3] == 128).all()
    assert (decoded[0, 0] == 0).all()  # zeroed, not just transparent


# ---------------------------------------------------------------------------
# Residual-ink metric
# ---------------------------------------------------------------------------


def test_residual_ink_pct_empty_mask_is_zero():
    mask = np.zeros((10, 10), dtype=bool)
    assert _residual_ink_pct(np.zeros((10, 10, 3), dtype=np.uint8), mask, threshold=0.3) == 0.0


@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_residual_ink_pct_matches_offline_definition(mock_segment_crop):
    # Half the masked footprint still scores above threshold -> 50% residual, restricted to
    # the mask (a hot pixel outside the mask must not count).
    prob = np.zeros((4, 4), dtype=np.float32)
    prob[0, 0] = 0.9
    prob[0, 1] = 0.9
    prob[3, 3] = 0.9  # outside the mask -- must not be counted
    mock_segment_crop.return_value = prob

    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 0] = True
    mask[1, 1] = True

    pct = _residual_ink_pct(np.zeros((4, 4, 3), dtype=np.uint8), mask, threshold=0.3)
    assert pct == 50.0


# ---------------------------------------------------------------------------
# AOT session loading (mirrors bubble_detector/glyph_mask's own tests)
# ---------------------------------------------------------------------------


@patch("worker.services.cleanup_reconstruct.os.path.exists")
def test_sha256_not_exists(mock_exists):
    mock_exists.return_value = False
    assert _sha256("dummy") is None


@patch("worker.services.cleanup_reconstruct.os.path.exists")
def test_get_aot_session_no_model(mock_exists):
    mock_exists.return_value = False

    import worker.services.cleanup_reconstruct as cr

    cr._aot_session = None

    with pytest.raises(FileNotFoundError):
        get_aot_session()


@patch("worker.services.cleanup_reconstruct.os.path.exists")
@patch("worker.services.cleanup_reconstruct._sha256")
@patch.dict("sys.modules", {"onnxruntime": MagicMock()})
def test_get_aot_session_success(mock_sha, mock_exists):
    import sys

    mock_ort = sys.modules["onnxruntime"]
    mock_exists.return_value = True
    mock_sha.return_value = "dummy_hash"
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    import worker.services.cleanup_reconstruct as cr

    cr._aot_session = None

    session = get_aot_session()
    assert session == mock_session
    assert cr._aot_session == mock_session


@patch("worker.services.cleanup_reconstruct.get_aot_session")
def test_reconstruct_aot_zeroes_the_hole_before_feeding_the_model(mock_get_session):
    """The measured contract (docs/archive/erasure_overhaul_plan_2026-08-26.md): zeroing the
    masked hole in the input image is what earns 35.44dB vs 18.34dB at the same [-1,1]
    normalization -- the model fills a blanked hole guided by the mask, it does not edit pixels
    that are still present. A regression here silently produces garbage/noise output that a
    generic shape/dtype test would not catch."""
    session = MagicMock()
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)  # a uniform crop so any non-zero input in
    # the hole is unambiguously the bug, not incidental content
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    session.run.return_value = [np.zeros((1, 3, 64, 64), dtype=np.float32)]
    mock_get_session.return_value = session

    _reconstruct_aot(crop, mask, CleanupConfig())

    fed_image = session.run.call_args[0][1]["image"]
    fed_mask = session.run.call_args[0][1]["mask"] >= 0.5
    # image is NCHW; broadcast the mask over channels the same way the implementation does.
    masked_region = fed_image[0][:, fed_mask[0, 0]]
    assert masked_region.size > 0
    assert np.allclose(masked_region, 0.0), "the hole must be zeroed before the model sees it"
    # Pixels outside the hole must still carry real (non-zero, normalized) content.
    unmasked_region = fed_image[0][:, ~fed_mask[0, 0]]
    assert not np.allclose(unmasked_region, 0.0)


# ---------------------------------------------------------------------------
# reconstruct_region orchestration
# ---------------------------------------------------------------------------


def _prob_map(shape: tuple[int, int], hot_slice: tuple[slice, slice]) -> np.ndarray:
    prob = np.zeros(shape, dtype=np.float32)
    prob[hot_slice] = 0.9
    return prob


def test_reconstruct_region_none_image_returns_none():
    assert reconstruct_region(None, 0, 0, 10, 10) is None


def test_reconstruct_region_degenerate_size_returns_none():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert reconstruct_region(img, 0, 0, 0, 0) is None


@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_none_when_ctd_fails(mock_segment_crop):
    mock_segment_crop.side_effect = RuntimeError("model missing")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert reconstruct_region(img, 10, 10, 20, 20) is None


@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_none_when_nothing_detected_in_footprint(mock_segment_crop):
    # CTD "detects" something, but only outside the region's own bbox+margin.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    config = CleanupConfig(crop_pad_px=10)
    crop, _crop_x0, _crop_y0 = _crop_with_context(img, 40, 40, 20, 20, config.crop_pad_px)
    assert crop is not None
    prob = np.zeros(crop.shape[:2], dtype=np.float32)
    prob[0, 0] = 0.9  # in the padding, not the region itself
    mock_segment_crop.return_value = prob
    assert reconstruct_region(img, 40, 40, 20, 20, config=config) is None


@patch("worker.services.cleanup_reconstruct._reconstruct_telea")
@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_routes_flat_interior_to_telea(mock_segment_crop, mock_telea):
    img = np.full((100, 100, 3), 200, dtype=np.uint8)  # flat -> low pixel_spread
    config = CleanupConfig(crop_pad_px=5, residual_ink_max_pct=100.0)
    crop, crop_x0, crop_y0 = _crop_with_context(img, 40, 40, 20, 20, config.crop_pad_px)
    assert crop is not None
    hot = (slice(5, 25), slice(5, 25))
    detect_prob = _prob_map(crop.shape[:2], hot)
    residual_prob = np.zeros(crop.shape[:2], dtype=np.float32)  # clean after "reconstruction"
    mock_segment_crop.side_effect = [detect_prob, residual_prob]
    mock_telea.return_value = crop.copy()

    result = reconstruct_region(img, 40, 40, 20, 20, config=config)

    assert result is not None
    mock_telea.assert_called_once()
    assert any("telea" in d for d in result.diagnostics)
    assert result.bounds == {"x": crop_x0, "y": crop_y0, "width": crop.shape[1], "height": crop.shape[0]}
    assert result.generator_sha256


@patch("worker.services.cleanup_reconstruct._reconstruct_aot")
@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_routes_structured_interior_to_aot(mock_segment_crop, mock_aot):
    img = np.random.default_rng(0).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)  # high spread
    config = CleanupConfig(crop_pad_px=5, residual_ink_max_pct=100.0)
    crop, _crop_x0, _crop_y0 = _crop_with_context(img, 40, 40, 20, 20, config.crop_pad_px)
    assert crop is not None
    hot = (slice(5, 25), slice(5, 25))
    detect_prob = _prob_map(crop.shape[:2], hot)
    residual_prob = np.zeros(crop.shape[:2], dtype=np.float32)
    mock_segment_crop.side_effect = [detect_prob, residual_prob]
    mock_aot.return_value = crop.copy()

    result = reconstruct_region(img, 40, 40, 20, 20, config=config)

    assert result is not None
    mock_aot.assert_called_once()
    assert any("aot" in d for d in result.diagnostics)


@patch("worker.services.cleanup_reconstruct._reconstruct_aot")
@patch("worker.services.cleanup_reconstruct._reconstruct_telea")
@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_falls_back_to_telea_when_aot_fails(mock_segment_crop, mock_telea, mock_aot):
    img = np.random.default_rng(0).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    config = CleanupConfig(crop_pad_px=5, residual_ink_max_pct=100.0)
    crop, _crop_x0, _crop_y0 = _crop_with_context(img, 40, 40, 20, 20, config.crop_pad_px)
    assert crop is not None
    hot = (slice(5, 25), slice(5, 25))
    mock_segment_crop.side_effect = [_prob_map(crop.shape[:2], hot), np.zeros(crop.shape[:2], dtype=np.float32)]
    mock_aot.side_effect = RuntimeError("AOT model missing")
    mock_telea.return_value = crop.copy()

    result = reconstruct_region(img, 40, 40, 20, 20, config=config)

    assert result is not None
    mock_telea.assert_called_once()
    assert any("telea-fallback" in d for d in result.diagnostics)


@patch("worker.services.cleanup_reconstruct._reconstruct_telea")
@patch("worker.services.cleanup_reconstruct.segment_crop")
def test_reconstruct_region_uses_one_ctd_pass_and_leaves_residual_check_offline(mock_segment_crop, mock_telea):
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    config = CleanupConfig(crop_pad_px=5, residual_ink_max_pct=15.0)
    crop, _crop_x0, _crop_y0 = _crop_with_context(img, 40, 40, 20, 20, config.crop_pad_px)
    assert crop is not None
    hot = (slice(5, 25), slice(5, 25))
    detect_prob = _prob_map(crop.shape[:2], hot)
    # The runtime cleanup path no longer performs a second CTD residual pass.
    mock_segment_crop.return_value = detect_prob
    mock_telea.return_value = crop.copy()

    assert reconstruct_region(img, 40, 40, 20, 20, config=config) is not None
    mock_segment_crop.assert_called_once()


CTD_MODEL_PATH_ON_DISK = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bootstrap", "ctd_seg_dyn.onnx")
AOT_MODEL_PATH_ON_DISK = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bootstrap", "lama_aot.onnx")


@pytest.mark.skipif(
    not (os.path.exists(CTD_MODEL_PATH_ON_DISK) and os.path.exists(AOT_MODEL_PATH_ON_DISK)),
    reason="ctd_seg_dyn.onnx / lama_aot.onnx not seeded at data/bootstrap/ on this host",
)
def test_reconstruct_region_real_models_end_to_end():
    """Runs the real CTD + (possibly) AOT models against a synthetic page with text-like dark
    strokes on a textured background, confirming the whole pipeline produces a well-formed
    result or a clean None -- not a crash -- against real ONNX Runtime inference."""
    import worker.services.cleanup_reconstruct as cr
    import worker.services.glyph_mask as gm

    gm._ort_session = None
    cr._aot_session = None
    with (
        patch("worker.config.CTD_MODEL_PATH", os.path.abspath(CTD_MODEL_PATH_ON_DISK)),
        patch("worker.services.glyph_mask.CTD_MODEL_PATH", os.path.abspath(CTD_MODEL_PATH_ON_DISK)),
        patch("worker.config.AOT_MODEL_PATH", os.path.abspath(AOT_MODEL_PATH_ON_DISK)),
        patch("worker.services.cleanup_reconstruct.AOT_MODEL_PATH", os.path.abspath(AOT_MODEL_PATH_ON_DISK)),
    ):
        rng = np.random.default_rng(0)
        img = rng.integers(60, 200, size=(300, 400, 3), dtype=np.uint8)
        # Paint dark "glyph-like" strokes into a sub-region to give CTD something to find.
        img[140:160, 150:250] = 10
        result = reconstruct_region(img, 150, 140, 100, 20, config=CleanupConfig(residual_ink_max_pct=100.0))

    if result is not None:
        assert isinstance(result, CleanupResult)
        assert result.bounds["width"] > 0 and result.bounds["height"] > 0
        assert len(result.mask_png) > 0
        assert len(result.patch_png) > 0
