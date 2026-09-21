"""Glyph-level text segmentation via comic-text-detector's `seg` head.

This module owns raw CTD inference only: model loading and one pure `segment_crop` call.
It has no opinion on region policy, reconstruction, or asset upload -- see
``services.cleanup_reconstruct`` for the orchestration that turns a probability map into a
``cleanup_artifact``. Mirrors ``services.bubble_detector``'s ONNX-session/pinned-checksum
pattern.

Measured contract (docs/archive/ctd_mask_validation_2026-08-26.md, verified empirically
against ``ctd_seg_dyn.onnx`` before writing this module): input tensor ``images``,
``[1, 3, H, W]`` float32 RGB scaled to ``[0, 1]`` (no mean/std normalization); output tensor
``seg``, ``[1, 1, H, W]``, a probability map (sigmoid applied here if the raw output falls
outside ``[0, 1]``, matching the reference `comic-text-detector` post-processing). The graph
requires **both spatial dimensions to be a multiple of 64** (not 32 -- verified by direct
probing: 96, 160, 224, 288, 300, 304 all fail; 64, 128, 192, 256, 320, 448 all succeed).
``segment_crop`` pads a native-scale crop up to that multiple by itself; do not *square*-pad
crops before calling it -- that was measured to cost extra runtime for no accuracy gain (up
to 3.6x, per the validation doc). That finding did not cover a large crop's total pixel count.

R3 live-run finding (2026-09-21, docs/output-quality-implementation-tracker.md R3 addendum):
CTD has no size cap and this graph's cost is ~70us/pixel with no internal downscale, so an
uncapped context-padded crop -- and `reconstruct_region` calls this twice per region, once for
the initial mask and once more for the residual-ink recheck -- turned an 18-region page into a
27-minute OCR stage. `CTD_MAX_SIDE` below downscales before inference and upscales the
probability map back, the same shape as AOT's already-shipped `aot_max_side` clamp. This is a
new, deliberate cost/accuracy trade rather than a re-confirmation of the padding finding above;
the residual-ink recheck (run at the same capped resolution) is what still catches a mask this
degrades and falls back to R2 behaviour, but the recheck's claimed parity with the offline
21-page median-6.1% baseline (measured at native resolution) no longer strictly holds and should
be re-verified before trusting its threshold as tightly as before.
"""

import hashlib
import logging
import os
import time

import cv2
import numpy as np

from worker.config import CTD_MODEL_PATH, CTD_PINNED_CHECKSUM

logger = logging.getLogger(__name__)

_ort_session = None

# The seg-only subgraph requires both spatial dims to be a multiple of this stride.
CTD_SIZE_MULTIPLE = 64

# Same bound and same reasoning as cleanup_reconstruct.CleanupConfig.aot_max_side: only ever
# downscale, never upscale a crop already under this size. CTD measured ~70us/pixel with no
# internal cap (docs/output-quality-implementation-tracker.md R3 addendum, 2026-09-21) --
# without this, a 1408x832 context-padded crop cost ~82s per call, twice per region.
CTD_MAX_SIDE = 1024


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        while chunk := model_file.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def get_ctd_session():
    """Lazily load and cache the CTD `seg`-head ONNX session (CPU execution provider)."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    if not CTD_MODEL_PATH or not os.path.exists(CTD_MODEL_PATH):
        raise FileNotFoundError(
            f"Required CTD glyph-segmentation model is not available at path: {CTD_MODEL_PATH}. "
            "Cannot proceed in offline mode without the required model."
        )

    current_checksum = _sha256(CTD_MODEL_PATH)
    if current_checksum != CTD_PINNED_CHECKSUM:
        logger.warning(f"[CTD] Pinned checksum mismatch! Expected: {CTD_PINNED_CHECKSUM}, got: {current_checksum}")
    else:
        logger.info("[CTD] Model checksum matches pinned checksum.")

    try:
        import onnxruntime as ort

        logger.info(f"[CTD] Loading ONNX model from {CTD_MODEL_PATH} via ONNX Runtime...")
        _ort_session = ort.InferenceSession(CTD_MODEL_PATH, providers=["CPUExecutionProvider"])
        logger.info("[CTD] ONNX Runtime session initialized successfully.")
        return _ort_session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model via ONNX Runtime: {e}") from e


def _pad_to_multiple(crop_bgr: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    """Zero-pad ``crop_bgr`` at the bottom/right up to the next multiple of ``multiple``.

    Native aspect ratio is preserved -- only enough padding to satisfy the graph's stride
    requirement is added, never a resize and never a pad to square.
    """
    h, w = crop_bgr.shape[:2]
    padded_h = ((h + multiple - 1) // multiple) * multiple
    padded_w = ((w + multiple - 1) // multiple) * multiple
    if padded_h == h and padded_w == w:
        return crop_bgr, h, w
    padded = np.zeros((padded_h, padded_w, 3), dtype=crop_bgr.dtype)
    padded[:h, :w] = crop_bgr
    return padded, h, w


def segment_crop(crop_bgr: np.ndarray, session=None) -> np.ndarray:
    """Run CTD's `seg` head on one BGR crop, capped to `CTD_MAX_SIDE` on the long side.

    Returns a float32 probability map the same (H, W) shape as the *input* ``crop_bgr``
    (upscaled back if the crop was downscaled for inference), values in ``[0, 1]``. Caller
    applies its own confidence threshold (``CTD_CONF_THRESHOLD``) and any region-set gating --
    this function makes no policy decision.
    """
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError(f"segment_crop expects an HxWx3 BGR crop, got shape {crop_bgr.shape}")
    if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
        raise ValueError("segment_crop received a degenerate (zero-area) crop")

    orig_h, orig_w = crop_bgr.shape[:2]
    scale = min(1.0, CTD_MAX_SIDE / max(orig_h, orig_w))
    if scale < 1.0:
        scaled_w, scaled_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        scaled_crop = cv2.resize(crop_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    else:
        scaled_crop = crop_bgr

    sess = session if session is not None else get_ctd_session()
    padded, padded_h, padded_w = _pad_to_multiple(scaled_crop, CTD_SIZE_MULTIPLE)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[None]  # NCHW

    started = time.perf_counter()
    outputs = sess.run(None, {"images": tensor})
    elapsed = time.perf_counter() - started
    fed_px = padded.shape[0] * padded.shape[1]
    # The cost of this graph is per fed pixel (~80us/px on a 2-core CPU, measured 2026-09-21), so
    # the fed shape is the number that explains the time; the crop shape says what the cap did.
    logger.info(
        f"[CTD] crop {orig_w}x{orig_h} -> fed {padded.shape[1]}x{padded.shape[0]} "
        f"({fed_px / 1e6:.2f} MP) in {elapsed:.1f}s ({elapsed / fed_px * 1e6:.0f} us/px)"
    )
    seg = outputs[0][0, 0]  # type: ignore
    if seg.min() < -0.01 or seg.max() > 1.01:
        seg = 1.0 / (1.0 + np.exp(-seg))

    seg = np.clip(seg[:padded_h, :padded_w], 0.0, 1.0).astype(np.float32)
    if scale < 1.0:
        seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return seg


def threshold_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean glyph mask from a CTD probability map at the given confidence threshold."""
    return prob >= threshold
