"""Glyph-mask cleanup orchestration -- R3.

Turns a CTD probability map into a `cleanup_artifact`'s two assets: a glyph-shaped alpha mask
and a reconstructed RGBA patch, routed to TELEA or an AOT GAN inpainter by the same
`pixel_spread` statistic `handlers.ocr` already uses to decide whether a background is flat
enough to fill (docs/archive/erasure_overhaul_plan_2026-08-26.md §8). Three recall-recovery
tiers guard the known residual-ink risk (CTD+TELEA alone leaves visible source strokes on
under-detected pages, docs/archive/ctd_mask_validation_2026-08-26.md Finding 7): the 0.3
detection threshold, a small bounded dilation of the gated mask, and a runtime residual-ink
re-check that falls back to `None` -- i.e. R2's current flat-plate / no-plate behaviour for
that region stands, never a worse result than today.

Public entry point: `reconstruct_region`. Everything else here is a private helper.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from worker.config import AOT_MODEL_PATH, AOT_PINNED_CHECKSUM, BACKGROUND_FILL_MAX_SPREAD, CTD_CONF_THRESHOLD
from worker.services.glyph_mask import segment_crop, threshold_mask
from worker.services.pixel_stats import pixel_spread

logger = logging.getLogger(__name__)

GENERATOR_ID = "ctd-seg+telea-aotgan-cleanup/v1"
GENERATOR_SHA256 = hashlib.sha256(GENERATOR_ID.encode()).hexdigest()

_aot_session = None


@dataclass(frozen=True)
class CleanupConfig:
    ctd_threshold: float = CTD_CONF_THRESHOLD
    crop_pad_px: int = 64
    # 3px left visible anti-aliased stroke fringes on dense small-glyph text -- measured on the
    # glyph-erasure-candidates corpus (docs/quality-runs/glyph-erasure-candidates/results/
    # 3_r0_dilation_sweep.png): +2px alone took a 10.6%-coverage region from 28.65dB/25.43dB
    # (AOT/LaMa-mpe, visible residual strokes) to 32.75dB/33.99dB (clean, matches Torii's own
    # plate); +6px more gained under 1dB further. 5 is the new floor, not the ceiling -- the
    # residual-ink recheck below still discards anything that isn't actually clean.
    mask_dilate_px: int = 5
    # Reuse the existing flat-vs-structured statistic and threshold rather than a second knob
    # (docs/archive/mask_precision_2026-08-27.md §4, archive erasure plan §8): flat interior ->
    # TELEA, structured/artwork interior -> AOT.
    spread_threshold: float = BACKGROUND_FILL_MAX_SPREAD
    # Starting bound from the 21-page validation (median 6.1%, 9/21 over 10%, 4/21 over 27%);
    # confirm/adjust against that same set before trusting it in production (R3b test plan).
    residual_ink_max_pct: float = 15.0
    aot_max_side: int = 1024


@dataclass(frozen=True)
class CleanupResult:
    mask_png: bytes
    patch_png: bytes
    bounds: dict = field(default_factory=dict)  # {x, y, width, height} in source pixel coords
    generator_sha256: str = GENERATOR_SHA256
    diagnostics: list[str] = field(default_factory=list)


_DEFAULT_CONFIG = CleanupConfig()


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        while chunk := model_file.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def get_aot_session():
    """Lazily load and cache the AOT GAN inpainting ONNX session (CPU execution provider)."""
    global _aot_session
    if _aot_session is not None:
        return _aot_session

    if not AOT_MODEL_PATH or not os.path.exists(AOT_MODEL_PATH):
        raise FileNotFoundError(
            f"Required AOT inpainting model is not available at path: {AOT_MODEL_PATH}. "
            "Cannot proceed in offline mode without the required model."
        )

    current_checksum = _sha256(AOT_MODEL_PATH)
    if current_checksum != AOT_PINNED_CHECKSUM:
        logger.warning(f"[AOT] Pinned checksum mismatch! Expected: {AOT_PINNED_CHECKSUM}, got: {current_checksum}")
    else:
        logger.info("[AOT] Model checksum matches pinned checksum.")

    try:
        import onnxruntime as ort

        logger.info(f"[AOT] Loading ONNX model from {AOT_MODEL_PATH} via ONNX Runtime...")
        _aot_session = ort.InferenceSession(AOT_MODEL_PATH, providers=["CPUExecutionProvider"])
        logger.info("[AOT] ONNX Runtime session initialized successfully.")
        return _aot_session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model via ONNX Runtime: {e}") from e


def _crop_with_context(
    img: np.ndarray, x: float, y: float, width: float, height: float, pad_px: int
) -> tuple[np.ndarray | None, int, int]:
    """Native-scale crop of the region plus `pad_px` of context, clamped to the image.

    Never square-padded and never resized -- both were measured to cost extra CTD runtime for
    no accuracy gain (docs/archive/ctd_mask_validation_2026-08-26.md Findings 3 and 6).
    """
    img_h, img_w = img.shape[:2]
    x0 = max(0, int(x) - pad_px)
    y0 = max(0, int(y) - pad_px)
    x1 = min(img_w, int(x + width) + pad_px)
    y1 = min(img_h, int(y + height) + pad_px)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    return img[y0:y1, x0:x1], x0, y0


def _gate_to_region_footprint(
    mask: np.ndarray, crop_x0: int, crop_y0: int, x: float, y: float, width: float, height: float, margin_px: int = 4
) -> np.ndarray:
    """Zero out anything CTD found outside this region's own bbox (plus a small margin).

    The crop carries context padding so CTD has surrounding pixels to work with, but glyphs it
    finds in that context belong to a *different* region -- this is the "gated by the region
    set" step from the validation doc, applied per-region rather than page-wide.
    """
    gated = np.zeros_like(mask)
    rx0 = max(0, int(x) - crop_x0 - margin_px)
    ry0 = max(0, int(y) - crop_y0 - margin_px)
    rx1 = min(mask.shape[1], int(x + width) - crop_x0 + margin_px)
    ry1 = min(mask.shape[0], int(y + height) - crop_y0 + margin_px)
    if rx1 > rx0 and ry1 > ry0:
        gated[ry0:ry1, rx0:rx1] = mask[ry0:ry1, rx0:rx1]
    return gated


def _dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_bool
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)


def _reconstruct_telea(crop_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8) * 255
    return cv2.inpaint(crop_bgr, mask_u8, 5, cv2.INPAINT_TELEA)


def _reconstruct_aot(crop_bgr: np.ndarray, mask_bool: np.ndarray, config: CleanupConfig) -> np.ndarray:
    """AOT GAN inpaint, following the validated contract measured against Torii's own plate
    (docs/archive/erasure_overhaul_plan_2026-08-26.md §8, corpus/gaps/manga-tl-erasure-eval/
    scripts/norm.py): resize so the long side is `aot_max_side` preserving aspect (never
    stretched), pad up to a multiple of 32 (a multiple of 4 is the model's own internal
    minimum -- probed directly -- 32 leaves comfortable margin and matches the measured
    recipe), normalize RGB to [-1, 1], run, denormalize, composite back onto the crop only
    inside the mask, resize to the crop's native size.
    """
    session = get_aot_session()
    h, w = crop_bgr.shape[:2]
    # Only ever downscale -- a crop already smaller than aot_max_side must not be blown up
    # (upstream MIT only resizes when the long side exceeds inpainting_size, never upsamples;
    # docs/archive/erasure_overhaul_plan_2026-08-26.md line 175-176). Missing this clamp meant
    # every production crop under 1024px on its long side was being upscaled toward 1024 with
    # cv2.INTER_AREA (which degrades toward nearest-neighbour when zooming in), inpainted at
    # that inflated size, then downscaled back -- paying up to ~8x the compute for a softer,
    # less detailed result than running at native resolution.
    scale = min(1.0, config.aot_max_side / max(h, w))
    scaled_w, scaled_h = max(1, round(w * scale)), max(1, round(h * scale))
    scaled_img = cv2.resize(crop_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(mask_bool.astype(np.uint8) * 255, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    padded_h = ((scaled_h + 31) // 32) * 32
    padded_w = ((scaled_w + 31) // 32) * 32
    padded_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
    padded_img[:scaled_h, :scaled_w] = scaled_img
    padded_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
    padded_mask[:scaled_h, :scaled_w] = scaled_mask

    rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_tensor = (rgb / 127.5 - 1.0).transpose(2, 0, 1)[None]
    mask_tensor = ((padded_mask.astype(np.float32) / 255.0) >= 0.5).astype(np.float32)[None, None]
    # Zeroing the hole is what matters (measured: 35.44dB with, 18.34dB without, at the same
    # [-1,1] normalization) -- the model is trained to fill a blanked hole guided by the mask
    # channel, not to edit pixels that are still present.
    image_tensor = image_tensor * (1.0 - mask_tensor)

    outputs = session.run(None, {"image": image_tensor, "mask": mask_tensor})
    out = outputs[0][0].transpose(1, 2, 0)  # type: ignore
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[:scaled_h, :scaled_w]
    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    composite = crop_bgr.copy()
    composite[mask_bool] = out_bgr[mask_bool]
    return composite


def _residual_ink_pct(reconstructed_bgr: np.ndarray, mask_bool: np.ndarray, threshold: float, session=None) -> float:
    """Re-run CTD on the reconstructed pixels; % of the masked footprint still above threshold.

    Same definition as the offline 21-page measurement (docs/archive/ctd_mask_validation_2026-
    08-26.md Finding 7) so this runtime gate and that median-6.1%/9-of-21-over-10% baseline are
    the same metric, not two things that happen to share a name.
    """
    if not mask_bool.any():
        return 0.0
    prob = segment_crop(reconstructed_bgr, session=session)
    residual = threshold_mask(prob, threshold) & mask_bool
    return 100.0 * float(residual.sum()) / float(mask_bool.sum())


def _encode_mask_png(mask_bool: np.ndarray) -> bytes:
    """White-on-transparent coverage mask, for provenance/audit -- see `cleanup_reconstruct`'s
    module docstring on why the visible cleanup effect comes from the patch's own alpha, not
    from this asset being dereferenced by any renderer."""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_bool] = (255, 255, 255, 255)
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("failed to encode glyph mask PNG")
    return buf.tobytes()


def _encode_patch_png(reconstructed_bgr: np.ndarray, mask_bool: np.ndarray) -> bytes:
    """Glyph-shaped RGBA patch: reconstructed colour where the mask hits, transparent (and
    zeroed, not just alpha-0) everywhere else -- same structural contract as the legacy flat
    fill (`page_scene_builder.rs`'s `legacy_patch_and_mask`), richer content."""
    h, w = reconstructed_bgr.shape[:2]
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[mask_bool, :3] = reconstructed_bgr[mask_bool]
    bgra[mask_bool, 3] = 255
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("failed to encode cleanup patch PNG")
    return buf.tobytes()


def reconstruct_region(
    img: np.ndarray | None,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    config: CleanupConfig | None = None,
    ctd_session=None,
) -> CleanupResult | None:
    """Erase and reconstruct one region's glyphs.

    Returns `None` -- R2's current flat-plate / no-plate-plus-halo behaviour for this region
    stands untouched, never a worse result than today -- on a degenerate crop, a CTD/AOT model
    failure, an empty gated mask, or the residual-ink recall-recovery check firing.
    """
    config = config if config is not None else _DEFAULT_CONFIG
    if img is None or width <= 0 or height <= 0:
        return None

    crop, crop_x0, crop_y0 = _crop_with_context(img, x, y, width, height, config.crop_pad_px)
    if crop is None or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None

    try:
        prob = segment_crop(crop, session=ctd_session)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.warning(f"[Cleanup] CTD segmentation failed, R2 behaviour stands: {exc}")
        return None

    raw_mask = threshold_mask(prob, config.ctd_threshold)
    gated_mask = _gate_to_region_footprint(raw_mask, crop_x0, crop_y0, x, y, width, height)
    if not gated_mask.any():
        return None

    dilated_mask = _dilate(gated_mask, config.mask_dilate_px)

    diagnostics: list[str] = []
    interior = crop[dilated_mask]
    spread = pixel_spread(interior) if len(interior) else 0.0
    if spread <= config.spread_threshold:
        method = "telea"
        reconstructed = _reconstruct_telea(crop, dilated_mask)
    else:
        try:
            method = "aot"
            reconstructed = _reconstruct_aot(crop, dilated_mask, config)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(f"[Cleanup] AOT reconstruction failed, falling back to TELEA: {exc}")
            method = "telea-fallback"
            reconstructed = _reconstruct_telea(crop, dilated_mask)
    diagnostics.append(f"reconstruction method: {method} (pixel_spread={spread:.1f})")

    residual_pct = _residual_ink_pct(reconstructed, dilated_mask, config.ctd_threshold, session=ctd_session)
    diagnostics.append(f"residual ink: {residual_pct:.1f}%")
    if residual_pct > config.residual_ink_max_pct:
        logger.info(
            f"[Cleanup] residual ink {residual_pct:.1f}% exceeds {config.residual_ink_max_pct}% "
            "bound; R2 behaviour stands for this region"
        )
        return None

    crop_h, crop_w = crop.shape[:2]
    return CleanupResult(
        mask_png=_encode_mask_png(dilated_mask),
        patch_png=_encode_patch_png(reconstructed, dilated_mask),
        bounds={"x": crop_x0, "y": crop_y0, "width": crop_w, "height": crop_h},
        diagnostics=diagnostics,
    )
