"""Hand-marked cleanup: the mask editor's repaint (2026-09-28).

The user brushes over what should go -- a leftover dash, a blob a cleanup pass grew -- and this
repaints exactly that, on the page as it looks now: the source with the visible patches already
drawn over it. No CTD pass; the user's mark is the mask.

The mark is tidied first. Page 19 of the SpaceBunny chapter showed why: a few unpainted pixels
of ink inside a mask are read by AOT as art and grown into black blobs. Closing small gaps and
filling enclosed holes took that bubble from 15,900 dark pixels left inside the mask to 0.

Public entry points: `tidy_mask`, `composite_underlay`, `reconstruct_manual`.
"""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from worker.services.cleanup_reconstruct import (
    CleanupConfig,
    CleanupResult,
    _encode_mask_png,
    _encode_patch_png,
    _reconstruct_aot,
    _reconstruct_telea,
)
from worker.services.pixel_stats import pixel_spread

# "restore" is the editor's eraser over an automatic patch: the original page goes back exactly
# where the user marked, so it is neither tidied nor repainted, and it reads the bare source.
MANUAL_MODES = ("auto", "aot", "telea", "flat", "restore")
GENERATOR_ID = "manual-mask+telea-aotgan-flat/v1"
GENERATOR_SHA256 = hashlib.sha256(GENERATOR_ID.encode()).hexdigest()

# A 15px closing (radius 7) is what the page-19 measurement used; it bridges the gaps a brush
# leaves between strokes without swallowing a separate mark a hand-width away.
CLOSE_RADIUS_PX = 7


@dataclass(frozen=True)
class UnderlayPatch:
    """One visible patch as the export draws it: the patch image stretched to its box."""

    bgra: np.ndarray
    x: float
    y: float
    width: float
    height: float
    opacity: float = 1.0


def tidy_mask(mask: np.ndarray, close_radius_px: int = CLOSE_RADIUS_PX) -> np.ndarray:
    """Close small gaps in the mark, then fill every hole it encloses."""
    marked = mask.astype(np.uint8)
    if close_radius_px > 0:
        # Padded so a mark touching the frame edge closes like one in the middle.
        pad = close_radius_px + 1
        padded = np.pad(marked, pad)
        size = 2 * close_radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        marked = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[pad:-pad, pad:-pad]
    outside = np.pad((marked == 0).astype(np.uint8), 1, constant_values=1)
    flood_mask = np.zeros((outside.shape[0] + 2, outside.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(outside, flood_mask, (0, 0), (2,))
    holes = outside[1:-1, 1:-1] == 1
    return marked.astype(bool) | holes


def composite_underlay(image_bgr: np.ndarray, patches: list[UnderlayPatch]) -> np.ndarray:
    """The page as the export shows it: each patch stretched to its box and alpha-blended, in order."""
    page = image_bgr.astype(np.float32)
    page_h, page_w = page.shape[:2]
    for patch in patches:
        w, h = round(patch.width), round(patch.height)
        x, y = round(patch.x), round(patch.y)
        if w <= 0 or h <= 0 or patch.bgra.ndim != 3 or patch.bgra.shape[2] != 4:
            continue
        stretched = cv2.resize(patch.bgra, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(page_w, x + w), min(page_h, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        piece = stretched[y0 - y : y1 - y, x0 - x : x1 - x]
        alpha = (piece[..., 3:4] / 255.0) * max(0.0, min(1.0, patch.opacity))
        page[y0:y1, x0:x1] = piece[..., :3] * alpha + page[y0:y1, x0:x1] * (1.0 - alpha)
    return np.clip(page + 0.5, 0, 255).astype(np.uint8)


def parse_fill_colour(value: str | None) -> tuple[int, int, int] | None:
    """`#rrggbb` as a BGR triple, or None."""
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        return None
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return None
    return b, g, r


def reconstruct_manual(
    page_bgr: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    *,
    mode: str = "auto",
    fill_bgr: tuple[int, int, int] | None = None,
    config: CleanupConfig | None = None,
) -> CleanupResult:
    """Repaint the marked area of `page_bgr` (already composited). `mask` sits at page (x, y).

    The patch is cut to the tidied mask's box and is transparent outside it. For "restore",
    `page_bgr` is the bare source and the mark is copied as drawn. Raises ValueError on an empty
    or off-page mark, or a flat fill with no colour.
    """
    config = config if config is not None else CleanupConfig()
    if mode not in MANUAL_MODES:
        raise ValueError(f"unknown manual cleanup method {mode!r}")
    if mode == "flat" and fill_bgr is None:
        raise ValueError("a flat fill needs a #rrggbb colour")
    page_h, page_w = page_bgr.shape[:2]
    mark_h, mark_w = mask.shape[:2]

    # A context crop around the mark, so AOT and Telea see what surrounds the hole.
    cx0, cy0 = max(0, x - config.crop_pad_px), max(0, y - config.crop_pad_px)
    cx1 = min(page_w, x + mark_w + config.crop_pad_px)
    cy1 = min(page_h, y + mark_h + config.crop_pad_px)
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError("the mark lies outside the page")
    crop = page_bgr[cy0:cy1, cx0:cx1]
    crop_mask = np.zeros(crop.shape[:2], dtype=bool)
    mx0, my0 = max(x, cx0), max(y, cy0)
    mx1, my1 = min(x + mark_w, cx1), min(y + mark_h, cy1)
    if mx1 > mx0 and my1 > my0:
        crop_mask[my0 - cy0 : my1 - cy0, mx0 - cx0 : mx1 - cx0] = mask[my0 - y : my1 - y, mx0 - x : mx1 - x]
    if mode != "restore":
        crop_mask = tidy_mask(crop_mask)
    if not crop_mask.any():
        raise ValueError("the mark is empty")

    diagnostics: list[str] = []
    if mode == "restore":
        method = "restore"
        reconstructed = crop.copy()
    elif mode == "flat":
        method = "flat"
        reconstructed = crop.copy()
        reconstructed[crop_mask] = fill_bgr
    elif mode == "telea" or (mode == "auto" and pixel_spread(crop[crop_mask]) <= config.spread_threshold):
        method = "telea"
        reconstructed = _reconstruct_telea(crop, crop_mask)
    else:
        try:
            method = "aot"
            reconstructed = _reconstruct_aot(crop, crop_mask, config)
        except (FileNotFoundError, RuntimeError) as exc:
            method = "telea-fallback"
            diagnostics.append(f"AOT failed, fell back to TELEA: {exc}")
            reconstructed = _reconstruct_telea(crop, crop_mask)
    diagnostics.insert(0, f"manual repaint: {method} (mode={mode})")

    ys, xs = np.nonzero(crop_mask)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    cut_mask = crop_mask[by0:by1, bx0:bx1]
    return CleanupResult(
        mask_png=_encode_mask_png(cut_mask),
        patch_png=_encode_patch_png(reconstructed[by0:by1, bx0:bx1], cut_mask),
        bounds={"x": cx0 + bx0, "y": cy0 + by0, "width": bx1 - bx0, "height": by1 - by0},
        generator_sha256=GENERATOR_SHA256,
        diagnostics=diagnostics,
    )
