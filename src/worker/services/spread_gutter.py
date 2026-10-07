"""Find the gutter of a two-page spread, so text on one page is never grouped with the other's.

Text that no balloon holds is grouped by distance within one panel partition. On a spread whose
panels were not detected (ja/sample93: full-bleed art on both pages, two tiny panels found), both
pages share one partition, and a shout on the left page chained to speech on the right page into
one region that translated as one jumbled sentence.

A gutter here is a vertical edge near the centre that runs almost the full height of a wide page:
the dark binding line, or the hard change of art where two pages meet. A wide single illustration
seldom has one; when its text runs over the seam, the seam is art and the guard stays off.
"""

import cv2
import numpy as np

from worker.config import (
    OCR_SPREAD_GUTTER,
    OCR_SPREAD_MIN_ASPECT,
    OCR_SPREAD_SEAM_COVERAGE,
)

# Search this far either side of the centre, as a share of the width. Bindings are seldom exact:
# sample93's sits at 0.517.
_SEARCH_BAND = 0.08
# Edges are found on a copy at most this wide; a gutter does not need full resolution.
_WORK_WIDTH = 1200
# A row counts for the seam when its horizontal gradient here clears this (0-255 scale, Sobel 3x3).
_EDGE_STRENGTH = 60.0
# A fragment with at least this share of its width on each side lies across the seam: the text
# continues over it, so it is no page boundary. sample93's shout balloon, drawn over the binding,
# has 22 % on the far side.
_STRADDLE_SHARE = 0.35


def find_spread_gutter(img, fragments, *, enabled=OCR_SPREAD_GUTTER) -> float | None:
    """Return the gutter's x in ``img`` coordinates, or None when the page is not a spread.

    ``fragments`` are the OCR boxes (``x``, ``width``) that might be split; one lying across the
    seam turns the guard off.
    """
    if not enabled or img is None:
        return None
    height, width = img.shape[:2]
    if height == 0 or width / height < OCR_SPREAD_MIN_ASPECT:
        return None

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, _WORK_WIDTH / width)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gradient = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    strong = gradient > _EDGE_STRENGTH
    # Allow the seam a pixel of wobble either way: scans and resizes are rarely perfectly straight.
    widened = strong.copy()
    widened[:, 1:] |= strong[:, :-1]
    widened[:, :-1] |= strong[:, 1:]

    work_width = gray.shape[1]
    lo = int(work_width * (0.5 - _SEARCH_BAND))
    hi = int(work_width * (0.5 + _SEARCH_BAND)) + 1
    coverage = widened[:, lo:hi].mean(axis=0)
    best = int(np.argmax(coverage))
    if coverage[best] < OCR_SPREAD_SEAM_COVERAGE:
        return None
    gutter_x = (lo + best + 0.5) / scale

    for fragment in fragments:
        left = fragment["x"]
        right = left + fragment["width"]
        if fragment["width"] <= 0 or not left < gutter_x < right:
            continue
        smaller_side = min(gutter_x - left, right - gutter_x)
        if smaller_side / fragment["width"] >= _STRADDLE_SHARE:
            return None
    return gutter_x


def split_fragments_at_gutter(fragments, gutter_x):
    """Split ``fragments`` into the pages their centres fall on, keeping their order.

    Without a gutter the fragments stay one partition. An empty page is dropped.
    """
    if gutter_x is None:
        return [fragments]
    left = [f for f in fragments if f["x"] + f["width"] / 2 < gutter_x]
    right = [f for f in fragments if f["x"] + f["width"] / 2 >= gutter_x]
    return [side for side in (left, right) if side]
