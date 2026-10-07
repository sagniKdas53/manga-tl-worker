"""Turning a rasterised balloon mask into the geometry `fragment_grouping` can consult.

This is the only place OpenCV and the grouping decision meet. `fragment_grouping` deliberately
takes a *callable* for clearance rather than an image, so it stays free of numpy and cv2 and can
be unit-tested with a lambda; this module is the production implementation of that callable.

The signal is a balloon's **waist**. Where two balloons touch inside one YOLO blob there is a
geometric constriction: the distance from the join to the outline collapses to roughly the stroke
width, while anywhere inside a single balloon every point between two fragments is at least half a
character from the edge, because balloons have margins. Text distance cannot separate those two
cases -- the gap distributions overlap -- and this can.
"""

import os
from collections.abc import Sequence
from typing import Any, overload

import cv2
import numpy as np

from worker.services.fragment_grouping import GroupingContext

# How many points along the segment between two fragments to sample the distance field at. The
# minimum over the samples is the waist; 32 is dense enough that a one-stroke constriction is not
# stepped over on a 1600px page, and the sampling is pure array indexing.
CLEARANCE_SAMPLES = 32


def mask_solidity(mask_polygon):
    """Polygon area / convex hull area. 1.0 means convex, i.e. there is no waist to find.

    This is the veto's applicability test rather than a tuning knob: a blob of two touching
    balloons is non-convex *by construction*, and a single balloon is nearly convex.
    """
    if not mask_polygon or len(mask_polygon) < 3:
        return 1.0
    pts = np.array(mask_polygon, dtype=np.int32)
    hull = cv2.contourArea(cv2.convexHull(pts))
    if hull <= 0:
        return 1.0
    return float(cv2.contourArea(pts) / hull)


def _sample_min(dt, p, q, samples=CLEARANCE_SAMPLES):
    """Smallest distance-to-outline along the segment p->q. Zero means it leaves the mask."""
    h, w = dt.shape[:2]
    xs = np.linspace(p[0], q[0], samples)
    ys = np.linspace(p[1], q[1], samples)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return float(dt[yi, xi].min())


def bubble_grouping_context(mask, mask_polygon):
    """A GroupingContext for one bubble, or None when the bubble has no usable mask.

    The distance transform is computed once here and closed over, so a bubble pays for it once
    however many fragment pairs are tested.
    """
    if mask is None or mask_polygon is None:
        return None

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    def clearance(p, q):
        return _sample_min(dt, p, q)

    return GroupingContext(clearance=clearance, solidity=mask_solidity(mask_polygon))


# How far a vertex may sit from the outline its neighbours describe before it is worth keeping,
# in page pixels. Absolute, not a fraction of the perimeter -- see `simplify_mask_polygon`.
MASK_POLYGON_TOLERANCE_PX = float(os.environ.get("MASK_POLYGON_TOLERANCE_PX", "2.0"))

# A polygon is never simplified below this. Three points is the minimum that still encloses area,
# and dropping to it would turn a balloon into a triangle if the tolerance were ever set absurdly.
MIN_MASK_POLYGON_POINTS = 4


# A pixel is ink when it is this many grey levels darker than the balloon's paper (its median).
WALL_INK_CONTRAST = 60


def gap_wall(gray, mask, min_stroke):
    """B3b's wall test for one balloon: ``(group a, group b, regions) -> bool``.

    True when one connected ink stroke inside the gap between the two groups reaches ``min_stroke``
    characters along the line. The gap is the space between the groups across the reading
    direction, over the stretch of the line both groups cover, inside the balloon mask. A balloon
    outline drawn between two speakers runs that whole stretch, straight or sloped (ja/sample24's
    fused bracket balloons: 5.2 characters); the glyphs of a column the OCR missed are separate
    strokes, one per character. The paper is measured over the whole balloon, so an outline that
    fills a narrow gap is still darker than it. ``gray`` and ``mask`` are page-sized; ``mask`` is
    non-zero inside the balloon.
    """
    height, width = gray.shape[:2]
    inside_balloon = mask > 0
    paper = float(np.median(gray[inside_balloon])) if inside_balloon.any() else 255.0

    def wall(group_a, group_b, regions):
        members = [regions[index] for index in (*group_a, *group_b)]
        vertical = sum(r["height"] > r["width"] for r in members) >= sum(r["width"] > r["height"] for r in members)
        char = float(np.median([r["width"] if vertical else r["height"] for r in members]))
        if char <= 0:
            return False

        def box(group):
            rs = [regions[index] for index in group]
            return (
                min(r["x"] for r in rs),
                min(r["y"] for r in rs),
                max(r["x"] + r["width"] for r in rs),
                max(r["y"] + r["height"] for r in rs),
            )

        a, b = box(group_a), box(group_b)
        across = 0 if vertical else 1  # columns stack along x, lines along y
        low, high = sorted((a, b), key=lambda r: r[across])
        gap_start, gap_end = int(low[across + 2]), int(high[across])
        along_start = int(max(a[1 - across], b[1 - across]))
        along_end = int(min(a[3 - across], b[3 - across]))
        if gap_end - gap_start < 2 or along_end - along_start < char:
            return False
        if vertical:
            rows, cols = (
                slice(max(0, along_start), min(height, along_end)),
                slice(max(0, gap_start), min(width, gap_end)),
            )
            strip, inside = gray[rows, cols], inside_balloon[rows, cols]
        else:
            rows, cols = (
                slice(max(0, gap_start), min(height, gap_end)),
                slice(max(0, along_start), min(width, along_end)),
            )
            strip, inside = gray[rows, cols].T, inside_balloon[rows, cols].T
        if strip.size == 0 or not inside.any():
            return False
        ink = ((strip.astype(np.int16) < paper - WALL_INK_CONTRAST) & inside).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if count < 2:
            return False
        # Rows of the strip run along the line, so a stroke's height is its length along the line.
        return float(stats[1:, cv2.CC_STAT_HEIGHT].max()) >= min_stroke * char

    return wall


@overload
def simplify_mask_polygon(points: None, tolerance_px: float | None = ...) -> None: ...


@overload
def simplify_mask_polygon(points: Sequence[Any] | np.ndarray, tolerance_px: float | None = ...) -> list: ...


def simplify_mask_polygon(points, tolerance_px=None):
    """Drop vertices that do not move the outline by more than `tolerance_px`.

    AUDIT-R7. Every contour in this pipeline used to be simplified with
    ``epsilon = 0.002 * cv2.arcLength(contour, True)`` -- a tolerance proportional to the
    perimeter. That is backwards. It gives a 3000px-perimeter balloon a 6px tolerance, which is
    fine, and a 200px-perimeter caption plate a **0.4px** tolerance, which is below one pixel and
    therefore removes nothing at all. The smaller and simpler the shape, the tighter the tolerance
    it was held to, so the shapes that should have come back as four points came back with dozens.

    Every one of those is a drag handle in reshape mode, is stored in `mask_polygon`, is
    re-serialised on every save, and is walked by `mask_solidity` and the merge hull.

    An absolute tolerance inverts that: 2px flattens rasterisation jitter along a straight edge at
    any size, while a balloon's tail -- which sticks out far more than 2px, that being the point of
    a tail -- survives untouched at every size.

    Returns a plain ``[[x, y], ...]`` list. Input may be that or a cv2 contour.
    """
    if points is None:
        return points
    tolerance = MASK_POLYGON_TOLERANCE_PX if tolerance_px is None else tolerance_px
    try:
        contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    except (ValueError, TypeError):
        return points
    if contour.shape[0] <= MIN_MASK_POLYGON_POINTS or tolerance <= 0:
        return np.rint(contour.reshape(-1, 2)).astype(int).tolist()

    simplified = cv2.approxPolyDP(contour, float(tolerance), True)
    # approxPolyDP can over-collapse a nearly-degenerate shape; keep the original rather than hand
    # back something that no longer encloses anything.
    if simplified.shape[0] < MIN_MASK_POLYGON_POINTS:
        simplified = contour
    return np.rint(simplified.reshape(-1, 2)).astype(int).tolist()
