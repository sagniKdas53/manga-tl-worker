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
