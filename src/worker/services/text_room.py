"""F5 (#244): how much room a text unit has beside it.

When several text units share one detector box, each one's balloon box is only a crop around its
own text (`get_split_polygon`), so the balloon's real width is never recorded and the English is set
into one source column. This measures the width from the page.

From just outside the text, a ray runs sideways along each of the text's middle rows to the first
ink. Flood fill and a rectangle grown from the text were both tried offline on B ja p10 and failed:
the fill ran 850 x 3600 through a whole page column, and the rectangle stopped at the text's own
edges (issue #244).

A low percentile of the rows decides, so one stray stroke doesn't. A ray reaching the page edge
found no outline, so the unit counts as unmeasured. The measurement is only recorded: masks,
balloon boxes and cleanup don't read it.
"""

import numpy as np

# Recorded with each measurement, so a region can be traced to the rule that measured it.
TEXT_ROOM_METHOD = "rays-v1"
# How far a pixel may be from the text's paper and still count as paper. Fitted on 92 pages.
PAPER_TOLERANCE = 28
# The share of rows allowed to run further than the width taken: high enough to ignore a stray
# stroke, low enough not to trust a gap in the outline.
RAY_PERCENTILE = 20
# Rays cover the middle of the text's height, where the sides of an oval balloon are widest apart
# without being pinched by its top and bottom. One in `ROW_SAMPLES` rows is enough.
ROW_SAMPLES = 40
# The gap kept from the text's own edge and from any ink, as a share of page width.
MARGIN_SHARE = 1 / 300
MIN_MARGIN_PX = 3
# A ray ending this close to the page edge reached it.
PAGE_EDGE_PX = 2


def _paper_level(crop):
    histogram, _ = np.histogram(crop, bins=32, range=(0, 256))
    return int(histogram.argmax()) * 8 + 4


def measure_text_room(gray, bbox, siblings=()):
    """The room beside a text unit, as `{"x", "width", "method"}`, or `None` if it can't be measured.

    `gray` is the page, `bbox` the text's `(x, y, w, h)`, and `siblings` the boxes of the other text
    units in the same detector box: each one takes its half of the gap between them. The height
    isn't measured. English gains from width, and the column already has its height.
    """
    x, y, w, h = bbox
    page_h, page_w = gray.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > page_w or y + h > page_h:
        return None

    is_paper = np.abs(gray.astype(np.int16) - _paper_level(gray[y : y + h, x : x + w])) <= PAPER_TOLERANCE
    margin = max(MIN_MARGIN_PX, round(page_w * MARGIN_SHARE))
    start_left = max(x - margin, 0)
    start_right = min(x + w + margin, page_w)

    left_runs, right_runs = [], []
    for row in range(y + h // 5, y + 4 * h // 5, max(1, h // ROW_SAMPLES)):
        ink = np.flatnonzero(~is_paper[row, :start_left])
        left_runs.append(start_left - (int(ink[-1]) + 1) if ink.size else start_left)
        ink = np.flatnonzero(~is_paper[row, start_right:])
        right_runs.append(int(ink[0]) if ink.size else page_w - start_right)
    if not left_runs:
        return None

    left = start_left - float(np.percentile(left_runs, RAY_PERCENTILE))
    right = start_right + float(np.percentile(right_runs, RAY_PERCENTILE))
    if left <= PAGE_EDGE_PX or right >= page_w - PAGE_EDGE_PX:
        return None

    left += margin
    right -= margin
    for other_x, other_y, other_w, other_h in siblings:
        if other_y >= y + h or other_y + other_h <= y:
            continue
        if other_x + other_w <= x:
            left = max(left, (other_x + other_w + x) / 2)
        elif other_x >= x + w:
            right = min(right, (x + w + other_x) / 2)

    left = int(min(np.ceil(left), x))
    right = int(max(np.floor(right), x + w))
    if left == x and right == x + w:
        return None
    return {"x": left, "width": right - left, "method": TEXT_ROOM_METHOD}


def text_room_provenance(provenance, gray, bbox, siblings):
    """`provenance` with the room beside this text unit added under `textRoom`, when it was measured."""
    room = measure_text_room(gray, bbox, siblings)
    return provenance | {"textRoom": room} if room else provenance
