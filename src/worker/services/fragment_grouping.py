"""Deciding which OCR line fragments belong to the same balloon.

Split out of `merge_regions.merge_ocr_regions`, which now does only dict assembly. The split is
deliberate: the *grouping decision* is under active investigation (see
`docs/region_waist_probe_2026-08-09.md`) while the assembly around it is settled. Keeping them
apart means an experiment can assert on `[[0], [1, 2], [3]]` instead of on union bounding boxes and
joined strings, and a configuration sweep does not pay for building dicts it discards.

**Every field of `GroupingConfig` defaults to the behaviour shipped before the split**, and
`test_fragment_grouping.py` holds a frozen copy of that behaviour and asserts equivalence over a
randomised corpus. Adding a phase means adding a field that is off by default; it cannot change
production until a call site opts in.
"""

from collections.abc import Callable
from dataclasses import dataclass

# Fallback when no threshold is supplied. Kept here beside the code that uses it; the environment
# lookup itself stays in merge_ocr_regions so this module has no ambient inputs.
DEFAULT_THRESHOLD_RATIO = 0.50

# Above this solidity a bubble is close enough to convex that its outline has no waist to measure,
# so the clearance veto is not applied. Measured over 15 annotated bubbles: below 0.90 the veto is
# exact (8/8), between 0.90 and 0.95 it is no better than distance, above 0.95 it is worse.
# See docs/region_waist_probe_2026-08-09.md.
DEFAULT_WAIST_MAX_SOLIDITY = 0.90


@dataclass(frozen=True)
class GroupingConfig:
    """How fragments are grouped. Defaults reproduce pre-split behaviour exactly.

    Args:
        threshold_ratio: Proximity budget as a multiple of the estimated character size.
        reading_direction: 'rtl' or 'ltr'. This is a *binding* setting (which way pages turn),
            and by default it is also read as a text-orientation flag, which is BUG-6.
        orientation: 'reading_direction' (the default, and the bug) or 'vote', which derives
            orientation from the fragments' own aspect ratios and falls back to the binding
            direction only when they carry no signal.
        waist_gate: Clearance veto, in characters. When set, two fragments inside a pinched
            balloon mask are kept apart if the path between them squeezes closer than this to the
            outline. `None` disables it, which is the shipped behaviour.
        waist_max_solidity: Only apply the veto to masks below this solidity. A convex mask has no
            waist, and forcing the measurement on one produces noise, not signal.
    """

    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    reading_direction: str = "rtl"
    orientation: str = "reading_direction"
    waist_gate: float | None = None
    waist_max_solidity: float = DEFAULT_WAIST_MAX_SOLIDITY


@dataclass(frozen=True)
class GroupingContext:
    """Per-call geometry that grouping can consult but does not compute.

    `clearance` is deliberately a callable rather than an image: it keeps this module free of
    numpy and OpenCV, keeps it unit-testable with a plain lambda, and lets the caller decide how
    the distance field is built and cached.

    Args:
        clearance: ((x, y), (x, y)) -> minimum distance to the mask outline along that segment,
            in pixels. Zero means the segment leaves the mask entirely.
        solidity: mask area / convex hull area. 1.0 means convex, i.e. no waist to find.
    """

    clearance: Callable[[tuple[float, float], tuple[float, float]], float] | None = None
    solidity: float = 1.0


def group_fragments(
    regions: list,
    config: GroupingConfig,
    context: GroupingContext | None = None,
) -> list[list[int]]:
    """Partition fragment indices into groups that should become one region each.

    Returns a list of index lists. Group order and the order within each group match what the
    pre-split connected-components pass produced, so callers may rely on either.
    """
    n = len(regions)
    if n == 0:
        return []

    veto = _waist_veto(config, context)

    # Character size, estimated once over every fragment in the call. Note this is per-bubble at
    # handlers/ocr.py:605 (that call site loops over bubbles) but genuinely page-wide at :663.
    avg_height = sum(r["height"] for r in regions) / n
    avg_width = sum(r["width"] for r in regions) / n

    # For vertical Japanese text the character/font size is the line's *width*, since a line is one
    # column, so the vertical gap budget scales from avg_width. For horizontal text it is the
    # line's height. Choosing the wrong axis inflates the budget by the length of a whole line --
    # measured on sample23, 214px instead of 33px.
    vertical = resolve_vertical(regions, config)
    char_size_vertical = avg_width if vertical else avg_height

    max_vertical_gap = char_size_vertical * config.threshold_ratio
    max_horizontal_gap = avg_width * config.threshold_ratio
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = regions[i], regions[j]
            if not _should_merge(r1, r2, max_horizontal_gap, max_vertical_gap):
                continue
            if veto is not None and veto(r1, r2, vertical):
                continue
            adj[i].append(j)
            adj[j].append(i)

    return _connected_components(adj, n)


# A box must be this much longer than it is tall (or vice versa) to vote on orientation. Below it
# the box is near-square -- a single CJK glyph or a short interjection -- and carries no evidence.
ORIENTATION_ASPECT = 1.2
# And the winning side must hold this share of the cast weight, otherwise the page is too mixed to
# call and the binding direction is the safer answer.
ORIENTATION_MAJORITY = 0.60


def resolve_vertical(regions: list, config: GroupingConfig) -> bool:
    """Is this group's text set vertically?

    `reading_direction` is a binding setting -- which way pages turn -- not a text-orientation one.
    They coincide for typical manga, so every Japanese job arrives as 'rtl' and every
    horizontally-set Japanese page (profile pages, narration blocks, author notes) gets vertical
    geometry applied to horizontal lines. That is BUG-6.

    The fragments already know the answer. Votes are weighted by the longer side, so one long
    narration line outweighs several square SFX glyphs rather than being outvoted by them.
    """
    binding_vertical = config.reading_direction != "ltr"
    if config.orientation != "vote":
        return binding_vertical

    vertical_weight = horizontal_weight = 0.0
    for r in regions:
        w, h = max(1, r["width"]), max(1, r["height"])
        weight = float(max(w, h))
        if h / w >= ORIENTATION_ASPECT:
            vertical_weight += weight
        elif w / h >= ORIENTATION_ASPECT:
            horizontal_weight += weight

    total = vertical_weight + horizontal_weight
    if total <= 0:
        return binding_vertical
    if vertical_weight / total >= ORIENTATION_MAJORITY:
        return True
    if horizontal_weight / total >= ORIENTATION_MAJORITY:
        return False
    return binding_vertical


def _waist_veto(config: GroupingConfig, context: GroupingContext | None):
    """Build the clearance veto, or None when it does not apply to this call.

    It is a veto and never a merger: it can only withhold an edge that distance already allowed.
    Where the mask is missing or convex it simply does not exist, so behaviour falls back to
    distance alone rather than to some other rule.
    """
    if config.waist_gate is None or context is None or context.clearance is None:
        return None
    if context.solidity >= config.waist_max_solidity:
        return None

    def veto(r1: dict, r2: dict, vertical: bool) -> bool:
        char_size = max(r1["width"], r2["width"]) if vertical else max(r1["height"], r2["height"])
        if char_size <= 0:
            return False
        p, q = _nearest_points(r1, r2)
        assert context.clearance is not None
        return context.clearance(p, q) < config.waist_gate * char_size

    return veto


def _nearest_points(r1: dict, r2: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """The closest pair of points on two axis-aligned boxes.

    On an axis where the boxes overlap, both points sit at the middle of that overlap, so the
    segment crosses the gap squarely instead of clipping a corner.
    """
    ax1, ay1, ax2, ay2 = r1["x"], r1["y"], r1["x"] + r1["width"], r1["y"] + r1["height"]
    bx1, by1, bx2, by2 = r2["x"], r2["y"], r2["x"] + r2["width"], r2["y"] + r2["height"]

    if ax2 < bx1:
        px, qx = float(ax2), float(bx1)
    elif bx2 < ax1:
        px, qx = float(ax1), float(bx2)
    else:
        px = qx = (max(ax1, bx1) + min(ax2, bx2)) / 2.0

    if ay2 < by1:
        py, qy = float(ay2), float(by1)
    elif by2 < ay1:
        py, qy = float(ay1), float(by2)
    else:
        py = qy = (max(ay1, by1) + min(ay2, by2)) / 2.0

    return (px, py), (qx, qy)


def _should_merge(r1: dict, r2: dict, max_horizontal_gap: float, max_vertical_gap: float) -> bool:
    """Whether two fragments are adjacent. Distance is currently the only criterion.

    Measured on hand-annotated pages, this signal separates same-balloon from cross-balloon pairs
    barely better than chance; the balloon mask's geometric clearance does far better. That is the
    next gate to land here -- as a veto, so a poor mask degrades toward today's behaviour.
    """
    r1_x2 = r1["x"] + r1["width"]
    r2_x2 = r2["x"] + r2["width"]
    x_overlap = max(0, min(r1_x2, r2_x2) - max(r1["x"], r2["x"]))
    x_dist = 0 if x_overlap > 0 else max(0, r2["x"] - r1_x2, r1["x"] - r2_x2)

    r1_y2 = r1["y"] + r1["height"]
    r2_y2 = r2["y"] + r2["height"]
    y_overlap = max(0, min(r1_y2, r2_y2) - max(r1["y"], r2["y"]))
    y_dist = 0 if y_overlap > 0 else max(0, r2["y"] - r1_y2, r1["y"] - r2_y2)

    # 1. overlap on both axes  2. horizontal overlap + vertical proximity
    # 3. vertical overlap + horizontal proximity  4. close diagonally
    return bool(
        (x_overlap > 0 and y_overlap > 0)
        or (x_overlap > 0 and y_dist <= max_vertical_gap)
        or (y_overlap > 0 and x_dist <= max_horizontal_gap)
        or (x_dist <= max_horizontal_gap and y_dist <= max_vertical_gap)
    )


def _connected_components(adj: dict, n: int) -> list[list[int]]:
    """BFS components, seeded in ascending index order.

    Adjacency is transitive and unbounded: A-B adjacent and B-C adjacent puts A, B and C in one
    group however far apart A and C are. Replacing this with a minimum spanning tree and an
    adaptive edge cut is the structural fix for chained groups on the unmatched path.
    """
    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        comp = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(comp)
    return components
