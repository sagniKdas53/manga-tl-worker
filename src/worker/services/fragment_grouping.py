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

from dataclasses import dataclass

# Fallback when no threshold is supplied. Kept here beside the code that uses it; the environment
# lookup itself stays in merge_ocr_regions so this module has no ambient inputs.
DEFAULT_THRESHOLD_RATIO = 0.50


@dataclass(frozen=True)
class GroupingConfig:
    """How fragments are grouped. Defaults reproduce pre-split behaviour exactly.

    Args:
        threshold_ratio: Proximity budget as a multiple of the estimated character size.
        reading_direction: 'rtl' or 'ltr'. Note this is a *binding* setting being used as a
            text-orientation flag -- see BUG-6 in docs/region_threshold_validation_2026-08-08.md.
            Deriving orientation from fragment geometry instead is a future field here.
    """

    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    reading_direction: str = "rtl"


def group_fragments(regions: list, config: GroupingConfig) -> list[list[int]]:
    """Partition fragment indices into groups that should become one region each.

    Returns a list of index lists. Group order and the order within each group match what the
    pre-split connected-components pass produced, so callers may rely on either.
    """
    n = len(regions)
    if n == 0:
        return []

    # Character size, estimated once over every fragment in the call. Note this is per-bubble at
    # handlers/ocr.py:605 (that call site loops over bubbles) but genuinely page-wide at :663.
    avg_height = sum(r["height"] for r in regions) / n
    avg_width = sum(r["width"] for r in regions) / n

    # For vertical Japanese text (typically reading_direction == "rtl"), the character/font size is
    # represented by the line's width, so the vertical gap threshold should be scaled relative to
    # avg_width rather than avg_height. For horizontal text (LTR), it is the line's height.
    char_size_vertical = avg_width if config.reading_direction == "rtl" else avg_height

    max_vertical_gap = char_size_vertical * config.threshold_ratio
    max_horizontal_gap = avg_width * config.threshold_ratio

    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(regions[i], regions[j], max_horizontal_gap, max_vertical_gap):
                adj[i].append(j)
                adj[j].append(i)

    return _connected_components(adj, n)


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
