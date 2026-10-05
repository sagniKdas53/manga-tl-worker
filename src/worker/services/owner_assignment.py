"""Conservative OCR fragment owner decisions.

An OCR grouping component is only a candidate.  It becomes a text owner when the captured
source-space evidence proves one bounded text unit; otherwise the decision remains explicitly
unknown.  Panels, conversations, bounding-box overlap, and proximity are deliberately not inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, Literal

OwnerState = Literal["assigned", "unknown"]

_MIN_ORIENTATION_ASPECT = 1.2
_MAX_ANGLE_DELTA_DEGREES = 15.0
_MIN_LATERAL_OVERLAP = 0.25
_MAX_LINE_GAP_MULTIPLIER = 2.0
# AUDIT-R20: a balloon is an ellipse and a text column is a rectangle, so the outermost column
# (ja/zh) or the top and bottom line (ko) of any multi-line balloon has corners past the curve.
# Requiring all four corners inside split every such balloon into one region per column on the
# R2 short list. Measured over the 42 fragments the three pages put in a balloon, every one that
# belongs is >= 0.875 inside its container; the only lower value (0.62) is a rotated quad that
# fails line continuity anyway. Three quarters keeps a margin under the real minimum and still
# rejects a quad that is half outside.
_MIN_QUAD_INSIDE_FRACTION = 0.75
_QUAD_SAMPLE_GRID = 16


@dataclass(frozen=True)
class OwnerDecision:
    """One conservative decision over an already-captured grouping candidate."""

    fragment_ids: tuple[str, ...]
    owner_id: str | None
    state: OwnerState
    reason: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _FragmentEvidence:
    fragment_id: str
    quad: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    style: str | None


def assign_captured_owners(
    *,
    fragment_ids: Sequence[str],
    raw_quads: Sequence[Any],
    recognition: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Sequence[int]],
    detector_masks: Sequence[Any],
    scale_transform: Mapping[str, Any],
    join_split_lines: bool = False,
) -> list[OwnerDecision]:
    """Assign only evidence-backed text owners for captured OCR grouping candidates.

    A multi-fragment owner requires one validated detector container (holding at least
    `_MIN_QUAD_INSIDE_FRACTION` of every member's quad), coherent oriented-line geometry, and a
    finite OCR-to-source scale. Available declared source style can veto
    contradictory fragments; absent style remains an explicit unknown feature, not a split.

    `join_split_lines` (AUDIT-R21, off by default) joins the pieces OCR broke one line into
    before the line-continuity checks, so a column read as ブラ | イダルなんて is one line rather
    than two that each fail to overlap their neighbour.
    """
    count = len(fragment_ids)
    if len(raw_quads) != count or len(recognition) != count or len(regions) != count:
        raise ValueError("fragment IDs, quads, recognition, and regions must have equal lengths")
    _validate_candidate_partition(candidate_groups, count)

    source_scale = _source_scale(scale_transform)
    containers = _validated_containers(detector_masks)
    fragments = [
        _fragment_evidence(fragment_ids[index], raw_quads[index], recognition[index], regions[index])
        for index in range(count)
    ]

    decisions: list[OwnerDecision] = []
    for group_index, group in enumerate(candidate_groups):
        members = [fragments[index] for index in group]
        fragment_group_ids = tuple(fragment_ids[index] for index in group)
        diagnostics: dict[str, Any] = {
            "candidate_group_index": group_index,
            "source_scale": source_scale,
            "validated_container_count": len(containers),
            "member_count": len(members),
        }

        if len(members) == 1:
            decisions.append(
                OwnerDecision(
                    fragment_ids=fragment_group_ids,
                    owner_id=_owner_id(fragment_group_ids),
                    state="assigned",
                    reason="single-fragment-owner",
                    diagnostics=diagnostics,
                )
            )
            continue

        if source_scale is None:
            decisions.append(_unknown(fragment_group_ids, "invalid-source-scale", diagnostics))
            continue

        if any(member is None for member in members):
            decisions.append(_unknown(fragment_group_ids, "invalid-oriented-quad", diagnostics))
            continue

        evidence = [member for member in members if member is not None]
        containment = [_containing_container(member.quad, containers) for member in evidence]
        container_ids = [identifier for identifier, _ in containment]
        diagnostics["container_ids"] = container_ids
        diagnostics["container_coverage"] = [round(fraction, 3) for _, fraction in containment]
        known_container_ids = {identifier for identifier in container_ids if identifier is not None}
        if len(known_container_ids) > 1:
            decisions.append(_unknown(fragment_group_ids, "different-validated-containers", diagnostics))
            continue

        styles = [member.style for member in evidence]
        diagnostics["source_styles"] = styles
        declared_styles = {style for style in styles if style is not None}
        diagnostics["style_evidence"] = (
            "matching-declared"
            if len(declared_styles) == 1 and all(style is not None for style in styles)
            else "unknown"
        )
        if len(declared_styles) > 1:
            decisions.append(_unknown(fragment_group_ids, "different-source-styles", diagnostics))
            continue

        continuity = _line_continuity(evidence, source_scale, join_split_lines=join_split_lines)
        diagnostics["line_continuity"] = continuity
        if not continuity["continuous"]:
            decisions.append(_unknown(fragment_group_ids, str(continuity["reason"]), diagnostics))
            continue

        known_container_ids = {identifier for identifier in container_ids if identifier is not None}
        if not known_container_ids:
            decisions.append(_unknown(fragment_group_ids, "missing-validated-container", diagnostics))
            continue

        # YOLO masks commonly stop at a balloon's ink edge, while PaddleOCR's oriented quad
        # covers an adjacent glyph/anti-aliased edge. Allow exactly one such neighbour to inherit
        # the one proven container only after the candidate grouping has supplied proximity and
        # reading-order continuity. Never bridge two detector containers or a larger component.
        if None in container_ids:
            if len(members) != 2 or container_ids.count(None) != 1:
                decisions.append(_unknown(fragment_group_ids, "incomplete-validated-container", diagnostics))
                continue
            diagnostics["geometry_attached_container"] = next(iter(known_container_ids))
            reason = "geometry-attached-continuous-lines"
        else:
            reason = "validated-container-continuous-lines"

        decisions.append(
            OwnerDecision(
                fragment_ids=fragment_group_ids,
                owner_id=_owner_id(fragment_group_ids),
                state="assigned",
                reason=reason,
                diagnostics=diagnostics,
            )
        )
    return decisions


def _unknown(fragment_ids: tuple[str, ...], reason: str, diagnostics: dict[str, Any]) -> OwnerDecision:
    return OwnerDecision(
        fragment_ids=fragment_ids,
        owner_id=None,
        state="unknown",
        reason=reason,
        diagnostics=diagnostics,
    )


def _validate_candidate_partition(groups: Sequence[Sequence[int]], count: int) -> None:
    indices = [index for group in groups for index in group]
    if sorted(indices) != list(range(count)):
        raise ValueError("candidate_groups must partition every fragment exactly once")


def _source_scale(scale_transform: Mapping[str, Any]) -> float | None:
    transform = scale_transform.get("ocr_to_source")
    if not isinstance(transform, Mapping):
        return None
    scale_x = transform.get("scale_x")
    scale_y = transform.get("scale_y")
    if (
        not isinstance(scale_x, int | float)
        or not isinstance(scale_y, int | float)
        or not math.isfinite(scale_x)
        or not math.isfinite(scale_y)
        or scale_x <= 0
        or scale_y <= 0
    ):
        return None
    return (float(scale_x) + float(scale_y)) / 2


def _fragment_evidence(
    fragment_id: str,
    raw_quad: Any,
    recognition: Mapping[str, Any],
    region: Mapping[str, Any],
) -> _FragmentEvidence | None:
    quad = _normalise_quad(raw_quad)
    if quad is None:
        return None
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    style = _style_signature(recognition, region)
    return _FragmentEvidence(fragment_id, quad, (min(xs), min(ys), max(xs), max(ys)), style)


def _normalise_quad(raw_quad: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(raw_quad, Sequence) or isinstance(raw_quad, str | bytes) or len(raw_quad) != 4:
        return None
    points: list[tuple[float, float]] = []
    for raw_point in raw_quad:
        if not isinstance(raw_point, Sequence) or isinstance(raw_point, str | bytes) or len(raw_point) != 2:
            return None
        x, y = raw_point
        if (
            not isinstance(x, int | float)
            or not isinstance(y, int | float)
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            return None
        points.append((float(x), float(y)))
    return tuple(points)


def _style_signature(recognition: Mapping[str, Any], region: Mapping[str, Any]) -> str | None:
    for record in (region, recognition):
        for field in ("sourceStyle", "source_style", "style"):
            value = record.get(field)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return None


def _validated_containers(detector_masks: Sequence[Any]) -> list[tuple[str, tuple[tuple[float, float], ...]]]:
    containers: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for index, mask in enumerate(detector_masks):
        if not isinstance(mask, Mapping) or mask.get("format") != "polygon":
            continue
        points = mask.get("points")
        if not isinstance(points, Sequence) or isinstance(points, str | bytes) or len(points) < 3:
            continue
        polygon: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, Sequence) or isinstance(point, str | bytes) or len(point) != 2:
                polygon = []
                break
            x, y = point
            if (
                not isinstance(x, int | float)
                or not isinstance(y, int | float)
                or not math.isfinite(x)
                or not math.isfinite(y)
            ):
                polygon = []
                break
            polygon.append((float(x), float(y)))
        if len(polygon) >= 3 and abs(_signed_area(polygon)) > 0:
            identifier = mask.get("id")
            containers.append(
                (
                    identifier if isinstance(identifier, str) and identifier else f"detector-container-{index}",
                    tuple(polygon),
                )
            )
    return containers


def _containing_container(
    quad: tuple[tuple[float, float], ...], containers: Sequence[tuple[str, tuple[tuple[float, float], ...]]]
) -> tuple[str | None, float]:
    """The one container holding at least `_MIN_QUAD_INSIDE_FRACTION` of the quad, and the best fraction seen.

    Two containers claiming the same quad is an ambiguity, not a match.
    """
    best = 0.0
    matches: list[str] = []
    for identifier, polygon in containers:
        fraction = _quad_inside_fraction(quad, polygon)
        best = max(best, fraction)
        if fraction >= _MIN_QUAD_INSIDE_FRACTION:
            matches.append(identifier)
    return (matches[0] if len(matches) == 1 else None), best


def _quad_inside_fraction(quad: Sequence[tuple[float, float]], polygon: Sequence[tuple[float, float]]) -> float:
    """Fraction of the quad's area inside the polygon, from a bilinear grid over the quad.

    The polygon is a detector mask and can be concave, so this is sampled rather than clipped;
    a 16x16 grid resolves 0.4 % of the quad, well under the floor it is compared with.
    """
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = quad
    inside = 0
    grid = _QUAD_SAMPLE_GRID
    for row in range(grid):
        v = (row + 0.5) / grid
        for column in range(grid):
            u = (column + 0.5) / grid
            x = (1 - v) * ((1 - u) * ax + u * bx) + v * ((1 - u) * dx + u * cx)
            y = (1 - v) * ((1 - u) * ay + u * by) + v * ((1 - u) * dy + u * cy)
            if _point_in_polygon((x, y), polygon):
                inside += 1
    return inside / (grid * grid)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for left, right in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        if _point_on_segment(point, left, right):
            return True
        left_x, left_y = left
        right_x, right_y = right
        if (left_y > y) != (right_y > y):
            crossing_x = (right_x - left_x) * (y - left_y) / (right_y - left_y) + left_x
            if x < crossing_x:
                inside = not inside
    return inside


def _point_on_segment(point: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> bool:
    x, y = point
    left_x, left_y = left
    right_x, right_y = right
    cross = (x - left_x) * (right_y - left_y) - (y - left_y) * (right_x - left_x)
    if abs(cross) > 1e-6:
        return False
    return min(left_x, right_x) <= x <= max(left_x, right_x) and min(left_y, right_y) <= y <= max(left_y, right_y)


def _signed_area(points: Sequence[tuple[float, float]]) -> float:
    return (
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, (*points[1:], points[0]), strict=True)
        )
        / 2
    )


# Two pieces are one line when they share at least this much of the narrower piece's line width
# (the same share `fragment_grouping` uses for "one block") ...
_SPLIT_LINE_CROSS_SHARE = 0.5
# ... and the white space between them along the line is at most this many line widths. OCR breaks
# a column where two glyphs sit close; on 4Oct p. 3 the pieces even overlap. A larger gap is a real
# gap, and the continuity check must still see it.
_SPLIT_LINE_MAX_GAP = 0.5
# ... and their centre lines are at most this share of the narrower piece apart. OCR's pieces of
# one column sit on one axis (4Oct p. 3: 0 and 1.5 px apart); the columns of two balloons stacked
# end to end inside one YOLO blob do not (Tests ch. 6 p. 5: 14 px, a fifth of the column).
_SPLIT_LINE_MAX_CENTRE_OFFSET = 0.15


def _join_line_pieces(
    boxes: Sequence[tuple[float, float, float, float]], horizontal: bool
) -> list[tuple[tuple[float, float, float, float], list[int]]]:
    """Union the boxes of pieces that lie on one line, until no two remaining boxes qualify.

    Returns each line's box with the indices of the pieces it holds. Neighbouring columns share
    their *height*, not their width, so this never joins two lines of one balloon; it only undoes
    a break OCR made inside one line.
    """
    lines = [(box, [index]) for index, box in enumerate(boxes)]
    joined = True
    while joined:
        joined = False
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                if _same_line(lines[first][0], lines[second][0], horizontal):
                    (a, a_members), (b, b_members) = lines[first], lines[second]
                    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    lines[first] = (box, a_members + b_members)
                    del lines[second]
                    joined = True
                    break
            if joined:
                break
    return lines


def _line_orientation(boxes: Sequence[tuple[float, float, float, float]]) -> bool | None:
    """True for horizontal lines, False for columns, None when the boxes are square or mixed."""
    horizontal_votes = sum((box[2] - box[0]) >= (box[3] - box[1]) * _MIN_ORIENTATION_ASPECT for box in boxes)
    vertical_votes = sum((box[3] - box[1]) >= (box[2] - box[0]) * _MIN_ORIENTATION_ASPECT for box in boxes)
    if bool(horizontal_votes) == bool(vertical_votes):
        return None
    return horizontal_votes > 0


def split_at_line_breaks(raw_quads: Sequence[Any], *, join_split_lines: bool) -> list[list[int]] | None:
    """Cut a candidate group where its line continuity breaks, keeping the runs on either side.

    The owner veto used to answer any break by splitting the whole group into single pieces: on
    chrome-box's TELEA p. 2 one handwritten aside set beside a two-column sentence (良くないけど)
    cost the sentence its own grouping. Lines are ordered and compared exactly as
    `_line_continuity` does, and a cut goes where it would reject: lateral overlap under
    `_MIN_LATERAL_OVERLAP`, or a gap over `_MAX_LINE_GAP_MULTIPLIER` mean line widths.

    Returns the parts as lists of indices into ``raw_quads``, or ``None`` when there is no break
    to cut at, or no line geometry to order (invalid quads, square or mixed orientation). Each
    part still has to pass the owner decision on its own.
    """
    quads = [_normalise_quad(raw) for raw in raw_quads]
    if len(quads) < 2 or any(quad is None for quad in quads):
        return None
    boxes = []
    for quad in quads:
        assert quad is not None
        xs, ys = [point[0] for point in quad], [point[1] for point in quad]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    horizontal = _line_orientation(boxes)
    if horizontal is None:
        return None
    lines = _join_line_pieces(boxes, horizontal) if join_split_lines else [(box, [i]) for i, box in enumerate(boxes)]
    ordered = sorted(lines, key=lambda line: _center(line[0])[1 if horizontal else 0])
    cross_lengths = [(box[3] - box[1]) if horizontal else (box[2] - box[0]) for box, _ in ordered]
    threshold = (sum(cross_lengths) / len(cross_lengths)) * _MAX_LINE_GAP_MULTIPLIER
    parts: list[list[int]] = [list(ordered[0][1])]
    for (previous, _), (current, members) in pairwise(ordered):
        previous_start, previous_end = _cross_interval(previous, horizontal)
        current_start, current_end = _cross_interval(current, horizontal)
        overlap = max(0.0, min(previous_end, current_end) - max(previous_start, current_start))
        share = overlap / max(1e-9, min(previous_end - previous_start, current_end - current_start))
        gap = max(0.0, _along_interval(current, horizontal)[0] - _along_interval(previous, horizontal)[1])
        if share < _MIN_LATERAL_OVERLAP or gap > threshold:
            parts.append(list(members))
        else:
            parts[-1].extend(members)
    return parts if len(parts) > 1 else None


def _same_line(a: tuple[float, float, float, float], b: tuple[float, float, float, float], horizontal: bool) -> bool:
    # Note the module's naming: `_along_interval` is the axis lines are stacked on (x for columns),
    # so it spans a line's width; `_cross_interval` runs the length of a line.
    a_width_span, b_width_span = _along_interval(a, horizontal), _along_interval(b, horizontal)
    a_width, b_width = a_width_span[1] - a_width_span[0], b_width_span[1] - b_width_span[0]
    if a_width <= 0 or b_width <= 0:
        return False
    shared = min(a_width_span[1], b_width_span[1]) - max(a_width_span[0], b_width_span[0])
    if shared / min(a_width, b_width) < _SPLIT_LINE_CROSS_SHARE:
        return False
    centre_offset = abs((a_width_span[0] + a_width_span[1]) - (b_width_span[0] + b_width_span[1])) / 2
    if centre_offset > _SPLIT_LINE_MAX_CENTRE_OFFSET * min(a_width, b_width):
        return False
    a_length_span, b_length_span = _cross_interval(a, horizontal), _cross_interval(b, horizontal)
    gap = max(a_length_span[0], b_length_span[0]) - min(a_length_span[1], b_length_span[1])
    return gap <= _SPLIT_LINE_MAX_GAP * (a_width + b_width) / 2


def _line_continuity(
    members: Sequence[_FragmentEvidence], source_scale: float, *, join_split_lines: bool = False
) -> dict[str, Any]:
    widths = [member.bbox[2] - member.bbox[0] for member in members]
    heights = [member.bbox[3] - member.bbox[1] for member in members]
    horizontal_votes = sum(
        width >= height * _MIN_ORIENTATION_ASPECT for width, height in zip(widths, heights, strict=True)
    )
    vertical_votes = sum(
        height >= width * _MIN_ORIENTATION_ASPECT for width, height in zip(widths, heights, strict=True)
    )
    if not horizontal_votes and not vertical_votes:
        return {"continuous": False, "reason": "ambiguous-line-orientation"}
    if horizontal_votes and vertical_votes:
        return {"continuous": False, "reason": "mixed-line-orientation"}

    horizontal = horizontal_votes > 0
    angles: list[float] = []
    for member in members:
        angle = _major_axis_angle(member.quad)
        if angle is None:
            return {"continuous": False, "reason": "degenerate-oriented-quad"}
        angles.append(angle)
    angle_delta = max(_angle_delta(angles[0], angle) for angle in angles[1:])
    if angle_delta > _MAX_ANGLE_DELTA_DEGREES:
        return {"continuous": False, "reason": "incoherent-oriented-lines", "max_angle_delta": angle_delta}

    boxes = [member.bbox for member in members]
    if join_split_lines:
        boxes = [box for box, _ in _join_line_pieces(boxes, horizontal)]
        widths = [box[2] - box[0] for box in boxes]
        heights = [box[3] - box[1] for box in boxes]
    ordered = sorted(boxes, key=lambda box: _center(box)[1 if horizontal else 0])
    cross_lengths = heights if horizontal else widths
    along_lengths = widths if horizontal else heights
    threshold = (sum(cross_lengths) / len(cross_lengths)) * _MAX_LINE_GAP_MULTIPLIER * source_scale
    gaps: list[float] = []
    overlaps: list[float] = []
    for previous, current in pairwise(ordered):
        previous_cross_start, previous_cross_end = _cross_interval(previous, horizontal)
        current_cross_start, current_cross_end = _cross_interval(current, horizontal)
        overlap = max(0.0, min(previous_cross_end, current_cross_end) - max(previous_cross_start, current_cross_start))
        overlap_share = overlap / min(
            previous_cross_end - previous_cross_start, current_cross_end - current_cross_start
        )
        _, previous_along_end = _along_interval(previous, horizontal)
        current_along_start, _ = _along_interval(current, horizontal)
        gap = max(0.0, current_along_start - previous_along_end)
        gaps.append(gap)
        overlaps.append(overlap_share)
        if overlap_share < _MIN_LATERAL_OVERLAP:
            return {
                "continuous": False,
                "reason": "insufficient-lateral-line-overlap",
                "lateral_overlap": overlap_share,
            }
        if gap > threshold:
            return {"continuous": False, "reason": "line-gap-too-large", "gap": gap, "max_gap": threshold}
    result: dict[str, Any] = {
        "continuous": True,
        "orientation": "horizontal" if horizontal else "vertical",
        "max_angle_delta": angle_delta,
        "max_gap": max(gaps, default=0.0),
        "min_lateral_overlap": min(overlaps, default=1.0),
        "gap_limit": threshold,
        "mean_along_length": sum(along_lengths) / len(along_lengths),
    }
    if join_split_lines:
        result["line_count"] = len(boxes)
    return result


def _major_axis_angle(quad: Sequence[tuple[float, float]]) -> float | None:
    edges = list(zip(quad, (*quad[1:], quad[0]), strict=True))
    start, end = max(edges, key=lambda edge: math.dist(*edge))
    if start == end:
        return None
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180


def _angle_delta(left: float, right: float) -> float:
    difference = abs(left - right) % 180
    return min(difference, 180 - difference)


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _cross_interval(bbox: tuple[float, float, float, float], horizontal: bool) -> tuple[float, float]:
    return (bbox[0], bbox[2]) if horizontal else (bbox[1], bbox[3])


def _along_interval(bbox: tuple[float, float, float, float], horizontal: bool) -> tuple[float, float]:
    return (bbox[1], bbox[3]) if horizontal else (bbox[0], bbox[2])


def _owner_id(fragment_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(fragment_ids), ensure_ascii=False, separators=(",", ":"))
    return "owner-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
