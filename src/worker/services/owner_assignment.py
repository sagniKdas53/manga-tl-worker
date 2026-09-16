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
) -> list[OwnerDecision]:
    """Assign only evidence-backed text owners for captured OCR grouping candidates.

    A multi-fragment owner requires one validated detector container, coherent oriented-line
    geometry, and a finite OCR-to-source scale. Available declared source style can veto
    contradictory fragments; absent style remains an explicit unknown feature, not a split.
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
        container_ids = [_containing_container(member.quad, containers) for member in evidence]
        diagnostics["container_ids"] = container_ids
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

        continuity = _line_continuity(evidence, source_scale)
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
) -> str | None:
    matches = [
        identifier for identifier, polygon in containers if all(_point_in_polygon(point, polygon) for point in quad)
    ]
    return matches[0] if len(matches) == 1 else None


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


def _line_continuity(members: Sequence[_FragmentEvidence], source_scale: float) -> dict[str, Any]:
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

    ordered = sorted(members, key=lambda member: _center(member.bbox)[1 if horizontal else 0])
    cross_lengths = heights if horizontal else widths
    along_lengths = widths if horizontal else heights
    threshold = (sum(cross_lengths) / len(cross_lengths)) * _MAX_LINE_GAP_MULTIPLIER * source_scale
    gaps: list[float] = []
    overlaps: list[float] = []
    for previous, current in pairwise(ordered):
        previous_cross_start, previous_cross_end = _cross_interval(previous.bbox, horizontal)
        current_cross_start, current_cross_end = _cross_interval(current.bbox, horizontal)
        overlap = max(0.0, min(previous_cross_end, current_cross_end) - max(previous_cross_start, current_cross_start))
        overlap_share = overlap / min(
            previous_cross_end - previous_cross_start, current_cross_end - current_cross_start
        )
        _, previous_along_end = _along_interval(previous.bbox, horizontal)
        current_along_start, _ = _along_interval(current.bbox, horizontal)
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
    return {
        "continuous": True,
        "orientation": "horizontal" if horizontal else "vertical",
        "max_angle_delta": angle_delta,
        "max_gap": max(gaps, default=0.0),
        "min_lateral_overlap": min(overlaps, default=1.0),
        "gap_limit": threshold,
        "mean_along_length": sum(along_lengths) / len(along_lengths),
    }


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
