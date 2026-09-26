"""Source-linked OCR fragment features used for conservative owner decisions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def stable_fragment_id(source_id: str, index: int, quad: Any, region: Mapping[str, Any]) -> str:
    payload = json.dumps([source_id, index, quad, region], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "fragment-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def capture_fragment_features(
    *,
    source_id: str,
    raw_quads: Sequence[Any],
    recognition: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic provenance; absent style is represented as ``None``, never inferred."""
    if len(raw_quads) != len(recognition) or len(raw_quads) != len(regions):
        raise ValueError("raw_quads, recognition, and regions must have equal lengths")
    return [
        _feature(source_id, index, raw_quad, recognition[index], regions[index])
        for index, raw_quad in enumerate(raw_quads)
    ]


def _feature(
    source_id: str, index: int, raw_quad: Any, recognition: Mapping[str, Any], region: Mapping[str, Any]
) -> dict[str, Any]:
    points = _quad(raw_quad)
    feature = {
        "id": stable_fragment_id(source_id, index, raw_quad, region),
        "sourceQuad": raw_quad,
        "sourceStyle": _style(recognition, region),
        "styleProvenance": "declared" if _style(recognition, region) is not None else "unknown",
    }
    if points is None:
        return feature | {"geometry": None}
    xs, ys = zip(*points, strict=True)
    edges = list(zip(points, (*points[1:], points[0]), strict=True))
    start, end = max(edges, key=lambda edge: math.dist(*edge))
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180
    return feature | {
        "geometry": {
            "bbox": {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)},
            "majorAxisDegrees": angle,
        }
    }


def _quad(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        return None
    points = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, str | bytes) or len(point) != 2:
            return None
        x, y = point
        if (
            not isinstance(x, int | float)
            or not isinstance(y, int | float)
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            return None
        points.append((float(x), float(y)))
    return tuple(points)


def _style(recognition: Mapping[str, Any], region: Mapping[str, Any]) -> Any:
    for record in (region, recognition):
        for key in ("sourceStyle", "source_style", "style"):
            if key in record and record[key] is not None:
                return record[key]
    return None
