"""Capture and deterministic replay for the current OCR/grouping seams.

This module is deliberately an observer. It records the values crossing the OCR and grouping
boundary and replays grouping from the recorded regions; it does not change either algorithm.
Capture files are JSON so a run can be copied out of a worker checkout without importing model
or image state.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from worker.services.fragment_grouping import GroupingConfig, group_fragments


def _json_value(value: Any) -> Any:
    """Convert ordinary and NumPy-like values to deterministic JSON values."""
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fragment_id(source_id: str, index: int, quad: Any, region: dict) -> str:
    return "fragment-" + _digest([source_id, index, quad, region])[:24]


def _owner_id(fragment_ids: list[str]) -> str:
    return "owner-" + _digest(sorted(fragment_ids))[:24]


@dataclass(frozen=True)
class ReplayResult:
    fragment_ids: list[str]
    fragment_geometry: list[dict[str, Any]]
    owner_ids: list[str]
    grouping_edges: list[dict[str, Any]]


@dataclass(frozen=True)
class OcrCapture:
    format: str
    source: dict[str, Any]
    raw_quads: list[Any]
    scale_transform: dict[str, Any]
    detector_masks: list[Any]
    recognition: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    grouping: dict[str, Any]
    grouping_edges: list[dict[str, Any]]
    final_owners: list[dict[str, Any]]
    paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> OcrCapture:
        return cls(**json.loads(Path(path).read_text()))

    def replay(self) -> ReplayResult:
        config = GroupingConfig(**self.grouping)
        groups = group_fragments(self.regions, config)
        fragment_ids = [
            _fragment_id(self.source["id"], index, quad, region)
            for index, (quad, region) in enumerate(zip(self.raw_quads, self.regions, strict=True))
        ]
        edges = [
            {"from": fragment_ids[left], "to": fragment_ids[right], "reason": "same-group-component"}
            for group in groups
            for offset, left in enumerate(group)
            for right in group[offset + 1 :]
        ]
        owners = [_owner_id([fragment_ids[index] for index in group]) for group in groups]
        geometry = [
            {"id": fragment_ids[index], "quad": _json_value(self.raw_quads[index]), "region": self.regions[index]}
            for index in range(len(self.regions))
        ]
        return ReplayResult(fragment_ids, geometry, owners, edges)


def capture_ocr_grouping(
    *,
    source_id: str,
    raw_quads: list[Any],
    scale_transform: dict[str, Any],
    detector_masks: list[Any],
    recognition: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    grouping: GroupingConfig,
    paths: dict[str, str] | None = None,
) -> OcrCapture:
    """Capture one page at the OCR/grouping boundary and immediately verify its owner set."""
    if len(raw_quads) != len(regions) or len(recognition) != len(regions):
        raise ValueError("raw_quads, recognition, and regions must have equal lengths")
    initial = OcrCapture(
        format="ocr-grouping-capture-v1",
        source={"id": source_id, "digest": _digest([source_id, raw_quads, regions])},
        raw_quads=_json_value(raw_quads),
        scale_transform=_json_value(scale_transform),
        detector_masks=_json_value(detector_masks),
        recognition=_json_value(recognition),
        regions=_json_value(regions),
        grouping=asdict(grouping),
        grouping_edges=[],
        final_owners=[],
        paths=paths or {"ocr": "live", "grouping": "live", "detector_masks": "live"},
    )
    replay = initial.replay()
    groups = group_fragments(initial.regions, grouping)
    fragment_ids = [
        _fragment_id(source_id, index, quad, region)
        for index, (quad, region) in enumerate(zip(initial.raw_quads, initial.regions, strict=True))
    ]
    return OcrCapture(
        **{
            **asdict(initial),
            "grouping_edges": replay.grouping_edges,
            "final_owners": [
                {"id": owner, "fragment_ids": [fragment_ids[index] for index in group]}
                for owner, group in zip(replay.owner_ids, groups, strict=True)
            ],
        }
    )


def capture_observed_ocr_grouping(
    *,
    source_id: str,
    raw_quads: list[Any],
    scale_transform: dict[str, Any],
    detector_masks: list[Any],
    recognition: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    grouping: GroupingConfig,
    observed_groups: list[list[int]],
    paths: dict[str, str] | None = None,
) -> OcrCapture:
    """Record groups observed at runtime without rerunning or changing the grouping decision."""
    if len(raw_quads) != len(regions) or len(recognition) != len(regions):
        raise ValueError("raw_quads, recognition, and regions must have equal lengths")
    flat_indices = [index for group in observed_groups for index in group]
    if sorted(flat_indices) != list(range(len(regions))):
        raise ValueError("observed_groups must partition every region exactly once")
    fragment_ids = [
        _fragment_id(source_id, index, quad, region)
        for index, (quad, region) in enumerate(zip(raw_quads, regions, strict=True))
    ]
    owners = []
    for group in observed_groups:
        member_regions = [regions[index] for index in group]
        owner_fragments = [fragment_ids[index] for index in group]
        owners.append(
            {
                "id": _owner_id(owner_fragments),
                "fragment_ids": owner_fragments,
                "bbox": {
                    "x": min(region["x"] for region in member_regions),
                    "y": min(region["y"] for region in member_regions),
                    "width": max(region["x"] + region["width"] for region in member_regions)
                    - min(region["x"] for region in member_regions),
                    "height": max(region["y"] + region["height"] for region in member_regions)
                    - min(region["y"] for region in member_regions),
                },
            }
        )
    return OcrCapture(
        format="ocr-grouping-capture-v1",
        source={"id": source_id, "digest": _digest([source_id, raw_quads, regions])},
        raw_quads=_json_value(raw_quads),
        scale_transform=_json_value(scale_transform),
        detector_masks=_json_value(detector_masks),
        recognition=_json_value(recognition),
        regions=_json_value(regions),
        grouping=asdict(grouping),
        grouping_edges=[
            {"from": fragment_ids[left], "to": fragment_ids[right], "reason": "observed-group-component"}
            for group in observed_groups
            for offset, left in enumerate(group)
            for right in group[offset + 1 :]
        ],
        final_owners=owners,
        paths=paths or {"ocr": "live", "grouping": "live", "detector_masks": "live"},
    )


def replay_twice(capture: OcrCapture) -> tuple[ReplayResult, ReplayResult, bool]:
    """Replay a capture twice and report equality of stable IDs, geometry, edges, and owners."""
    first, second = capture.replay(), capture.replay()
    return first, second, first == second
