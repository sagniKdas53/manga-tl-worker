"""The OCR handler actually reaches the grouping fixes it was given.

`fragment_grouping` ships every phase off by default, on purpose: adding one cannot change
production until a call site opts in. That safety property has an obvious failure mode -- the
phases stay unreachable and the pipeline keeps the old behaviour while the tests, the probe and
the bench all report the new one. These tests pin the opt-in itself.

Measurements behind the values: docs/region_waist_probe_2026-08-09.md and
corpus/runs/2026-08-09/region-grouping/.
"""

import numpy as np
import pytest

from worker.handlers.ocr import (
    MERGE_THRESHOLD_RANGE,
    OCR_MERGE_THRESHOLD,
    attach_live_owner_decisions,
    grouping_config,
    merge_threshold_for,
    owner_aware_grouping_context,
    partition_unmatched_fragments_by_panel,
)
from worker.services.bubble_geometry import bubble_grouping_context, mask_solidity
from worker.services.fragment_grouping import GroupingConfig, group_fragments
from worker.services.merge_regions import merge_ocr_regions


def _frag(x, y, w, h, text="あ"):
    return {
        "text": text,
        "detectedLanguage": "ja",
        "confidence": 0.9,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    }


def test_unmatched_fragments_are_partitioned_by_smallest_containing_panel():
    fragments = [_frag(25, 25, 20, 20), _frag(125, 25, 20, 20), _frag(225, 25, 20, 20)]
    panels = [
        {"x": 0, "y": 0, "width": 300, "height": 100},
        {"x": 0, "y": 0, "width": 100, "height": 100},
        {"x": 100, "y": 0, "width": 100, "height": 100},
    ]

    partitions = partition_unmatched_fragments_by_panel(fragments, panels)

    assert partitions[1] == [fragments[0]]
    assert partitions[2] == [fragments[1]]
    assert partitions[0] == [fragments[2]]


def test_handler_config_enables_every_measured_phase():
    """A regression guard on the wiring, not on the constants.

    If someone reverts a call site to positional arguments this still passes, so it is paired
    with test_call_sites_pass_a_configuration below.
    """
    cfg = grouping_config("rtl")
    assert cfg.orientation == "vote", "BUG-6: orientation must not come from the binding direction"
    assert cfg.waist_gate is not None, "the clearance veto is what separates touching balloons"
    assert cfg.threshold_ratio < 2.0, "2.0 was the hardcoded in-bubble value that fused speakers"
    assert cfg.reading_direction == "rtl"


def test_waist_gate_can_be_switched_off_by_configuration():
    """OCR_WAIST_GATE=0 has to mean off, not a gate of zero characters (which vetoes nothing but
    still pays for the distance transform on every bubble)."""
    import worker.handlers.ocr as ocr_handler

    original = ocr_handler.OCR_WAIST_GATE
    try:
        ocr_handler.OCR_WAIST_GATE = 0.0
        assert grouping_config("rtl").waist_gate is None
    finally:
        ocr_handler.OCR_WAIST_GATE = original


def test_call_sites_pass_a_configuration():
    """All three merge calls in the handler must hand over a GroupingConfig.

    The in-bubble call used to hardcode threshold_ratio=2.0 while the other two read the
    environment, so tuning the deployment moved two of three paths and left the one where BUG-2
    lives untouched. Reading the source is crude, but the alternative is running the whole
    handler, and this catches exactly the mistake that was made before.
    """
    import inspect

    import worker.handlers.ocr as ocr_handler

    source = inspect.getsource(ocr_handler)
    calls = [line for line in source.splitlines() if "merge_ocr_regions(" in line and "def " not in line]
    calls = [c for c in calls if "import" not in c]
    assert len(calls) == 3, f"expected 3 merge call sites, found {len(calls)}: {calls}"
    assert "threshold_ratio=2.0" not in source, "the hardcoded in-bubble threshold is back"


def test_orientation_vote_reaches_a_page_with_no_bubbles():
    """sample23's failure: 61 horizontal fragments, zero detected bubbles.

    Read as vertical text the page collapsed to two regions; the vote recovers the rows. This is
    the unmatched path, which has no mask, so the vote is the only lever that reaches it.
    """
    # Wide lines, 180x30, stacked 40px apart. Orientation decides which side *is* the character
    # size: read as vertical the budget is 180*ratio and every row chains into one region; read as
    # horizontal it is 30*ratio and the rows stay apart. The gap is deliberately inside that band,
    # which is the whole failure -- on sample23 it inflated the budget to 214px instead of 33px.
    rows = [_frag(100 + 200 * col, 40 + 70 * row, 180, 30) for row in range(4) for col in range(2)]
    cfg = grouping_config("rtl")

    voted = group_fragments(rows, cfg)
    binding = group_fragments(rows, type(cfg)(threshold_ratio=cfg.threshold_ratio, reading_direction="rtl"))

    assert len(binding) == 1, "the bug: vertical geometry fuses a horizontally-set page"
    assert len(voted) == 4, f"the vote must recover one region per line, got {len(voted)}"


def test_clearance_veto_separates_two_balloons_in_one_mask():
    """Two circles joined by a narrow isthmus -- the shape YOLO returns for touching balloons.

    Distance alone merges the fragments; the waist between them is far tighter than a character,
    so the veto keeps them apart. End-to-end through merge_ocr_regions, including the mask.
    """
    h, w = 400, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    import cv2

    cv2.circle(mask, (200, 110), 90, (255,), -1)
    cv2.circle(mask, (200, 290), 90, (255,), -1)
    cv2.rectangle(mask, (188, 190), (212, 210), (255,), -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = np.asarray(contours[0]).reshape(-1, 2).tolist()

    frags = [_frag(180, 70, 40, 80, "うえ"), _frag(180, 250, 40, 80, "した")]
    cfg = grouping_config("rtl")
    ctx = bubble_grouping_context(mask, polygon)

    assert mask_solidity(polygon) < cfg.waist_max_solidity, "the pinched blob must be non-convex"
    with_mask = merge_ocr_regions(frags, grouping=cfg, context=ctx)
    without_mask = merge_ocr_regions(frags, grouping=cfg)

    assert len(with_mask) == 2, "the waist must keep two balloons apart"
    assert len(without_mask) <= len(with_mask), "the veto may only ever withhold a merge"


def test_context_is_none_when_the_bubble_has_no_polygon():
    """A missing mask must degrade to distance alone, never raise. The handler passes whatever
    YOLO returned, and the fallback detector does not always produce a polygon."""
    assert bubble_grouping_context(None, None) is None
    assert bubble_grouping_context(np.zeros((10, 10), dtype=np.uint8), None) is None


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        ([[0, 0], [100, 0], [100, 100], [0, 100]], 1.0),
        (None, 1.0),
        ([[0, 0], [10, 0]], 1.0),
    ],
)
def test_solidity_defaults_to_convex_when_it_cannot_be_measured(polygon, expected):
    """1.0 disables the veto, so every unmeasurable case must land there rather than at 0."""
    assert mask_solidity(polygon) == pytest.approx(expected, abs=0.01)


def _provenanced_fragment(index, x, y):
    fragment = _frag(x, y, 60, 20, f"line-{index}")
    fragment["fragmentId"] = f"fragment-{index}"
    fragment["sourceQuad"] = [[x, y], [x + 60, y], [x + 60, y + 20], [x, y + 20]]
    fragment["ownershipProvenance"] = {"id": f"fragment-{index}", "sourceQuad": fragment["sourceQuad"]}
    return fragment


def test_live_owner_context_assigns_a_continuous_shared_container_without_capture_mode():
    fragments = [_provenanced_fragment(0, 10, 10), _provenanced_fragment(1, 10, 35)]
    context = owner_aware_grouping_context(
        None,
        [{"format": "polygon", "id": "bubble-0", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
    )

    groups = group_fragments(
        fragments,
        GroupingConfig(threshold_ratio=1.0, reading_direction="ltr", orientation="vote"),
        context,
    )

    assert groups == [[0, 1]]
    decisions = [fragment["ownershipProvenance"]["ownerDecision"] for fragment in fragments]
    assert all(decision["state"] == "assigned" for decision in decisions)
    assert decisions[0]["owner_id"] == decisions[1]["owner_id"]


def test_live_owner_context_attaches_one_continuous_line_at_a_bubble_edge():
    fragments = [_provenanced_fragment(0, 10, 10), _provenanced_fragment(1, 10, 55)]
    decisions = attach_live_owner_decisions(
        fragments,
        [[0, 1]],
        [{"format": "polygon", "id": "bubble-0", "points": [[0, 0], [100, 0], [100, 65], [0, 65]]}],
    )

    assert decisions[0].reason == "geometry-attached-continuous-lines"
    assert {fragment["ownershipProvenance"]["ownerDecision"]["reason"] for fragment in fragments} == {
        "geometry-attached-continuous-lines"
    }


def test_live_owner_context_keeps_an_uncontained_component_unresolved():
    fragments = [_provenanced_fragment(0, 10, 10), _provenanced_fragment(1, 10, 35)]

    groups = group_fragments(
        fragments,
        GroupingConfig(threshold_ratio=1.0, reading_direction="ltr", orientation="vote"),
        owner_aware_grouping_context(None, []),
    )

    assert groups == [[0], [1]]
    decisions = [fragment["ownershipProvenance"]["ownerDecision"] for fragment in fragments]
    assert all(decision["state"] == "unknown" for decision in decisions)
    assert {decision["reason"] for decision in decisions} == {"missing-validated-container"}


def test_the_job_threshold_reaches_grouping_and_is_clamped():
    """System Settings and series/chapter overrides send `ocrMergeThreshold` on the OCR job."""
    assert merge_threshold_for({"ocrMergeThreshold": 0.8}) == 0.8
    assert merge_threshold_for({"ocrMergeThreshold": "0.5"}) == 0.5
    assert merge_threshold_for({"ocrMergeThreshold": 0}) == MERGE_THRESHOLD_RANGE[0]
    assert merge_threshold_for({"ocrMergeThreshold": 99}) == MERGE_THRESHOLD_RANGE[1]
    # A job queued before the setting existed, or a bad value, keeps the deployment default.
    for job in ({}, {"ocrMergeThreshold": None}, {"ocrMergeThreshold": "x"}, {"ocrMergeThreshold": float("nan")}):
        assert merge_threshold_for(job) == OCR_MERGE_THRESHOLD
    assert grouping_config("rtl", 0.8).threshold_ratio == 0.8
    assert grouping_config("rtl").threshold_ratio == OCR_MERGE_THRESHOLD


def test_a_higher_threshold_joins_what_the_default_keeps_apart():
    # Two 40px-wide columns 30px apart: under the default 0.35-character budget (14px) they are
    # two regions; at 1.0 (40px) they are one.
    columns = [_frag(200, 100, 40, 200, "あ"), _frag(130, 100, 40, 200, "い")]
    assert len(merge_ocr_regions(columns, grouping=grouping_config("rtl", 0.35))) == 2
    assert len(merge_ocr_regions(columns, grouping=grouping_config("rtl", 1.0))) == 1


def test_every_grouping_call_in_the_handler_uses_the_job_threshold():
    import inspect

    import worker.handlers.ocr as ocr_handler

    source = inspect.getsource(ocr_handler.process_ocr)
    calls = [line.strip() for line in source.splitlines() if "grouping_config(" in line]
    assert calls, "process_ocr no longer builds a grouping configuration"
    assert all("merge_threshold" in call for call in calls), calls
