import math

import pytest

from worker.services.owner_assignment import _point_in_polygon as _inside
from worker.services.owner_assignment import assign_captured_owners

_CONTAINER = {"id": "bubble-a", "format": "polygon", "points": [[0, 0], [300, 0], [300, 300], [0, 300]]}


def _quad(x, y, width=80, height=20):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def _decision(*, groups, quads=None, masks=None, recognition=None, scale=1.0):
    quads = quads or [_quad(20, 30), _quad(25, 65)]
    recognition = recognition or [{"sourceStyle": {"fill": "black", "weight": "regular"}} for _ in quads]
    return assign_captured_owners(
        fragment_ids=[f"fragment-{index}" for index in range(len(quads))],
        raw_quads=quads,
        recognition=recognition,
        regions=[{"panelId": "same-panel", "conversationId": "same-conversation"} for _ in quads],
        candidate_groups=groups,
        detector_masks=masks if masks is not None else [_CONTAINER],
        scale_transform={"ocr_to_source": {"scale_x": scale, "scale_y": scale}},
    )


def test_assigns_continuous_styled_lines_in_one_validated_container():
    decision = _decision(groups=[[0, 1]])[0]

    assert decision.state == "assigned"
    assert decision.owner_id is not None
    assert decision.reason == "validated-container-continuous-lines"
    assert decision.diagnostics["container_ids"] == ["bubble-a", "bubble-a"]
    assert decision.diagnostics["line_continuity"]["orientation"] == "horizontal"


def test_same_panel_and_conversation_do_not_assign_an_owner_without_a_container():
    decision = _decision(groups=[[0, 1]], masks=[])[0]

    assert decision.state == "unknown"
    assert decision.owner_id is None
    assert decision.reason == "missing-validated-container"


def test_assigns_one_adjacent_line_just_outside_its_validated_bubble():
    decision = _decision(
        groups=[[0, 1]],
        quads=[_quad(280, 30, width=20, height=80), _quad(305, 30, width=20, height=80)],
        recognition=[{}, {}],
    )[0]

    assert decision.state == "assigned"
    assert decision.reason == "geometry-attached-continuous-lines"
    assert decision.diagnostics["container_ids"] == ["bubble-a", None]
    assert decision.diagnostics["geometry_attached_container"] == "bubble-a"


def _ellipse(cx, cy, rx, ry, points=64):
    return [
        [cx + rx * math.cos(2 * math.pi * index / points), cy + ry * math.sin(2 * math.pi * index / points)]
        for index in range(points)
    ]


def test_assigns_columns_whose_corners_poke_past_an_elliptical_balloon():
    """AUDIT-R20: the outer columns of a balloon are rectangles in an ellipse; their corners are outside."""
    balloon = {"id": "balloon", "format": "polygon", "points": _ellipse(150, 150, 100, 140)}
    columns = [_quad(60 + 30 * index, 60, width=26, height=180) for index in range(5)]
    # The geometry the test is about: at least one column has a corner outside the curve.
    assert not all(_inside((corner[0], corner[1]), balloon["points"]) for column in columns for corner in column)

    decision = _decision(groups=[[0, 1, 2, 3, 4]], quads=columns, recognition=[{}] * 5, masks=[balloon])[0]

    assert decision.state == "assigned"
    assert decision.reason == "validated-container-continuous-lines"
    assert decision.diagnostics["container_ids"] == ["balloon"] * 5
    assert min(decision.diagnostics["container_coverage"]) >= 0.75


def test_a_column_mostly_outside_the_balloon_is_still_not_contained():
    balloon = {"id": "balloon", "format": "polygon", "points": _ellipse(150, 150, 100, 140)}
    # Three adjacent columns; the last straddles the right edge with under half of it inside.
    columns = [
        _quad(170, 60, width=26, height=180),
        _quad(200, 60, width=26, height=180),
        _quad(230, 60, width=26, height=180),
    ]

    decision = _decision(groups=[[0, 1, 2]], quads=columns, recognition=[{}] * 3, masks=[balloon])[0]

    assert decision.state == "unknown"
    assert decision.reason == "incomplete-validated-container"
    assert decision.diagnostics["container_ids"] == ["balloon", "balloon", None]
    assert decision.diagnostics["container_coverage"][2] < 0.75


def test_overlapping_boxes_do_not_assign_an_owner_without_a_container():
    decision = _decision(
        groups=[[0, 1]],
        quads=[_quad(20, 30), _quad(25, 35)],
        recognition=[{}, {}],
        masks=[],
    )[0]

    assert decision.state == "unknown"
    assert decision.owner_id is None
    assert decision.reason == "missing-validated-container"


def test_missing_source_style_remains_an_unknown_feature_not_a_split():
    decision = _decision(groups=[[0, 1]], recognition=[{}, {}])[0]

    assert decision.state == "assigned"
    assert decision.owner_id is not None
    assert decision.reason == "validated-container-continuous-lines"
    assert decision.diagnostics["source_styles"] == [None, None]
    assert decision.diagnostics["style_evidence"] == "unknown"


def test_conflicting_declared_source_styles_veto_a_candidate_group():
    decision = _decision(
        groups=[[0, 1]],
        recognition=[{"sourceStyle": {"fill": "black"}}, {"sourceStyle": {"fill": "white"}}],
    )[0]

    assert decision.state == "unknown"
    assert decision.owner_id is None
    assert decision.reason == "different-source-styles"
    assert decision.diagnostics["style_evidence"] == "unknown"


def test_distinct_validated_containers_veto_a_candidate_group():
    masks = [
        {"id": "bubble-a", "format": "polygon", "points": [[0, 0], [120, 0], [120, 120], [0, 120]]},
        {"id": "bubble-b", "format": "polygon", "points": [[180, 0], [300, 0], [300, 120], [180, 120]]},
    ]
    decision = _decision(groups=[[0, 1]], quads=[_quad(20, 30), _quad(200, 65)], masks=masks)[0]

    assert decision.state == "unknown"
    assert decision.reason == "different-validated-containers"


def test_discontinuous_lines_remain_explicitly_unknown():
    decision = _decision(groups=[[0, 1]], quads=[_quad(20, 30), _quad(25, 180)])[0]

    assert decision.state == "unknown"
    assert decision.reason == "line-gap-too-large"
    assert decision.diagnostics["line_continuity"]["gap"] > decision.diagnostics["line_continuity"]["max_gap"]


def test_single_fragment_owner_does_not_require_container_or_style():
    decision = _decision(groups=[[0], [1]], masks=[], recognition=[{}, {}])

    assert [item.state for item in decision] == ["assigned", "assigned"]
    assert [item.reason for item in decision] == ["single-fragment-owner", "single-fragment-owner"]


def test_rejects_a_candidate_group_that_loses_or_duplicates_a_fragment():
    with pytest.raises(ValueError, match="partition every fragment exactly once"):
        _decision(groups=[[0, 0]])
