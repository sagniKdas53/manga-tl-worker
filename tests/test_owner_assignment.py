import pytest

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
