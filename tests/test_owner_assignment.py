import math

import pytest

from worker.services.owner_assignment import _join_line_pieces, assign_captured_owners, split_at_line_breaks
from worker.services.owner_assignment import _point_in_polygon as _inside

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


# AUDIT-R21, 4Oct ch. 1 p. 3 (the user's screenshot): seven OCR pieces of one balloon, as detected.
# OCR broke two columns in two -- ブラ | イダルなんて and アイ | ドルを -- so sorted by x, each top
# piece's neighbour is the *next* column and the lateral-overlap check vetoes the whole balloon.
_SPLIT_COLUMNS = [
    ("でしょう：", 1628, 501, 118, 498),
    ("縁が無かった", 1708, 487, 152, 633),
    ("していなければ", 1811, 491, 142, 726),
    ("アイ", 1936, 508, 104, 207),
    ("ブラ", 2026, 501, 114, 235),
    ("ドルを", 1919, 677, 135, 332),
    ("イダルなんて", 2002, 670, 162, 626),
]
_SPLIT_COLUMNS_BALLOON = {
    "id": "bubble-2",
    "format": "polygon",
    "points": [[1593, 60], [2435, 60], [2435, 1357], [1593, 1357]],
}


def _joined_decision(quads, *, join_split_lines, masks=None):
    return assign_captured_owners(
        fragment_ids=[f"fragment-{index}" for index in range(len(quads))],
        raw_quads=quads,
        recognition=[{} for _ in quads],
        regions=[{} for _ in quads],
        candidate_groups=[list(range(len(quads)))],
        detector_masks=masks if masks is not None else [_CONTAINER],
        scale_transform={"ocr_to_source": {"scale_x": 1.0, "scale_y": 1.0}},
        join_split_lines=join_split_lines,
    )[0]


def _split_columns():
    return [_quad(x, y, width=w, height=h) for _, x, y, w, h in _SPLIT_COLUMNS]


def test_a_column_broken_in_two_vetoes_its_balloon_by_default():
    decision = _joined_decision(_split_columns(), join_split_lines=False, masks=[_SPLIT_COLUMNS_BALLOON])

    assert decision.state == "unknown"
    assert decision.reason == "insufficient-lateral-line-overlap"


def test_joining_a_columns_pieces_first_keeps_the_balloon_one_owner():
    decision = _joined_decision(_split_columns(), join_split_lines=True, masks=[_SPLIT_COLUMNS_BALLOON])

    assert decision.state == "assigned"
    assert decision.reason == "validated-container-continuous-lines"
    continuity = decision.diagnostics["line_continuity"]
    assert continuity["orientation"] == "vertical"
    # Seven pieces, five columns: each broken column became one line.
    assert continuity["line_count"] == 5


def test_side_by_side_columns_are_not_joined_into_one_line():
    """Neighbouring columns share their height, not their width; joining runs along the line only."""
    columns = [_quad(60 + 30 * index, 60, width=26, height=180) for index in range(3)]
    decision = _joined_decision(columns, join_split_lines=True)

    assert decision.state == "assigned"
    assert decision.diagnostics["line_continuity"]["line_count"] == 3


# chrome-box TELEA chapter, p. 2 (the 良くないけど gate), as the live worker read it 2026-10-05.
# Balloon 1: two columns of one sentence, and a handwritten aside set lower beside them.
_ASIDE_BALLOON = [
    (2562, 1347, 124, 1183),  # クソ兄貴のお手製フリップは
    (2454, 1347, 104, 767),  # まだいいとして：
    (2334, 1986, 148, 576),  # 良くないけど
]
# Balloon 4: two two-column blocks, the second set lower and to the left of the first.
_STEPPED_BALLOON = [
    (763, 1826, 104, 812),  # ジッパー式のバニーが
    (659, 1830, 96, 700),  # 好きなんだけどさ
    (523, 2386, 104, 1311),  # あれ引っかかったら痛そうだから
    (415, 2382, 96, 1183),  # ボタンで手作りしてみたぞノ
]


def _as_sets(parts):
    return sorted(sorted(part) for part in parts)


def test_a_vetoed_balloon_splits_where_its_lines_break_not_into_single_pieces():
    quads = [_quad(x, y, width=w, height=h) for x, y, w, h in _ASIDE_BALLOON]

    assert _as_sets(split_at_line_breaks(quads, join_split_lines=True)) == [[0, 1], [2]]


def test_two_stepped_blocks_in_one_balloon_split_into_the_two_blocks():
    quads = [_quad(x, y, width=w, height=h) for x, y, w, h in _STEPPED_BALLOON]

    assert _as_sets(split_at_line_breaks(quads, join_split_lines=True)) == [[0, 1], [2, 3]]


def test_continuous_lines_have_no_break_to_split_at():
    columns = [_quad(60 + 30 * index, 60, width=26, height=180) for index in range(3)]

    assert split_at_line_breaks(columns, join_split_lines=True) is None


def test_split_points_keep_a_broken_columns_pieces_together():
    """The p. 3 balloon is continuous once its pieces are joined, so there is nothing to cut."""
    assert split_at_line_breaks(_split_columns(), join_split_lines=True) is None


def test_mixed_orientation_gives_no_split_points():
    quads = [_quad(100, 20, width=26, height=180), _quad(140, 20, width=180, height=26)]

    assert split_at_line_breaks(quads, join_split_lines=True) is None


def test_columns_of_two_stacked_balloons_are_not_joined_as_one_broken_column():
    """Tests ch. 6 p. 5: three echo balloons fused by YOLO, stacked top to bottom.

    The middle balloon's 終わらせる ends 2 px into the bottom balloon's 速攻で and they share most
    of their width, but a broken column's pieces sit on one centre line (4Oct p. 3: 0 and 1.5 px
    apart); these are 14 px apart, a fifth of the column.
    """
    stacked = [
        _quad(92, 172, width=68, height=274),  # 終わらせる (middle balloon)
        _quad(96, 444, width=88, height=214),  # 速攻で (bottom balloon)
    ]

    assert _join_line_pieces([_box(quad) for quad in stacked], horizontal=False) == [
        (_box(stacked[0]), [0]),
        (_box(stacked[1]), [1]),
    ]


def _box(quad):
    return (quad[0][0], quad[0][1], quad[2][0], quad[2][1])


def test_pieces_of_one_column_far_apart_are_not_joined_over_the_gap():
    """Joining must not hide a gap: two pieces of one column far apart along it stay two lines."""
    quads = [_quad(100, 20, width=26, height=60), _quad(100, 200, width=26, height=60)]
    decision = _joined_decision(quads, join_split_lines=True)

    # Not joined, so still two lines in one column, which share no length: vetoed as before.
    assert decision.state == "unknown"
    assert decision.reason == "insufficient-lateral-line-overlap"
