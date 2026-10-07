"""B3: a balloon whose lines are staggered is one text, not one per x-neighbour pair.

The line-continuity check sorts a balloon's lines across the reading direction and needs every
neighbouring pair to overlap along the line. A balloon set as two short paragraphs, one higher
than the other, has neighbours that do not overlap even though every line overlaps the line beside
it. Corpus pages, chrome-box captures of 2026-10-06 (worker dacd7e8):

- ja/sample78: 「ちょっと男子ぃ」 over 「いま峯森さん撮ったでしょ？」 -- one balloon, three regions.
- ja/sample24: one YOLO container holding two bracket balloons: three columns above, then the reply
  「人違いじゃない？」 below. The upper three columns are one text; the reply is another.

With ``staggered_lines`` the lines are continuous when they form one connected chain: two lines are
linked when they overlap along the line and sit less than half a line apart across it.
"""

import pytest

from worker.services.owner_assignment import assign_captured_owners


def _quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _decide(boxes, *, staggered_lines):
    quads = [_quad(*box) for box in boxes]
    xs = [p[0] for q in quads for p in q]
    ys = [p[1] for q in quads for p in q]
    container = {
        "id": "balloon",
        "format": "polygon",
        "points": [
            [min(xs) - 20, min(ys) - 20],
            [max(xs) + 20, min(ys) - 20],
            [max(xs) + 20, max(ys) + 20],
            [min(xs) - 20, max(ys) + 20],
        ],
    }
    return assign_captured_owners(
        fragment_ids=[f"fragment-{index}" for index in range(len(quads))],
        raw_quads=quads,
        recognition=[{} for _ in quads],
        regions=[{} for _ in quads],
        candidate_groups=[list(range(len(quads)))],
        detector_masks=[container],
        scale_transform={"ocr_to_source": {"scale_x": 1.0, "scale_y": 1.0}},
        join_split_lines=True,
        staggered_lines=staggered_lines,
    )[0]


SAMPLE78 = [
    (1114, 79, 48, 115),  # 男子ぃ
    (1152, 73, 50, 292),  # ちょっといま峯森 (OCR read two columns' worth as one)
    (1095, 229, 39, 134),  # でしょ？
    (1125, 225, 44, 168),  # さん撮った
]
SAMPLE24_UPPER = [
    (26, 2188, 58, 397),  # 習なんて必要なの？
    (78, 2185, 65, 473),  # 有名な天才じゃない？補
    (134, 2188, 61, 456),  # ねえこの子、隣の学校で
]
SAMPLE24_REPLY = [
    (0, 2667, 40, 135),  # ない？
    (32, 2661, 61, 222),  # 人違いじゃ
]
# chrome-box TELEA p. 2, balloon 1 (tests/test_owner_assignment.py): the aside 良くないけど is set
# lower beside the two columns and must stay its own text.
TELEA_ASIDE = [(2562, 1347, 124, 1183), (2454, 1347, 104, 767), (2334, 1986, 148, 576)]


def test_staggered_paragraphs_are_vetoed_by_default():
    decision = _decide(SAMPLE78, staggered_lines=False)
    assert decision.state == "unknown"
    assert decision.reason == "insufficient-lateral-line-overlap"


def test_staggered_paragraphs_in_one_balloon_are_one_text():
    decision = _decide(SAMPLE78, staggered_lines=True)
    assert decision.state == "assigned"
    assert decision.diagnostics["line_continuity"]["staggered"] is True


def test_a_reply_below_the_columns_is_not_joined_to_them():
    assert _decide(SAMPLE24_UPPER, staggered_lines=True).state == "assigned"
    decision = _decide(SAMPLE24_UPPER + SAMPLE24_REPLY, staggered_lines=True)
    assert decision.state == "unknown"


def test_an_aside_set_beside_the_columns_stays_apart():
    decision = _decide(TELEA_ASIDE, staggered_lines=True)
    assert decision.state == "unknown"


@pytest.mark.parametrize("staggered_lines", [False, True])
def test_ordinary_columns_are_continuous_either_way(staggered_lines):
    columns = [(60 + 30 * index, 60, 26, 180) for index in range(3)]
    decision = _decide(columns, staggered_lines=staggered_lines)
    assert decision.state == "assigned"
    assert "staggered" not in decision.diagnostics["line_continuity"]


@pytest.mark.parametrize(("env", "expected"), [(None, "True"), ("false", "False")])
def test_the_setting_defaults_on_and_can_be_switched_off(env, expected):
    import os
    import subprocess
    import sys

    import worker

    environment = {key: value for key, value in os.environ.items() if key != "OCR_STAGGERED_LINES"}
    environment["PYTHONPATH"] = os.path.dirname(os.path.dirname(worker.__file__))
    if env is not None:
        environment["OCR_STAGGERED_LINES"] = env
    result = subprocess.run(
        [sys.executable, "-c", "from worker.config import OCR_STAGGERED_LINES; print(OCR_STAGGERED_LINES)"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == expected


@pytest.mark.parametrize("staggered_lines", [False, True])
def test_the_owner_veto_and_split_receive_the_setting(monkeypatch, staggered_lines):
    import worker.handlers.ocr as ocr_handler

    received = {}

    class Decision:
        def to_dict(self):
            return {"state": "unknown", "reason": "test"}

    def fake_attach(*args, **kwargs):
        received["veto"] = kwargs.get("staggered_lines")
        return [Decision()]

    def fake_split(quads, **kwargs):
        received["split"] = kwargs.get("staggered_lines")
        return None

    monkeypatch.setattr(ocr_handler, "attach_live_owner_decisions", fake_attach)
    monkeypatch.setattr(ocr_handler, "split_at_line_breaks", fake_split)
    context = ocr_handler.owner_aware_grouping_context(None, [], staggered_lines=staggered_lines)
    assert context.owner_veto is not None and context.owner_split is not None
    regions = [{"sourceQuad": _quad(*box)} for box in SAMPLE78]
    context.owner_veto([0, 1, 2, 3], regions)
    context.owner_split([0, 1, 2, 3], regions)
    assert received == {"veto": staggered_lines, "split": staggered_lines}


# 4Oct ch. 1 p. 3, the owner's hand merge (worker #53): 仕事とはいえ, high in the same balloon, stays
# apart from the seven-piece sentence ブライダルなんて…でしょう. Its overlap along the line with the
# top of the sentence's first columns is under a fifth, so the chain does not link it.
FOURTH_OCT_P3_SENTENCE = [
    (1628, 501, 118, 498),  # でしょう：
    (1708, 487, 152, 633),  # 縁が無かった
    (1811, 491, 142, 726),  # していなければ
    (1936, 508, 104, 207),  # アイ
    (2026, 501, 114, 235),  # ブラ
    (1919, 677, 135, 332),  # ドルを
    (2002, 670, 162, 626),  # イダルなんて
]
FOURTH_OCT_P3_ASIDE = (2161, 110, 231, 436)  # 仕事とはいえ (its region box on the dev stack)


def test_the_owners_hand_merge_on_4oct_p3_still_holds():
    assert _decide(FOURTH_OCT_P3_SENTENCE, staggered_lines=True).state == "assigned"
    decision = _decide([*FOURTH_OCT_P3_SENTENCE, FOURTH_OCT_P3_ASIDE], staggered_lines=True)
    assert decision.state == "unknown"


def _region(index, box):
    quad = _quad(*box)
    x, y, w, h = box
    return {
        "text": f"line-{index}",
        "detectedLanguage": "ja",
        "confidence": 0.9,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "fragmentId": f"fragment-{index}",
        "sourceQuad": quad,
        "ownershipProvenance": {"id": f"fragment-{index}", "sourceQuad": quad},
    }


def _grouped(boxes, *, staggered_lines):
    """Group ``boxes`` as the handler does inside one balloon: owner veto, then the split at breaks."""
    from worker.handlers.ocr import grouping_config, owner_aware_grouping_context
    from worker.services.fragment_grouping import group_fragments

    regions = [_region(index, box) for index, box in enumerate(boxes)]
    xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    left, top, right, bottom = min(xs) - 20, min(ys) - 20, max(xs) + 20, max(ys) + 20
    balloon = {
        "format": "polygon",
        "id": "balloon",
        "points": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }
    context = owner_aware_grouping_context(
        None, [balloon], join_split_lines=True, split_at_breaks=True, staggered_lines=staggered_lines
    )
    return sorted(sorted(group) for group in group_fragments(regions, grouping_config("rtl", 0.35), context))


@pytest.mark.parametrize(
    "stray",
    [
        (1055, 370, 38, 120),  # below and left of the lower paragraph
        (1205, 380, 40, 120),  # below and right of the upper paragraph
    ],
)
def test_a_stray_line_in_the_component_does_not_cut_the_staggered_balloon(stray):
    """CodeRabbit on #56: a line that proximity joins to sample78's balloon gets the whole component
    vetoed, and the split that follows cut only between neighbours, so the balloon came back as three
    pieces. The split now keeps lines the staggered chain links together."""
    boxes = [*SAMPLE78, stray]
    assert _grouped(boxes, staggered_lines=True) == [[0, 1, 2, 3], [4]]
    # Without B3 the balloon is pieces either way: the split only changes when the setting is on.
    assert _grouped(boxes, staggered_lines=False) == [[0], [1, 3], [2], [4]]


@pytest.mark.parametrize(
    ("boxes", "expected"),
    [
        (TELEA_ASIDE, [[0, 1], [2]]),
        ([*FOURTH_OCT_P3_SENTENCE, FOURTH_OCT_P3_ASIDE], [[0, 1, 2, 3, 4, 5, 6], [7]]),
        ([*SAMPLE24_UPPER, *SAMPLE24_REPLY], [[0, 1, 2], [3, 4]]),
    ],
    ids=["telea-aside", "4oct-p3-hand-merge", "sample24-reply"],
)
def test_the_guard_pages_hold_through_grouping_and_the_split(boxes, expected):
    """CodeRabbit on #56: the guards above call the owner decision directly. Through group_fragments
    and the split at breaks, the aside and the reply must still come out as texts of their own."""
    assert _grouped(boxes, staggered_lines=True) == expected


def test_the_split_without_staggered_lines_is_unchanged():
    from worker.services.owner_assignment import split_at_line_breaks

    quads = [_quad(*box) for box in [*SAMPLE78, (1055, 370, 38, 120)]]
    plain = split_at_line_breaks(quads, join_split_lines=True)
    assert plain is not None and sorted(sorted(part) for part in plain) == [[0], [1, 3], [2], [4]]
    staggered = split_at_line_breaks(quads, join_split_lines=True, staggered_lines=True)
    assert staggered is not None and sorted(sorted(part) for part in staggered) == [[0, 1, 2, 3], [4]]
