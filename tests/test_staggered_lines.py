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


def test_the_handler_passes_the_setting_to_the_owner_veto():
    import inspect

    import worker.handlers.ocr as ocr_handler
    from worker.config import OCR_STAGGERED_LINES

    assert OCR_STAGGERED_LINES is True
    assert "staggered_lines=OCR_STAGGERED_LINES" in inspect.getsource(ocr_handler)


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
