"""B4 + B5: text no balloon holds gets the owner decision's line checks, without the balloon.

Geometry from the chrome-box captures of 2026-10-06 (worker df87949), the pieces' OCR quads:

- ja/sample218 p. 9: YOLO found no balloon around the left speech, so its three columns took the
  no-balloon path and chained with four misreads of the art into one 1140 x 1995 region, typeset
  under the balloon.
- ja/sample83: two upright columns of one text, with a square misread "M" and a tilted おぁ nearby.
- ja/sample104 p. 19: a title on tilted tiles (one text), and a four-line caption whose lines were
  split to singletons because the tall 右中大 beside them made the group oversized.
"""

import pytest

from worker.handlers.ocr import grouping_config, no_balloon_grouping_context
from worker.services.fragment_grouping import GroupingContext, group_fragments
from worker.services.owner_assignment import (
    has_rotated_lines,
    split_at_column_steps,
    split_by_character_size,
    split_by_orientation,
    split_off_squares,
)

SAMPLE218_P9 = {
    "page": (2844, 4014),
    "pieces": [
        ("せそっ…♡らりっっっ♡", [[122, 2093], [216, 2093], [223, 2885], [133, 2885]]),
        ("ヌネせそっ…♡", [[200, 2093], [294, 2093], [294, 2560], [200, 2560]]),
        ("るれろわっ…！♡", [[282, 2101], [376, 2101], [376, 2630], [282, 2630]]),
        ("(gftgs grgitgt", [[466, 2062], [698, 2089], [600, 2956], [368, 2928]]),
        ("一", [[627, 2042], [878, 2031], [890, 2379], [639, 2391]]),
        ("BOFE", [[118, 2909], [408, 2952], [263, 3904], [0, 3861]]),
    ],
}
SAMPLE83 = {
    "page": (1412, 2000),
    "pieces": [
        ("けこ…かっきゃ下くけこ", [[119, 186], [209, 186], [209, 1033], [119, 1033]]),
        ("万上となにゃ…", [[25, 225], [117, 225], [117, 721], [25, 721]]),
        ("M", [[258, 719], [314, 719], [314, 775], [258, 775]]),
        ("おぁ，", [[318, 561], [389, 537], [436, 680], [365, 703]]),
    ],
}
SAMPLE104_TITLE = {
    "page": (1474, 1989),
    "pieces": [
        ("七八九十", [[1373, 1496], [1428, 1669], [1375, 1686], [1321, 1511]]),
        ("六七", [[1154, 1533], [1251, 1480], [1340, 1647], [1243, 1700]]),
        ("口目手も", [[1288, 1515], [1338, 1501], [1379, 1670], [1329, 1682]]),
        ("十百", [[1208, 1624], [1472, 1657], [1472, 1849], [1187, 1816]]),
        ("ア", [[1323, 1801], [1430, 1801], [1430, 1890], [1323, 1890]]),
    ],
}
SAMPLE104_CAPTION = {
    "page": (1474, 1989),
    "pieces": [
        ("【八九十百ヤユ千】", [[19, 140], [489, 82], [497, 150], [29, 208]]),
        ("あいうっ月火い水木金え", [[31, 206], [1165, 58], [1173, 130], [41, 280]]),
        ("キクケコサシスし口すセ", [[39, 276], [1179, 132], [1187, 200], [49, 346]]),
        ("タ千万チツテ上のはひふへ", [[49, 342], [763, 256], [771, 328], [58, 416]]),
        ("右中大", [[1043, 206], [1410, 188], [1447, 946], [1080, 963]]),
    ],
}


def _regions(case):
    regions = []
    for index, (text, quad) in enumerate(case["pieces"]):
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        regions.append(
            {
                "text": text,
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "x": min(xs),
                "y": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
                "fragmentId": f"fragment-{index}",
                "sourceQuad": quad,
                "ownershipProvenance": {"id": f"fragment-{index}", "sourceQuad": quad},
            }
        )
    return regions


def _groups(case, *, veto=True):
    width, height = case["page"]
    context = GroupingContext(page_area=float(width * height))
    if veto:
        context = no_balloon_grouping_context(context)
    groups = group_fragments(_regions(case), grouping_config("rtl"), context)
    return sorted(sorted(group) for group in groups)


def test_distance_alone_chains_speech_with_misreads_of_the_art():
    assert [0, 1, 2] in [g[:3] for g in _groups(SAMPLE218_P9, veto=False) if len(g) > 3]


def test_the_speech_comes_apart_from_the_misreads():
    groups = _groups(SAMPLE218_P9)
    assert [0, 1, 2] in groups
    assert all(not ({0, 1, 2} & set(group)) for group in groups if group != [0, 1, 2])


def test_a_square_glyph_does_not_break_two_upright_columns():
    """B4: the square "M" read as 0 degrees against the columns' 90, and the pair fell apart."""
    together = next(group for group in _groups(SAMPLE83) if 0 in group)
    assert 1 in together


def test_a_title_on_tilted_tiles_stays_one_text():
    assert _groups(SAMPLE104_TITLE) == [[0, 1, 2, 3, 4]]


def test_a_caption_no_longer_falls_apart_beside_a_tall_column():
    """The caption and 右中大 were one oversized group, halved down to single lines. Cut by
    direction first, the four lines are small enough to stay one text."""
    assert _groups(SAMPLE104_CAPTION, veto=False) == [[0], [1], [2], [3], [4]]
    assert _groups(SAMPLE104_CAPTION) == [[0, 1, 2, 3], [4]]


def test_a_straight_multi_line_caption_still_joins():
    caption = {
        "page": (1200, 1600),
        "pieces": [
            (
                f"line-{index}",
                [[100, 100 + 40 * index], [700, 100 + 40 * index], [700, 132 + 40 * index], [100, 132 + 40 * index]],
            )
            for index in range(5)
        ],
    }
    assert _groups(caption) == [[0, 1, 2, 3, 4]]


def test_splitters():
    def q(x, y, w, h):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    # sizes 30, 32, 90: the jump 32 -> 90 is 2.8x
    assert split_by_character_size([q(0, 0, 30, 300), q(40, 0, 32, 300), q(90, 0, 90, 400)], 2.2) == [[0, 1], [2]]
    assert split_by_character_size([q(0, 0, 30, 300), q(40, 0, 50, 300)], 2.2) is None
    assert split_by_orientation([q(0, 0, 30, 300), q(0, 400, 300, 30), q(500, 500, 40, 40)]) == [[0], [1], [2]]
    assert split_by_orientation([q(0, 0, 30, 300), q(40, 0, 30, 300)]) is None
    assert split_off_squares([q(0, 0, 30, 300), q(40, 0, 30, 300), q(80, 0, 40, 40)]) == [[0, 1], [2]]
    assert split_off_squares([q(0, 0, 40, 40)]) is None
    assert has_rotated_lines([quad for _, quad in SAMPLE104_TITLE["pieces"]])
    assert not has_rotated_lines([quad for _, quad in SAMPLE104_CAPTION["pieces"][:4]])


@pytest.mark.parametrize(("env", "expected"), [(None, "True 2.2"), ("false", "False 2.2")])
def test_the_settings_default_on_and_can_be_switched_off(env, expected):
    import os
    import subprocess
    import sys

    import worker

    environment = {key: value for key, value in os.environ.items() if not key.startswith("OCR_NO_BALLOON_")}
    environment["PYTHONPATH"] = os.path.dirname(os.path.dirname(worker.__file__))
    if env is not None:
        environment["OCR_NO_BALLOON_VETO"] = env
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from worker.config import OCR_NO_BALLOON_VETO as v, OCR_NO_BALLOON_SIZE_RATIO as r; print(v, r)",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == expected


def test_the_handler_gives_the_no_balloon_path_the_veto():
    import inspect

    import worker.handlers.ocr as ocr_handler

    source = inspect.getsource(ocr_handler.process_ocr)
    assert "page_context = no_balloon_grouping_context(page_context)" in source


# Fixture sample61 (a game UI page): eight lines of skill text in two blocks side by side, with two
# font sizes. Lines that do not stack neatly are a free layout, not two texts: the first B5 cut
# this page's paragraphs into chunks, which the fixture gate forbids.
SAMPLE61_SKILL_TEXT = {
    "page": (2000, 2000),
    "pieces": [
        ("四五·火水み木い金土本、", [[984, 471], [1408, 471], [1408, 484], [984, 484]]),
        ("金メャモよ土本、土本め人", [[982, 486], [1408, 486], [1408, 506], [982, 506]]),
        ("目手ソ足一せ二三四五そ【", [[986, 508], [1406, 508], [1406, 521], [986, 521]]),
        ("下左ほ右中トナーニい大ま", [[1449, 500], [1617, 500], [1617, 514], [1449, 514]]),
        ("【火水木金】い三四ゆ五六", [[1455, 516], [1713, 516], [1713, 535], [1455, 535]]),
        ("【八九】い2Pt山川田、", [[1457, 537], [1920, 537], [1920, 551], [1457, 551]]),
        ("山川ヨ田日ほ月ま、万上つ", [[984, 541], [1400, 543], [1400, 562], [984, 561]]),
        ("四やゆよ1三をあ四五六七", [[986, 527], [1137, 527], [1137, 541], [986, 541]]),
    ],
}


def test_free_layout_text_keeps_its_distance_grouping():
    assert _groups(SAMPLE61_SKILL_TEXT) == _groups(SAMPLE61_SKILL_TEXT, veto=False) == [[0, 1, 2, 3, 4, 5, 6, 7]]


# #321: two speakers side by side with no balloon between them. Pieces (x, y, w, h) from the owner's
# one-shot `test`, OCR captures of 2026-11-07. Within one speaker the column tops line up; where the
# speaker changes, they jump.
TEST_P16_TWO_SPEAKERS = [
    (386, 387, 29, 78),  # へほまみ
    (367, 387, 27, 140),  # そたミムちつて
    (346, 388, 27, 139),  # にぬねのはひふ？
    (337, 526, 23, 116),  # 千そたちつて？
    (316, 524, 27, 93),  # 足一も二や
    (298, 525, 25, 115),  # 木っかうえやゆ
]
TEST_P5_TWO_SPEAKERS = [
    (634, 46, 35, 234),  # うえおかき八九くけ
    (605, 47, 29, 255),  # 大小むあ山めもやゆよ
    (572, 47, 33, 107),  # 五ら六り
    (540, 46, 35, 283),  # 川田らり日るれろわを？
    (495, 202, 33, 59),  # す～
    (463, 200, 35, 135),  # 田ひ日月ふ
    (428, 199, 40, 185),  # へほま五ら六り
    (397, 199, 40, 183),  # 人口らりるれゃ
    (369, 202, 36, 158),  # をあうえい？
]
TEST_P5_ONE_SPEAKER = [
    (799, 504, 36, 160),  # よらりるれろ
    (762, 501, 44, 165),  # 目み手むめ！
    (734, 503, 40, 213),  # そた中大ちうえつ
    (706, 504, 34, 136),  # 下左右中た
]
TEST_P5_ONE_SPEAKER_LONG = [
    (1010, 467, 29, 252),  # にぬ左ねヲアイ右のは
    (974, 463, 38, 260),  # レロワ十百ちっつてw
    (944, 464, 36, 162),  # うえお九かき
    (912, 463, 37, 186),  # 月火みむめ水も
    (885, 465, 31, 356),  # ミムメモヤユヨ田っかろわをw
]
# test p13, OCR capture of 2026-11-08: a long column, and a short one starting 234 px lower.
TEST_P13_TWO_SPEAKERS = [
    (1847, 41, 63, 576),
    (1763, 275, 88, 205),
]


def _column_quads(boxes):
    return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]] for x, y, w, h in boxes]


def _column_fragments(boxes):
    return [
        {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "text": "あ" * max(1, h // w),
            "confidence": 0.99,
            "fragmentId": f"fragment-{index}",
            "sourceQuad": quad,
            "ownershipProvenance": {"id": f"fragment-{index}", "sourceQuad": quad},
        }
        for index, ((x, y, w, h), quad) in enumerate(zip(boxes, _column_quads(boxes), strict=True))
    ]


def _column_groups(boxes, **context_settings):
    fragments = _column_fragments(boxes)
    context = no_balloon_grouping_context(GroupingContext(page_area=1500.0 * 1500.0), **context_settings)
    groups = group_fragments(fragments, grouping_config("rtl"), context)
    return sorted(sorted(group) for group in groups), fragments


def test_a_step_in_the_column_tops_separates_two_speakers():
    assert split_at_column_steps(_column_quads(TEST_P16_TWO_SPEAKERS), 2.5) == [[0, 1, 2], [3, 4, 5]]
    assert split_at_column_steps(_column_quads(TEST_P5_TWO_SPEAKERS), 2.5) == [[0, 1, 2, 3], [4, 5, 6, 7, 8]]


def test_columns_whose_tops_line_up_are_one_speaker():
    assert split_at_column_steps(_column_quads(TEST_P5_ONE_SPEAKER), 2.5) is None
    assert split_at_column_steps(_column_quads(TEST_P5_ONE_SPEAKER_LONG), 2.5) is None


def test_a_column_ocr_read_as_two_pieces_is_not_a_step():
    # The same column, cut in two along its length, beside a neighbour starting level with it.
    boxes = [(500, 100, 40, 150), (500, 260, 40, 150), (450, 100, 40, 300)]
    assert split_at_column_steps(_column_quads(boxes), 2.5) is None


def test_horizontal_lines_are_never_cut_by_a_column_step():
    # A caption's lines, each starting further along: not columns.
    boxes = [(100, 100, 300, 40), (260, 150, 300, 40), (420, 200, 300, 40)]
    assert split_at_column_steps(_column_quads(boxes), 2.5) is None


def test_a_step_under_the_threshold_does_not_cut():
    # 2 characters of 40 px: an indent, not a new speaker.
    boxes = [(500, 100, 40, 300), (450, 180, 40, 300)]
    assert split_at_column_steps(_column_quads(boxes), 2.5) is None


def test_the_no_balloon_path_separates_the_two_speakers():
    groups, _ = _column_groups(TEST_P16_TWO_SPEAKERS)
    assert groups == [[0, 1, 2], [3, 4, 5]]
    groups, _ = _column_groups(TEST_P5_TWO_SPEAKERS)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7, 8]]


def test_the_no_balloon_path_keeps_one_speaker_whole():
    groups, _ = _column_groups(TEST_P5_ONE_SPEAKER)
    assert groups == [[0, 1, 2, 3]]


def test_the_rule_can_be_switched_off():
    groups, fragments = _column_groups(TEST_P16_TWO_SPEAKERS, column_step=0)
    assert groups == [[0, 1, 2, 3, 4, 5]]
    assert not any("splitBy" in fragment["ownershipProvenance"] for fragment in fragments)


def test_a_group_it_cut_says_so_on_its_pieces():
    _, fragments = _column_groups(TEST_P16_TWO_SPEAKERS)
    assert all(fragment["ownershipProvenance"]["splitBy"] == "column-step" for fragment in fragments)


def test_two_pieces_it_cut_apart_say_so_too():
    # A two-piece group is never offered to owner_split (only one way to cut it), so the veto has to
    # mark it. Live, test p13's pair was cut with no marker anywhere (2026-11-08 sample run).
    groups, fragments = _column_groups(TEST_P13_TWO_SPEAKERS)
    assert groups == [[0], [1]]
    assert all(fragment["ownershipProvenance"]["splitBy"] == "column-step" for fragment in fragments)


def test_two_pieces_kept_together_have_no_marker():
    # Two columns of one speaker, tops level.
    _, fragments = _column_groups([(500, 100, 40, 300), (450, 100, 40, 300)])
    assert not any("splitBy" in fragment["ownershipProvenance"] for fragment in fragments)


def test_a_group_it_did_not_cut_has_no_marker():
    _, fragments = _column_groups(TEST_P5_ONE_SPEAKER)
    assert not any("splitBy" in fragment["ownershipProvenance"] for fragment in fragments)
