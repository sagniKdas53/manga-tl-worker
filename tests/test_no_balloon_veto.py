"""B4 + B5: text no balloon holds gets the owner decision's line checks, without the balloon.

Geometry from the chrome-box captures of 2026-10-06 (worker df87949), the pieces' OCR quads:

- ja/sample218 p. 9: YOLO found no balloon around the left speech, so its three columns took the
  no-balloon path and chained with four misreads of the art into one 1140 x 1995 region, typeset
  under the balloon.
- ja/sample83: two upright columns of one text, with a square misread "M" and a tilted はぁ nearby.
- ja/sample104 p. 19: a title on tilted tiles (one text), and a four-line caption whose lines were
  split to singletons because the tall 祝発売 beside them made the group oversized.
"""

import pytest

from worker.handlers.ocr import grouping_config, no_balloon_grouping_context
from worker.services.fragment_grouping import GroupingContext, group_fragments
from worker.services.owner_assignment import (
    has_rotated_lines,
    split_by_character_size,
    split_by_orientation,
    split_off_squares,
)

SAMPLE218_P9 = {
    "page": (2844, 4014),
    "pieces": [
        ("ああっ…♡ううっっっ♡", [[122, 2093], [216, 2093], [223, 2885], [133, 2885]]),
        ("クリやめっ…♡", [[200, 2093], [294, 2093], [294, 2560], [200, 2560]]),
        ("これだめっ…！♡", [[282, 2101], [376, 2101], [376, 2630], [282, 2630]]),
        ("(gftgs grgitgt", [[466, 2062], [698, 2089], [600, 2956], [368, 2928]]),
        ("谷", [[627, 2042], [878, 2031], [890, 2379], [639, 2391]]),
        ("BOFE", [[118, 2909], [408, 2952], [263, 3904], [0, 3861]]),
    ],
}
SAMPLE83 = {
    "page": (1412, 2000),
    "pieces": [
        ("くそ…むっちゃ痛いけど", [[119, 186], [209, 186], [209, 1033], [119, 1033]]),
        ("我慢しなきゃ…", [[25, 225], [117, 225], [117, 721], [25, 721]]),
        ("M", [[258, 719], [314, 719], [314, 775], [258, 775]]),
        ("はぁ，", [[318, 561], [389, 537], [436, 680], [365, 703]]),
    ],
}
SAMPLE104_TITLE = {
    "page": (1474, 1989),
    "pieces": [
        ("性格最悪", [[1373, 1496], [1428, 1669], [1375, 1686], [1321, 1511]]),
        ("強制", [[1154, 1533], [1251, 1480], [1340, 1647], [1243, 1700]]),
        ("高慢女が", [[1288, 1515], [1338, 1501], [1379, 1670], [1329, 1682]]),
        ("無様", [[1208, 1624], [1472, 1657], [1472, 1849], [1187, 1816]]),
        ("セ", [[1323, 1801], [1430, 1801], [1430, 1890], [1323, 1890]]),
    ],
}
SAMPLE104_CAPTION = {
    "page": (1474, 1989),
    "pieces": [
        ("【下品低音オホ声】", [[19, 140], [489, 82], [497, 150], [29, 208]]),
        ("いじめっ子女を徹底的に", [[31, 206], [1165, 58], [1173, 130], [41, 280]]),
        ("キモオタチンポに腰へコ", [[39, 276], [1179, 132], [1187, 200], [49, 346]]),
        ("ド変態マンコ女になるまで", [[49, 342], [763, 256], [771, 328], [58, 416]]),
        ("祝発売", [[1043, 206], [1410, 188], [1447, 946], [1080, 963]]),
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
    """The caption and 祝発売 were one oversized group, halved down to single lines. Cut by
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
        ("重擎·蒼刻の型を発動時、", [[984, 471], [1408, 471], [1408, 484], [984, 484]]),
        ("場キャラの場合、今国の重", [[982, 486], [1408, 486], [1408, 506], [982, 506]]),
        ("重撃・蒼剣の型終了時に【", [[986, 508], [1406, 508], [1406, 521], [986, 521]]),
        ("強力な重掣ダメージを与え", [[1449, 500], [1617, 500], [1617, 514], [1449, 514]]),
        ("【虚减効果】を自身の有利", [[1455, 516], [1713, 516], [1713, 535], [1455, 535]]),
        ("【蒼铜】を2Pt所持時、", [[1457, 537], [1920, 537], [1920, 551], [1457, 551]]),
        ("重撃・置刻の型は、命中し", [[984, 541], [1400, 543], [1400, 562], [984, 561]]),
        ("秒ごとに1回のみ握得可能", [[986, 527], [1137, 527], [1137, 541], [986, 541]]),
    ],
}


def test_free_layout_text_keeps_its_distance_grouping():
    assert _groups(SAMPLE61_SKILL_TEXT) == _groups(SAMPLE61_SKILL_TEXT, veto=False) == [[0, 1, 2, 3, 4, 5, 6, 7]]
