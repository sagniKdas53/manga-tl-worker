"""B3b: inside one balloon, the groups a 0.35-character budget leaves apart are joined.

Geometry from the chrome-box captures of 2026-10-06 (worker 1889ac5), drawn on blank pages:

- ja/sample25: one speaker's two-lobed balloon, 「そんなことしないよ…？」 beside 「だって今からサンドローネは…」.
  Two regions before; #231 wants one.
- ja/sample24 b4: two bracket balloons YOLO fused into one blob, 「あなたが補習に来てる子？可愛すぎない！？」
  and 「はじめましてお姉さんたちこんにちは～」: two speakers. By geometry it looks like sample25; only the
  bracket stroke drawn between them (4.3 characters long on the page) tells them apart.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from worker.handlers.ocr import grouping_config, owner_aware_grouping_context
from worker.services.balloon_join import balloon_join
from worker.services.bubble_geometry import bubble_grouping_context, gap_wall
from worker.services.fragment_grouping import group_fragments

SAMPLE25 = {
    "page": (1451, 2048),
    "boxes": [(300, 54, 66, 322), (368, 54, 68, 272), (90, 116, 68, 420), (158, 112, 68, 320)],
    "outline": [
        [421, 0], [332, 0], [276, 54], [245, 126], [171, 90], [129, 90], [98, 103], [58, 144], [39, 194],
        [19, 281], [19, 381], [45, 489], [78, 548], [94, 560], [105, 588], [115, 574], [178, 568],
        [221, 548], [254, 509], [289, 421], [353, 447], [431, 440], [452, 425], [484, 348], [497, 295],
        [498, 140], [471, 60], [446, 23],
    ],
}  # fmt: skip
SAMPLE24_B4 = {
    "page": (2648, 2992),
    "boxes": [(2375, 187, 76, 566), (2448, 192, 73, 556), (2173, 350, 82, 526), (2241, 330, 78, 549)],
    "outline": [
        [2522, 150], [2448, 133], [2380, 135], [2358, 173], [2336, 365], [2316, 368], [2306, 352],
        [2288, 346], [2203, 339], [2185, 351], [2179, 367], [2154, 390], [2143, 435], [2143, 701],
        [2153, 834], [2174, 921], [2204, 913], [2233, 917], [2259, 934], [2322, 929], [2338, 904],
        [2349, 804], [2372, 779], [2400, 783], [2427, 797], [2467, 794], [2499, 802], [2552, 783],
        [2560, 770], [2572, 622], [2572, 285], [2553, 190], [2543, 170],
    ],
    # The right balloon's bracket, down the gap between the two groups.
    "stroke": ((2345, 370), (2345, 760)),
}  # fmt: skip


def _page(case, stroke=None):
    width, height = case["page"]
    gray = np.full((height, width), 250, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(case["outline"], dtype=np.int32)], 255)
    if stroke is not None:
        cv2.line(gray, stroke[0], stroke[1], 120, 5)
    return gray, mask


def _regions(case):
    regions = []
    for index, (x, y, w, h) in enumerate(case["boxes"]):
        quad = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        regions.append(
            {
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
        )
    return regions


def _groups(case, *, budget=1.5, stroke=None, wall=True, max_lines=8, regions=None):
    """Group one balloon the way process_ocr does, with B3b at ``budget`` (0 = off)."""
    gray, mask = _page(case, stroke)
    width, height = case["page"]
    grouping = grouping_config("rtl")
    base = bubble_grouping_context(mask, case["outline"])
    assert base is not None
    base = replace(base, page_area=float(width * height))
    context = owner_aware_grouping_context(base, [{"format": "polygon", "id": "balloon", "points": case["outline"]}])
    if budget > 0:
        test = gap_wall(gray, mask, 3.0) if wall else None
        context = replace(context, group_join=balloon_join(grouping, context, budget, test, max_lines))
    regions = _regions(case) if regions is None else regions
    return sorted(sorted(group) for group in group_fragments(regions, grouping, context))


def test_the_tight_budget_leaves_both_balloons_in_two_groups():
    assert _groups(SAMPLE25, budget=0) == [[0, 1], [2, 3]]
    assert _groups(SAMPLE24_B4, budget=0) == [[0, 1], [2, 3]]


def test_one_speakers_two_lobed_balloon_is_joined():
    assert _groups(SAMPLE25) == [[0, 1, 2, 3]]


def test_two_fused_balloons_look_the_same_without_the_wall():
    """Why the wall exists: by geometry alone, sample24 b4 joins like sample25."""
    assert _groups(SAMPLE24_B4, stroke=SAMPLE24_B4["stroke"], wall=False) == [[0, 1, 2, 3]]


def test_a_stroke_between_the_groups_keeps_two_balloons_apart():
    assert _groups(SAMPLE24_B4, stroke=SAMPLE24_B4["stroke"]) == [[0, 1], [2, 3]]


def test_a_missed_columns_glyphs_are_no_wall():
    """The hole a missed column leaves still holds its glyphs: short marks, broken every character."""
    gray, mask = _page(SAMPLE25)
    for top in range(140, 500, 75):  # one glyph-sized mark per character down the gap
        cv2.rectangle(gray, (262, top), (282, top + 40), 30, -1)
    regions = _regions(SAMPLE25)
    assert gap_wall(gray, mask, 3.0)([0, 1], [2, 3], regions) is False
    cv2.line(gray, (272, 140), (272, 520), 30, 3)
    assert gap_wall(gray, mask, 3.0)([0, 1], [2, 3], regions) is True


def test_a_sloped_outline_between_the_groups_is_a_wall():
    """CodeRabbit on #59: an outline crossing the gap at a slant has no long run in any one column."""
    assert _groups(SAMPLE24_B4, stroke=((2322, 370), (2372, 760))) == [[0, 1], [2, 3]]


def test_an_outline_that_fills_a_narrow_gap_is_still_a_wall():
    """CodeRabbit on #59: measured over the gap alone, a thick outline was its own "paper"."""
    gray, mask = _page(SAMPLE24_B4)
    cv2.rectangle(gray, (2319, 370), (2375, 760), 120, -1)
    assert gap_wall(gray, mask, 3.0)([0, 1], [2, 3], _regions(SAMPLE24_B4)) is True


def test_no_joined_group_holds_more_than_max_lines():
    """ja/sample9: one speaker's three connected balloons of 7, 6 and 3 columns are separate texts."""
    assert _groups(SAMPLE25, max_lines=3) == [[0, 1], [2, 3]]
    assert _groups(SAMPLE25, max_lines=4) == [[0, 1, 2, 3]]


def test_the_joined_groups_owner_decision_is_recorded_on_its_pieces():
    regions = _regions(SAMPLE25)
    assert _groups(SAMPLE25, regions=regions) == [[0, 1, 2, 3]]
    decisions = [region["ownershipProvenance"]["ownerDecision"] for region in regions]
    assert {decision["state"] for decision in decisions} == {"assigned"}
    assert len({decision["owner_id"] for decision in decisions}) == 1


def test_trial_regroupings_do_not_write_decisions_for_pairs_that_stay_apart():
    regions = _regions(SAMPLE24_B4)
    assert _groups(SAMPLE24_B4, stroke=SAMPLE24_B4["stroke"], regions=regions) == [[0, 1], [2, 3]]
    owners = {region["ownershipProvenance"].get("ownerDecision", {}).get("owner_id") for region in regions}
    assert len(owners) != 1  # never the joined pair's single owner


@pytest.mark.parametrize(("env", "expected"), [(None, "1.5 3.0 8"), ("0", "0.0 3.0 8")])
def test_the_settings_default_on_and_can_be_switched_off(env, expected):
    import os
    import subprocess
    import sys

    import worker

    environment = {key: value for key, value in os.environ.items() if not key.startswith("OCR_BALLOON_")}
    environment["PYTHONPATH"] = os.path.dirname(os.path.dirname(worker.__file__))
    if env is not None:
        environment["OCR_BALLOON_JOIN_BUDGET"] = env
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from worker.config import OCR_BALLOON_JOIN_BUDGET as b, OCR_BALLOON_WALL_STROKE as w, "
            "OCR_BALLOON_JOIN_MAX_LINES as m; print(b, w, m)",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == expected


def test_the_handler_gives_the_balloon_path_the_join():
    import inspect

    import worker.handlers.ocr as ocr_handler

    source = inspect.getsource(ocr_handler.process_ocr)
    assert "group_join=balloon_join(" in source
    assert "gap_wall(page_gray, bubble_mask, OCR_BALLOON_WALL_STROKE)" in source
