"""A two-page spread's gutter: no-balloon text on one page is never grouped with the other's.

ja/sample93 (a 6764x4961 spread, one of the six fixtures) came back with one region holding the
left page's shout and the right page's speech, translated as one jumbled sentence. YOLO found no
balloons there and panel detection found two small panels, so every fragment shared one partition
and proximity chained across the gutter.
"""

import numpy as np

from worker.services.spread_gutter import find_spread_gutter, split_fragments_at_gutter


def _frag(x, y=100, w=60, h=400, text="t"):
    return {"x": x, "y": y, "width": w, "height": h, "text": text}


def _spread(width=1400, height=1000, seam_x=700, line=True):
    """Two flat pages with different tones, joined at ``seam_x``, optionally with a dark gutter line."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    img[:, seam_x:] = 120
    if line:
        img[:, seam_x - 2 : seam_x + 2] = 10
    return img


def test_finds_the_gutter_of_a_spread():
    gutter = find_spread_gutter(_spread(), [_frag(300), _frag(900)])
    assert gutter is not None
    assert abs(gutter - 700) <= 8


def test_portrait_page_has_no_gutter():
    img = _spread(width=900, height=1300, seam_x=450)
    assert find_spread_gutter(img, [_frag(100), _frag(600)]) is None


def test_wide_page_without_a_seam_has_no_gutter():
    img = np.full((1000, 1400, 3), 200, dtype=np.uint8)
    assert find_spread_gutter(img, [_frag(300), _frag(900)]) is None


def test_seam_far_from_the_centre_is_not_a_gutter():
    assert find_spread_gutter(_spread(seam_x=250), [_frag(100), _frag(900)]) is None


def test_seam_broken_over_most_of_the_height_is_not_a_gutter():
    img = np.full((1000, 1400, 3), 200, dtype=np.uint8)
    img[:400, 698:702] = 10  # a pole or a panel edge, not a page boundary
    assert find_spread_gutter(img, [_frag(300), _frag(900)]) is None


def test_a_line_drawn_across_both_pages_turns_the_guard_off():
    # One fragment sits half on each page: the seam runs through the text, so it is art, not a
    # page boundary, and grouping must stay as it was.
    assert find_spread_gutter(_spread(), [_frag(300), _frag(600, w=200)]) is None


def test_balloon_drawn_over_the_gutter_keeps_the_guard():
    # sample93: the left page's shout balloon crosses the gutter line by about a fifth of its
    # width. Its centre is on the left, so it still belongs to the left page.
    assert find_spread_gutter(_spread(), [_frag(300), _frag(560, w=180)]) is not None


def test_split_puts_each_fragment_on_its_centres_side():
    left_a, left_b, right_a = _frag(300), _frag(560, w=180), _frag(900)
    sides = split_fragments_at_gutter([right_a, left_a, left_b], 700)
    assert sides == [[left_a, left_b], [right_a]]


def test_split_without_a_gutter_is_one_partition():
    frags = [_frag(300), _frag(900)]
    assert split_fragments_at_gutter(frags, None) == [frags]


def test_split_drops_an_empty_side():
    frags = [_frag(100), _frag(300)]
    assert split_fragments_at_gutter(frags, 700) == [frags]


def test_columns_either_side_of_the_gutter_stay_apart():
    from worker.handlers.ocr import grouping_config
    from worker.services.merge_regions import merge_ocr_regions

    # Two columns 10px apart join by distance; split at a gutter between them, they cannot.
    columns = [_frag(640, w=50, text="あ"), _frag(700, w=50, text="い")]
    for column in columns:
        column.update({"detectedLanguage": "ja", "confidence": 0.9})
    grouping = grouping_config("rtl", 0.35)
    assert len(merge_ocr_regions(columns, grouping=grouping)) == 1
    pages = split_fragments_at_gutter(columns, 695)
    assert sum(len(merge_ocr_regions(page, grouping=grouping)) for page in pages) == 2


def test_the_handler_splits_no_balloon_text_at_the_gutter():
    import inspect

    import worker.handlers.ocr as ocr_handler

    source = inspect.getsource(ocr_handler.process_ocr)
    assert "find_spread_gutter(" in source
    assert "split_fragments_at_gutter(" in source


def _fallback_region(index, x):
    region = _frag(x, w=50, text="あ")
    quad = [[x, 100], [x + 50, 100], [x + 50, 500], [x, 500]]
    region.update(
        {
            "detectedLanguage": "ja",
            "confidence": 0.9,
            "fragmentId": f"fragment-{index}",
            "sourceQuad": quad,
            "ownershipProvenance": {"id": f"fragment-{index}", "sourceQuad": quad},
        }
    )
    return region


def test_the_fallback_path_groups_each_page_of_a_spread_apart():
    """CodeRabbit on #55: without YOLO, process_ocr groups in its fallback branch, which had no gutter."""
    from worker.handlers.ocr import group_fallback_regions, grouping_config
    from worker.services.fragment_grouping import GroupingContext

    # Two columns 10px apart, one on each side of the seam, and a third on the left page.
    regions = [_fallback_region(0, 640), _fallback_region(1, 700), _fallback_region(2, 580)]
    grouping = grouping_config("rtl", 0.35)
    context = GroupingContext(page_area=1400 * 1000)

    groups, merged = group_fallback_regions(_spread(), regions, grouping, context)
    assert sorted(sorted(group) for group in groups) == [[0, 2], [1]]
    assert len(merged) == 2

    # On a single page the same three columns are one group, as before.
    groups, merged = group_fallback_regions(None, regions, grouping, context)
    assert sorted(sorted(group) for group in groups) == [[0, 1, 2]]
    assert len(merged) == 1
