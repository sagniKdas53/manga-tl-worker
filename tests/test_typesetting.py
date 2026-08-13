import json

import pytest

from worker.handlers.render import (
    BAND_SAMPLE_FRACTIONS,
    break_word_to_width,
    fit_text_in_box_py,
    hyphen_positions,
    load_font,
    mask_span_for_band,
    reassemble,
)


def widest_line(res, font_name="Comic Neue", bold=False):
    font = load_font(res["fontSize"], font_name=font_name, bold=bold)
    # load_font falls back all the way to ImageFont.load_default(), so None means the environment
    # has no usable font at all -- worth failing on loudly rather than measuring against nothing.
    assert font is not None
    return max(font.getlength(line) for line in res["lines"])


def test_fit_text_rectangular():
    text = "Hello world this is a test"
    # Try fitting in a 200x100 rectangular box
    res = fit_text_in_box_py(
        text=text,
        max_width=200,
        max_height=100,
        font_name="Comic Neue",
        default_font_size=16,
        shape="rectangular",
        box_x=10,
        box_y=10,
    )
    assert "fontSize" in res
    assert len(res["lines"]) > 0
    assert not res["overflow"]
    assert len(res["lineCenters"]) == len(res["lines"])
    # All centers should be box_x + max_width / 2 = 10 + 200/2 = 110
    for c in res["lineCenters"]:
        assert abs(c - 110.0) < 1e-3


def test_fit_text_polygon():
    text = "Longer text inside a diamond speech bubble"
    # Create a diamond shape mask polygon
    # Vertices: top (50, 0), right (100, 50), bottom (50, 100), left (0, 50)
    polygon = [[50, 0], [100, 50], [50, 100], [0, 50]]
    mask_polygon_str = json.dumps(polygon)

    res = fit_text_in_box_py(
        text=text,
        max_width=100,
        max_height=100,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=0,
        box_y=0,
        mask_polygon=mask_polygon_str,
    )
    assert "fontSize" in res
    assert len(res["lines"]) > 0
    assert len(res["lineCenters"]) == len(res["lines"])
    # Center X coordinates should be close to 50 (middle of the diamond horizontal span)
    for c in res["lineCenters"]:
        assert abs(c - 50.0) < 10.0


def test_sets_a_long_word_across_two_lines_rather_than_mangling_or_shrinking_it():
    """The size search used to test height only.

    A word wider than the line it lands on was split per character, and that split is what made
    every size "fit": the search kept growing the font until the height ran out and returned the
    largest size that mangles the text. Page 22 of Openrouter ch. 11 rendered "collection...??"
    as "collect" / "ion...??" this way, in a box tall enough for far more.

    **Contract changed 2026-08-13 (D7 hyphenation).** The answer to an over-wide word used to be
    to shrink the type until it fit whole, which in this 95px box means 12px. It is now to break
    it at a legal point and carry a hyphen, which fits the same box at 21px. So the rule is no
    longer "never split a word" -- it is "never split a word *illegally*": whatever comes out must
    reassemble into exactly what went in.
    """
    res = fit_text_in_box_py(
        text="...How did I become a target for collection...??",
        max_width=95,
        max_height=1180,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=45,
        box_y=298,
        bold=True,
    )

    assert reassemble(res["lines"]) == ["...How", "did", "I", "become", "a", "target", "for", "collection...??"]
    assert widest_line(res, bold=True) <= 95
    assert res["fontSize"] > 12, "must use the hyphen to grow, not shrink to fit the word whole"
    assert any(line.endswith("-") for line in res["lines"])


def test_grows_past_the_old_width_over_three_cap_in_a_narrow_tall_box():
    """D7 (docs/render_quality_gap_2026-08-05.md): `max_width // 3` capped the search before it
    ever ran, on the assumption that a line needs roughly 3 characters' worth of width. That
    punished tall narrow boxes -- the shape of a vertical Japanese speech bubble -- far more than
    wide ones. sample1's `safe_text_w=145, safe_text_h=259` bubble capped at 48px under the old
    rule (`min(259//2, 145//3, 72)`); short text in the same box should now clear that.
    """
    res = fit_text_in_box_py(
        text="Big bro...",
        max_width=145,
        max_height=259,
        font_name="Comic Neue",
        default_font_size=16,
        shape="rectangular",
        box_x=0,
        box_y=0,
    )

    assert res["fontSize"] > 48
    assert not res["overflow"]
    assert widest_line(res) <= 145


def test_prefers_a_smaller_size_over_a_line_that_overhangs_the_box():
    """The polygon and ellipse paths fall back to a wrap that ignores their own line spans.

    That fallback keeps words whole but wraps to the full box width, so a size whose lines spill
    sideways out of the bubble used to report exactly the same "fits" as one that does not.
    """
    diamond = json.dumps([[50, 0], [100, 50], [50, 100], [0, 50]])

    res = fit_text_in_box_py(
        text="Immediately hand over your belongings",
        max_width=100,
        max_height=100,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=0,
        box_y=0,
        mask_polygon=diamond,
    )

    assert widest_line(res) <= 100
    assert len(res["lines"]) * res["fontSize"] * 1.2 <= 100


def test_still_sets_text_that_cannot_fit_cleanly_at_the_largest_legible_size():
    """A single word wider than the box at 6px has no clean layout. Splitting it is then the only
    option left, and the text should still get the largest size that fits rather than collapse."""
    res = fit_text_in_box_py(
        text="Onomatopoeia",
        max_width=18,
        max_height=300,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=0,
        box_y=0,
    )

    assert res["fontSize"] > 6
    assert len(res["lines"]) > 1


def test_ignores_a_mask_that_is_narrower_than_the_box_it_flows_into():
    """The mask does two jobs: it is painted over the source text, and it is the shape lines wrap
    to. For free-floating text the mask is the tight rectangle round the vertical Japanese column,
    so wrapping to it would set English straight back down that column and undo the box."""
    ink_column = json.dumps([[71, 675], [120, 675], [120, 1164], [71, 1164]])

    res = fit_text_in_box_py(
        text="...How did I become a target for collection...??",
        max_width=163,
        max_height=185,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=13,
        box_y=822,
        mask_polygon=ink_column,
        bold=True,
    )

    assert widest_line(res, bold=True) > 49, "must use the box it was given, not the ink column"
    assert widest_line(res, bold=True) <= 163


def test_keeps_using_a_mask_that_contains_the_box():
    """A balloon's mask is bigger than the inset box drawn into it, and there it still shapes the
    text -- that is what makes lines follow the outline instead of squaring off inside it."""
    balloon = json.dumps([[0, 0], [200, 0], [200, 200], [0, 200]])

    res = fit_text_in_box_py(
        text="Wait a second please",
        max_width=180,
        max_height=180,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=10,
        box_y=10,
        mask_polygon=balloon,
    )

    assert len(res["lineCenters"]) == len(res["lines"])


def ellipse_polygon(cx, cy, rx, ry, n=64):
    """A closed elliptical mask, the shape most speech balloons actually are."""
    import math

    return [[cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)] for i in range(n)]


def span_over_band(polygon, box_x, max_width, y_top, y_bottom):
    """The span available across a band, measured independently of the renderer's own sampling.

    Deliberately does not go through `BAND_SAMPLE_FRACTIONS`: a rod that moves with the code
    under test cannot catch the code moving. Each call below asks for a single row (a degenerate
    band), so the fractions cannot affect where it samples.
    """
    left = None
    right = None
    for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.85):
        y = y_top + t * (y_bottom - y_top)
        row = mask_span_for_band(polygon, box_x, max_width, y, y)
        if row is None:
            continue
        left = row[0] if left is None else max(left, row[0])
        right = row[1] if right is None else min(right, row[1])
    if left is None or right is None or right <= left:
        return None
    return (left, right)


def lines_outside_mask(res, polygon, box_x, box_y, max_width, max_height, font_name="Comic Neue"):
    """Lines whose drawn extent leaves the mask, measured the way the renderer draws them."""
    font = load_font(res["fontSize"], font_name=font_name)
    assert font is not None
    line_height = res["fontSize"] * 1.2
    y_start = box_y + (max_height - len(res["lines"]) * line_height) / 2
    outside = []
    for idx, line in enumerate(res["lines"]):
        if not line.strip():
            continue
        width = font.getlength(line)
        center = res["lineCenters"][idx]
        top = y_start + idx * line_height
        span = span_over_band(polygon, box_x, max_width, top, top + line_height)
        if span is None:
            continue
        if center - width / 2 < span[0] - 0.5 or center + width / 2 > span[1] + 0.5:
            outside.append(line)
    return outside


def test_mask_span_is_measured_over_the_line_band_not_its_centreline():
    """D6: a curved balloon closes in above and below a line's midline, so the span at the centre
    row overstates what the line can use. sample1's bottom-left balloon is the measured case."""
    diamond = [[50, 0], [100, 50], [50, 100], [0, 50]]

    # A band from y=30 to y=54, on a diamond that is widest at its waist (y=50). The narrowest
    # ink row in that band is its top, where the diamond is 60 wide; its centre row is 84 wide.
    narrowest_row = mask_span_for_band(diamond, 0, 100, 33.6, 33.6)
    band = mask_span_for_band(diamond, 0, 100, 30, 54)

    assert narrowest_row is not None and band is not None
    assert band[1] - band[0] == pytest.approx(narrowest_row[1] - narrowest_row[0])


def test_no_line_escapes_an_oval_balloon():
    """The D6 regression, in the shape that produced it: an oval mask, a box that is its bounding
    rectangle, and text long enough to want the full width. Every line has to stay inside the
    outline over its own band, not merely inside the box."""
    polygon = ellipse_polygon(100, 130, 90, 120)
    box_x, box_y, box_w, box_h = 10, 10, 180, 240

    res = fit_text_in_box_py(
        text="I'll cover for us with Mom and the others.",
        max_width=box_w,
        max_height=box_h,
        font_name="Comic Neue",
        default_font_size=16,
        shape="elliptical",
        box_x=box_x,
        box_y=box_y,
        mask_polygon=json.dumps(polygon),
    )

    assert lines_outside_mask(res, polygon, box_x, box_y, box_w, box_h) == []
    # And it has not paid for that by collapsing the type -- the box is 240 tall.
    assert res["fontSize"] >= 20


def test_band_sampling_spans_the_ink_not_the_leading():
    """The band is the ascender-to-descender run, not the whole 1.2em line box. Sampling the full
    box would let a tail row that no glyph reaches veto a line that fits."""
    assert min(BAND_SAMPLE_FRACTIONS) > 0.0
    assert max(BAND_SAMPLE_FRACTIONS) < 1.0
    assert 0.5 in BAND_SAMPLE_FRACTIONS


def test_breaks_a_word_rather_than_setting_lines_wider_than_the_box():
    """When no size sets the text whole, the text still has to stay inside its box.

    The fallback used to test height alone, so it grew the type until the *height* ran out while
    the lines got arbitrarily wider than the region. sample27's "(PLAYFUL RETRACTION)" gloss is a
    44px-wide box that came out at 43px type -- lines twice the width of the box, drawn across the
    neighbouring balloon. A broken word is ugly and stays put; overflow lands on someone else's
    panel.
    """
    res = fit_text_in_box_py(
        text="...Just kidding! (PLAYFUL RETRACTION)",
        max_width=44,
        max_height=208,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=515,
        box_y=972,
        bold=True,
    )

    assert widest_line(res, bold=True) <= 44
    assert res["fontSize"] < 43


def test_hyphen_positions_ignores_surrounding_punctuation():
    """Positions come from the word's letters, not from whatever punctuation is stuck to it.

    Measured against the raw token, pyphen counts a leading quote toward its two-character
    minimum and offers `"O-taku-kun` -- a break after a single letter of the actual word.
    """
    assert hyphen_positions("Opportunities") == [2, 5, 7, 9]
    assert all(p >= 3 for p in hyphen_positions('"Otaku-kun'))
    assert hyphen_positions("don't") == []
    assert hyphen_positions("a") == []


def test_break_word_prefers_a_legal_point_and_carries_the_hyphen():
    font = load_font(20)
    assert font is not None

    head, tail = break_word_to_width("Opportunities", font.getlength, font.getlength("Opportun"))

    assert head.endswith("-")
    assert head[:-1] + tail == "Opportunities"
    assert font.getlength(head) <= font.getlength("Opportun")


def test_break_word_still_splits_when_no_legal_point_fits():
    """A box narrower than the shortest legal head has no good answer, and the old
    character-splitting behaviour is the least bad one -- better than a line outside the region."""
    font = load_font(20)
    assert font is not None

    head, tail = break_word_to_width("Opportunities", font.getlength, font.getlength("Op"))

    assert not head.endswith("-")
    assert head + tail == "Opportunities"


def test_reassemble_puts_a_hyphenated_word_back_but_not_a_mangled_one():
    assert reassemble(["Op-", "portunities"]) == ["Opportunities"]
    assert reassemble(["Otaku-", "kun"]) == ["Otakukun"]
    assert reassemble(["collect", "ion"]) == ["collect", "ion"]
    assert reassemble(["two", "words"]) == ["two", "words"]


def test_hyphenation_fills_the_balloon_a_long_word_used_to_starve():
    """D7's underfill: 190 of 351 corpus elements were held at median 0.38 fill by one word that
    would not fit the width. sample1's third balloon is the shape of it."""
    res = fit_text_in_box_py(
        text="Opportunities like this don't come often... You're excited, right?",
        max_width=125,
        max_height=207,
        font_name="Comic Neue",
        default_font_size=12,
        shape="rectangular",
        box_x=0,
        box_y=0,
    )

    fill = len(res["lines"]) * res["fontSize"] * 1.2 / 207
    assert fill > 0.7, f"balloon still two-thirds empty (fill={fill:.2f})"
    assert reassemble(res["lines"]) == [
        w.replace("-", "")
        for w in ["Opportunities", "like", "this", "don't", "come", "often...", "You're", "excited,", "right?"]
    ]


def test_lettering_stays_legible_on_a_backdrop_sampled_from_artwork():
    """R2 gave the renderer backdrops it never used to see.

    A region's text colour is decided without reference to its fill and defaults to black, so a
    covering balloon sampled from a dark panel arrived as black text on near-black. sample10's
    bottom-left corner: white-stroked lettering on a dark panel, no balloon, dominant colour
    #33272d.
    """
    from worker.handlers.render import readable_text_color

    assert readable_text_color("#33272d", "#000000") == "#ffffff"
    assert readable_text_color("#ffffff", "#000000") == "#000000", "already legible, leave it"
    assert readable_text_color("#fddd54", "#000000") == "#000000", "black on the yellow is fine"
    assert readable_text_color("#101010", "#ffffff") == "#ffffff", "already legible, leave it"


def test_a_deliberate_low_contrast_pairing_is_left_alone():
    """The floor is legibility, not taste. Mid-grey on white is a choice somebody could have made;
    black on black is not."""
    from worker.handlers.render import readable_text_color

    assert readable_text_color("#ffffff", "#767676") == "#767676"
    assert readable_text_color(None, "#000000") == "#000000", "no fill: nothing to contrast against"
    assert readable_text_color("not-a-colour", "#000000") == "#000000"
