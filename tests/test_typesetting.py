import json

from worker.handlers.render import fit_text_in_box_py, load_font


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


def test_shrinks_to_fit_the_width_instead_of_breaking_words():
    """The size search used to test height only.

    A word wider than the line it lands on is split per character, and that split is what made
    every size "fit": the search kept growing the font until the height ran out and returned the
    largest size that mangles the text. Page 22 of Openrouter ch. 11 rendered "collection...??"
    as "collect" / "ion...??" this way, in a box tall enough for far more.
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

    assert " ".join(res["lines"]).split() == ["...How", "did", "I", "become", "a", "target", "for", "collection...??"]
    assert widest_line(res, bold=True) <= 95


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
