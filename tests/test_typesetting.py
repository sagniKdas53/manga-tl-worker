import json

from worker.handlers.render import fit_text_in_box_py


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
