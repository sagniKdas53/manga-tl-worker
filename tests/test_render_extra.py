import io
import json
from unittest.mock import MagicMock, patch

from worker.handlers.render import (
    draw_wrapped_text,
    fit_text_in_box_py,
    halo_stroke_for,
    has_detected_bubble,
    load_font,
    process_render,
    wrap_text,
)


def test_load_font_registry_hit():
    with (
        patch("worker.handlers.render.os.path.exists", return_value=True),
        patch("worker.handlers.render.ImageFont.truetype") as mock_tt,
    ):
        mock_font = MagicMock()
        mock_tt.return_value = mock_font
        font = load_font(12, "Comic Neue", bold=True)
        assert font == mock_font
        mock_tt.assert_called_with("/usr/share/fonts/opentype/comic-neue/ComicNeue-Bold.otf", 12)


def test_load_font_registry_miss_fallback():
    with (
        patch("worker.handlers.render.os.path.exists", return_value=True),
        patch("worker.handlers.render.ImageFont.truetype") as mock_tt,
    ):
        mock_font = MagicMock()
        mock_tt.side_effect = [Exception("error"), mock_font]
        font = load_font(12, "UnknownFont")
        assert font == mock_font


def test_load_font_default_fallback():
    with (
        patch("worker.handlers.render.os.path.exists", return_value=False),
        patch("worker.handlers.render.ImageFont.truetype", side_effect=Exception("error")),
        patch("worker.handlers.render.ImageFont.load_default") as mock_def,
    ):
        mock_font = MagicMock()
        mock_def.return_value = mock_font
        font = load_font(12, "UnknownFont")
        assert font == mock_font


def test_wrap_text_empty():
    assert wrap_text("", MagicMock(), 100) == []


def test_wrap_text_with_getbbox():
    mock_font = MagicMock()

    # word "hello" fits, "world" fits, but together > 100
    def mock_getbbox(text):
        length = len(text) * 10
        return (0, 0, length, 10)

    mock_font.getbbox.side_effect = mock_getbbox

    lines = wrap_text("hello world longworddddddddddddddddddd", mock_font, 100)
    assert lines == ["hello", "world", "longworddddddddddddddddddd"]


def test_wrap_text_with_exception():
    mock_font = MagicMock()
    mock_font.getbbox.side_effect = Exception("error")
    mock_font.getsize.side_effect = Exception("error")
    # fallbacks to len(text) * 6
    lines = wrap_text("hello world", mock_font, 100)
    assert lines == ["hello world"]


def test_draw_wrapped_text():
    mock_draw = MagicMock()
    mock_font = MagicMock()
    mock_font.getbbox.return_value = (0, 0, 50, 10)

    draw_wrapped_text(mock_draw, "hello world", mock_font, "#000", 0, 0, 100, 100)
    assert mock_draw.text.called


def test_fit_text_in_box_elliptical():
    res = fit_text_in_box_py("hello world", 100, 100, "Comic Neue", shape="elliptical")
    assert res["fontSize"] > 0
    assert len(res["lines"]) > 0


def test_fit_text_in_box_polygon_complex():
    polygon = [[10, 10], [90, 10], [90, 90], [10, 90]]
    res = fit_text_in_box_py(
        "hello world",
        100,
        100,
        "Comic Neue",
        shape="rectangular",
        mask_polygon=json.dumps(polygon),
    )
    assert res["fontSize"] > 0


@patch("worker.handlers.render.requests")
@patch("worker.handlers.render.redis_client")
@patch("worker.handlers.render.render_image_core")
def test_process_render_qa_mode_llm(mock_render_core, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_render_core.return_value = True

    with patch("worker.config.QA_MODE", "llm"):
        process_render({"imageId": "123"})
        mock_requests.post.assert_called()


@patch("worker.handlers.render.requests")
@patch("worker.handlers.render.minio_client")
@patch("worker.handlers.render.download_image")
@patch("worker.handlers.render.os.makedirs")
@patch("builtins.open")
def test_process_render_success(mock_open, mock_makedirs, mock_download, mock_minio, mock_requests):
    from PIL import Image

    mock_redis = MagicMock()
    mock_redis.llen.return_value = 0

    # Create a real small image
    img = Image.new("RGB", (100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mock_download.return_value = buf.getvalue()

    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "layerElements": [
            {
                "visible": True,
                "layerVisible": True,
                "layerType": "translation",
                "text": "hello",
                "boxShape": "elliptical",
                "x": 10,
                "y": 10,
                "maxWidth": 80,
                "maxHeight": 80,
                "backgroundColor": "#ffffff",
                "textColor": "#000000",
            },
            {
                "visible": True,
                "layerVisible": True,
                "layerType": "sfx",
                "text": "BOOM",
                "boxShape": "rectangular",
                "x": 20,
                "y": 20,
                "maxWidth": 50,
                "maxHeight": 50,
                "backgroundColor": "#ff0000",
                "maskPolygon": json.dumps([[20, 20], [70, 20], [70, 70], [20, 70]]),
            },
        ]
    }
    mock_requests.get.return_value = mock_res

    with patch("worker.config.QA_MODE", "none"):
        process_render({"imageId": "123"})
        # Should exit early in none mode

    with patch("worker.config.QA_MODE", "normal"):
        process_render({"imageId": "123", "pageNumber": 1, "chapterNumber": 1})
        assert mock_minio.put_object.called


@patch("worker.handlers.render.requests")
def test_process_render_fail_api(mock_requests):
    import pytest

    mock_res = MagicMock()
    mock_res.status_code = 500
    mock_requests.get.return_value = mock_res
    with (
        patch("worker.config.QA_MODE", "normal"),
        pytest.raises(Exception, match=r".*"),
    ):
        process_render({"imageId": "123"})


# The real geometry from HKXfexLbAAAN7IE p4, the caption that reads "Talk about peak laziness":
# written down the character's hair with no balloon at all, so the region came back as
# direct_text_0 at confidence 0.0 and `free_text_box` squared its 91x293 column into a 186x187 box
# anchored at x=0.
IUNO_P4_PLATE = [[2, 1005], [93, 1005], [93, 1298], [2, 1298]]
IUNO_P4_BOX = (0, 1058, 186, 187)


def test_halo_fires_when_the_widened_box_escapes_its_plate():
    assert halo_stroke_for(IUNO_P4_PLATE, IUNO_P4_BOX, 37) == 4


def test_halo_accepts_the_polygon_as_a_json_string():
    assert halo_stroke_for(json.dumps(IUNO_P4_PLATE), IUNO_P4_BOX, 37) == 4


def test_no_halo_when_the_box_is_an_inset_of_the_plate():
    """The plate covers the box, so every glyph already has a backdrop."""
    plate = [[100, 40], [240, 40], [240, 260], [100, 260]]
    assert halo_stroke_for(plate, (110, 50, 120, 200), 30) == 0


def test_no_halo_inside_a_detected_balloon_even_when_the_box_escapes_the_mask():
    """A bubble's mask hugs the glyphs while the box is the bubble inset, so the box escapes on
    nearly every balloon element -- and what it escapes onto is blank balloon interior."""
    glyph_mask = [[150, 80], [190, 80], [190, 220], [150, 220]]
    assert halo_stroke_for(glyph_mask, (110, 50, 120, 200), 30) > 0
    assert halo_stroke_for(glyph_mask, (110, 50, 120, 200), 30, in_bubble=True) == 0


def test_has_detected_bubble_reads_the_bbox_echo_as_no_bubble():
    # p4's caption: the worker copies the bbox into bubble* when YOLO matched no balloon.
    assert not has_detected_bubble(
        {"bubbleW": 91, "bubbleH": 293, "bboxW": 91, "bboxH": 293, "bubbleId": "direct_text_0"}
    )
    # p4's first balloon, bubble_1 at 0.96.
    assert has_detected_bubble({"bubbleW": 122, "bubbleH": 220, "bboxW": 67, "bboxH": 114, "bubbleId": "bubble_1"})
    assert not has_detected_bubble({"bboxW": 91, "bboxH": 293})


def test_no_halo_when_the_box_only_grazes_the_plate_edge():
    """Two pixels of slack, so rounding in the geometry chain does not outline a contained box."""
    plate = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert halo_stroke_for(plate, (-1, -1, 102, 102), 30) == 0
    assert halo_stroke_for(plate, (-4, 0, 100, 100), 30) > 0


def test_halo_when_nothing_was_erased_at_all():
    assert halo_stroke_for(None, IUNO_P4_BOX, 20) == 2
    assert halo_stroke_for([], IUNO_P4_BOX, 20) == 2


def test_halo_scales_with_font_size_but_never_hairline():
    assert halo_stroke_for(IUNO_P4_PLATE, IUNO_P4_BOX, 80) == 8
    assert halo_stroke_for(IUNO_P4_PLATE, IUNO_P4_BOX, 10) == 2
    assert halo_stroke_for(IUNO_P4_PLATE, IUNO_P4_BOX, 0) == 0


def test_malformed_polygon_does_not_halo_or_raise():
    assert halo_stroke_for("not json", IUNO_P4_BOX, 30) == 0
    assert halo_stroke_for([["a", "b"]], IUNO_P4_BOX, 30) == 0


# ---------------------------------------------------------------------------
# AUDIT-R5 — an element's angle
# ---------------------------------------------------------------------------


def test_rotate_point_deg_turns_clockwise_like_the_editor():
    """The editor is SVG/canvas: y grows downward, so a positive angle turns clockwise on screen.

    Getting this backwards is invisible in a unit test that only checks magnitudes, and shows up
    on a page as text leaning the opposite way from the plate it is supposed to sit on.
    """
    from worker.handlers.render import rotate_point_deg

    # A point directly above the centre swings to the centre's right at +90°.
    x, y = rotate_point_deg(0.0, -10.0, 0.0, 0.0, 90.0)
    assert round(x, 6) == 10.0
    assert round(y, 6) == 0.0

    # ...and to its left at -90°.
    x, y = rotate_point_deg(0.0, -10.0, 0.0, 0.0, -90.0)
    assert round(x, 6) == -10.0
    assert round(y, 6) == 0.0


def test_box_outline_points_returns_the_unrotated_rect_when_flat():
    from worker.handlers.render import box_outline_points

    assert box_outline_points(10, 20, 100, 50, 0) == [
        (10, 20),
        (110, 20),
        (110, 70),
        (10, 70),
    ]


def test_box_outline_points_turns_the_rect_about_its_own_centre():
    from worker.handlers.render import box_outline_points

    points = box_outline_points(0, 0, 100, 20, 90)
    # A 90° turn swaps the extents about the centre (50, 10): a 100x20 box becomes 20x100.
    xs = [round(px, 6) for px, _ in points]
    ys = [round(py, 6) for _, py in points]
    assert min(xs) == 40.0 and max(xs) == 60.0
    assert min(ys) == -40.0 and max(ys) == 60.0


def test_box_outline_points_samples_an_ellipse_rather_than_its_bounding_box():
    from worker.handlers.render import box_outline_points

    points = box_outline_points(0, 0, 100, 40, 0, elliptical=True, segments=16)
    assert len(points) == 16
    # Every sampled point is on the ellipse, so none sits at a bounding-box corner.
    for px, py in points:
        norm = ((px - 50) / 50.0) ** 2 + ((py - 20) / 20.0) ** 2
        assert round(norm, 6) == 1.0


def test_element_rotation_ignores_noise_and_junk():
    """Below a tenth of a degree an element is treated as flat, so the fast path stays the default."""
    from worker.handlers.render import element_rotation

    assert element_rotation({}) == 0.0
    assert element_rotation({"rotation": None}) == 0.0
    assert element_rotation({"rotation": "not a number"}) == 0.0
    assert element_rotation({"rotation": 0.05}) == 0.0
    assert element_rotation({"rotation": float("nan")}) == 0.0
    assert element_rotation({"rotation": 37.5}) == 37.5
    assert element_rotation({"rotation": "37.5"}) == 37.5


# ---------------------------------------------------------------------------
# AUDIT-R1 / AUDIT-F16 — one fitted rectangle, shared with the frontend
# ---------------------------------------------------------------------------


def test_text_fit_box_matches_the_frontend_parity_table():
    """The same table is asserted in `frontend/src/__tests__/utils/textFitBox.test.ts`.

    The two implementations are in different languages and nothing else can catch them drifting
    apart -- which is exactly what happened when each side owned its own literal: the live reader
    used the raw box, the frontend's exports insetted 4px, and render.py insetted 4px and then took
    95%. Three rectangles, and the one on screen was not the one that shipped.
    """
    from worker.handlers.render import text_fit_box

    cases = [
        # width, height, padding, safety, expected_w, expected_h
        (300, 120, 4, 95, 277, 106),
        (100, 40, 4, 95, 87, 30),
        (91, 293, 4, 95, 78, 270),
        (50, 50, 0, 100, 50, 50),
        (9, 9, 4, 95, 1, 1),
        (1, 1, 4, 95, 1, 1),
    ]
    for width, height, padding, safety, expected_w, expected_h in cases:
        _, _, box_w, box_h = text_fit_box(0, 0, width, height, padding, safety)
        assert (box_w, box_h) == (expected_w, expected_h), f"{width}x{height} @ {padding}/{safety}"


def test_text_fit_box_reproduces_the_old_literals():
    """The default inset is exactly what the pipeline used before it was configurable."""
    from worker.handlers.render import text_fit_box

    box_x, box_y, box_w, box_h = text_fit_box(100, 200, 300, 120, 4, 95)
    assert (box_x, box_y) == (104, 204)
    assert box_w == int((300 - 8) * 0.95)
    assert box_h == int((120 - 8) * 0.95)


def test_text_fit_box_never_insets_a_box_away_to_nothing():
    from worker.handlers.render import text_fit_box

    _, _, box_w, box_h = text_fit_box(0, 0, 6, 6, 4, 95)
    assert box_w > 0 and box_h > 0


def test_resolve_text_box_inset_defaults_and_clamps():
    """A job payload is not a trusted source of numbers: a 0% safety margin fits everything into
    a zero-width box, which would stop the whole library typesetting."""
    from worker.handlers.render import resolve_text_box_inset

    assert resolve_text_box_inset(None) == (4, 95)
    assert resolve_text_box_inset({}) == (4, 95)
    assert resolve_text_box_inset({"textBoxPaddingPx": 12, "textBoxSafetyPercent": 80}) == (12, 80)
    assert resolve_text_box_inset({"textBoxPaddingPx": "oops"}) == (4, 95)
    assert resolve_text_box_inset({"textBoxSafetyPercent": 0}) == (4, 1)
    assert resolve_text_box_inset({"textBoxSafetyPercent": 500}) == (4, 100)
    assert resolve_text_box_inset({"textBoxPaddingPx": -5}) == (0, 95)
    assert resolve_text_box_inset({"textBoxPaddingPx": 9999}) == (64, 95)
