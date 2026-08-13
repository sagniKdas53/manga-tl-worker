import cv2
import numpy as np

from worker.handlers.ocr import (
    detect_background_color,
    detect_background_color_poly,
    detect_bubble_contour,
    get_split_polygon,
)


def test_detect_background_color():
    # Create a 100x100 BGR image with light gray background (#e0e0e0)
    img = np.full((100, 100, 3), 224, dtype=np.uint8)  # 224 BGR -> #e0e0e0
    # Draw some black text strokes in the center (foreground)
    cv2.putText(img, "TEST", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Run background color detection
    color = detect_background_color(img, 20, 20, 60, 60)
    # The borders of the 60x60 region at (20,20) should be untouched by the text and have value #e0e0e0
    assert color is not None
    assert color.lower() == "#e0e0e0"


def test_detect_background_color_refuses_textured_border():
    # Random noise stands in for real art/photo detail: no flat colour to sample, so the caller
    # should get None (skip the fill) rather than a misleading median (D1/D3). A checkerboard
    # would not exercise this properly -- an exact 50/50 split sits right at the median's
    # breakdown point, where a slight sampling imbalance collapses MAD back to 0.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    assert detect_background_color(img, 20, 20, 60, 60) is None


def test_detect_background_color_poly_refuses_textured_interior():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    assert detect_background_color_poly(img, poly) is None


def test_detect_bubble_contour():
    # Create a 200x200 BGR image with gray background (#808080)
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Draw a white speech bubble (filled circle at 100,100 with radius 40)
    cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
    # Draw some black text in the center
    cv2.putText(img, "TXT", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # OCR bounding box of text is around (80, 95, 40, 20)
    bubble_box = detect_bubble_contour(img, 80, 95, 40, 20)

    assert bubble_box is not None
    # Bounding box of a circle centered at 100,100 with radius 40 should be close to (60, 60, 80, 80)
    assert abs(bubble_box["x"] - 60) <= 5
    assert abs(bubble_box["y"] - 60) <= 5
    assert abs(bubble_box["width"] - 80) <= 5
    assert abs(bubble_box["height"] - 80) <= 5
    assert len(bubble_box["maskPolygon"]) >= 3


def test_detect_background_color_poly():
    # Create a 100x100 BGR image with light gray background (#e0e0e0)
    img = np.full((100, 100, 3), 224, dtype=np.uint8)  # 224 BGR -> #e0e0e0
    # Draw some black text strokes in the center (foreground)
    cv2.circle(img, (50, 50), 10, (10, 10, 10), -1)

    # Polygon mask for the whole region
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    color = detect_background_color_poly(img, poly)
    assert color is not None
    assert color.lower() == "#e0e0e0"


def test_split_polygon_and_safe_area():
    # Create a 200x200 binary mask with two white circles (representing two speech bubbles)
    # Circle 1: center (50, 50), radius 30
    # Circle 2: center (150, 150), radius 30
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)  # type: ignore
    cv2.circle(mask, (150, 150), 30, 255, -1)  # type: ignore

    # We want to split out Circle 1 using a bounding box around it
    bbox = [30, 30, 40, 40]
    poly = get_split_polygon(mask, bbox, 200, 200, margin=10)
    assert poly is not None
    assert len(poly) >= 3

    # Check that all points in the split polygon are around Circle 1 (X < 100, Y < 100)
    for p in poly:
        assert p[0] < 100
        assert p[1] < 100


# --- R1/R2 (docs/issues.md) -------------------------------------------------------------------


def test_a_balloon_that_does_not_cover_its_text_is_rejected():
    """R1. sample10's 待って: dark type with a white stroke around each glyph, on a yellow burst.

    YOLO scores the stroke as a bubble. It sits exactly on the glyphs, so it beats every real
    balloon on the page on overlap -- and the geometry that came back was 0.60x the area of its
    own text, which is impossible for a container. Accepting it painted a glyph-shaped white slab
    onto the burst and set "WAIT!" in the sliver left over.
    """
    from worker.handlers.ocr import bubble_covers_text

    text_box = (100, 100, 200, 300)  # x1, y1, x2, y2

    stroke = np.zeros((400, 400), dtype=np.uint8)
    stroke[140:260, 130:170] = 255  # a glyph-shaped sliver inside the text box
    assert not bubble_covers_text(stroke, *text_box)

    balloon = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(balloon, (150, 200), (90, 140), 0, 0, 360, (255,), -1)
    assert bubble_covers_text(balloon, *text_box)


def test_a_tight_but_real_balloon_still_passes():
    """The reason the test is coverage and not an area ratio.

    A lone "!?" in a balloon barely bigger than itself is a real balloon, and an area-ratio test
    tuned to reject the glyph stroke would throw it away too.
    """
    from worker.handlers.ocr import bubble_covers_text

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (48, 48), (104, 104), (255,), -1)
    assert bubble_covers_text(mask, 50, 50, 102, 102)


def test_a_shaded_region_gets_covered_instead_of_abandoned():
    """R2. The old contract was "not flat -> return None -> draw nothing", which left English on
    top of unerased Japanese. sample10's yellow blanket is the case: a vertical gradient, no
    balloon, and every individual decision defensible. Now it comes back with a colour and a
    covering shape."""
    from worker.handlers.ocr import cover_fill_for_region

    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for row in range(300):  # a lit-to-shadowed yellow gradient, far too spread to sample flat
        img[row, :] = (20 + row // 8, max(0, 250 - int(row * 0.8)), max(0, 255 - int(row * 0.7)))

    poly = [[80, 80], [220, 80], [220, 220], [80, 220]]
    assert detect_background_color_poly(img, poly) is None, "precondition: no flat colour here"

    color, shape = cover_fill_for_region(img, poly, 90, 90, 120, 120)

    assert color is not None and color.startswith("#")
    assert shape is not None and len(shape) > 4, "expected a synthesized covering balloon"
    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]
    assert min(xs) <= 90 and max(xs) >= 210, "must cover the source text horizontally"
    assert min(ys) <= 90 and max(ys) >= 210, "must cover the source text vertically"


def test_a_flat_region_is_left_on_the_old_path():
    """Cover fill is the fallback, not a replacement: a genuinely flat balloon still gets its own
    median painted over its own outline, and keeps the mask it came in with."""
    from worker.handlers.ocr import cover_fill_for_region

    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    poly = [[40, 40], [160, 40], [160, 160], [40, 160]]

    color, shape = cover_fill_for_region(img, poly, 50, 50, 100, 100)

    assert color == "#f0f0f0"
    assert shape == poly


def test_dominant_colour_names_the_colour_not_the_average():
    """Why the mode and not the median. Over a two-tone region the median lands between the tones,
    and a balloon painted in it reads as dirty; the reference's synthesized shape is the lit
    colour, which is the mode."""
    from worker.handlers.ocr import dominant_color

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:70] = (40, 220, 250)  # BGR: bright yellow, the larger share
    img[70:] = (10, 40, 60)  # a dark shadowed strip

    color = dominant_color(img, box=(0, 0, 100, 100))

    assert color is not None
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    assert r > 200 and g > 180 and b < 90, f"expected the lit yellow, got {color}"


def test_an_unenclosed_region_always_gets_a_shape_even_when_the_colour_was_easy():
    """The bug this encodes: right colour, wrong shape.

    After R1 rejects a balloon there is no mask, but the *border* of the text box can still sample
    flat -- sample10's 待って sits on a burst, so the yellow comes back cleanly. Returning that
    colour with no polygon left the renderer painting the element's own box, which is the sliver
    the glyphs occupy, so the source lettering still showed around the edges. No mask in means a
    synthesized shape out, however the colour was found.
    """
    from worker.handlers.ocr import cover_fill_for_region

    img = np.full((400, 400, 3), 200, dtype=np.uint8)  # a flat background the border test likes
    cv2.rectangle(img, (150, 150), (250, 250), (255, 255, 255), -1)  # the glyphs' white stroke

    color, shape = cover_fill_for_region(img, None, 150, 150, 100, 100)

    assert color is not None
    assert shape is not None and len(shape) > 4, "no mask in, so a synthesized shape must come out"
    xs = [p[0] for p in shape]
    assert min(xs) < 150 and max(xs) > 250, "the shape must be bigger than the text it covers"
