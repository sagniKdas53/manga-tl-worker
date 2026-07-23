import cv2
import numpy as np

from worker.handlers.ocr import (
    detect_background_color,
    detect_background_color_poly,
    detect_bubble_contour,
    get_split_polygon,
    sort_fragments_vertical,
)


def test_sort_fragments_vertical():
    fragments = [
        {"x": 10, "y": 10, "width": 10, "height": 10},
        {"x": 15, "y": 50, "width": 10, "height": 10},
        {"x": 50, "y": 20, "width": 10, "height": 10},
    ]
    # LTR sort should put the leftmost one (x=10, 15) first, then x=50
    sorted_ltr = sort_fragments_vertical(fragments, "ltr")
    assert sorted_ltr[0]["x"] == 10

    sorted_rtl = sort_fragments_vertical(fragments, "rtl")
    assert sorted_rtl[0]["x"] == 50


def test_detect_background_color():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # BGR (Blue)
    assert detect_background_color(img, 10, 10, 20, 20) == "#0000ff"

    assert detect_background_color(None, 0, 0, 10, 10) == "#ffffff"
    assert detect_background_color(img, 200, 200, 10, 10) == "#ffffff"  # out of bounds


def test_detect_background_color_poly():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)  # BGR (Green)
    poly = [[10, 10], [50, 10], [50, 50], [10, 50]]
    assert detect_background_color_poly(img, poly) == "#00ff00"

    assert detect_background_color_poly(None, poly) == "#ffffff"
    assert detect_background_color_poly(img, "invalid") == "#ffffff"
    assert detect_background_color_poly(img, "[[1,1]]") == "#ffffff"  # <3 pts


def test_get_split_polygon():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)

    poly = get_split_polygon(mask, (20, 20, 10, 10), 100, 100)
    assert poly is not None
    assert len(poly) >= 4

    assert get_split_polygon(None, (0, 0, 1, 1), 100, 100) is None


def test_detect_bubble_contour():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

    _ = detect_bubble_contour(img, 45, 45, 10, 10)
    # the function is just returning None or contour bbox depending on if it finds white background contour
    # let's just make sure it runs without exception
    pass
