"""Guards on the contour fallback used when YOLO matches a text fragment to no bubble.

The detector is single-class and only confident on canonical enclosed balloons — on irregular
thought clouds it scores 0.04-0.21 against 0.92 for a normal bubble. Those fall below threshold and
their text would otherwise be typeset into the tight vertical Japanese column. This fallback runs
the OpenCV contour search on the fragment instead, but only accepts a result that is plausibly the
bubble rather than a panel border or a run of artwork.
"""

import cv2
import numpy as np
import pytest

import worker.handlers.ocr as ocr_mod
from worker.handlers.ocr import contour_bubble_for_unmatched


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_FALLBACK", True)
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_MAX_GROWTH", 5.0)
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_MAX_PAGE_FRACTION", 0.35)


def bubble_page(radius=40, size=400):
    """Gray page with one white bubble and black text inside it."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (255, 255, 255), -1)
    cv2.putText(img, "TXT", (size // 2 - 20, size // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def test_recovers_the_shape_yolo_missed(flag_on):
    img = bubble_page()
    # The OCR text extent, far narrower than the bubble containing it.
    found = contour_bubble_for_unmatched(img, 180, 192, 40, 20, 400, 400)

    assert found is not None
    assert found["width"] > 40, "must be wider than the text column it replaces"
    assert found["x"] <= 180 and found["x"] + found["width"] >= 220, "must contain the text"


def test_returns_none_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_FALLBACK", False)
    assert contour_bubble_for_unmatched(bubble_page(), 180, 192, 40, 20, 400, 400) is None


def test_rejects_a_contour_that_does_not_contain_the_text(flag_on):
    img = bubble_page()
    # Text sitting outside the bubble: whatever the search latches onto is not this text's bubble.
    assert contour_bubble_for_unmatched(img, 10, 10, 30, 15, 400, 400) is None


def test_rejects_a_contour_that_grew_past_the_growth_cap(flag_on, monkeypatch):
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_MAX_GROWTH", 1.2)
    assert contour_bubble_for_unmatched(bubble_page(), 180, 192, 40, 20, 400, 400) is None


def test_rejects_a_contour_covering_too_much_of_the_page(flag_on, monkeypatch):
    # A contour tracing most of the page is a panel border, not a bubble.
    monkeypatch.setattr(ocr_mod, "BUBBLE_CONTOUR_MAX_PAGE_FRACTION", 0.001)
    assert contour_bubble_for_unmatched(bubble_page(), 180, 192, 40, 20, 400, 400) is None


def test_returns_none_when_nothing_is_found(flag_on):
    # Uniform image: no contour to find.
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    assert contour_bubble_for_unmatched(flat, 90, 90, 20, 10, 200, 200) is None


def test_rejects_the_search_window_itself(flag_on):
    """Free-floating text sits on the page background, and the background is what the threshold
    finds. That blob has no boundary inside the crop, so its bounding box comes back as the crop --
    which passes every other guard by construction: a search window contains the text it was built
    around, is within 2 * pad of its size, and is a small part of the page. Measured over 300
    unmatched fragments, 171 of the 172 results accepted before this guard were their own window.
    """
    paper = np.full((400, 400, 3), 255, dtype=np.uint8)  # no balloon anywhere on it
    cv2.putText(paper, "TXT", (180, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    assert contour_bubble_for_unmatched(paper, 180, 192, 40, 20, 400, 400) is None


def test_accepts_a_balloon_that_runs_off_the_edge_of_the_page(flag_on):
    """The window is clamped by the paper there, so the blob touching its edge says nothing about
    whether the shape is enclosed -- a balloon drawn to the edge of the page is a real balloon."""
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    cv2.circle(img, (0, 200), 45, (255, 255, 255), -1)
    cv2.putText(img, "TX", (5, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    found = contour_bubble_for_unmatched(img, 5, 192, 35, 20, 400, 400)

    assert found is not None
    assert found["x"] == 0, "clipped by the paper, not by the search window"
