"""F5 (#244): how much room a text unit has beside it, when its balloon box says nothing about that."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.handlers.ocr import process_ocr
from worker.services.text_room import TEXT_ROOM_METHOD, measure_text_room, text_room_provenance

PAPER, INK = 255, 0


def page(width=900, height=900, paper=PAPER):
    return np.full((height, width), paper, dtype=np.uint8)


def draw_text(gray, bbox, ink=INK):
    """Vertical strokes filling the box, as a column of text does."""
    x, y, w, h = bbox
    gray[y : y + h, x : x + w : 4] = ink


def draw_balloon(gray, left, top, right, bottom, ink=INK, stroke=4):
    gray[top : bottom + stroke, left : left + stroke] = ink
    gray[top : bottom + stroke, right : right + stroke] = ink
    gray[top : top + stroke, left : right + stroke] = ink
    gray[bottom : bottom + stroke, left : right + stroke] = ink


def measured(gray, bbox, siblings=()):
    """The room, for a test where it must have been measured."""
    room = measure_text_room(gray, bbox, siblings)
    assert room is not None
    return room


def test_the_room_runs_from_outline_to_outline():
    gray = page()
    draw_balloon(gray, 200, 100, 700, 800)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)

    room = measure_text_room(gray, bbox)

    # A margin of 3 px either side of the 4 px outline, at x 200-203 and 700-703.
    assert room == {"x": 207, "width": 490, "method": TEXT_ROOM_METHOD}


def test_it_stops_at_ink_that_runs_along_the_text():
    gray = page()
    draw_balloon(gray, 100, 100, 800, 800)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    draw_text(gray, (600, 200, 60, 500))

    room = measured(gray, bbox)

    assert room["x"] + room["width"] <= 600


def test_text_reaching_the_page_edge_is_unmeasured_because_no_outline_was_found():
    gray = page()
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    assert measure_text_room(gray, bbox) is None


def test_one_side_reaching_the_page_edge_is_enough_to_leave_it_unmeasured():
    gray = page()
    gray[100:800, 200:204] = INK
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    assert measure_text_room(gray, bbox) is None


def test_it_reads_a_paper_that_is_not_white():
    gray = page(paper=190)
    draw_balloon(gray, 200, 100, 700, 800)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    assert measure_text_room(gray, bbox) == {"x": 207, "width": 490, "method": TEXT_ROOM_METHOD}


def test_it_reads_white_text_on_a_dark_balloon():
    gray = page(paper=20)
    draw_balloon(gray, 200, 100, 700, 800, ink=255)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox, ink=255)
    assert measure_text_room(gray, bbox) == {"x": 207, "width": 490, "method": TEXT_ROOM_METHOD}


def test_a_short_stroke_beside_the_text_does_not_decide_it():
    gray = page()
    draw_balloon(gray, 200, 100, 700, 800)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    # Crosses only a few of the rows, like a stray mark or a tail tip.
    gray[440:446, 300:304] = INK

    assert measured(gray, bbox)["x"] == 207


def test_two_texts_side_by_side_split_the_gap_between_them():
    gray = page()
    draw_balloon(gray, 100, 100, 800, 800)
    left, right = (300, 300, 60, 300), (560, 300, 60, 300)
    draw_text(gray, left)
    draw_text(gray, right)

    room_left = measured(gray, left, siblings=[right])
    room_right = measured(gray, right, siblings=[left])

    # The gap runs from 360 to 560: each stops at its middle, 460.
    assert room_left["x"] + room_left["width"] <= 460
    assert room_right["x"] >= 460


def test_a_sibling_above_or_below_does_not_limit_the_width():
    gray = page()
    draw_balloon(gray, 100, 100, 800, 800)
    bbox = (400, 150, 60, 200)
    other = (600, 500, 60, 200)
    draw_text(gray, bbox)
    draw_text(gray, other)

    assert measured(gray, bbox, siblings=[other])["width"] > 600


def test_it_never_returns_less_than_the_text_itself():
    gray = page()
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    # Ink right against the text on both sides.
    gray[100:800, 395:398] = INK
    gray[100:800, 462:465] = INK

    room = measure_text_room(gray, bbox)

    assert room is None or (room["x"] <= 400 and room["x"] + room["width"] >= 460)


def test_a_box_off_the_page_or_empty_is_unmeasured():
    gray = page()
    assert measure_text_room(gray, (2000, 2000, 60, 60)) is None
    assert measure_text_room(gray, (10, 10, 0, 0)) is None


def test_the_marker_is_added_and_the_rest_of_the_provenance_is_kept():
    gray = page()
    draw_balloon(gray, 200, 100, 700, 800)
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    provenance = {"fragments": ["kept"], "containerResolution": "resolved-local-split"}

    marked = text_room_provenance(provenance, gray, bbox, [])

    assert marked["fragments"] == ["kept"]
    assert marked["containerResolution"] == "resolved-local-split"
    assert marked["textRoom"]["method"] == TEXT_ROOM_METHOD


def test_an_unmeasured_region_gets_no_marker():
    gray = page()
    bbox = (400, 300, 60, 300)
    draw_text(gray, bbox)
    provenance = {"containerResolution": "resolved-local-split"}
    assert text_room_provenance(provenance, gray, bbox, []) == provenance


def _ocr_regions(fragments, detector_polygon, enabled=True):
    """The regions the whole OCR handler sends, on a white page with a drawn balloon at x 50-950."""
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    image[:, 50:54] = 0
    image[:, 950:954] = 0
    x_values = [point[0] for point in detector_polygon]
    y_values = [point[1] for point in detector_polygon]
    bubble = {
        "bbox": [min(x_values), min(y_values), max(x_values) - min(x_values), max(y_values) - min(y_values)],
        "confidence": 0.9,
        "mask_polygon": detector_polygon,
        "safe_rect": [
            min(x_values) + 5,
            min(y_values) + 5,
            max(x_values) - min(x_values) - 10,
            max(y_values) - min(y_values) - 10,
        ],
    }
    with (
        patch("worker.handlers.ocr.redis_client"),
        patch("worker.handlers.ocr.requests.post") as post,
        patch("worker.handlers.ocr.requests.get") as get,
        patch("worker.handlers.ocr.download_image", return_value=b"dummy"),
        patch("worker.handlers.ocr.downscale_for_ocr", return_value=(image, 1.0)),
        patch("worker.handlers.ocr.parse_paddle_ocr_results", return_value=fragments),
        patch("worker.handlers.ocr.model_manager.get_paddle_ocr_reader", return_value=MagicMock()),
        patch("worker.handlers.ocr.detect_bubbles_yolo", return_value=[bubble]),
        patch("worker.handlers.ocr.OCR_TEXT_ROOM", enabled),
    ):
        get.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"panels": []}))
        post.return_value = MagicMock(status_code=200)
        process_ocr(
            {"imageId": "text-room", "imageUrl": "http://dummy", "sourceLanguage": "ja", "readingDirection": "rtl"}
        )
        post.assert_called_once()
        return post.call_args.kwargs["json"]["regions"]


TWO_TEXTS = [
    ([[400, 100], [460, 100], [460, 300], [400, 300]], "top", 0.99),
    ([[400, 700], [460, 700], [460, 900], [400, 900]], "bottom", 0.99),
]
ONE_TEXT = TWO_TEXTS[:1]
BALLOON = [[50, 0], [950, 0], [950, 1000], [50, 1000]]


def test_the_handler_records_the_room_on_text_sharing_a_detector_box():
    regions = _ocr_regions(TWO_TEXTS, BALLOON)

    assert len(regions) == 2
    for region in regions:
        assert region["ownershipProvenance"]["containerResolution"] == "resolved-local-split"
        assert region["ownershipProvenance"]["textRoom"]["method"] == TEXT_ROOM_METHOD
        # A margin of 3 px inside the 4 px outlines at x 50-53 and 950-953.
        assert region["ownershipProvenance"]["textRoom"]["x"] == 57
        assert region["ownershipProvenance"]["textRoom"]["width"] == 890


def test_the_handler_records_no_room_on_text_alone_in_its_balloon():
    [region] = _ocr_regions(ONE_TEXT, BALLOON)
    assert region["ownershipProvenance"]["containerResolution"] == "resolved-bubble-container"
    assert "textRoom" not in region["ownershipProvenance"]


def test_the_handler_records_no_room_on_free_standing_text():
    # A detector box far from the text, so no balloon claims it.
    [region] = _ocr_regions(ONE_TEXT, [[600, 600], [700, 600], [700, 700], [600, 700]])
    assert "textRoom" not in (region.get("ownershipProvenance") or {})


@pytest.mark.parametrize("fragments", [TWO_TEXTS, ONE_TEXT])
def test_recording_the_room_changes_nothing_else_the_handler_sends(fragments):
    # Cleanup reads the masks and balloon boxes, so they must be the same with the feature on or off.
    with_room = _ocr_regions(fragments, BALLOON, enabled=True)
    without_room = _ocr_regions(fragments, BALLOON, enabled=False)

    assert len(with_room) == len(without_room)
    for on, off in zip(with_room, without_room, strict=True):
        on = on | {"ownershipProvenance": {k: v for k, v in on["ownershipProvenance"].items() if k != "textRoom"}}
        assert on == off
