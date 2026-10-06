"""The series' reading direction reaches the worker in either spelling.

The UI stores `rtl` / `ltr` / `ttb`; older series and the corpus scripts store `rightToLeft` /
`leftToRight`, and the backend sends the value lowercased. The worker compared against the short
forms only, so on a `rightToLeft` series merge_ocr_regions read a joined balloon's columns left to
right (sample78's 「でかした友人！アイツが…」 came out as 「ウチに俺も…図になってる友人！アイツがでかした」),
and on a `leftToRight` series panel order and the binding check took it for a manga.
"""

import inspect

import pytest

from worker.services.merge_regions import merge_ocr_regions
from worker.utils.reading_direction import normalize_reading_direction


@pytest.mark.parametrize(
    "stored, expected",
    [
        ("rtl", "rtl"),
        ("RTL", "rtl"),
        (" rightToLeft ", "rtl"),
        ("righttoleft", "rtl"),
        ("right-to-left", "rtl"),
        ("ltr", "ltr"),
        ("leftToRight", "ltr"),
        ("lefttoright", "ltr"),
        ("ttb", "ttb"),
        ("topToBottom", "ttb"),
        ("vertical", "ttb"),
        ("webtoon", "ttb"),
        (None, "rtl"),
        ("", "rtl"),
        ("sideways", "rtl"),
    ],
)
def test_every_spelling_maps_to_one_value(stored, expected):
    assert normalize_reading_direction(stored) == expected


def _column(x, text):
    return {"text": text, "detectedLanguage": "ja", "confidence": 0.9, "x": x, "y": 100, "width": 40, "height": 200}


def test_a_right_to_left_series_reads_its_columns_right_to_left():
    # Two columns 10 px apart in one balloon; the right one is read first.
    columns = [_column(150, "二"), _column(200, "一")]
    merged = merge_ocr_regions(columns, reading_direction=normalize_reading_direction("righttoleft"))
    assert [region["text"] for region in merged] == ["一二"]


@pytest.mark.parametrize(
    "handler", ["worker.handlers.ocr:process_ocr", "worker.handlers.panel:process_panel_detection"]
)
def test_handlers_normalize_the_direction_they_receive(handler):
    import importlib

    module_name, function_name = handler.split(":")
    source = inspect.getsource(getattr(importlib.import_module(module_name), function_name))
    assert "normalize_reading_direction(" in source
    assert '.get("readingDirection") or "rtl").strip().lower()' not in source
