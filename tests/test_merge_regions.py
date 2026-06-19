import os
from worker.services.merge_regions import merge_ocr_regions


def test_merge_no_regions():
    assert merge_ocr_regions([]) == []


def test_merge_single_region():
    regions = [
        {
            "text": "Hello",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 10,
            "width": 50,
            "height": 20,
        }
    ]
    result = merge_ocr_regions(regions)
    assert len(result) == 1
    assert result[0]["text"] == "Hello"


def test_merge_overlapping_regions():
    regions = [
        {
            "text": "World",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 12,
            "y": 15,
            "width": 48,
            "height": 18,
        },
        {
            "text": "Hello",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 10,
            "width": 50,
            "height": 20,
        },
    ]
    # LTR merge: Hello (at x=10) should come before World (at x=12)
    result = merge_ocr_regions(regions, reading_direction="ltr")
    assert len(result) == 1
    assert result[0]["text"] == "Hello World"
    assert result[0]["x"] == 10
    assert result[0]["y"] == 10
    assert result[0]["width"] == 50
    assert result[0]["height"] == 23  # Union y extends to 15 + 18 = 33


def test_merge_rtl_regions():
    regions = [
        {
            "text": "右",  # Right
            "detectedLanguage": "ja",
            "confidence": 0.9,
            "x": 100,
            "y": 10,
            "width": 20,
            "height": 50,
        },
        {
            "text": "左",  # Left
            "detectedLanguage": "ja",
            "confidence": 0.8,
            "x": 70,
            "y": 12,
            "width": 20,
            "height": 48,
        },
    ]
    # RTL merge: Right (at larger X = 100) should come before Left (at smaller X = 70)
    result = merge_ocr_regions(regions, reading_direction="rtl")
    assert len(result) == 1
    assert result[0]["text"] == "右左"
    assert result[0]["x"] == 70
    assert result[0]["y"] == 10
    assert result[0]["width"] == 50
    assert result[0]["height"] == 50
