import os
import json
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


def test_merge_preserves_shared_mask_polygon_and_safe_area():
    polygon = [[40, 20], [130, 20], [130, 120], [40, 120]]
    regions = [
        {
            "text": "Hello",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 70,
            "y": 50,
            "width": 30,
            "height": 20,
            "backgroundColor": "#ffffff",
            "bubbleX": 40,
            "bubbleY": 20,
            "bubbleWidth": 90,
            "bubbleHeight": 100,
            "detectionConfidence": 0.8,
            "maskPolygon": json.dumps(polygon),
            "safeTextX": 50,
            "safeTextY": 30,
            "safeTextW": 70,
            "safeTextH": 80,
        },
        {
            "text": "World",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 72,
            "y": 72,
            "width": 34,
            "height": 20,
            "backgroundColor": "#ffffff",
            "bubbleX": 40,
            "bubbleY": 20,
            "bubbleWidth": 90,
            "bubbleHeight": 100,
            "detectionConfidence": 0.6,
            "maskPolygon": json.dumps(polygon),
            "safeTextX": 50,
            "safeTextY": 30,
            "safeTextW": 70,
            "safeTextH": 80,
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="ltr")

    assert len(result) == 1
    assert result[0]["text"] == "Hello World"
    assert json.loads(result[0]["maskPolygon"]) == polygon
    assert result[0]["safeTextX"] == 50
    assert result[0]["safeTextY"] == 30
    assert result[0]["safeTextW"] == 70
    assert result[0]["safeTextH"] == 80
    assert result[0]["detectionConfidence"] == 0.7


def test_merge_v10_split_middle_sample_bubble():
    regions = [
        {
            "text": "Bro... are you really gonna do it?",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 642,
            "y": 145,
            "width": 74,
            "height": 213,
        },
        {
            "text": "Yep. Get changed and meet me at the rocks over there.",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 586,
            "y": 542,
            "width": 111,
            "height": 225,
            "safeTextX": 586,
            "safeTextY": 542,
            "safeTextW": 111,
            "safeTextH": 225,
        },
        {
            "text": "We don't get a chance like this often... Fun, right?",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 496,
            "y": 798,
            "width": 112,
            "height": 198,
            "safeTextX": 496,
            "safeTextY": 798,
            "safeTextW": 112,
            "safeTextH": 198,
        },
        {
            "text": "I'll just make something up for Mom and them.",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 43,
            "y": 972,
            "width": 113,
            "height": 178,
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="rtl")

    assert len(result) == 3
    middle = result[1]
    assert (
        middle["text"]
        == "Yep. Get changed and meet me at the rocks over there. We don't get a chance like this often... Fun, right?"
    )
    assert middle["x"] == 496
    assert middle["y"] == 542
    assert middle["width"] == 201
    assert middle["height"] == 454
    assert middle["safeTextX"] == 496
    assert middle["safeTextY"] == 542
    assert middle["safeTextW"] == 201
    assert middle["safeTextH"] == 454
