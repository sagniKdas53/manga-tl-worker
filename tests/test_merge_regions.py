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


def test_merge_adjacent_vertical_fragments_without_merging_distant_bubbles():
    regions = [
        {
            "text": "top bubble",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 300,
            "y": 20,
            "width": 60,
            "height": 120,
        },
        {
            "text": "middle upper",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 240,
            "y": 260,
            "width": 80,
            "height": 120,
            "safeTextX": 240,
            "safeTextY": 260,
            "safeTextW": 80,
            "safeTextH": 120,
        },
        {
            "text": "middle lower",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 170,
            "y": 395,
            "width": 80,
            "height": 115,
            "safeTextX": 170,
            "safeTextY": 395,
            "safeTextW": 80,
            "safeTextH": 115,
        },
        {
            "text": "bottom bubble",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 20,
            "y": 620,
            "width": 75,
            "height": 120,
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="rtl")

    assert len(result) == 3
    middle = result[1]
    assert middle["text"] == "middle upper middle lower"
    assert middle["x"] == 170
    assert middle["y"] == 260
    assert middle["width"] == 150
    assert middle["height"] == 250
    assert middle["safeTextX"] == 170
    assert middle["safeTextY"] == 260
    assert middle["safeTextW"] == 150
    assert middle["safeTextH"] == 250
