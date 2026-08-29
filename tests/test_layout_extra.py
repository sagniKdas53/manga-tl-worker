from worker.services.layout import (
    bubble_compare,
    chunk_regions_by_conversation,
    classify_region_type,
    group_conversations,
)


def test_bubble_compare():
    a = {"x": 100, "y": 100}
    b = {"x": 200, "y": 100}
    # RTL same row: larger x is first, so b should be before a?
    # function: x_diff = b["x"] - a["x"] -> 200 - 100 = 100 > 0 -> returns 1
    assert bubble_compare(a, b, "rtl") == 1

    assert bubble_compare(a, b, "ltr") == -1

    c = {"x": 100, "y": 500}
    # y diff large -> y_diff = a['y'] - c['y'] = 100 - 500 = -400 -> returns -1
    assert bubble_compare(a, c, "rtl") == -1
    assert bubble_compare(a, c, "ttb") == -1


def test_classify_region_type():
    reg_sfx = {"text": "ああ", "width": 10, "height": 100, "confidence": 0.9}
    assert classify_region_type(reg_sfx, None, 1000, 1000) == "sfx"

    reg_narration = {"text": "hello", "width": 500, "height": 50, "bboxY": 50}
    assert classify_region_type(reg_narration, None, 1000, 1000) == "caption"

    panel = {"bboxX": 0, "bboxY": 0, "bboxW": 1000, "bboxH": 1000}
    assert classify_region_type(reg_narration, panel, 1000, 1000) == "narration"


def test_group_conversations():
    regions = [
        {
            "id": "1",
            "panelReadingOrder": 1,
            "bubbleReadingOrder": 1,
            "regionType": "speech",
            "bboxY": 100,
            "bboxH": 50,
        },
        {
            "id": "2",
            "panelReadingOrder": 1,
            "bubbleReadingOrder": 2,
            "regionType": "speech",
            "bboxY": 120,
            "bboxH": 50,
        },
        {
            "id": "3",
            "panelReadingOrder": 1,
            "bubbleReadingOrder": 3,
            "regionType": "sfx",
            "bboxY": 500,
            "bboxH": 50,
        },
        {"id": "4", "panelReadingOrder": 0, "regionType": "speech"},
    ]
    convs = group_conversations(regions, None)
    assert len(convs) == 3
    assert convs[0]["regionIds"] == ["1", "2"]
    assert convs[1]["regionIds"] == ["3"]
    assert convs[2]["regionIds"] == ["4"]


def test_chunk_regions_by_conversation():
    regions = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    convs = [{"regionIds": ["1", "2"]}]
    chunks = chunk_regions_by_conversation(regions, convs, 2)
    assert len(chunks) == 2
    assert chunks[0][0]["id"] == "1"
    assert chunks[1][0]["id"] == "3"


# --- Regression: dialogue must not be typed as a sound effect -------------------------------
#
# Both cases below are real regions from corpus/gaps/unfiled/iuno, where the pipeline left the
# Japanese sitting in a clean balloon while every neighbouring bubble came out in English. A
# region typed "sfx" is dropped from the translation batch entirely by should_typeset_region, so
# a false positive here costs the bubble its translation outright.

IUNO_PAGE1_WATASHI = {
    "text": "私だって…",
    "bboxX": 1234, "bboxY": 1455, "bboxW": 92, "bboxH": 291,
    "confidence": 0.9814,
    "bubbleId": "bubble_3",
    "detectionConfidence": 0.9662,
}

IUNO_PAGE2_TOMODACHI = {
    "text": "え？友達",
    "bboxX": 294, "bboxY": 1332, "bboxW": 70, "bboxH": 227,
    "confidence": 0.9752,
    "bubbleId": "bubble_4",
    "detectionConfidence": 0.8440,
}

PANEL = {"bboxX": 0, "bboxY": 0, "bboxW": 1447, "bboxH": 2039}


def test_tall_vertical_dialogue_in_a_balloon_is_speech_not_sfx():
    """The deleted rule typed both of these sfx on shape alone: tall_aspect 3.16 and 3.24 with
    five and four characters. Vertical Japanese is always tall and narrow."""
    assert classify_region_type(IUNO_PAGE1_WATASHI, PANEL, 1447, 2039) == "speech"
    assert classify_region_type(IUNO_PAGE2_TOMODACHI, PANEL, 1447, 2039) == "speech"


def test_short_kana_interjection_in_a_balloon_is_speech():
    """'え？' and 'ん?' are among the most common utterances in manga and the kana-only rule ate
    them. Inside a detected balloon they are dialogue."""
    for text in ("え？", "いや", "ん?", "はは"):
        region = {
            "text": text,
            "bboxW": 60, "bboxH": 90, "confidence": 0.95,
            "bubbleId": "bubble_1", "detectionConfidence": 0.9,
        }
        assert classify_region_type(region, PANEL, 1000, 1000) == "speech", text


def test_unenclosed_kana_is_still_sfx():
    """The kana rule keeps its job outside balloons — these are the real sound effects."""
    for text in ("ドキ", "ザワ", "ガチャ", "ガサ"):
        region = {"text": text, "bboxW": 80, "bboxH": 80, "confidence": 0.95}
        assert classify_region_type(region, PANEL, 1000, 1000) == "sfx", text


def test_lettering_the_detector_did_not_enclose_is_not_treated_as_enclosed():
    """direct_text_N is the synthetic id the OCR stage gives text found outside any balloon; it
    must not count as enclosure or the kana rule would never fire again."""
    region = {
        "text": "ドキ", "bboxW": 80, "bboxH": 80, "confidence": 0.95,
        "bubbleId": "direct_text_0", "detectionConfidence": 0.0,
    }
    assert classify_region_type(region, PANEL, 1000, 1000) == "sfx"


def test_low_confidence_bubble_detection_does_not_count_as_enclosure():
    region = {
        "text": "ドキ", "bboxW": 80, "bboxH": 80, "confidence": 0.95,
        "bubbleId": "bubble_9", "detectionConfidence": 0.11,
    }
    assert classify_region_type(region, PANEL, 1000, 1000) == "sfx"


def test_kanji_bearing_vertical_text_is_never_sfx():
    """55 regions like these were typed sfx across 397 corpus exports; not one was onomatopoeia."""
    for text in ("失礼します。", "当然です！", "行きますか", "それは即ち…", "真的嗎？"):
        region = {"text": text, "bboxW": 40, "bboxH": 200, "confidence": 0.95}
        assert classify_region_type(region, PANEL, 1000, 1000) != "sfx", text
