from worker.services.layout import (
    bubble_compare,
    classify_region_type,
    group_conversations,
    chunk_regions_by_conversation,
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
