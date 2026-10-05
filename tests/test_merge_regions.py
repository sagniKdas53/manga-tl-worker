import json

from worker.services.fragment_grouping import GroupingConfig
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


def _piece(text, x, y, width, height):
    return {"text": text, "detectedLanguage": "ja", "confidence": 0.9, "x": x, "y": y, "width": width, "height": height}


# Tests ch. 4 p. 63: a three-line horizontal caption whose lines all start at about the same x.
_CAPTION = [
    _piece("ここまで追跡してくるのは骨が", 552, 564, 455, 39),
    _piece("折れたけど，大魔族討伐の手柄", 546, 605, 440, 41),
    _piece("はアタシがもらうよ,ゼンゼ.", 550, 652, 410, 33),
]
# Tests ch. 4 p. 54: four columns, the last broken by OCR into 手ブラ above a narrower での.
_BROKEN_COLUMNS = [
    _piece("手ブラ", 111, 548, 39, 97),
    _piece("長富蓮実ち", 137, 547, 47, 157),
    _piece("での", 112, 628, 37, 73),
    _piece("本日の挑戦者は", 169, 548, 45, 208),
]


def _one_group(regions, **config):
    grouping = GroupingConfig(threshold_ratio=1e9, reading_direction="rtl", orientation="vote", **config)
    (merged,) = merge_ocr_regions([dict(region) for region in regions], grouping=grouping)
    return merged["text"]


def test_legacy_order_reads_a_horizontal_caption_by_x():
    """The shipped order sorts by -x first, so left-aligned lines come out by their start offset."""
    assert _one_group(_CAPTION) == "ここまで追跡してくるのは骨がはアタシがもらうよ,ゼンゼ.折れたけど，大魔族討伐の手柄"


def test_line_order_reads_horizontal_lines_top_to_bottom():
    assert _one_group(_CAPTION, line_reading_order=True) == (
        "ここまで追跡してくるのは骨が折れたけど，大魔族討伐の手柄はアタシがもらうよ,ゼンゼ."
    )


def test_line_order_reads_a_broken_column_top_to_bottom():
    """での is narrower than 手ブラ above it, so sorting by -x put it first."""
    assert _one_group(_BROKEN_COLUMNS) == "本日の挑戦者は長富蓮実ちでの手ブラ"
    assert _one_group(_BROKEN_COLUMNS, line_reading_order=True) == "本日の挑戦者は長富蓮実ち手ブラでの"


def test_line_order_keeps_overlapping_wide_columns_apart():
    """Tests ch. 4 p. 64: OCR boxes wider than the column pitch overlap their neighbours by half.

    Overlap alone put ボクの言う and 通りにすれば on one line; their centres are 90 px apart.
    """
    columns = [
        _piece("大丈夫だよん", 1209, 1398, 155, 456),
        _piece("通りにすれば", 1288, 1405, 159, 456),
        _piece("ボクの言う", 1367, 1409, 181, 403),
    ]
    assert _one_group(columns, line_reading_order=True) == "ボクの言う通りにすれば大丈夫だよん"


def test_line_order_reads_a_tilted_sign_line_by_line():
    """Tests ch. 5 p. 19 (AUDIT-R23's sign, tilted 25°): two lines overlap in y but sit apart."""
    sign = [_piece("キュアット", 487, 106, 250, 173), _piece("探偵事所", 446, 153, 296, 223)]
    assert _one_group(sign, line_reading_order=True) == "キュアット探偵事所"


def test_line_order_keeps_plain_columns_right_to_left():
    regions = [_piece("一", 200, 10, 30, 120), _piece("二", 160, 14, 30, 110), _piece("三", 120, 12, 30, 90)]
    assert _one_group(regions, line_reading_order=True) == _one_group(regions) == "一二三"


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
    assert result[0]["rawFragmentMembership"] == [
        {"index": 0, "fragment_id": None},
        {"index": 1, "fragment_id": None},
    ]
    assert result[0]["containerResolution"] == "resolved-shared-container"


def test_merge_refuses_to_grant_a_fused_component_a_hulled_container():
    left_polygon = [[0, 0], [55, 0], [55, 100], [0, 100]]
    right_polygon = [[45, 0], [100, 0], [100, 100], [45, 100]]
    regions = [
        {
            "text": "left",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-left",
            "maskPolygon": json.dumps(left_polygon),
        },
        {
            "text": "right",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 40,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-right",
            "maskPolygon": json.dumps(right_polygon),
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="ltr")

    assert len(result) == 1
    assert result[0]["maskPolygon"] is None
    assert result[0]["containerResolution"] == "review-incompatible-containers"
    assert result[0]["rawFragmentMembership"] == [
        {"index": 0, "fragment_id": "fragment-left"},
        {"index": 1, "fragment_id": "fragment-right"},
    ]


def test_merge_keeps_separate_dark_containers_review_only():
    left_polygon = [[0, 0], [55, 0], [55, 100], [0, 100]]
    right_polygon = [[45, 0], [100, 0], [100, 100], [45, 100]]
    regions = [
        {
            "text": "left",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-dark-left",
            "backgroundColor": "#171717",
            "maskPolygon": json.dumps(left_polygon),
        },
        {
            "text": "right",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 40,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-dark-right",
            "backgroundColor": "#171717",
            "maskPolygon": json.dumps(right_polygon),
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="ltr")

    assert len(result) == 1
    assert result[0]["maskPolygon"] is None
    assert result[0]["containerResolution"] == "review-incompatible-containers"
    assert [member["fragment_id"] for member in result[0]["rawFragmentMembership"]] == [
        "fragment-dark-left",
        "fragment-dark-right",
    ]


def test_merge_requires_a_container_for_every_component_member():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    regions = [
        {
            "text": "contained",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-contained",
            "maskPolygon": json.dumps(polygon),
        },
        {
            "text": "uncontained",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 40,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-uncontained",
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="ltr")

    assert len(result) == 1
    assert result[0]["maskPolygon"] is None
    assert result[0]["containerResolution"] == "review-missing-container"
    assert [member["fragment_id"] for member in result[0]["rawFragmentMembership"]] == [
        "fragment-contained",
        "fragment-uncontained",
    ]


def test_merge_never_crosses_known_panel_boundaries():
    polygon = [[0, 0], [120, 0], [120, 100], [0, 100]]
    regions = [
        {
            "text": "left panel",
            "detectedLanguage": "en",
            "confidence": 0.9,
            "x": 10,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-panel-left",
            "panelId": "panel-left",
            "maskPolygon": json.dumps(polygon),
        },
        {
            "text": "right panel",
            "detectedLanguage": "en",
            "confidence": 0.8,
            "x": 40,
            "y": 20,
            "width": 50,
            "height": 30,
            "fragmentId": "fragment-panel-right",
            "panelId": "panel-right",
            "maskPolygon": json.dumps(polygon),
        },
    ]

    result = merge_ocr_regions(regions, reading_direction="ltr")

    assert [region["text"] for region in result] == ["left panel", "right panel"]
    assert [region["rawFragmentMembership"] for region in result] == [
        [{"index": 0, "fragment_id": "fragment-panel-left"}],
        [{"index": 1, "fragment_id": "fragment-panel-right"}],
    ]


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
