from worker.services.fragment_grouping import GroupingConfig
from worker.services.ocr_capture import capture_observed_ocr_grouping, capture_ocr_grouping, replay_twice


def test_capture_round_trip_replays_stable_geometry_and_owners(tmp_path):
    regions = [
        {"x": 10, "y": 10, "width": 20, "height": 10, "text": "あ"},
        {"x": 10, "y": 25, "width": 20, "height": 10, "text": "い"},
    ]
    capture = capture_ocr_grouping(
        source_id="fixture-page-1",
        raw_quads=[[[10, 10], [30, 10], [30, 20], [10, 20]], [[10, 25], [30, 25], [30, 35], [10, 35]]],
        scale_transform={"ocr_to_source": {"scale_x": 2.0, "scale_y": 2.0}},
        detector_masks=[{"format": "polygon", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
        recognition=[{"text": "あ", "confidence": 0.9}, {"text": "い", "confidence": 0.8}],
        regions=regions,
        grouping=GroupingConfig(threshold_ratio=1.0, reading_direction="rtl"),
        paths={"ocr": "live", "grouping": "live", "detector_masks": "cached"},
    )
    path = tmp_path / "capture.json"
    capture.write(path)
    loaded = type(capture).read(path)
    first, second, equal = replay_twice(loaded)
    assert equal
    assert first.fragment_ids == second.fragment_ids
    assert first.fragment_geometry == second.fragment_geometry
    assert first.owner_ids == second.owner_ids
    assert loaded.raw_quads and loaded.scale_transform and loaded.detector_masks
    assert loaded.recognition and loaded.grouping_edges and loaded.final_owners
    assert loaded.paths["detector_masks"] == "cached"


def test_observed_capture_preserves_runtime_groups_without_replay_assumptions():
    regions = [
        {"x": 10, "y": 10, "width": 20, "height": 10, "text": "あ"},
        {"x": 10, "y": 25, "width": 20, "height": 10, "text": "い"},
        {"x": 50, "y": 10, "width": 10, "height": 10, "text": "う"},
    ]
    capture = capture_observed_ocr_grouping(
        source_id="fixture-page-1",
        raw_quads=[
            [[10, 10], [30, 10], [30, 20], [10, 20]],
            [[10, 25], [30, 25], [30, 35], [10, 35]],
            [[50, 10], [60, 10], [60, 20], [50, 20]],
        ],
        scale_transform={"ocr_to_source": {"scale_x": 1.0, "scale_y": 1.0}},
        detector_masks=[],
        recognition=[{"text": "あ"}, {"text": "い"}, {"text": "う"}],
        regions=regions,
        grouping=GroupingConfig(threshold_ratio=1.0, reading_direction="rtl"),
        observed_groups=[[0, 1], [2]],
    )
    assert len(capture.grouping_edges) == 1
    assert capture.final_owners[0]["bbox"] == {"x": 10, "y": 10, "width": 20, "height": 25}
    assert capture.final_owners[0]["fragment_ids"]
