"""E2 (#180): the angle rule and the box along the text.

The rule cases are the settled E0 rule's (``scripts/e0/angle_rule.py`` in the parent repo); each
expected value is what that script returns for the same pieces.
"""

import math

import pytest

from worker.services.text_angle import (
    DEAD_BAND,
    NO_LONG_PIECE,
    PIECES_DISAGREE,
    SWITCHED_OFF,
    TURNED,
    apply_text_angles,
    piece_quads,
    text_angle,
    text_area,
)


def quad(cx, cy, length, thickness, degrees):
    """A line piece's quad: `length` along `degrees`, centred on (cx, cy)."""
    a = math.radians(degrees)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    hl, ht = length / 2, thickness / 2
    return [
        [cx - ux * hl - vx * ht, cy - uy * hl - vy * ht],
        [cx + ux * hl - vx * ht, cy + uy * hl - vy * ht],
        [cx + ux * hl + vx * ht, cy + uy * hl + vy * ht],
        [cx - ux * hl + vx * ht, cy - uy * hl + vy * ht],
    ]


def merged(*quads):
    """A merged region's provenance: one fragment per piece."""
    return {
        "containerResolution": "detector",
        "fragments": [{"index": i, "provenance": {"sourceQuad": q}} for i, q in enumerate(quads)],
    }


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        (merged(quad(0, 0, 20, 20, 10)), (0.0, NO_LONG_PIECE)),
        (merged(quad(0, 0, 25, 20, 20)), (0.0, NO_LONG_PIECE)),
        (merged(quad(0, 0, 200, 20, 44), quad(0, 40, 200, 20, 46)), (45.0, TURNED)),
        (
            merged(
                quad(0, 0, 200, 20, 15),
                quad(0, 40, 200, 20, 16),
                quad(0, 80, 200, 20, 14),
                quad(0, 120, 200, 20, -20),
            ),
            (15.0, TURNED),
        ),
        (merged(quad(0, 0, 200, 20, -89), quad(0, 40, 200, 20, 89), quad(0, 80, 200, 20, 50)), (-90.0, TURNED)),
        (merged(quad(0, 0, 20, 200, 92), quad(40, 0, 20, 200, 93)), (0.0, DEAD_BAND)),
        (merged(quad(0, 0, 200, 20, 3)), (0.0, DEAD_BAND)),
        (merged(quad(0, 0, 200, 20, 10), quad(0, 40, 200, 20, 30)), (0.0, PIECES_DISAGREE)),
        ({"sourceQuad": quad(0, 0, 200, 20, -12)}, (-12.0, TURNED)),
    ],
    ids=["square", "short", "near-45", "stray-dropped", "wrap-at-90", "vertical", "dead-band", "disagree", "single"],
)
def test_rule_matches_the_settled_e0_cases(provenance, expected):
    angle, why = text_angle(piece_quads(provenance))
    assert why == expected[1]
    assert angle == pytest.approx(expected[0], abs=1e-6)


def test_no_provenance_stays_level():
    assert text_angle(piece_quads(None)) == (0.0, NO_LONG_PIECE)
    assert text_angle(piece_quads({"fragments": [{"index": 0, "provenance": {}}]})) == (0.0, NO_LONG_PIECE)


def test_text_area_is_the_box_along_the_text():
    # Two 100x20 lines, 30 px apart, turned 30° about (200, 200): along the text they span
    # 100 x 50, centred on (200, 200).
    a = math.radians(30)
    offset = (-math.sin(a) * 15, math.cos(a) * 15)
    quads = [
        quad(200 - offset[0], 200 - offset[1], 100, 20, 30),
        quad(200 + offset[0], 200 + offset[1], 100, 20, 30),
    ]
    box = text_area(quads, 30.0)
    assert box is not None
    x, y, w, h = box
    assert (w, h) == (pytest.approx(100), pytest.approx(50))
    assert (x + w / 2, y + h / 2) == (pytest.approx(200), pytest.approx(200))


def test_text_area_is_smaller_than_the_level_box():
    quads = [quad(500, 300, 400, 40, -20)]
    box = text_area(quads, -20.0)
    assert box is not None
    _, _, w, h = box
    xs = [p[0] for p in quads[0]]
    ys = [p[1] for p in quads[0]]
    assert w * h < (max(xs) - min(xs)) * (max(ys) - min(ys)) / 2


def test_apply_turns_a_region_and_records_why():
    region = {"rotation": 0.0, "ownershipProvenance": {"sourceQuad": quad(300, 300, 300, 30, 12)}}
    apply_text_angles([region], enabled=True)
    assert region["rotation"] == pytest.approx(12.0)
    assert region["textAreaW"] == pytest.approx(300, abs=0.1)
    assert region["textAreaH"] == pytest.approx(30, abs=0.1)
    assert region["ownershipProvenance"]["textAngle"] == {"angle": pytest.approx(12.0), "why": TURNED}
    assert region["ownershipProvenance"]["sourceQuad"]  # the piece record itself is kept


def test_apply_leaves_a_level_region_without_a_text_area():
    region = {"rotation": 0.0, "ownershipProvenance": merged(quad(0, 0, 200, 20, 2))}
    apply_text_angles([region], enabled=True)
    assert region["rotation"] == 0.0
    assert "textAreaX" not in region
    assert region["ownershipProvenance"]["textAngle"]["why"] == DEAD_BAND
    assert region["ownershipProvenance"]["fragments"]


def test_switched_off_keeps_every_region_level_but_says_what_it_would_do():
    region = {"rotation": 0.0, "ownershipProvenance": {"sourceQuad": quad(300, 300, 300, 30, 12)}}
    apply_text_angles([region], enabled=False)
    assert region["rotation"] == 0.0
    assert "textAreaX" not in region
    assert region["ownershipProvenance"]["textAngle"] == {
        "angle": pytest.approx(12.0),
        "why": SWITCHED_OFF,
        "wouldTurn": True,
    }


def test_region_without_provenance_is_left_level():
    region = {"rotation": 0.0}
    apply_text_angles([region], enabled=True)
    assert region == {"rotation": 0.0}


def test_the_job_switch_wins_and_an_old_job_uses_the_env_default(monkeypatch):
    """System Settings and series/chapter overrides send `ocrTextAngle` on the OCR job."""
    from worker.handlers import ocr

    assert ocr.text_angle_enabled_for({"ocrTextAngle": False}) is False
    assert ocr.text_angle_enabled_for({"ocrTextAngle": True}) is True
    assert ocr.text_angle_enabled_for({"ocrTextAngle": "false"}) is False
    monkeypatch.setattr(ocr, "OCR_TEXT_ANGLE", False)
    for job in ({}, {"ocrTextAngle": None}, {"ocrTextAngle": "maybe"}):
        assert ocr.text_angle_enabled_for(job) is False
    monkeypatch.setattr(ocr, "OCR_TEXT_ANGLE", True)
    assert ocr.text_angle_enabled_for({}) is True
