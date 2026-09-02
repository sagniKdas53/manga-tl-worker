# ---------------------------------------------------------------------------
# AUDIT-R7 — mask polygon simplification
# ---------------------------------------------------------------------------


def _jittery_rectangle():
    """A 100x40 rectangle whose edges wobble by a pixel, as a rasterised contour does."""
    points = []
    for x in range(0, 101, 5):
        points.append([x, 0 + (1 if (x // 5) % 2 else 0)])
    for y in range(5, 41, 5):
        points.append([100 - (1 if (y // 5) % 2 else 0), y])
    for x in range(95, -1, -5):
        points.append([x, 40 + (1 if (x // 5) % 2 else 0)])
    for y in range(35, 0, -5):
        points.append([0 + (1 if (y // 5) % 2 else 0), y])
    return points


def test_a_rectangle_comes_back_as_a_rectangle():
    """The reported symptom: a box that should have four handles arriving with dozens."""
    from worker.services.bubble_geometry import simplify_mask_polygon

    points = _jittery_rectangle()
    assert len(points) > 30
    assert len(simplify_mask_polygon(points)) == 4


def test_the_old_relative_tolerance_is_what_made_small_shapes_worst():
    """0.002 * perimeter is sub-pixel on a small plate, so it removed nothing.

    This is the whole bug in one assertion: at the tolerance the pipeline used to apply to this
    shape, the jitter survives; at an absolute 2px it does not.
    """
    import cv2
    import numpy as np

    from worker.services.bubble_geometry import simplify_mask_polygon

    points = _jittery_rectangle()
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    old_epsilon = 0.002 * cv2.arcLength(contour, True)
    assert old_epsilon < 1.0, "the old tolerance was below one pixel on a shape this size"
    assert len(cv2.approxPolyDP(contour, old_epsilon, True)) > 20
    assert len(simplify_mask_polygon(points)) == 4


def test_a_balloon_tail_survives_because_it_is_not_jitter():
    """The tolerance that flattens rasterisation noise must not flatten a real feature."""
    import math

    from worker.services.bubble_geometry import simplify_mask_polygon

    circle = [
        [round(50 + 40 * math.cos(a * math.pi / 18)), round(50 + 40 * math.sin(a * math.pi / 18))] for a in range(36)
    ]
    with_tail = [*circle[:9], [110, 50], [95, 70], *circle[9:]]
    simplified = simplify_mask_polygon(with_tail)
    assert any(px >= 105 for px, _ in simplified), "the tail is gone"
    assert len(simplified) < len(with_tail), "nothing was simplified at all"


def test_a_shape_already_minimal_is_left_alone():
    from worker.services.bubble_geometry import simplify_mask_polygon

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert simplify_mask_polygon(square) == square


def test_junk_input_is_handed_back_rather_than_raising():
    from worker.services.bubble_geometry import simplify_mask_polygon

    assert simplify_mask_polygon(None) is None
    assert simplify_mask_polygon([]) == []
    ragged = [[0, 0], [1], [2, 2, 2]]
    assert simplify_mask_polygon(ragged) == ragged
