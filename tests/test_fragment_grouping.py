"""Equivalence guard for the grouping seam.

`group_fragments` was extracted verbatim out of `merge_ocr_regions`. `_legacy_components` below is
a frozen copy of the pre-split adjacency + BFS pass. These tests assert the two agree exactly --
group membership AND ordering -- over a randomised corpus plus hand-written edge cases.

**Do not "fix" `_legacy_components` when a phase changes grouping behaviour.** It is a fossil, and
its whole value is that it never changes. When a phase deliberately alters grouping, that phase
goes behind an off-by-default `GroupingConfig` field and these tests keep passing at the defaults;
the phase gets its own tests. If a default-configuration equivalence test ever has to be edited to
pass, something has changed production behaviour by accident.
"""

import dataclasses
import random

import pytest

from worker.services.fragment_grouping import GroupingConfig, group_fragments


def _legacy_components(regions, reading_direction, threshold_ratio):
    """Frozen copy of merge_ocr_regions' grouping pass as it stood before the extraction."""
    avg_height = sum(r["height"] for r in regions) / len(regions)
    avg_width = sum(r["width"] for r in regions) / len(regions)
    char_size_vertical = avg_width if reading_direction == "rtl" else avg_height
    max_vertical_gap = char_size_vertical * threshold_ratio
    max_horizontal_gap = avg_width * threshold_ratio

    n = len(regions)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = regions[i], regions[j]
            r1_x2 = r1["x"] + r1["width"]
            r2_x2 = r2["x"] + r2["width"]
            x_overlap = max(0, min(r1_x2, r2_x2) - max(r1["x"], r2["x"]))
            x_dist = 0 if x_overlap > 0 else max(0, r2["x"] - r1_x2, r1["x"] - r2_x2)
            r1_y2 = r1["y"] + r1["height"]
            r2_y2 = r2["y"] + r2["height"]
            y_overlap = max(0, min(r1_y2, r2_y2) - max(r1["y"], r2["y"]))
            y_dist = 0 if y_overlap > 0 else max(0, r2["y"] - r1_y2, r1["y"] - r2_y2)
            if (
                (x_overlap > 0 and y_overlap > 0)
                or (x_overlap > 0 and y_dist <= max_vertical_gap)
                or (y_overlap > 0 and x_dist <= max_horizontal_gap)
                or (x_dist <= max_horizontal_gap and y_dist <= max_vertical_gap)
            ):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    return components


def _random_page(rng, n):
    """Fragments on a 1000x1500 page, sized like real OCR line boxes.

    Half the pages are vertical-ish (tall narrow columns), half horizontal-ish (wide short lines),
    so both branches of the char-size choice are exercised. Boxes are allowed to overlap.
    """
    vertical = rng.random() < 0.5
    frags = []
    for _ in range(n):
        if vertical:
            w = rng.randint(15, 70)
            h = rng.randint(w, 400)
        else:
            h = rng.randint(12, 45)
            w = rng.randint(h, 500)
        frags.append(
            {
                "x": rng.randint(0, max(1, 1000 - w)),
                "y": rng.randint(0, max(1, 1500 - h)),
                "width": w,
                "height": h,
            }
        )
    return frags


THRESHOLDS = (0.15, 0.35, 0.5, 1.0, 2.0)
DIRECTIONS = ("rtl", "ltr")


def test_matches_legacy_on_random_corpus():
    """200 seeded pages x every threshold x both directions, exact match including ordering."""
    rng = random.Random(20260809)
    checked = 0
    for _ in range(200):
        frags = _random_page(rng, rng.randint(1, 12))
        for direction in DIRECTIONS:
            for thr in THRESHOLDS:
                expected = _legacy_components(frags, direction, thr)
                actual = group_fragments(frags, GroupingConfig(threshold_ratio=thr, reading_direction=direction))
                assert actual == expected, (
                    f"grouping diverged at direction={direction} threshold={thr}\n"
                    f"  fragments={frags}\n  legacy={expected}\n  actual={actual}"
                )
                checked += 1
    assert checked == 200 * len(DIRECTIONS) * len(THRESHOLDS)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_matches_legacy_on_degenerate_pages(direction, threshold):
    """Cases a random generator is unlikely to produce."""
    cases = [
        [{"x": 0, "y": 0, "width": 10, "height": 10}],  # single fragment
        [{"x": 0, "y": 0, "width": 10, "height": 10}] * 3,  # exact duplicates
        [
            {"x": 0, "y": 0, "width": 1, "height": 1},  # degenerate + huge
            {"x": 500, "y": 700, "width": 400, "height": 600},
        ],
        [{"x": i * 300, "y": 0, "width": 20, "height": 20} for i in range(4)],  # far apart
        [{"x": 0, "y": i * 12, "width": 60, "height": 10} for i in range(6)],  # a stacked column
    ]
    for frags in cases:
        expected = _legacy_components(frags, direction, threshold)
        actual = group_fragments(frags, GroupingConfig(threshold_ratio=threshold, reading_direction=direction))
        assert actual == expected, f"diverged on {frags}"


def test_empty_input():
    assert group_fragments([], GroupingConfig()) == []


def test_config_defaults_are_the_shipped_values():
    """A config built with no arguments must be the pre-split behaviour, not a new policy."""
    cfg = GroupingConfig()
    assert cfg.threshold_ratio == 0.50
    assert cfg.reading_direction == "rtl"


def test_config_is_immutable():
    """Frozen so a caller cannot mutate a shared config and silently change another call site."""
    cfg = GroupingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.threshold_ratio = 1.0  # type: ignore[misc]
