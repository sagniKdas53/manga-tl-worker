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

from worker.services.fragment_grouping import (
    GroupingConfig,
    GroupingContext,
    group_fragments,
    resolve_vertical,
)


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


TWO_COLUMNS = [
    {"x": 100, "y": 100, "width": 40, "height": 200},
    {"x": 150, "y": 100, "width": 40, "height": 200},
]


def _grouped(config, context=None):
    return group_fragments(TWO_COLUMNS, config, context)


def test_two_adjacent_columns_merge_without_the_gate():
    """Baseline for the gate tests: distance alone joins these."""
    assert _grouped(GroupingConfig(threshold_ratio=2.0)) == [[0, 1]]


def test_context_without_gate_changes_nothing():
    """A context is inert unless waist_gate is set, so callers may always pass one."""
    ctx = GroupingContext(clearance=lambda p, q: 0.0, solidity=0.5)
    assert _grouped(GroupingConfig(threshold_ratio=2.0), ctx) == [[0, 1]]


def test_gate_without_context_changes_nothing():
    """And a gate is inert without geometry to evaluate it against."""
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    assert _grouped(cfg) == [[0, 1]]
    assert _grouped(cfg, GroupingContext()) == [[0, 1]]


def test_gate_splits_when_the_path_is_pinched():
    """Low clearance on a pinched mask withholds the merge distance would have allowed."""
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    ctx = GroupingContext(clearance=lambda p, q: 5.0, solidity=0.7)  # 5px << 1.0 * 40px
    assert _grouped(cfg, ctx) == [[0], [1]]


def test_gate_allows_when_the_path_is_open():
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    ctx = GroupingContext(clearance=lambda p, q: 80.0, solidity=0.7)  # 80px >> 1.0 * 40px
    assert _grouped(cfg, ctx) == [[0, 1]]


def test_gate_does_not_apply_to_a_convex_mask():
    """The headline safety property: a mask with no waist cannot trigger a split.

    sample27's borderless shout sits at solidity 0.958, where the clearance measurement is noise.
    """
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    ctx = GroupingContext(clearance=lambda p, q: 0.0, solidity=0.958)
    assert _grouped(cfg, ctx) == [[0, 1]]


def test_gate_never_splits_fragments_whose_boxes_overlap():
    """The gate measures the space *between* two fragments. Overlapping boxes have none.

    _nearest_points collapses to a single point when the boxes overlap on both axes, so the
    clearance call reports how close that point sits to the outline rather than anything about
    separation -- and inside a tight balloon every column is within a character of the edge. The
    gate therefore split adjacent columns of one sentence (sample9 r13, sample27 r10), sending the
    translator a fragment with no verb. Overlap means one block, and the gate must stay silent.
    """
    overlapping = [
        {"x": 100, "y": 100, "width": 40, "height": 200},
        {"x": 130, "y": 120, "width": 40, "height": 200},  # 10px of x overlap, 180px of y
    ]
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    ctx = GroupingContext(clearance=lambda p, q: 0.0, solidity=0.5)  # maximally hostile
    assert group_fragments(overlapping, cfg, ctx) == [[0, 1]]

    # Touching edges are the boundary case: still a zero gap, still nothing to measure.
    touching = [
        {"x": 100, "y": 100, "width": 40, "height": 200},
        {"x": 140, "y": 100, "width": 40, "height": 200},
    ]
    assert group_fragments(touching, cfg, ctx) == [[0, 1]]

    # But a real gap must still be vetoable, or the fix has disabled the gate outright.
    assert _grouped(cfg, ctx) == [[0], [1]]


def test_gate_still_applies_to_boxes_that_only_clip_corners():
    """The exemption above is for one block, not for any intersection at all.

    Fragments in two different balloons of one fused blob can still cross at the corners. On
    sample9 such a pair shared 10% of its shorter side while the two columns of a single sentence
    shared 100%, and letting the corner case through the exemption cost a merger -- the expensive
    direction -- to save a split.
    """
    corner = [
        {"x": 100, "y": 100, "width": 40, "height": 200},
        {"x": 130, "y": 280, "width": 40, "height": 200},  # 10px x, 20px y: corners only
    ]
    cfg = GroupingConfig(threshold_ratio=2.0, waist_gate=1.0)
    ctx = GroupingContext(clearance=lambda p, q: 0.0, solidity=0.5)
    assert group_fragments(corner, cfg, ctx) == [[0], [1]]


def test_gate_can_only_withhold_merges_never_create_them():
    """Over the random corpus, gated grouping is always a refinement of ungated grouping."""
    rng = random.Random(7)
    for _ in range(60):
        frags = _random_page(rng, rng.randint(2, 10))
        base = group_fragments(frags, GroupingConfig(threshold_ratio=1.0))
        gated = group_fragments(
            frags,
            GroupingConfig(threshold_ratio=1.0, waist_gate=1.0),
            GroupingContext(clearance=lambda p, q: rng.uniform(0, 60), solidity=0.6),
        )
        assert len(gated) >= len(base)
        base_pairs = {(i, j) for comp in base for i in comp for j in comp if i < j}
        gated_pairs = {(i, j) for comp in gated for i in comp for j in comp if i < j}
        assert gated_pairs <= base_pairs, "the gate created a grouping distance did not allow"


def _cols(n, w=40, h=200):
    """n vertical columns, i.e. tall narrow boxes."""
    return [{"x": 100 + i * 60, "y": 100, "width": w, "height": h} for i in range(n)]


def _lines(n, w=200, h=30):
    """n horizontal lines, i.e. wide short boxes."""
    return [{"x": 100, "y": 100 + i * 50, "width": w, "height": h} for i in range(n)]


def test_orientation_defaults_to_the_binding_direction():
    """Off by default: 'rtl' still means vertical even when every box is a horizontal line."""
    assert resolve_vertical(_lines(5), GroupingConfig(reading_direction="rtl")) is True
    assert resolve_vertical(_cols(5), GroupingConfig(reading_direction="ltr")) is False


def test_vote_reads_orientation_off_the_fragments():
    cfg = GroupingConfig(reading_direction="rtl", orientation="vote")
    assert resolve_vertical(_lines(5), cfg) is False, "5 wide lines are horizontal despite rtl"
    assert resolve_vertical(_cols(5), cfg) is True


def test_vote_falls_back_when_the_page_is_genuinely_mixed():
    """No side reaching 60% of the cast weight means the binding direction is the safer answer."""
    mixed = _cols(2, w=40, h=200) + _lines(2, w=200, h=40)
    assert resolve_vertical(mixed, GroupingConfig(reading_direction="rtl", orientation="vote")) is True
    assert resolve_vertical(mixed, GroupingConfig(reading_direction="ltr", orientation="vote")) is False


def test_square_fragments_abstain_rather_than_voting():
    """Single CJK glyphs are near-square and carry no orientation evidence either way."""
    squares = [{"x": i * 40, "y": 0, "width": 30, "height": 30} for i in range(6)]
    cfg = GroupingConfig(reading_direction="rtl", orientation="vote")
    assert resolve_vertical(squares, cfg) is True, "abstentions must not flip the fallback"
    # One long horizontal line among squares still wins: votes are weighted by the longer side.
    assert resolve_vertical(squares + _lines(1, w=400, h=30), cfg) is False


def test_vote_does_not_disturb_a_genuinely_vertical_page():
    """C3's ship gate: detection must not flip the common case."""
    frags = _cols(4)
    plain = group_fragments(frags, GroupingConfig(threshold_ratio=0.35))
    voted = group_fragments(frags, GroupingConfig(threshold_ratio=0.35, orientation="vote"))
    assert voted == plain


def test_nearest_points_on_overlapping_and_disjoint_axes():
    from worker.services.fragment_grouping import _nearest_points

    a = {"x": 0, "y": 0, "width": 10, "height": 10}
    b = {"x": 30, "y": 0, "width": 10, "height": 10}
    # Disjoint in x, fully overlapping in y: the segment runs along the middle of the overlap.
    assert _nearest_points(a, b) == ((10.0, 5.0), (30.0, 5.0))
    # Overlapping on both axes: a zero-length segment at the centre of the intersection.
    c = {"x": 5, "y": 5, "width": 10, "height": 10}
    p, q = _nearest_points(a, c)
    assert p == q == (7.5, 7.5)


def test_config_is_immutable():
    """Frozen so a caller cannot mutate a shared config and silently change another call site."""
    cfg = GroupingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.threshold_ratio = 1.0  # type: ignore[misc]
