"""Tilted text at its angle (E2, #180): each region's angle and the box measured along its text.

The rule is E0's, settled by the owner on 2026-10-09 and kept in step with
``scripts/e0/angle_rule.py`` in the parent repo (the tests here pin the same cases). A region's
angle is the length-weighted direction of its OCR line pieces:

1. A piece's direction is its quad's long edge. A piece votes only when that edge is at least
   ``MIN_LINE_THICKNESSES`` times its short edge: a single glyph has no direction.
2. A voter's angle is how far it is off the nearest axis (±45°). Once any voter is more than
   ``READ_ALONG_PAST_DEGREES`` off, every voter is read as a line along its long edge instead
   (±90°), or text near 45° reads +44° on one piece and -44° on the next (#265).
3. Directions wrap at ±90°, so they are unwrapped around the voter closest to all the others.
4. With three or more voters spread wider than ``AGREE_DEGREES``, the one farthest from the
   length-weighted median is dropped, once (#266). Still spread: the region stays level.
5. The angle is the length-weighted mean; under ``DEAD_BAND_DEGREES`` the region stays level.

Positive is clockwise on the page (y runs down), as ``rotation`` is in the editor and the scene.

A turned region also gets a *text area*: the box around its pieces measured along the text, given
as the level box that, turned by the angle about its own centre, covers the text. That is how the
editor (``rectToPolygon``) and the scene draw an element, so the backend can use it as the text
box as it is. A level region gets none, and keeps today's box.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

DEAD_BAND_DEGREES = 5.0
MIN_LINE_THICKNESSES = 1.5
AGREE_DEGREES = 6.0
READ_ALONG_PAST_DEGREES = 35.0

TURNED = "turned"
DEAD_BAND = "dead band"
PIECES_DISAGREE = "pieces disagree"
NO_LONG_PIECE = "no long piece"
SWITCHED_OFF = "switched off"

Point = Sequence[float]
Quad = Sequence[Point]


def _fold90(angle: float) -> float:
    """A line's direction, in [-90, 90)."""
    return ((angle + 90) % 180) - 90


def _axis_deviation(angle: float) -> float:
    """How far a line direction is off the nearest axis (0° or 90°), in [-45, 45)."""
    return ((angle + 45) % 90) - 45


def _gap(a: float, b: float) -> float:
    """The angle between two line directions, which wrap at ±90°."""
    return abs(((a - b + 90) % 180) - 90)


def provenance_pieces(provenance) -> list[dict]:
    """The OCR pieces of a region's ownership provenance.

    It comes in two shapes: one piece's own record, or a merged region's
    ``{"containerResolution": ..., "fragments": [{"index": i, "provenance": {...}}, ...]}``.
    """
    if not isinstance(provenance, dict):
        return []
    if "fragments" in provenance:
        pieces = []
        for fragment in provenance.get("fragments") or []:
            if isinstance(fragment, dict):
                piece = fragment.get("provenance", fragment)
                if isinstance(piece, dict):
                    pieces.append(piece)
        return pieces
    return [provenance]


def piece_quads(provenance) -> list[list[list[float]]]:
    """Each piece's ``sourceQuad`` (page pixels), skipping pieces without a four-point quad."""
    quads = []
    for piece in provenance_pieces(provenance):
        quad = piece.get("sourceQuad")
        if isinstance(quad, (list, tuple)) and len(quad) == 4:
            try:
                quads.append([[float(point[0]), float(point[1])] for point in quad])
            except (TypeError, ValueError, IndexError):
                continue
    return quads


def _long_edge(quad: Quad) -> tuple[float, float, float]:
    """(direction in degrees, long edge length, short edge length) of a quad."""
    e0 = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    e1 = (quad[2][0] - quad[1][0], quad[2][1] - quad[1][1])
    l0, l1 = math.hypot(*e0), math.hypot(*e1)
    edge, long_len, short_len = (e0, l0, l1) if l0 >= l1 else (e1, l1, l0)
    return math.degrees(math.atan2(edge[1], edge[0])), long_len, max(short_len, 1e-6)


def _votes(quads: Iterable[Quad]) -> list[tuple[float, float]]:
    """(angle, length) of each piece long enough to vote."""
    long_pieces = []
    for quad in quads:
        direction, long_len, short_len = _long_edge(quad)
        if long_len >= MIN_LINE_THICKNESSES * short_len:
            long_pieces.append((direction, long_len))
    deviations = [_axis_deviation(direction) for direction, _ in long_pieces]
    # Past 35° every piece is read along its long edge. That assumes horizontal lines: a vertical
    # column tilted this far would read about 90° off.
    if any(abs(d) > READ_ALONG_PAST_DEGREES for d in deviations):
        deviations = [_fold90(direction) for direction, _ in long_pieces]
    return [(d, length) for d, (_, length) in zip(deviations, long_pieces, strict=True)]


def _spread(votes: list[tuple[float, float]]) -> float:
    return max(d for d, _ in votes) - min(d for d, _ in votes)


def text_angle(quads: Iterable[Quad]) -> tuple[float, str]:
    """(angle, why) for one region's piece quads; the angle is 0.0 unless ``why`` is ``turned``."""
    votes = _votes(quads)
    if not votes:
        return 0.0, NO_LONG_PIECE
    centre = min(votes, key=lambda c: sum(length * _gap(c[0], d) for d, length in votes))[0]
    votes = [(centre + ((d - centre + 90) % 180) - 90, length) for d, length in votes]
    if _spread(votes) > AGREE_DEGREES and len(votes) >= 3:
        total, running = sum(length for _, length in votes), 0.0
        median = votes[0][0]
        for d, length in sorted(votes):
            running += length
            if running >= total / 2:
                median = d
                break
        farthest = max(votes, key=lambda v: abs(v[0] - median))
        votes = [v for v in votes if v is not farthest]
    if _spread(votes) > AGREE_DEGREES:
        return 0.0, PIECES_DISAGREE
    angle = _fold90(sum(d * length for d, length in votes) / sum(length for _, length in votes))
    if abs(angle) < DEAD_BAND_DEGREES:
        return 0.0, DEAD_BAND
    return angle, TURNED


def text_area(quads: Sequence[Quad], angle: float) -> tuple[float, float, float, float] | None:
    """The box along the text: (x, y, width, height) of the level box that, turned by ``angle``
    about its own centre, is the tightest box at that angle around every piece."""
    points = [point for quad in quads for point in quad]
    if not points:
        return None
    radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    # Into the text's frame: turn every point by -angle about the origin.
    along = [(x * cos_a + y * sin_a, -x * sin_a + y * cos_a) for x, y in points]
    u_min, u_max = min(u for u, _ in along), max(u for u, _ in along)
    v_min, v_max = min(v for _, v in along), max(v for _, v in along)
    width, height = u_max - u_min, v_max - v_min
    centre_u, centre_v = (u_min + u_max) / 2, (v_min + v_max) / 2
    # The box's centre back on the page (turn by +angle).
    centre_x = centre_u * cos_a - centre_v * sin_a
    centre_y = centre_u * sin_a + centre_v * cos_a
    return centre_x - width / 2, centre_y - height / 2, width, height


def apply_text_angles(regions: Iterable[dict], enabled: bool) -> None:
    """Set each region's ``rotation`` and, for a turned one, its ``textArea*`` box, in place.

    Every region records the angle the rule found and why in its ``ownershipProvenance`` under
    ``textAngle``, so *Show debug -> OCR fragments* can show why a region turned or didn't, even
    with the switch off.
    """
    for region in regions:
        provenance = region.get("ownershipProvenance")
        quads = piece_quads(provenance)
        angle, why = text_angle(quads)
        turned = enabled and why == TURNED
        region["rotation"] = round(angle, 2) if turned else 0.0
        for key in ("textAreaX", "textAreaY", "textAreaW", "textAreaH"):
            region.pop(key, None)
        if turned:
            box = text_area(quads, angle)
            if box is not None:
                x, y, w, h = box
                region.update(
                    {
                        "textAreaX": round(x, 1),
                        "textAreaY": round(y, 1),
                        "textAreaW": round(w, 1),
                        "textAreaH": round(h, 1),
                    }
                )
        if isinstance(provenance, dict):
            record = {"angle": round(angle, 2), "why": why if enabled else SWITCHED_OFF}
            if not enabled and why == TURNED:
                record["wouldTurn"] = True
            region["ownershipProvenance"] = provenance | {"textAngle": record}
