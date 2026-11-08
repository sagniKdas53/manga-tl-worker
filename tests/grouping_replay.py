"""Grouping from stored piece positions, with no page image and no OCR.

Follows `process_ocr`'s steps through the worker's own functions: assign pieces to balloons, group
inside each balloon, then group what's left by panel and spread. The only inputs it takes from the
page itself are stored as numbers (text containers and the spread gutter). The balloon join's wall
check reads the page, so a page where it matters isn't kept in the fixture.
"""

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from worker.handlers.ocr import (
    OCR_BALLOON_JOIN_BUDGET,
    OCR_BALLOON_JOIN_MAX_LINES,
    OCR_BALLOON_WALL_STROKE,
    OCR_NO_BALLOON_VETO,
    bubble_covers_text,
    no_balloon_grouping_context,
    owner_aware_grouping_context,
    partition_unmatched_fragments_by_panel,
)
from worker.services.balloon_join import balloon_join
from worker.services.bubble_geometry import bubble_grouping_context, gap_wall
from worker.services.fragment_grouping import GroupingConfig, GroupingContext, group_fragments
from worker.services.spread_gutter import split_fragments_at_gutter


def group_page(page):
    """The page's groups of piece ids. `page` is one entry of the fixture."""
    width, height = page["width"], page["height"]
    gray = np.full((height, width), 255, dtype=np.uint8)
    pieces: list[dict[str, Any]] = [
        {**piece, "text": "あ", "confidence": 1.0, "detectedLanguage": "ja"} for piece in page["pieces"]
    ]

    masks = []
    for points in page["balloons"]:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], (255,))
        masks.append(mask)

    for piece in pieces:
        best, most = -1, 0
        x1, y1 = max(0, min(width - 1, piece["x"])), max(0, min(height - 1, piece["y"]))
        x2, y2 = max(0, min(width, piece["x"] + piece["width"])), max(0, min(height, piece["y"] + piece["height"]))
        if x2 > x1 and y2 > y1:
            for index, mask in enumerate(masks):
                overlap = np.sum(mask[y1:y2, x1:x2] > 0)
                if overlap > most:
                    best, most = index, overlap
            if best >= 0 and not bubble_covers_text(masks[best], x1, y1, x2, y2):
                best = -1
        piece["bubble_idx"] = best

    grouping = GroupingConfig(**page["grouping"])
    page_area = float(width * height)
    groups = []
    for index, mask in enumerate(masks):
        assigned = [piece for piece in pieces if piece["bubble_idx"] == index]
        if not assigned:
            continue
        context = bubble_grouping_context(mask, page["balloons"][index])
        if context is not None:
            context = replace(context, page_area=page_area)
        context = owner_aware_grouping_context(
            context, [{"format": "polygon", "id": f"bubble-{index}", "points": page["balloons"][index]}]
        )
        if OCR_BALLOON_JOIN_BUDGET > 0 and len(assigned) > 1:
            wall = gap_wall(gray, mask, OCR_BALLOON_WALL_STROKE)
            context = replace(
                context,
                group_join=balloon_join(grouping, context, OCR_BALLOON_JOIN_BUDGET, wall, OCR_BALLOON_JOIN_MAX_LINES),
            )
        groups.extend(
            [assigned[i]["fragmentId"] for i in group] for group in group_fragments(assigned, grouping, context)
        )

    unmatched = [piece for piece in pieces if piece["bubble_idx"] == -1]
    page_context = GroupingContext(page_area=page_area)
    if OCR_NO_BALLOON_VETO:
        page_context = no_balloon_grouping_context(page_context)
    for partition in partition_unmatched_fragments_by_panel(unmatched, page["containers"]).values():
        for part in split_fragments_at_gutter(partition, page["gutter"]):
            groups.extend(
                [part[i]["fragmentId"] for i in group] for group in group_fragments(part, grouping, page_context)
            )
    return sorted(sorted(group) for group in groups)
