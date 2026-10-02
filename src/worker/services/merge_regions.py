import json
import logging
import os
import re

from worker.services.fragment_grouping import (
    DEFAULT_THRESHOLD_RATIO,
    GroupingConfig,
    GroupingContext,
    group_fragments,
)

logger = logging.getLogger("translation")


def _parse_polygon(mask_polygon):
    if not mask_polygon:
        return None
    try:
        pts = json.loads(mask_polygon) if isinstance(mask_polygon, str) else mask_polygon
    except Exception:
        return None
    if not isinstance(pts, list) or len(pts) < 3:
        return None
    polygon = []
    for pt in pts:
        if not isinstance(pt, list) or len(pt) != 2:
            return None
        polygon.append([int(pt[0]), int(pt[1])])
    return polygon


def resolve_component_container(regions, comp):
    """Return a trusted component container or an explicit review state.

    A convex hull spanning separate detector polygons is not a local container. It grants cleanup
    authority over the gutter and both bubbles. Components with incompatible or absent polygons
    therefore retain their local raw membership and enter review without any replacement mask.
    """
    polygons = [_parse_polygon(regions[idx].get("maskPolygon")) for idx in comp]
    if any(polygon is None for polygon in polygons):
        return None, "review-missing-container"
    first = polygons[0]
    if all(polygon == first for polygon in polygons[1:]):
        return json.dumps(first), "resolved-shared-container"
    return None, "review-incompatible-containers"


def _split_cross_panel_components(components, regions):
    """Veto a candidate that spans known distinct panels.

    A shared panel is not owner evidence, but distinct panels are incompatible source geometry.
    Preserve raw local regions rather than concatenate text across a gutter or panel boundary.
    """

    bounded = []
    for component in components:
        panel_ids = {regions[index].get("panelId") for index in component if regions[index].get("panelId") is not None}
        if len(panel_ids) > 1:
            bounded.extend([[index] for index in component])
        else:
            bounded.append(component)
    return bounded


def merge_ocr_regions(
    regions: list,
    reading_direction: str = "rtl",
    threshold_ratio: float | None = None,
    grouping: GroupingConfig | None = None,
    context: GroupingContext | None = None,
) -> list:
    """Merge OCR line-level detections into logical speech balloon groups.

    Args:
        regions: List of OCR region dicts with x, y, width, height, text keys
        reading_direction: 'rtl' or 'ltr'
        threshold_ratio: Optional override for the merge proximity threshold multiplier
        grouping: Optional full grouping configuration. When given it supersedes
            reading_direction and threshold_ratio, which are then ignored.
        context: Optional per-call geometry (balloon clearance field and mask solidity). Inert
            unless the configuration enables a gate that consults it.

    Returns:
        Merged region list with concatenated text and union bounding boxes.
    """
    if not regions:
        return []

    if grouping is None:
        # Legacy entry: no configuration object, so fall back to the environment. Every call site
        # in handlers/ocr.py now passes `grouping=` (see its grouping_config), so this path serves
        # direct callers and tests only, and it deliberately keeps the pre-split default rather
        # than tracking config.OCR_MERGE_THRESHOLD -- the frozen equivalence test in
        # test_fragment_grouping.py asserts that no-argument behaviour is unchanged.
        if threshold_ratio is None:
            try:
                threshold_ratio = float(os.environ.get("OCR_MERGE_THRESHOLD", str(DEFAULT_THRESHOLD_RATIO)))
            except ValueError:
                threshold_ratio = DEFAULT_THRESHOLD_RATIO
        grouping = GroupingConfig(threshold_ratio=threshold_ratio, reading_direction=reading_direction)

    reading_direction = grouping.reading_direction
    threshold_ratio = grouping.threshold_ratio

    n = len(regions)
    components = _split_cross_panel_components(group_fragments(regions, grouping, context), regions)

    # Merge each component into a single region
    merged_regions = []
    cjk_pattern = re.compile(r"[\u3040-\u9FFF\uF900-\uFAFF]")

    for comp in components:
        component_mask, container_resolution = resolve_component_container(regions, comp)
        raw_membership = [{"index": index, "fragment_id": regions[index].get("fragmentId")} for index in comp]
        if len(comp) == 1:
            merged_regions.append(
                {
                    **regions[comp[0]],
                    "rawFragmentMembership": raw_membership,
                    "containerResolution": container_resolution,
                }
            )
            continue

        # Sort indices in reading order inside the component
        if reading_direction == "rtl":
            # Right-to-left: larger X first, then top-to-bottom (smaller Y)
            comp.sort(key=lambda idx: (-regions[idx]["x"], regions[idx]["y"]))
        else:
            # Left-to-right: smaller X first, then top-to-bottom (smaller Y)
            comp.sort(key=lambda idx: (regions[idx]["x"], regions[idx]["y"]))

        texts_to_join = []
        for idx in comp:
            t = regions[idx]["text"].strip()
            if t:
                texts_to_join.append(t)

        raw_membership = [{"index": index, "fragment_id": regions[index].get("fragmentId")} for index in comp]
        for member, index in zip(raw_membership, comp, strict=True):
            provenance = regions[index].get("ownershipProvenance")
            if provenance is not None:
                member["provenance"] = provenance
        has_cjk = any(cjk_pattern.search(t) for t in texts_to_join)
        joined_text = "".join(texts_to_join) if has_cjk else " ".join(texts_to_join)

        # Calculate union bounding box
        x_min = min(regions[idx]["x"] for idx in comp)
        y_min = min(regions[idx]["y"] for idx in comp)
        x_max = max(regions[idx]["x"] + regions[idx]["width"] for idx in comp)
        y_max = max(regions[idx]["y"] + regions[idx]["height"] for idx in comp)

        # Average confidence
        avg_conf = sum(regions[idx]["confidence"] for idx in comp) / len(comp)

        # Most common detected language
        langs = [regions[idx]["detectedLanguage"] for idx in comp]
        most_common_lang = max(set(langs), key=langs.count)

        # Get background color of the first region in the component
        bg_color = regions[comp[0]].get("backgroundColor", "#ffffff")

        # Bubble coordinates (union of bubble coordinates of elements in component)
        bx_min = min(regions[idx].get("bubbleX", regions[idx]["x"]) for idx in comp)
        by_min = min(regions[idx].get("bubbleY", regions[idx]["y"]) for idx in comp)
        bx_max = max(
            regions[idx].get("bubbleX", regions[idx]["x"]) + regions[idx].get("bubbleWidth", regions[idx]["width"])
            for idx in comp
        )
        by_max = max(
            regions[idx].get("bubbleY", regions[idx]["y"]) + regions[idx].get("bubbleHeight", regions[idx]["height"])
            for idx in comp
        )

        # Safe area coordinates
        sx_min = min(regions[idx].get("safeTextX", regions[idx]["x"]) for idx in comp)
        sy_min = min(regions[idx].get("safeTextY", regions[idx]["y"]) for idx in comp)
        sx_max = max(
            regions[idx].get("safeTextX", regions[idx]["x"]) + regions[idx].get("safeTextW", regions[idx]["width"])
            for idx in comp
        )
        sy_max = max(
            regions[idx].get("safeTextY", regions[idx]["y"]) + regions[idx].get("safeTextH", regions[idx]["height"])
            for idx in comp
        )
        merged_mask_polygon = component_mask

        merged_regions.append(
            {
                "text": joined_text,
                "detectedLanguage": most_common_lang,
                "confidence": float(avg_conf),
                "rotation": 0.0,
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "panelId": None,
                "bubbleReadingOrder": 0,
                "backgroundColor": bg_color,
                "bubbleX": bx_min,
                "bubbleY": by_min,
                "bubbleWidth": bx_max - bx_min,
                "bubbleHeight": by_max - by_min,
                "bubbleId": None,
                "detectionConfidence": float(
                    sum(regions[idx].get("detectionConfidence", 0.0) for idx in comp) / len(comp)
                ),
                "maskPolygon": merged_mask_polygon,
                "safeTextX": sx_min,
                "safeTextY": sy_min,
                "safeTextW": sx_max - sx_min,
                "safeTextH": sy_max - sy_min,
                "rawFragmentMembership": raw_membership,
                "ownershipProvenance": {"fragments": raw_membership, "containerResolution": container_resolution},
                "containerResolution": container_resolution,
            }
        )

    logger.info(f"[OCR] Merged {n} regions into {len(merged_regions)} regions (threshold={threshold_ratio})")
    return merged_regions
