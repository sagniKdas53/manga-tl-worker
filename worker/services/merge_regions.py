import os
import re
import logging
import json

logger = logging.getLogger("translation")


def _parse_polygon(mask_polygon):
    if not mask_polygon:
        return None
    try:
        pts = (
            json.loads(mask_polygon)
            if isinstance(mask_polygon, str)
            else mask_polygon
        )
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


def _polygon_area(points):
    area = 0
    for idx, p1 in enumerate(points):
        p2 = points[(idx + 1) % len(points)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(area) / 2


def _convex_hull(points):
    unique = sorted({(p[0], p[1]) for p in points})
    if len(unique) <= 1:
        return [[p[0], p[1]] for p in unique]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [[p[0], p[1]] for p in hull]


def _merged_mask_polygon(regions, comp):
    polygons = [
        polygon
        for polygon in (
            _parse_polygon(regions[idx].get("maskPolygon")) for idx in comp
        )
        if polygon
    ]
    if not polygons:
        return None
    if len(polygons) == 1:
        return json.dumps(polygons[0])

    first = polygons[0]
    if all(poly == first for poly in polygons[1:]):
        return json.dumps(first)

    points = [pt for polygon in polygons for pt in polygon]
    hull = _convex_hull(points)
    if len(hull) >= 3:
        return json.dumps(hull)

    largest = max(polygons, key=_polygon_area)
    return json.dumps(largest)


def merge_ocr_regions(regions: list, reading_direction: str = "rtl") -> list:
    """Merge OCR line-level detections into logical speech balloon groups.

    Args:
        regions: List of OCR region dicts with x, y, width, height, text keys
        reading_direction: 'rtl' or 'ltr'

    Returns:
        Merged region list with concatenated text and union bounding boxes.
    """
    if not regions:
        return []

    # Get configuration threshold
    try:
        threshold_ratio = float(os.environ.get("OCR_MERGE_THRESHOLD", "0.50"))
    except ValueError:
        threshold_ratio = 0.50

    # Compute average height and width to establish relative proximity guidelines
    avg_height = sum(r["height"] for r in regions) / len(regions)
    avg_width = sum(r["width"] for r in regions) / len(regions)

    # For vertical Japanese text (typically reading_direction == "rtl"),
    # the character/font size is represented by the line's width, so the vertical gap
    # threshold should be scaled relative to avg_width rather than avg_height.
    # For horizontal text (LTR), the character size is represented by the line's height.
    if reading_direction == "rtl":
        char_size_vertical = avg_width
    else:
        char_size_vertical = avg_height

    max_vertical_gap = char_size_vertical * threshold_ratio
    max_horizontal_gap = avg_width * threshold_ratio

    n = len(regions)
    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            r1 = regions[i]
            r2 = regions[j]

            # Calculate horizontal gap
            r1_x2 = r1["x"] + r1["width"]
            r2_x2 = r2["x"] + r2["width"]
            x_overlap = max(0, min(r1_x2, r2_x2) - max(r1["x"], r2["x"]))
            x_dist = 0 if x_overlap > 0 else max(0, r2["x"] - r1_x2, r1["x"] - r2_x2)

            # Calculate vertical gap
            r1_y2 = r1["y"] + r1["height"]
            r2_y2 = r2["y"] + r2["height"]
            y_overlap = max(0, min(r1_y2, r2_y2) - max(r1["y"], r2["y"]))
            y_dist = 0 if y_overlap > 0 else max(0, r2["y"] - r1_y2, r1["y"] - r2_y2)

            # Conditions to merge:
            # 1. Overlap both horizontally and vertically
            # 2. Horizontal overlap and vertical proximity
            # 3. Vertical overlap and horizontal proximity
            # 4. Close diagonally
            should_merge = False
            if x_overlap > 0 and y_overlap > 0:
                should_merge = True
            elif x_overlap > 0 and y_dist <= max_vertical_gap:
                should_merge = True
            elif y_overlap > 0 and x_dist <= max_horizontal_gap:
                should_merge = True
            elif x_dist <= max_horizontal_gap and y_dist <= max_vertical_gap:
                should_merge = True

            if should_merge:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components (clusters) using BFS
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

    # Merge each component into a single region
    merged_regions = []
    cjk_pattern = re.compile(r"[\u3040-\u9FFF\uF900-\uFAFF]")

    for comp in components:
        if len(comp) == 1:
            merged_regions.append(regions[comp[0]])
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

        if not texts_to_join:
            joined_text = ""
        else:
            # Check if any part contains CJK characters to decide on spacer-less join
            has_cjk = any(cjk_pattern.search(t) for t in texts_to_join)
            if has_cjk:
                joined_text = "".join(texts_to_join)
            else:
                joined_text = " ".join(texts_to_join)

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
            regions[idx].get("bubbleX", regions[idx]["x"])
            + regions[idx].get("bubbleWidth", regions[idx]["width"])
            for idx in comp
        )
        by_max = max(
            regions[idx].get("bubbleY", regions[idx]["y"])
            + regions[idx].get("bubbleHeight", regions[idx]["height"])
            for idx in comp
        )

        # Safe area coordinates
        sx_min = min(regions[idx].get("safeTextX", regions[idx]["x"]) for idx in comp)
        sy_min = min(regions[idx].get("safeTextY", regions[idx]["y"]) for idx in comp)
        sx_max = max(
            regions[idx].get("safeTextX", regions[idx]["x"])
            + regions[idx].get("safeTextW", regions[idx]["width"])
            for idx in comp
        )
        sy_max = max(
            regions[idx].get("safeTextY", regions[idx]["y"])
            + regions[idx].get("safeTextH", regions[idx]["height"])
            for idx in comp
        )
        merged_mask_polygon = _merged_mask_polygon(regions, comp)

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
                    sum(regions[idx].get("detectionConfidence", 0.0) for idx in comp)
                    / len(comp)
                ),
                "maskPolygon": merged_mask_polygon,
                "safeTextX": sx_min,
                "safeTextY": sy_min,
                "safeTextW": sx_max - sx_min,
                "safeTextH": sy_max - sy_min,
            }
        )

    logger.info(
        f"[OCR] Merged {n} regions into {len(merged_regions)} regions (threshold={threshold_ratio})"
    )
    return merged_regions
