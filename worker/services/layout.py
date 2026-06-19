import re


def bubble_compare(a, b, reading_direction="rtl"):
    """Sort OCR bubbles within a panel according to *reading_direction*.

    Supported values (matching the series table field):
      'rtl' — right-to-left, top-before-bottom  (Japanese manga default)
      'ltr' — left-to-right, top-before-bottom  (Western comics)
      'ttb' — top-to-bottom strip               (webtoons / manhwa)
    """
    y_diff = a["y"] - b["y"]

    if reading_direction == "ttb":
        # Pure top-to-bottom: y position decides everything
        return 1 if y_diff > 0 else (-1 if y_diff < 0 else 0)

    # For both RTL and LTR, cluster into rows first (within 100 px)
    if abs(y_diff) > 100:
        return 1 if y_diff > 0 else -1

    # Within the same row: RTL puts rightmost bubble first, LTR puts leftmost first
    if reading_direction == "ltr":
        x_diff = a["x"] - b["x"]
    else:  # default: rtl
        x_diff = b["x"] - a["x"]

    return 1 if x_diff > 0 else -1


def classify_region_type(region, panel, image_width, image_height):
    """Classify an OCR region as speech/narration/sfx/caption/sign.

    Uses heuristic rules based on geometry, position, and text content.
    Returns one of: 'speech', 'narration', 'sfx', 'caption', 'sign'
    """
    text = region.get("text", "")
    confidence = region.get("confidence") or 1.0
    rx = region.get("bboxX") or region.get("x", 0)
    ry = region.get("bboxY") or region.get("y", 0)
    rw = region.get("bboxW") or region.get("width", 1)
    rh = region.get("bboxH") or region.get("height", 1)

    # Aspect ratios
    aspect = rw / max(rh, 1)
    tall_aspect = rh / max(rw, 1)

    # Check if text is kana-only (hiragana/katakana) — strong SFX signal
    cleaned = re.sub(r"[\s！？\?!\.\,\-\_\"]", "", text.strip())
    is_kana_only = False
    if cleaned:
        is_kana_only = bool(
            re.match(r"^[\u3040-\u309F\u30A0-\u30FF\u30FC\uFF66-\uFF9F]+$", cleaned)
        )

    # --- SFX detection ---
    # Kana-only text or very tall narrow region (vertical SFX)
    if is_kana_only and len(cleaned) <= 5:
        return "sfx"
    if tall_aspect > 3.0 and len(text.strip()) <= 6:
        return "sfx"

    # --- Check if region is inside any panel ---
    in_panel = panel is not None

    # --- Caption: outside all panels, at page top or bottom edges ---
    if not in_panel:
        # Top 8% or bottom 8% of the image
        if ry < image_height * 0.08 or (ry + rh) > image_height * 0.92:
            return "caption"
        # Outside panels but not at page edges — could be narration box overlay
        if aspect > 2.5:
            return "narration"
        return "caption"

    # --- Narration: very wide region or at panel edge ---
    if panel is not None:
        px = panel.get("bboxX", 0)
        py = panel.get("bboxY", 0)
        pw = panel.get("bboxW", 1)
        ph = panel.get("bboxH", 1)

        # Wide aspect ratio relative to panel width — narration box
        if aspect > 3.0 and rw > pw * 0.6:
            return "narration"

        # At very top or bottom edge of panel (within 8% of panel height)
        rel_top = (ry - py) / max(ph, 1)
        rel_bottom = ((py + ph) - (ry + rh)) / max(ph, 1)
        if (rel_top < 0.08 or rel_bottom < 0.08) and aspect > 2.0:
            return "narration"

    # --- Sign: inside panel but low confidence and small ---
    if in_panel and confidence < 0.50:
        region_area = rw * rh
        panel_area = panel.get("bboxW", 1) * panel.get("bboxH", 1) if panel else 1
        if panel_area > 0 and region_area / panel_area < 0.05:
            return "sign"

    # Default: speech bubble
    return "speech"


def _finish_conversation_group(group, panel_ids):
    """Convert a group of regions into a conversation dict."""
    region_ids = [r.get("id", "") for r in group]
    scene_type = "monologue" if len(region_ids) == 1 else "dialogue"
    return {
        "regionIds": region_ids,
        "sceneType": scene_type,
        "panelIds": list(panel_ids),
    }


def group_conversations(regions, panels, reading_direction="rtl"):
    """Group OCR regions into conversation clusters.

    Algorithm:
    1. For each panel, collect its assigned regions sorted by bubble reading order.
    2. Within a panel, group regions into conversations using spatial proximity:
       - Two regions belong to the same conversation if their vertical gap is
         ≤ 1.5× the average bubble height in the panel.
       - Narration and SFX regions start their own group.
    3. Assign scene_type based on the region types within each group.

    Returns list of:
      {"regionIds": [region_id, ...], "sceneType": "dialogue"|..., "panelIds": [...]}
    """
    # Build panel → regions mapping
    panel_map = {}  # panel_reading_order → list of regions
    unmapped = []

    for r in regions:
        panel_order = r.get("panelReadingOrder") or 0
        if panel_order > 0:
            if panel_order not in panel_map:
                panel_map[panel_order] = []
            panel_map[panel_order].append(r)
        else:
            unmapped.append(r)

    conversations = []

    for panel_order in sorted(panel_map.keys()):
        panel_regions = panel_map[panel_order]
        # Sort by bubble reading order
        panel_regions.sort(key=lambda r: r.get("bubbleReadingOrder", 0))

        # Calculate average bubble height for proximity threshold
        heights = [r.get("bboxH") or r.get("height", 50) for r in panel_regions]
        avg_height = sum(heights) / len(heights) if heights else 50
        proximity_threshold = avg_height * 1.5

        current_group = []
        current_panel_ids = set()

        for r in panel_regions:
            region_type = r.get("regionType") or r.get("region_type") or "speech"
            rid = r.get("id", "")
            ry = r.get("bboxY") or r.get("y", 0)

            # Narration and SFX always start their own group
            if region_type in ("narration", "sfx", "caption", "sign"):
                # Flush current dialogue group
                if current_group:
                    conversations.append(
                        _finish_conversation_group(current_group, current_panel_ids)
                    )
                    current_group = []
                    current_panel_ids = set()

                # Single-region group for narration/sfx
                scene = (
                    "narration"
                    if region_type in ("narration", "caption")
                    else "sfx_cluster"
                )
                conversations.append(
                    {
                        "regionIds": [rid],
                        "sceneType": scene,
                        "panelIds": [str(panel_order)],
                    }
                )
                continue

            # Speech/thought — group by spatial proximity
            if current_group:
                last_r = current_group[-1]
                last_bottom = (last_r.get("bboxY") or last_r.get("y", 0)) + (
                    last_r.get("bboxH") or last_r.get("height", 0)
                )
                gap = ry - last_bottom
                if gap > proximity_threshold:
                    # Start new group
                    conversations.append(
                        _finish_conversation_group(current_group, current_panel_ids)
                    )
                    current_group = []
                    current_panel_ids = set()

            current_group.append(r)
            current_panel_ids.add(str(panel_order))

        # Flush remaining group
        if current_group:
            conversations.append(
                _finish_conversation_group(current_group, current_panel_ids)
            )

    # Handle unmapped regions (outside all panels)
    for r in unmapped:
        rid = r.get("id", "")
        region_type = r.get("regionType") or r.get("region_type") or "speech"
        scene = (
            "narration"
            if region_type in ("narration", "caption")
            else ("sfx_cluster" if region_type == "sfx" else "dialogue")
        )
        conversations.append(
            {
                "regionIds": [rid],
                "sceneType": scene,
                "panelIds": [],
            }
        )

    return conversations


def chunk_regions_by_conversation(unmatched_regions, conversations, max_batch_size):
    region_map = {r["id"]: r for r in unmatched_regions}
    grouped_rids = set()
    chunks = []
    current_chunk = []

    for conv in conversations:
        conv_regions = []
        for rid in conv.get("regionIds", []):
            if rid in region_map:
                conv_regions.append(region_map[rid])
                grouped_rids.add(rid)

        if not conv_regions:
            continue

        if len(current_chunk) + len(conv_regions) > max_batch_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []

        current_chunk.extend(conv_regions)

    for r in unmatched_regions:
        if r["id"] not in grouped_rids:
            if len(current_chunk) >= max_batch_size:
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(r)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
