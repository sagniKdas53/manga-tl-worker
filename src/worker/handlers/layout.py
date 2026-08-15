import logging

import requests

from worker.config import CALLBACK_URL, backend_headers, redis_client
from worker.services.layout import classify_region_type, group_conversations

logger = logging.getLogger(__name__)


def process_layout(job_data):
    """Layout analysis: classify region types and group conversations."""
    image_id = job_data.get("imageId")
    page_id = job_data.get("pageId")

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:layout")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    logger.info(f"[Layout] Processing page: {page_id or image_id}{progress_str}")

    # 1. Fetch OCR regions + panels from backend
    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[Layout] Failed to get page/image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        panels = image_info.get("panels", [])
    except Exception as e:
        logger.error(f"[Layout] Error fetching image details: {e}")
        raise

    if not ocr_regions:
        logger.warning("[Layout] No OCR regions found, skipping layout analysis.")
        # Still send callback so pipeline continues
        callback_payload = {
            "jobId": job_data.get("jobId"),
            "imageId": image_id,
            "pageId": page_id,
            "regionTypes": [],
            "conversations": [],
        }
        try:
            res = requests.post(f"{CALLBACK_URL}/layout", json=callback_payload, headers=backend_headers())
            logger.debug(f"[Layout] Callback status code: {res.status_code}")
        except Exception as e:
            logger.error(f"[Layout] Failed to post callback: {e}")
        return

    # Get image dimensions from the first panel or estimate from regions
    image_width = max(
        (p.get("bboxX", 0) + p.get("bboxW", 0) for p in panels),
        default=max((r.get("bboxX", 0) + r.get("bboxW", 0) for r in ocr_regions), default=1000),
    )
    image_height = max(
        (p.get("bboxY", 0) + p.get("bboxH", 0) for p in panels),
        default=max((r.get("bboxY", 0) + r.get("bboxH", 0) for r in ocr_regions), default=1400),
    )

    # Build panel lookup by ID
    panel_by_id = {}
    for p in panels:
        pid = p.get("id") or p.get("panelId")
        if pid:
            panel_by_id[str(pid)] = p

    # 2. Classify each region type
    region_types = []
    for r in ocr_regions:
        # Find matching panel for this region
        panel_id = r.get("panelId") or r.get("panel_id")
        panel = panel_by_id.get(str(panel_id)) if panel_id else None

        rtype = classify_region_type(r, panel, image_width, image_height)
        r["regionType"] = rtype  # Annotate in-memory for conversation grouping
        region_types.append(
            {
                "regionId": str(r.get("id", "")),
                "regionType": rtype,
            }
        )
        logger.info(
            f"[Layout] Region {str(r.get('id', ''))[:8]}... type={rtype} text='{(r.get('text', '') or '')[:30]}'"
        )

    logger.info(
        "[Layout] Region types: "
        + ", ".join(
            f"{t}: {sum(1 for rt in region_types if rt['regionType'] == t)}"
            for t in set(rt["regionType"] for rt in region_types)
        )
    )

    # 3. Group conversations
    reading_direction = "rtl"  # Default; could be passed in job_data if needed
    conversations = group_conversations(ocr_regions, panels, reading_direction)
    logger.info(f"[Layout] Grouped {len(ocr_regions)} regions into {len(conversations)} conversations")

    # Detailed logging for the grouped conversations
    logger.info("[Layout] --- Conversation Grouping Details ---")
    for idx, conv in enumerate(conversations):
        region_details = []
        for rid in conv["regionIds"]:
            reg = next((r for r in ocr_regions if str(r.get("id")) == rid), None)
            if reg:
                text = reg.get("text", "").strip().replace("\n", " ")
                rtype = reg.get("regionType") or reg.get("region_type") or "speech"
                region_details.append(f"[{rtype}] '{text}'")
        panel_info = f"panels={conv['panelIds']}" if conv.get("panelIds") else "unmapped"
        logger.info(
            f"[Layout] Conversation #{idx + 1} ({conv['sceneType']}, {panel_info}): " + " -> ".join(region_details)
        )
    logger.info("[Layout] -------------------------------------")

    # 4. Send enriched layout callback
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": page_id,
        "regionTypes": region_types,
        "conversations": [
            {
                "regionIds": conv["regionIds"],
                "sceneType": conv["sceneType"],
            }
            for conv in conversations
        ],
    }
    try:
        res = requests.post(f"{CALLBACK_URL}/layout", json=callback_payload, headers=backend_headers())
        logger.debug(f"[Layout] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[Layout] Failed to post callback to backend: {e}")
