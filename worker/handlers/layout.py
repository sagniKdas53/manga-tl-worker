import requests
from worker.config import CALLBACK_URL, BACKEND_HEADERS, redis_client
from worker.services.layout import classify_region_type, group_conversations


def process_layout(job_data):
    """Layout analysis: classify region types and group conversations."""
    image_id = job_data["imageId"]

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:layout")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    print(f"[Layout] Processing image: {image_id}{progress_str}", flush=True)

    # 1. Fetch OCR regions + panels from backend
    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[Layout] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        panels = image_info.get("panels", [])
    except Exception as e:
        print(f"[Layout] Error fetching image details: {e}", flush=True)
        raise

    if not ocr_regions:
        print("[Layout] No OCR regions found, skipping layout analysis.", flush=True)
        # Still send callback so pipeline continues
        callback_payload = {"imageId": image_id, "regionTypes": [], "conversations": []}
        try:
            res = requests.post(
                f"{CALLBACK_URL}/layout", json=callback_payload, headers=BACKEND_HEADERS
            )
            print(f"[Layout] Callback status code: {res.status_code}", flush=True)
        except Exception as e:
            print(f"[Layout] Failed to post callback: {e}", flush=True)
        raise

    # Get image dimensions from the first panel or estimate from regions
    image_width = max(
        (p.get("bboxX", 0) + p.get("bboxW", 0) for p in panels),
        default=max(
            (r.get("bboxX", 0) + r.get("bboxW", 0) for r in ocr_regions), default=1000
        ),
    )
    image_height = max(
        (p.get("bboxY", 0) + p.get("bboxH", 0) for p in panels),
        default=max(
            (r.get("bboxY", 0) + r.get("bboxH", 0) for r in ocr_regions), default=1400
        ),
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
        print(
            f"[Layout] Region {str(r.get('id', ''))[:8]}... "
            f"type={rtype} text='{(r.get('text', '') or '')[:30]}'",
            flush=True,
        )

    print(
        "[Layout] Region types: "
        + ", ".join(
            f"{t}: {sum(1 for rt in region_types if rt['regionType'] == t)}"
            for t in set(rt["regionType"] for rt in region_types)
        ),
        flush=True,
    )

    # 3. Group conversations
    reading_direction = "rtl"  # Default; could be passed in job_data if needed
    conversations = group_conversations(ocr_regions, panels, reading_direction)
    print(
        f"[Layout] Grouped {len(ocr_regions)} regions into {len(conversations)} conversations",
        flush=True,
    )

    # Detailed logging for the grouped conversations
    print("[Layout] --- Conversation Grouping Details ---", flush=True)
    for idx, conv in enumerate(conversations):
        region_details = []
        for rid in conv["regionIds"]:
            reg = next((r for r in ocr_regions if str(r.get("id")) == rid), None)
            if reg:
                text = reg.get("text", "").strip().replace("\n", " ")
                rtype = reg.get("regionType") or reg.get("region_type") or "speech"
                region_details.append(f"[{rtype}] '{text}'")
        panel_info = (
            f"panels={conv['panelIds']}" if conv.get("panelIds") else "unmapped"
        )
        print(
            f"[Layout] Conversation #{idx + 1} ({conv['sceneType']}, {panel_info}): "
            + " -> ".join(region_details),
            flush=True,
        )
    print("[Layout] -------------------------------------", flush=True)

    # 4. Send enriched layout callback
    callback_payload = {
        "imageId": image_id,
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
        res = requests.post(
            f"{CALLBACK_URL}/layout", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[Layout] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[Layout] Failed to post callback to backend: {e}", flush=True)
