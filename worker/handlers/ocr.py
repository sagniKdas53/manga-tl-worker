import gc
import cv2
import numpy as np
import requests
from PIL import Image
import logging
import re
import json
import os
import base64
from functools import cmp_to_key

from worker.config import (
    CALLBACK_URL,
    BACKEND_HEADERS,
    logger,
    YOLO_MASK_EROSION,
    redis_client,
)
from worker.model_manager import model_manager
from worker.utils.image import downscale_for_ocr, calculate_overlap_area, download_image
from worker.utils.text import detect_language
from worker.services.ocr import parse_paddle_ocr_results
from worker.services.layout import bubble_compare
from worker.utils.lock import acquire_lock
from worker.services.bubble_detector import detect_bubbles_yolo
from worker.services.translation import try_cloud_ai_vision, try_local_vlm_vision, LANG_MAP


def sort_fragments_vertical(fragments, reading_direction="rtl"):
    if not fragments:
        return []
    if len(fragments) == 1:
        return fragments

    # Calculate average width
    avg_w = sum(f["width"] for f in fragments) / len(fragments)
    col_threshold = max(20, avg_w * 0.7)

    # Calculate center coordinates
    for f in fragments:
        f["cx"] = f["x"] + f["width"] / 2
        f["cy"] = f["y"] + f["height"] / 2

    # Sort by horizontal center
    if reading_direction == "ltr":
        sorted_by_x = sorted(fragments, key=lambda f: f["cx"])
    else:  # default RTL: right to left
        sorted_by_x = sorted(fragments, key=lambda f: -f["cx"])

    # Group into columns
    columns = []
    for f in sorted_by_x:
        placed = False
        for col in columns:
            col_avg_cx = sum(c["cx"] for c in col) / len(col)
            if abs(f["cx"] - col_avg_cx) <= col_threshold:
                col.append(f)
                placed = True
                break
        if not placed:
            columns.append([f])

    # Sort within each column top-to-bottom (ascending cy)
    sorted_fragments = []
    for col in columns:
        col.sort(key=lambda f: f["cy"])
        sorted_fragments.extend(col)

    return sorted_fragments


def detect_background_color(img, x, y, w, h):
    """Auto-detect the background color of a region using border pixels of the crop."""
    if img is None:
        return "#ffffff"
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return "#ffffff"

    crop = img[y1:y2, x1:x2]

    # We take a small border margin to sample the background color (usually solid color)
    margin = min(2, crop.shape[1] // 4, crop.shape[0] // 4)
    if margin < 1:
        margin = 1

    border_pixels = []
    # Top and bottom margin rows
    border_pixels.extend(crop[0:margin, :].reshape(-1, 3))
    border_pixels.extend(crop[-margin:, :].reshape(-1, 3))
    # Left and right margin columns
    border_pixels.extend(crop[margin:-margin, 0:margin].reshape(-1, 3))
    border_pixels.extend(crop[margin:-margin, -margin:].reshape(-1, 3))

    if len(border_pixels) == 0:
        median_bgr = np.median(crop.reshape(-1, 3), axis=0)
    else:
        border_pixels = np.array(border_pixels)
        median_bgr = np.median(border_pixels, axis=0)

    # Convert BGR to RGB and format as hex
    r, g, b = int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"


def detect_background_color_poly(img, mask_polygon):
    """Detect the background color of a region using its polygon mask.
    If it fails, defaults to white (#ffffff).
    """
    if img is None or not mask_polygon:
        return "#ffffff"
    try:
        if isinstance(mask_polygon, str):
            pts = json.loads(mask_polygon)
        else:
            pts = mask_polygon
        if not isinstance(pts, list) or len(pts) < 3:
            return "#ffffff"

        h, w = img.shape[:2]
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

        # Erode mask slightly to avoid sampling bubble borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(mask_eroded) > 0:
            mask = mask_eroded

        pixels = img[mask == 255]
        if len(pixels) == 0:
            return "#ffffff"

        median_bgr = np.median(pixels, axis=0)
        r, g, b = int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as e:
        print(f"[OCR] Error detecting color from poly: {e}", flush=True)
        return "#ffffff"


def get_split_polygon(mask, bbox, img_w, img_h, margin=20):
    """Crop the main mask to bbox with a margin, find and return its simplified contour."""
    if mask is None or not bbox:
        return None
    try:
        rx, ry, rw, rh = bbox
        x1 = max(0, rx - margin)
        y1 = max(0, ry - margin)
        x2 = min(img_w, rx + rw + margin)
        y2 = min(img_h, ry + rh + margin)

        crop_mask = np.zeros_like(mask)
        crop_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        contours, _ = cv2.findContours(
            crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]
    except Exception as e:
        print(f"[OCR] Error splitting polygon: {e}", flush=True)
        return None


def detect_bubble_contour(img, ocr_x, ocr_y, ocr_w, ocr_h):
    """Find the contour of the speech bubble containing the OCR region and return its bounding box."""
    if img is None:
        return None
    h, w = img.shape[:2]

    # Expand search window to find the surrounding bubble edges
    pad_x = max(40, int(ocr_w * 0.8))
    pad_y = max(40, int(ocr_h * 0.8))

    x1 = max(0, ocr_x - pad_x)
    y1 = max(0, ocr_y - pad_y)
    x2 = min(w, ocr_x + ocr_w + pad_x)
    y2 = min(h, ocr_y + ocr_h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Smooth out text using median blur (kernel size must be odd and <= crop dims)
    ksize = 11
    if ksize >= min(gray.shape[0], gray.shape[1]):
        ksize = max(3, (min(gray.shape[0], gray.shape[1]) // 2) * 2 - 1)

    blurred = cv2.medianBlur(gray, ksize)

    # Check if the local background is light or dark
    median_val = np.median(blurred)
    is_light = median_val > 127

    if is_light:
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # OCR center in crop coordinates
    cx = (ocr_x + ocr_w / 2) - x1
    cy = (ocr_y + ocr_h / 2) - y1

    best_rect = None
    max_overlap_area = 0

    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        page_bx = x1 + bx
        page_by = y1 + by

        # Calculate overlap area with OCR region
        overlap_x = max(0, min(ocr_x + ocr_w, page_bx + bw) - max(ocr_x, page_bx))
        overlap_y = max(0, min(ocr_y + ocr_h, page_by + bh) - max(ocr_y, page_by))
        overlap_area = overlap_x * overlap_y

        if overlap_area > max_overlap_area:
            max_overlap_area = overlap_area
            best_rect = (bx, by, bw, bh)

    if best_rect is not None and max_overlap_area > 0:
        bx, by, bw, bh = best_rect
        return {"x": x1 + bx, "y": y1 + by, "width": bw, "height": bh}

    return None


def process_ocr(job_data):
    image_id = job_data["imageId"]
    # The backend sets these from the series context when it enqueues the job.
    # Defaults preserve the original behaviour (Japanese RTL) when not supplied.
    source_language = (job_data.get("sourceLanguage") or "ja").strip().lower()
    reading_direction = (job_data.get("readingDirection") or "rtl").strip().lower()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[OCR] Inputs: job_data={job_data}")

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:ocr")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    print(
        f"[OCR] Processing image: {image_id} (lang={source_language}, direction={reading_direction}){progress_str}",
        flush=True,
    )

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[OCR] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        panels = image_info.get("panels", [])
    except Exception as e:
        print(f"[OCR] Error fetching image details: {e}", flush=True)
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[OCR] Error downloading image: {e}", flush=True)
        return

    try:
        with acquire_lock("ocr"):
            results = []
            ocr_upscale = 1.0  # multiplier to map OCR coords back to original image
            img_decoded = None  # decoded image reused by both PaddleOCR and MangaOCR
            img_original = None  # full-resolution image for MangaOCR crops

            # Check if we should disable local OCR
            disable_local_ocr = os.environ.get("DISABLE_LOCAL_OCR", "").strip().lower() in ("true", "1", "yes")

            # Try PaddleOCR (PP-OCRv5) first — reader is lazily created per language
            paddle_ocr_reader = None if disable_local_ocr else model_manager.get_paddle_ocr_reader(source_language)
            if paddle_ocr_reader is not None:
                try:
                    print(
                        f"[OCR] Running PaddleOCR (PP-OCRv5 Mobile, lang={source_language}).",
                        flush=True,
                    )

                    try:
                        import psutil

                        rss = psutil.Process().memory_info().rss / 1024 / 1024
                        print(f"[OCR] Memory before OCR: {rss:.1f} MB", flush=True)
                    except Exception:
                        pass

                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    img_decoded, ocr_upscale = downscale_for_ocr(
                        img_original, max_dim=1024
                    )

                    if ocr_upscale != 1.0:
                        print(
                            f"[OCR] Downscaled image for OCR (upscale factor: {ocr_upscale:.2f}x)",
                            flush=True,
                        )

                    del nparr  # free compressed buffer immediately
                    if img_decoded is not None:
                        print("[OCR] Calling PaddleOCR...", flush=True)
                        raw_results = paddle_ocr_reader.predict(img_decoded)
                        print("[OCR] PaddleOCR returned.", flush=True)
                        results = parse_paddle_ocr_results(raw_results)
                        del raw_results
                        gc.collect()
                    else:
                        print(
                            "[OCR] OpenCV failed to decode image for PaddleOCR",
                            flush=True,
                        )
                except Exception as ocr_err:
                    print(
                        f"[OCR] PaddleOCR failed with exception: {ocr_err}. Falling back...",
                        flush=True,
                    )

            # Fallback to EasyOCR if results are empty and reader is available
            easy_reader = None if disable_local_ocr else model_manager.get_easy_ocr_reader(source_language)
            if not results and easy_reader is not None:
                try:
                    print("[OCR] Running EasyOCR fallback...", flush=True)
                    results = easy_reader.readtext(img_bytes)
                except Exception as ocr_err:
                    print(f"[OCR] EasyOCR failed: {ocr_err}", flush=True)

            if not results:
                print("[OCR] No text regions detected", flush=True)
                results = []

            # Force GC to reclaim any large temporary tensors created during inference
            gc.collect()

            # Use the full-resolution original image for MangaOCR crops
            # (img_decoded may be downscaled, so we use img_original instead)
            img = img_original if img_original is not None else img_decoded
            manga_ocr_reader = None if disable_local_ocr else model_manager.get_manga_ocr_reader()
            if img is None and manga_ocr_reader is not None:
                try:
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    del nparr
                except Exception as e:
                    print(f"[OCR] Error decoding image for MangaOCR: {e}", flush=True)

            img_h, img_w = img.shape[:2] if img is not None else (0, 0)
            detected_bubbles = None
            if img is not None:
                try:
                    detected_bubbles = detect_bubbles_yolo(img)
                except Exception as e:
                    print(f"[OCR] Failed to run YOLO bubble detection: {e}", flush=True)

            regions = []
            is_yolo_active = detected_bubbles is not None

            if is_yolo_active and disable_local_ocr:
                print(f"[OCR] VLM OCR Mode active for {len(detected_bubbles)} bubbles.", flush=True)
                provider = os.environ.get("MODEL_PROVIDER", "").lower().strip()
                api_key = os.environ.get("API_KEY", "").strip()
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or (api_key if provider == "openrouter" else "")
                gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or (api_key if provider == "gemini" else "")
                nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip() or (api_key if provider == "nvidia" else "")

                for b_idx, bubble in enumerate(detected_bubbles):
                    bx, by, bw, bh = bubble["bbox"]
                    bx1, by1 = max(0, bx), max(0, by)
                    bx2, by2 = min(img_w, bx + bw), min(img_h, by + bh)

                    if (bx2 - bx1) <= 0 or (by2 - by1) <= 0:
                        continue

                    crop = img[by1:by2, bx1:bx2]
                    _, buffer = cv2.imencode('.jpg', crop)
                    base64_image = base64.b64encode(buffer).decode('utf-8')

                    lang_name = LANG_MAP.get(source_language.lower(), source_language)
                    sys_prompt = f"You are an expert manga OCR system. Extract all text from the provided image crop exactly as it appears. The source language is {lang_name}. Return ONLY a valid JSON object."
                    user_prompt = "Extract the text from this speech bubble."
                    
                    schema = {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The extracted text"},
                        },
                        "required": ["text"]
                    }

                    res_text = None
                    if provider == "openrouter" and openrouter_key:
                        vlm_model = os.environ.get("PREFERRED_VLM_MODEL", "").strip() or "qwen/qwen3-vl-8b-instruct"
                        res_text = try_cloud_ai_vision("openrouter", openrouter_key, vlm_model, user_prompt, base64_image, schema, system_prompt=sys_prompt)
                    elif provider == "gemini" and gemini_key:
                        vlm_model = os.environ.get("PREFERRED_VLM_MODEL", "").strip() or "gemini-1.5-flash"
                        res_text = try_cloud_ai_vision("gemini", gemini_key, vlm_model, user_prompt, base64_image, schema, system_prompt=sys_prompt)
                    elif provider == "nvidia" and nvidia_key:
                        vlm_model = os.environ.get("PREFERRED_VLM_MODEL", "").strip() or "nvidia/nemotron-nano-12b-v2-vl"
                        res_text = try_cloud_ai_vision("nvidia", nvidia_key, vlm_model, user_prompt, base64_image, schema, system_prompt=sys_prompt)
                    else:
                        local_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
                        if local_model:
                            res_text = try_local_vlm_vision(local_model, user_prompt, base64_image, schema, system_prompt=sys_prompt)

                    extracted_text = ""
                    if res_text:
                        try:
                            parsed = json.loads(res_text.strip().removeprefix('```json').removesuffix('```').strip())
                            extracted_text = parsed.get("text", "")
                        except Exception:
                            extracted_text = res_text

                    if extracted_text and len(extracted_text.strip()) > 0:
                        bg_color = detect_background_color_poly(img, bubble["mask_polygon"])
                        regions.append({
                            "text": extracted_text,
                            "detectedLanguage": detect_language(extracted_text),
                            "confidence": 0.99,
                            "rotation": 0.0,
                            "x": bx,
                            "y": by,
                            "width": bw,
                            "height": bh,
                            "panelId": None,
                            "bubbleReadingOrder": 0,
                            "backgroundColor": bg_color,
                            "bubbleX": bx,
                            "bubbleY": by,
                            "bubbleWidth": bw,
                            "bubbleHeight": bh,
                            "bubbleId": f"bubble_{b_idx}",
                            "detectionConfidence": bubble["confidence"],
                            "maskPolygon": json.dumps(bubble["mask_polygon"]),
                            "safeTextX": bubble["safe_rect"][0],
                            "safeTextY": bubble["safe_rect"][1],
                            "safeTextW": bubble["safe_rect"][2],
                            "safeTextH": bubble["safe_rect"][3],
                        })
            
            elif is_yolo_active and not disable_local_ocr:
                # 1. Map raw PaddleOCR fragments to original image dimensions
                raw_fragments = []
                for bbox, text, confidence in results:
                    xs = [pt[0] * ocr_upscale for pt in bbox]
                    ys = [pt[1] * ocr_upscale for pt in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    width, height = int(max(xs) - x), int(max(ys) - y)
                    raw_fragments.append(
                        {
                            "text": text,
                            "detectedLanguage": detect_language(text),
                            "confidence": float(confidence),
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                    )

                # 2. Pre-generate binary masks for bubbles to compute exact pixel overlap
                bubble_masks = []
                for bubble in detected_bubbles:
                    poly = np.array(bubble["mask_polygon"], dtype=np.int32)
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    bubble_masks.append(mask)

                # 3. Assign each raw fragment to exactly one bubble by mask overlap
                for frag in raw_fragments:
                    best_b_idx = -1
                    max_overlap = 0
                    fx1 = max(0, min(img_w - 1, frag["x"]))
                    fy1 = max(0, min(img_h - 1, frag["y"]))
                    fx2 = max(0, min(img_w, frag["x"] + frag["width"]))
                    fy2 = max(0, min(img_h, frag["y"] + frag["height"]))

                    if fx2 > fx1 and fy2 > fy1:
                        for b_idx, mask in enumerate(bubble_masks):
                            overlap = np.sum(mask[fy1:fy2, fx1:fx2] > 0)
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_b_idx = b_idx
                    frag["bubble_idx"] = best_b_idx

                # 4. Group fragments for each bubble and merge them
                for b_idx, bubble in enumerate(detected_bubbles):
                    bx, by, bw, bh = bubble["bbox"]
                    bubble_mask = bubble_masks[b_idx]
                    assigned_frags = [
                        f for f in raw_fragments if f.get("bubble_idx", -1) == b_idx
                    ]

                    if not assigned_frags:
                        # Attempt to run MangaOCR on the empty bubble crop to see if Paddle missed it!
                        manga_text = None
                        if manga_ocr_reader is not None:
                            bx1 = max(0, bx)
                            by1 = max(0, by)
                            bx2 = min(img_w, bx + bw)
                            by2 = min(img_h, by + bh)
                            if (bx2 - bx1) > 0 and (by2 - by1) > 0:
                                try:
                                    crop = img[by1:by2, bx1:bx2]
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    pil_img = Image.fromarray(crop_rgb)
                                    manga_text = manga_ocr_reader(pil_img)
                                except Exception:
                                    pass
                        if manga_text and len(manga_text.strip()) > 0:
                            print(
                                f"[OCR] Found text '{manga_text}' in YOLO bubble {b_idx} that PaddleOCR missed!",
                                flush=True,
                            )
                            regions.append(
                                {
                                    "text": manga_text,
                                    "detectedLanguage": detect_language(manga_text),
                                    "confidence": 0.8,
                                    "rotation": 0.0,
                                    "x": bx,
                                    "y": by,
                                    "width": bw,
                                    "height": bh,
                                    "panelId": None,
                                    "bubbleReadingOrder": 0,
                                    "backgroundColor": detect_background_color_poly(
                                        img, bubble["mask_polygon"]
                                    ),
                                    "bubbleX": bx,
                                    "bubbleY": by,
                                    "bubbleWidth": bw,
                                    "bubbleHeight": bh,
                                    "bubbleId": f"bubble_{b_idx}",
                                    "detectionConfidence": bubble["confidence"],
                                    "maskPolygon": json.dumps(bubble["mask_polygon"]),
                                    "safeTextX": bubble["safe_rect"][0],
                                    "safeTextY": bubble["safe_rect"][1],
                                    "safeTextW": bubble["safe_rect"][2],
                                    "safeTextH": bubble["safe_rect"][3],
                                }
                            )
                        continue

                    # Run proximity merging inside the bubble to separate multiple semantic bubbles
                    from worker.services.merge_regions import merge_ocr_regions

                    merged_bubble_regions = merge_ocr_regions(
                        assigned_frags, reading_direction
                    )

                    for r_sub in merged_bubble_regions:
                        # 1. Get split polygon for this merged region
                        r_box = [
                            r_sub["x"],
                            r_sub["y"],
                            r_sub["width"],
                            r_sub["height"],
                        ]
                        poly_pts = get_split_polygon(
                            bubble_mask, r_box, img_w, img_h, margin=20
                        )
                        if not poly_pts:
                            poly_pts = bubble["mask_polygon"]

                        # 2. Bounding box of the split polygon
                        sp_x, sp_y, sp_w, sp_h = cv2.boundingRect(
                            np.array(poly_pts, dtype=np.int32)
                        )

                        # 3. Bounding box of the eroded mask (safe area)
                        split_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(
                            split_mask, [np.array(poly_pts, dtype=np.int32)], 255
                        )
                        erosion_px = YOLO_MASK_EROSION
                        kernel_erode = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
                        )
                        eroded_split_mask = cv2.erode(
                            split_mask, kernel_erode, iterations=1
                        )
                        if cv2.countNonZero(eroded_split_mask) == 0:
                            eroded_split_mask = split_mask
                        sx, sy, sw, sh = cv2.boundingRect(eroded_split_mask)

                        # 4. Crop split bubble region and run MangaOCR
                        manga_text = None
                        is_manga_ocr = False
                        if manga_ocr_reader is not None:
                            bx1 = max(0, sp_x)
                            by1 = max(0, sp_y)
                            bx2 = min(img_w, sp_x + sp_w)
                            by2 = min(img_h, sp_y + sp_h)
                            if (bx2 - bx1) > 0 and (by2 - by1) > 0:
                                try:
                                    crop = img[by1:by2, bx1:bx2]
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    pil_img = Image.fromarray(crop_rgb)
                                    manga_text = manga_ocr_reader(pil_img)
                                    if manga_text and len(manga_text.strip()) > 0:
                                        is_manga_ocr = True
                                except Exception as e:
                                    print(
                                        f"[OCR] MangaOCR failed on split bubble {b_idx} crop: {e}",
                                        flush=True,
                                    )

                        final_text = manga_text if is_manga_ocr else r_sub["text"]

                        # 5. Background color detection using split polygon
                        bg_color = detect_background_color_poly(img, poly_pts)

                        regions.append(
                            {
                                "text": final_text,
                                "detectedLanguage": (
                                    detect_language(final_text) if final_text else "ja"
                                ),
                                "confidence": (
                                    1.0 if is_manga_ocr else r_sub["confidence"]
                                ),
                                "rotation": 0.0,
                                "x": r_sub["x"],
                                "y": r_sub["y"],
                                "width": r_sub["width"],
                                "height": r_sub["height"],
                                "panelId": None,
                                "bubbleReadingOrder": 0,
                                "backgroundColor": bg_color,
                                "bubbleX": sp_x,
                                "bubbleY": sp_y,
                                "bubbleWidth": sp_w,
                                "bubbleHeight": sp_h,
                                "bubbleId": f"bubble_{b_idx}",
                                "detectionConfidence": bubble["confidence"],
                                "maskPolygon": json.dumps(poly_pts),
                                "safeTextX": sx,
                                "safeTextY": sy,
                                "safeTextW": sw,
                                "safeTextH": sh,
                            }
                        )

                # 5. Add unmatched fragments as separate standalone regions (SFX/sign)
                for f in raw_fragments:
                    if f.get("bubble_idx", -1) == -1:
                        regions.append(
                            {
                                "text": f["text"],
                                "detectedLanguage": f["detectedLanguage"],
                                "confidence": f["confidence"],
                                "rotation": 0.0,
                                "x": f["x"],
                                "y": f["y"],
                                "width": f["width"],
                                "height": f["height"],
                                "panelId": None,
                                "bubbleReadingOrder": 0,
                                "backgroundColor": detect_background_color(
                                    img, f["x"], f["y"], f["width"], f["height"]
                                ),
                                "bubbleX": f["x"],
                                "bubbleY": f["y"],
                                "bubbleWidth": f["width"],
                                "bubbleHeight": f["height"],
                                "bubbleId": None,
                                "detectionConfidence": 0.0,
                                "maskPolygon": None,
                                "safeTextX": f["x"],
                                "safeTextY": f["y"],
                                "safeTextW": f["width"],
                                "safeTextH": f["height"],
                            }
                        )

            else:
                # Fallback mode (legacy OpenCV bubble search)
                for bbox, text, confidence in results:
                    xs = [pt[0] * ocr_upscale for pt in bbox]
                    ys = [pt[1] * ocr_upscale for pt in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    width, height = int(max(xs) - x), int(max(ys) - y)

                    lang = detect_language(text)
                    is_manga_ocr = False
                    if (
                        lang in ("ja", "zh-TW")
                        and manga_ocr_reader is not None
                        and img is not None
                    ):
                        try:
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(img_w, x + width), min(img_h, y + height)
                            if (x2 - x1) > 0 and (y2 - y1) > 0:
                                crop = img[y1:y2, x1:x2]
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(crop_rgb)
                                manga_text = manga_ocr_reader(pil_img)
                                if manga_text and len(manga_text.strip()) > 0:
                                    text = manga_text
                                    is_manga_ocr = True
                        except Exception:
                            pass

                    bg_color = detect_background_color(img, x, y, width, height)
                    bubble_box = detect_bubble_contour(img, x, y, width, height)

                    if (
                        bubble_box
                        and bubble_box["width"] <= width * 2.5
                        and bubble_box["height"] <= height * 2.5
                    ):
                        bx, by, bw, bh = (
                            bubble_box["x"],
                            bubble_box["y"],
                            bubble_box["width"],
                            bubble_box["height"],
                        )
                    else:
                        bx, by, bw, bh = x, y, width, height

                    regions.append(
                        {
                            "text": text,
                            "detectedLanguage": lang,
                            "confidence": 1.0 if is_manga_ocr else float(confidence),
                            "rotation": 0.0,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "panelId": None,
                            "bubbleReadingOrder": 0,
                            "backgroundColor": bg_color,
                            "bubbleX": bx,
                            "bubbleY": by,
                            "bubbleWidth": bw,
                            "bubbleHeight": bh,
                            "bubbleId": None,
                            "detectionConfidence": 0.0,
                            "maskPolygon": None,
                            "safeTextX": bx,
                            "safeTextY": by,
                            "safeTextW": bw,
                            "safeTextH": bh,
                        }
                    )

                from worker.services.merge_regions import merge_ocr_regions

                regions = merge_ocr_regions(regions, reading_direction)

            panel_regions_map = {}
            unmapped_regions = []

            for r in regions:
                best_panel_idx = -1
                max_overlap = 0
                for idx, p in enumerate(panels):
                    overlap = calculate_overlap_area(r, p)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_panel_idx = idx

                if best_panel_idx != -1:
                    if best_panel_idx not in panel_regions_map:
                        panel_regions_map[best_panel_idx] = []
                    panel_regions_map[best_panel_idx].append(r)
                else:
                    unmapped_regions.append(r)

            ordered_regions = []
            sorted_panel_indices = sorted(
                panel_regions_map.keys(), key=lambda idx: panels[idx]["readingOrder"]
            )

            # Curry the reading direction into the comparator so sort is direction-aware
            def _bubble_cmp(a, b):
                return bubble_compare(a, b, reading_direction)

            for panel_idx in sorted_panel_indices:
                panel_bubbles = panel_regions_map[panel_idx]
                panel_bubbles.sort(key=cmp_to_key(_bubble_cmp))

                for b_order, r in enumerate(panel_bubbles, start=1):
                    r["bubbleReadingOrder"] = b_order
                    ordered_regions.append(r)

            unmapped_regions.sort(key=cmp_to_key(_bubble_cmp))
            for b_order, r in enumerate(unmapped_regions, start=1):
                r["bubbleReadingOrder"] = b_order
                ordered_regions.append(r)

            print(
                f"[OCR] Completed OCR. Found {len(ordered_regions)} text regions (lang={source_language}, direction={reading_direction})",
                flush=True,
            )

            avg_conf = (
                sum(r["confidence"] for r in ordered_regions) / len(ordered_regions)
                if ordered_regions
                else 1.0
            )

            callback_payload = {
                "imageId": image_id,
                "modelIdentifier": "MangaOCR/PaddleOCR",
                "confidence": avg_conf,
                "sourceLanguage": source_language,
                "readingDirection": reading_direction,
                "regions": ordered_regions,
            }
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[OCR] Outputs: callback_payload={callback_payload}")
            try:
                res = requests.post(
                    f"{CALLBACK_URL}/ocr",
                    json=callback_payload,
                    headers=BACKEND_HEADERS,
                )
                print(f"[OCR] Callback status code: {res.status_code}", flush=True)
            except Exception as e:
                print(f"[OCR] Failed to post callback to backend: {e}", flush=True)
    except Exception as e:
        print(f"[OCR] Error during locked OCR process: {e}", flush=True)
        return
