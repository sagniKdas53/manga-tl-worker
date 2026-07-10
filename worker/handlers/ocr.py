import gc
import cv2
import numpy as np
import requests
import logging
import json
import os
import base64
import concurrent.futures
from functools import cmp_to_key

from worker.config import (
    CALLBACK_URL,
    BACKEND_HEADERS,
    logger,
    YOLO_MASK_EROSION,
    redis_client,
    OCR_CONFIG,
)
from worker.model_manager import model_manager
from worker.utils.image import downscale_for_ocr, calculate_overlap_area, download_image
from worker.utils.text import detect_language
from worker.services.ocr import parse_paddle_ocr_results
from worker.services.layout import bubble_compare
from worker.services.merge_regions import merge_ocr_regions
from worker.utils.lock import acquire_lock
from worker.services.bubble_detector import detect_bubbles_yolo
from worker.services.translation import (
    try_cloud_ai_vision_batch,
    try_local_vlm_vision,
    LANG_MAP,
)


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
    (ocr_x + ocr_w / 2) - x1
    (ocr_y + ocr_h / 2) - y1

    best_contour = None
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
            best_contour = c
            best_rect = (bx, by, bw, bh)

    if best_rect is not None and best_contour is not None and max_overlap_area > 0:
        bx, by, bw, bh = best_rect
        epsilon = 0.002 * cv2.arcLength(best_contour, True)
        simplified = cv2.approxPolyDP(best_contour, epsilon, True)
        polygon = [[int(x1 + pt[0][0]), int(y1 + pt[0][1])] for pt in simplified]
        return {
            "x": x1 + bx,
            "y": y1 + by,
            "width": bw,
            "height": bh,
            "maskPolygon": polygon,
        }

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
        results = []
        ocr_upscale = 1.0
        img_decoded = None
        img_original = None
        detected_bubbles = None
        img = None

        disable_local_ocr = os.environ.get("DISABLE_LOCAL_OCR", "").strip().lower() in (
            "true",
            "1",
            "yes",
        )

        provider = (
            (job_data.get("ocrProvider") or OCR_CONFIG.provider or "local")
            .lower()
            .strip()
        )
        use_paddle_ocr = (provider == "local") and not disable_local_ocr

        # WARNING: Even when using Cloud VLM OCR (where transcription is offloaded), local models
        # (PP-OCR-Det for text detection and YOLO for bubble detection) still execute locally on
        # this host. We must serialize these local predictions using the "ocr" lock to avoid CPU/GPU
        # overload and OOM crashes. This local bottleneck will be resolved when remote workers on
        # dedicated machines are supported, allowing parallel detection and full OCR job queues.
        with acquire_lock("ocr"):
            # Try PaddleOCR (PP-OCRv5) first — reader is lazily created per language
            paddle_ocr_reader = (
                model_manager.get_paddle_ocr_reader(source_language)
                if use_paddle_ocr
                else None
            )
            paddle_ocr_detector = (
                model_manager.get_paddle_ocr_detector(source_language)
                if not use_paddle_ocr
                else None
            )

            if use_paddle_ocr and paddle_ocr_reader is None:
                raise RuntimeError(
                    f"Required local PaddleOCR model failed to initialize for language: {source_language}. "
                    "Cannot proceed in offline mode without the required model."
                )
            if not use_paddle_ocr and paddle_ocr_detector is None:
                raise RuntimeError(
                    f"Required local PaddleOCR detector failed to initialize for language: {source_language}."
                )

            if paddle_ocr_reader is not None:
                try:
                    det_model = os.environ.get(
                        "PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
                    ).strip()
                    rec_model = os.environ.get(
                        "PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec"
                    ).strip()
                    print(
                        f"[OCR] Running PaddleOCR ({det_model}/{rec_model}, lang={source_language}).",
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
                        f"[OCR] PaddleOCR failed with exception: {ocr_err}.",
                        flush=True,
                    )
                    raise ocr_err

            if paddle_ocr_detector is not None:
                try:
                    det_model = os.environ.get(
                        "PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
                    ).strip()
                    print(
                        f"[OCR] Running PaddleOCR Detector ({det_model}, lang={source_language}).",
                        flush=True,
                    )
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_decoded, ocr_upscale = downscale_for_ocr(
                        img_original, max_dim=1024
                    )
                    del nparr
                    if img_decoded is not None:
                        raw_results = paddle_ocr_detector.predict(img_decoded)
                        results = parse_paddle_ocr_results(raw_results)
                        del raw_results
                        gc.collect()
                    else:
                        print(
                            "[OCR] OpenCV failed to decode image for PaddleOCR Detector",
                            flush=True,
                        )
                except Exception as ocr_err:
                    print(
                        f"[OCR] PaddleOCR Detector failed with exception: {ocr_err}.",
                        flush=True,
                    )
                    raise ocr_err

            if not results:
                print("[OCR] No text regions detected", flush=True)

            # Force GC to reclaim any large temporary tensors created during inference
            gc.collect()

            # Use the full-resolution original image
            img = img_original if img_original is not None else img_decoded

            if img is None:
                try:
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    del nparr
                except Exception as e:
                    print(f"[OCR] Error decoding image: {e}", flush=True)

            img_h, img_w = img.shape[:2] if img is not None else (0, 0)
            detected_bubbles = None
            if img is not None:
                detected_bubbles = detect_bubbles_yolo(img)

        regions = []
        is_yolo_active = detected_bubbles is not None

        if is_yolo_active:
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

            # 4. Group fragments for each bubble and merge them (or create default crop if empty and we are using Cloud VLM)
            candidate_regions = []  # regions we need to OCR/transcribe

            for b_idx, bubble in enumerate(detected_bubbles):
                bx, by, bw, bh = bubble["bbox"]
                bubble_mask = bubble_masks[b_idx]
                assigned_frags = [
                    f for f in raw_fragments if f.get("bubble_idx", -1) == b_idx
                ]

                if not assigned_frags:
                    # If Cloud VLM is active, we STILL want to crop and VLM-OCR empty bubbles to be safe!
                    if not use_paddle_ocr:
                        candidate_regions.append(
                            {
                                "type": "bubble",
                                "bubble_idx": b_idx,
                                "x": bx,
                                "y": by,
                                "width": bw,
                                "height": bh,
                                "poly_pts": bubble["mask_polygon"],
                                "safe_rect": bubble["safe_rect"],
                                "text": "",
                                "confidence": 1.0,
                                "bubble": bubble,
                            }
                        )
                    continue

                # Run proximity merging inside the bubble to separate multiple semantic bubbles
                merged_bubble_regions = merge_ocr_regions(
                    assigned_frags, reading_direction, threshold_ratio=2.0
                )

                for r_sub in merged_bubble_regions:
                    if len(merged_bubble_regions) == 1:
                        poly_pts = bubble["mask_polygon"]
                        sp_x, sp_y, sp_w, sp_h = bx, by, bw, bh
                        sx, sy, sw, sh = bubble["safe_rect"]
                    else:
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
                            cv2.MORPH_ELLIPSE,
                            (2 * erosion_px + 1, 2 * erosion_px + 1),
                        )
                        eroded_split_mask = cv2.erode(
                            split_mask, kernel_erode, iterations=1
                        )
                        if cv2.countNonZero(eroded_split_mask) == 0:
                            eroded_split_mask = split_mask
                        sx, sy, sw, sh = cv2.boundingRect(eroded_split_mask)

                    candidate_regions.append(
                        {
                            "type": "bubble",
                            "bubble_idx": b_idx,
                            "x": r_sub["x"],
                            "y": r_sub["y"],
                            "width": r_sub["width"],
                            "height": r_sub["height"],
                            "poly_pts": poly_pts,
                            "safe_rect": [sx, sy, sw, sh],
                            "text": r_sub["text"],
                            "confidence": r_sub["confidence"],
                            "bubbleX": sp_x,
                            "bubbleY": sp_y,
                            "bubbleWidth": sp_w,
                            "bubbleHeight": sp_h,
                            "bubble": bubble,
                        }
                    )

            # 5. Add unmatched fragments as merged standalone regions (direct text / SFX)
            unmatched_frags = [
                f for f in raw_fragments if f.get("bubble_idx", -1) == -1
            ]
            if unmatched_frags:
                merged_unmatched = merge_ocr_regions(unmatched_frags, reading_direction)

                for idx, r_sub in enumerate(merged_unmatched):
                    rx, ry, rw, rh = (
                        r_sub["x"],
                        r_sub["y"],
                        r_sub["width"],
                        r_sub["height"],
                    )

                    # Generate tight padded "virtual bubble" mask to allow typesetter inpainting / background cleaning
                    pad = 6
                    px1 = max(0, rx - pad)
                    py1 = max(0, ry - pad)
                    px2 = min(img_w, rx + rw + pad)
                    py2 = min(img_h, ry + rh + pad)
                    mask_polygon = [[px1, py1], [px2, py1], [px2, py2], [px1, py2]]

                    candidate_regions.append(
                        {
                            "type": "direct_text",
                            "direct_idx": idx,
                            "x": rx,
                            "y": ry,
                            "width": rw,
                            "height": rh,
                            "poly_pts": mask_polygon,
                            "safe_rect": [px1, py1, px2 - px1, py2 - py1],
                            "text": r_sub["text"],
                            "confidence": r_sub["confidence"],
                            "detectedLanguage": r_sub["detectedLanguage"],
                        }
                    )

            # 6. Now, recognize candidates
            if not use_paddle_ocr:
                # CLOUD OCR MODE (VLM Batching)
                if candidate_regions:
                    print(
                        f"[OCR] VLM OCR Mode active (batched) for {len(candidate_regions)} regions.",
                        flush=True,
                    )
                    provider = job_data.get("ocrProvider") or OCR_CONFIG.provider
                    api_key = OCR_CONFIG.resolve_key(provider)

                    # Generate base64 crops for all candidate regions
                    crops_payload = []
                    for cr_idx, r in enumerate(candidate_regions):
                        rx, ry, rw, rh = r["x"], r["y"], r["width"], r["height"]
                        rx1, ry1 = max(0, rx), max(0, ry)
                        rx2, ry2 = min(img_w, rx + rw), min(img_h, ry + rh)

                        crop = img[ry1:ry2, rx1:rx2]
                        if crop.size > 0:
                            _, buffer = cv2.imencode(".jpg", crop)
                            base64_image = base64.b64encode(buffer).decode("utf-8")
                            crops_payload.append(
                                {"id": f"region_{cr_idx}", "base64": base64_image}
                            )

                    schema = {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "text": {"type": "string"},
                                    },
                                    "required": ["id", "text"],
                                },
                            }
                        },
                        "required": ["results"],
                    }

                    if crops_payload:
                        vlm_model = job_data.get("ocrModel") or OCR_CONFIG.vlm_model
                        # Default model depending on provider
                        if not vlm_model:
                            if provider == "openrouter":
                                vlm_model = "google/gemini-2.5-flash"
                            elif provider == "gemini":
                                vlm_model = "gemini-1.5-flash"
                            elif provider == "nvidia":
                                vlm_model = "nvidia/nemotron-nano-12b-v2-vl"

                        lang_name = LANG_MAP.get(
                            source_language.lower(), source_language
                        )
                        sys_prompt = (
                            f"You are an expert manga OCR system. Perform OCR on each of the provided image crops. "
                            f"The source language is {lang_name}. Return ONLY a valid JSON object matching the schema."
                        )

                        transcriptions = {}

                        def chunk_list(lst, n):
                            return [lst[i : i + n] for i in range(0, len(lst), n)]

                        crop_chunks = chunk_list(crops_payload, 10)

                        def process_crop_chunk(chunk_idx, chunk):
                            print(
                                f"[OCR] Processing cloud OCR batch chunk {chunk_idx + 1}/{len(crop_chunks)} ({len(chunk)} crops)...",
                                flush=True,
                            )
                            results_list = []
                            if (
                                provider
                                in (
                                    "openai",
                                    "openrouter",
                                    "gemini",
                                    "anthropic",
                                    "nvidia",
                                )
                                and api_key
                            ):
                                from worker.config import OCR_CONFIG

                                models_to_try = []
                                if vlm_model:
                                    models_to_try.append(vlm_model)
                                for m in getattr(OCR_CONFIG, "vlm_model_list", []):
                                    if m not in models_to_try:
                                        models_to_try.append(m)

                                for current_model in models_to_try:
                                    try:
                                        chunk_res = try_cloud_ai_vision_batch(
                                            provider,
                                            api_key,
                                            current_model,
                                            chunk,
                                            schema,
                                            system_prompt=sys_prompt,
                                        )
                                        if chunk_res:
                                            parsed = json.loads(
                                                chunk_res.strip()
                                                .removeprefix("```json")
                                                .removesuffix("```")
                                                .strip()
                                            )
                                            results_list = parsed.get("results", [])
                                            if results_list:
                                                print(
                                                    f"[OCR] Successfully processed chunk {chunk_idx + 1} using model '{current_model}'",
                                                    flush=True,
                                                )
                                                break
                                    except Exception as parse_err:
                                        print(
                                            f"[OCR] Failed for model '{current_model}' on chunk {chunk_idx + 1}: {parse_err}",
                                            flush=True,
                                        )
                            else:
                                local_model = (
                                    job_data.get("ocrModel")
                                    or os.environ.get("LOCAL_VLM_MODEL", "").strip()
                                )
                                if local_model:
                                    user_prompt = (
                                        "Extract the text from this speech bubble."
                                    )
                                    crop_schema = {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "The extracted text",
                                            },
                                        },
                                        "required": ["text"],
                                    }
                                    for crop_info in chunk:
                                        try:
                                            crop_res = try_local_vlm_vision(
                                                local_model,
                                                user_prompt,
                                                crop_info["base64"],
                                                crop_schema,
                                                system_prompt=sys_prompt,
                                            )
                                            if crop_res:
                                                try:
                                                    parsed = json.loads(
                                                        crop_res.strip()
                                                        .removeprefix("```json")
                                                        .removesuffix("```")
                                                        .strip()
                                                    )
                                                    results_list.append(
                                                        {
                                                            "id": crop_info["id"],
                                                            "text": parsed.get(
                                                                "text", ""
                                                            ),
                                                        }
                                                    )
                                                except Exception:
                                                    results_list.append(
                                                        {
                                                            "id": crop_info["id"],
                                                            "text": crop_res,
                                                        }
                                                    )
                                        except Exception as local_vlm_err:
                                            print(
                                                f"[OCR] Local VLM failed for crop {crop_info['id']}: {local_vlm_err}",
                                                flush=True,
                                            )
                            return results_list

                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor:
                            futures = {
                                executor.submit(process_crop_chunk, idx, chunk): chunk
                                for idx, chunk in enumerate(crop_chunks)
                            }
                            for future in concurrent.futures.as_completed(futures):
                                results_list = future.result()
                                for item in results_list:
                                    item_id = item.get("id", "")
                                    item_text = item.get("text", "")
                                    transcriptions[item_id] = item_text

                    # Create regions list
                    for cr_idx, r in enumerate(candidate_regions):
                        final_text = transcriptions.get(f"region_{cr_idx}", "").strip()
                        if final_text:
                            bg_color = detect_background_color_poly(img, r["poly_pts"])
                            if r["type"] == "bubble":
                                regions.append(
                                    {
                                        "text": final_text,
                                        "detectedLanguage": detect_language(final_text),
                                        "confidence": 0.99,
                                        "rotation": 0.0,
                                        "x": r["x"],
                                        "y": r["y"],
                                        "width": r["width"],
                                        "height": r["height"],
                                        "panelId": None,
                                        "bubbleReadingOrder": 0,
                                        "backgroundColor": bg_color,
                                        "bubbleX": r.get("bubbleX", r["x"]),
                                        "bubbleY": r.get("bubbleY", r["y"]),
                                        "bubbleWidth": r.get("bubbleWidth", r["width"]),
                                        "bubbleHeight": r.get(
                                            "bubbleHeight", r["height"]
                                        ),
                                        "bubbleId": f"bubble_{r['bubble_idx']}",
                                        "detectionConfidence": r["bubble"][
                                            "confidence"
                                        ],
                                        "maskPolygon": json.dumps(r["poly_pts"]),
                                        "safeTextX": r["safe_rect"][0],
                                        "safeTextY": r["safe_rect"][1],
                                        "safeTextW": r["safe_rect"][2],
                                        "safeTextH": r["safe_rect"][3],
                                    }
                                )
                            else:
                                # direct text / free-floating
                                regions.append(
                                    {
                                        "text": final_text,
                                        "detectedLanguage": detect_language(final_text),
                                        "confidence": 0.99,
                                        "rotation": 0.0,
                                        "x": r["x"],
                                        "y": r["y"],
                                        "width": r["width"],
                                        "height": r["height"],
                                        "panelId": None,
                                        "bubbleReadingOrder": 0,
                                        "backgroundColor": bg_color,
                                        "bubbleX": r["x"],
                                        "bubbleY": r["y"],
                                        "bubbleWidth": r["width"],
                                        "bubbleHeight": r["height"],
                                        "bubbleId": f"direct_text_{r['direct_idx']}",
                                        "detectionConfidence": 0.0,
                                        "maskPolygon": json.dumps(r["poly_pts"]),
                                        "safeTextX": r["safe_rect"][0],
                                        "safeTextY": r["safe_rect"][1],
                                        "safeTextW": r["safe_rect"][2],
                                        "safeTextH": r["safe_rect"][3],
                                    }
                                )

            else:
                # LOCAL OCR MODE (Use already-recognized texts in candidates)
                for r in candidate_regions:
                    final_text = r["text"]
                    if final_text:
                        bg_color = detect_background_color_poly(img, r["poly_pts"])
                        if r["type"] == "bubble":
                            regions.append(
                                {
                                    "text": final_text,
                                    "detectedLanguage": detect_language(final_text)
                                    if final_text
                                    else "ja",
                                    "confidence": r["confidence"],
                                    "rotation": 0.0,
                                    "x": r["x"],
                                    "y": r["y"],
                                    "width": r["width"],
                                    "height": r["height"],
                                    "panelId": None,
                                    "bubbleReadingOrder": 0,
                                    "backgroundColor": bg_color,
                                    "bubbleX": r.get("bubbleX", r["x"]),
                                    "bubbleY": r.get("bubbleY", r["y"]),
                                    "bubbleWidth": r.get("bubbleWidth", r["width"]),
                                    "bubbleHeight": r.get("bubbleHeight", r["height"]),
                                    "bubbleId": f"bubble_{r['bubble_idx']}",
                                    "detectionConfidence": r["bubble"]["confidence"],
                                    "maskPolygon": json.dumps(r["poly_pts"]),
                                    "safeTextX": r["safe_rect"][0],
                                    "safeTextY": r["safe_rect"][1],
                                    "safeTextW": r["safe_rect"][2],
                                    "safeTextH": r["safe_rect"][3],
                                }
                            )
                        else:
                            regions.append(
                                {
                                    "text": final_text,
                                    "detectedLanguage": detect_language(final_text)
                                    if final_text
                                    else r["detectedLanguage"],
                                    "confidence": r["confidence"],
                                    "rotation": 0.0,
                                    "x": r["x"],
                                    "y": r["y"],
                                    "width": r["width"],
                                    "height": r["height"],
                                    "panelId": None,
                                    "bubbleReadingOrder": 0,
                                    "backgroundColor": bg_color,
                                    "bubbleX": r["x"],
                                    "bubbleY": r["y"],
                                    "bubbleWidth": r["width"],
                                    "bubbleHeight": r["height"],
                                    "bubbleId": f"direct_text_{r['direct_idx']}",
                                    "detectionConfidence": 0.0,
                                    "maskPolygon": json.dumps(r["poly_pts"]),
                                    "safeTextX": r["safe_rect"][0],
                                    "safeTextY": r["safe_rect"][1],
                                    "safeTextW": r["safe_rect"][2],
                                    "safeTextH": r["safe_rect"][3],
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
                bubble_box = detect_bubble_contour(img, x, y, width, height)

                use_bubble_contour = (
                    bubble_box
                    and bubble_box["width"] <= width * 2.5
                    and bubble_box["height"] <= height * 2.5
                )
                if use_bubble_contour:
                    bx, by, bw, bh = (
                        bubble_box["x"],
                        bubble_box["y"],
                        bubble_box["width"],
                        bubble_box["height"],
                    )
                else:
                    bx, by, bw, bh = x, y, width, height

                mask_polygon = (
                    bubble_box.get("maskPolygon") if use_bubble_contour else None
                )
                bg_color = (
                    detect_background_color_poly(img, mask_polygon)
                    if mask_polygon
                    else detect_background_color(img, x, y, width, height)
                )

                regions.append(
                    {
                        "text": text,
                        "detectedLanguage": lang,
                        "confidence": float(confidence),
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
                        "maskPolygon": (
                            json.dumps(mask_polygon) if mask_polygon else None
                        ),
                        "safeTextX": bx,
                        "safeTextY": by,
                        "safeTextW": bw,
                        "safeTextH": bh,
                    }
                )

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

        rec_model = os.environ.get("PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec").strip()
        callback_payload = {
            "imageId": image_id,
            "modelIdentifier": f"MangaOCR/PaddleOCR({rec_model})",
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
            raise e
    except Exception as e:
        print(f"[OCR] Error during OCR process: {e}", flush=True)
        raise e
