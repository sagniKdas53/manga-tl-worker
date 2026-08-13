import base64
import concurrent.futures
import gc
import json
import logging
import math
import os
from functools import cmp_to_key

import cv2
import numpy as np
import requests

from worker.config import (
    BACKEND_HEADERS,
    BACKGROUND_FILL_MAX_SPREAD,
    BUBBLE_CONTOUR_FALLBACK,
    BUBBLE_CONTOUR_MAX_GROWTH,
    BUBBLE_CONTOUR_MAX_PAGE_FRACTION,
    BUBBLE_MAX_SELF_CONTAINMENT,
    BUBBLE_MIN_TEXT_COVERAGE,
    CALLBACK_URL,
    COVER_FILL_ENABLED,
    COVER_FILL_PAD_FRACTION,
    COVER_FILL_QUANT,
    COVER_FILL_RING_FRACTION,
    OCR_CONFIG,
    OCR_MERGE_THRESHOLD,
    OCR_ORIENTATION,
    OCR_WAIST_GATE,
    OCR_WAIST_MAX_SOLIDITY,
    YOLO_MASK_EROSION,
    logger,
    redis_client,
)
from worker.model_manager import model_manager
from worker.services.bubble_detector import detect_bubbles_yolo
from worker.services.bubble_geometry import bubble_grouping_context
from worker.services.fragment_grouping import GroupingConfig
from worker.services.layout import bubble_compare
from worker.services.merge_regions import merge_ocr_regions
from worker.services.ocr import parse_paddle_ocr_results
from worker.services.translation import (
    LANG_MAP,
    try_cloud_ai_vision_batch,
    try_local_vlm_vision,
)
from worker.utils.image import calculate_overlap_area, download_image, downscale_for_ocr
from worker.utils.lock import acquire_lock
from worker.utils.text import detect_language


def grouping_config(reading_direction):
    """One configuration for all three merge call sites in this handler.

    They used to disagree by accident -- the in-bubble path hardcoded 2.0 while the other two read
    OCR_MERGE_THRESHOLD -- so a deployment could tune one and silently leave the others alone.
    The clearance veto is inert wherever no mask is passed, so the same object is correct on the
    paths that have no balloon.
    """
    return GroupingConfig(
        threshold_ratio=OCR_MERGE_THRESHOLD,
        reading_direction=reading_direction,
        orientation=OCR_ORIENTATION,
        waist_gate=OCR_WAIST_GATE if OCR_WAIST_GATE > 0 else None,
        waist_max_solidity=OCR_WAIST_MAX_SOLIDITY,
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


def _pixel_spread(pixels: np.ndarray) -> float:
    """Per-channel median absolute deviation, maxed across channels.

    Robust in the two ways a plain stddev over the flattened array is not: a handful of
    anti-aliased text-edge pixels in an otherwise-flat sample barely moves a median-based
    measure, and computing per-channel (then taking the max) rather than over B/G/R mixed
    together means a saturated solid colour reads as flat instead of "spread" by its own
    channel separation.
    """
    medians = np.median(pixels, axis=0)
    return float(np.max(np.median(np.abs(pixels - medians), axis=0)))


def detect_background_color(img, x, y, w, h):
    """Auto-detect the background color of a region using border pixels of the crop.

    Returns None when the sampled border is not close to a flat colour (D1/D3,
    docs/render_quality_gap_2026-08-05.md). Painting a median colour over line art or a busy
    photo replaces detail with a solid slab; a caller that gets None should skip the backdrop
    fill entirely rather than fall back to white, which is just as wrong a guess as the median
    on real artwork.
    """
    if img is None:
        return None
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return None

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

    sample = np.array(border_pixels) if len(border_pixels) > 0 else crop.reshape(-1, 3)
    if _pixel_spread(sample) > BACKGROUND_FILL_MAX_SPREAD:
        return None

    median_bgr = np.median(sample, axis=0)

    # Convert BGR to RGB and format as hex
    r, g, b = int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"


def detect_background_color_poly(img, mask_polygon):
    """Detect the background color of a region using its polygon mask.

    Returns None -- meaning "do not fill" -- when there is no polygon, detection fails, or the
    sampled interior is not close to flat. See detect_background_color for why None and not a
    white fallback: on real artwork white is as wrong a guess as the median (D1/D3,
    docs/render_quality_gap_2026-08-05.md).
    """
    if img is None or not mask_polygon:
        return None
    try:
        pts = json.loads(mask_polygon) if isinstance(mask_polygon, str) else mask_polygon
        if not isinstance(pts, list) or len(pts) < 3:
            return None

        h, w = img.shape[:2]
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)  # type: ignore

        # Erode mask slightly to avoid sampling bubble borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(mask_eroded) > 0:
            mask = mask_eroded

        pixels = img[mask == 255]
        if len(pixels) == 0:
            return None

        if _pixel_spread(pixels) > BACKGROUND_FILL_MAX_SPREAD:
            return None

        median_bgr = np.median(pixels, axis=0)
        r, g, b = int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as e:
        print(f"[OCR] Error detecting color from poly: {e}", flush=True)
        return None


def ring_pixels(img, x, y, width, height):
    """Pixels in the band just outside a text box: what the text is sitting *on*.

    Sampling inside the box samples the lettering. Unenclosed manga text is drawn with a thick
    white stroke around each glyph precisely so it reads against artwork, and on sample10's yellow
    blanket that stroke is the single most common colour in the box -- so "the dominant colour of
    this region" came back white, and R2's covering balloon would have been a white slab. The band
    outside the text is background by construction.
    """
    h, w = img.shape[:2]
    pad = max(3, round(min(width, height) * COVER_FILL_RING_FRACTION))
    ox1, oy1 = max(0, int(x) - pad), max(0, int(y) - pad)
    ox2, oy2 = min(w, int(x + width) + pad), min(h, int(y + height) + pad)
    if ox2 <= ox1 or oy2 <= oy1:
        return None

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[oy1:oy2, ox1:ox2] = 255
    ix1, iy1 = max(0, int(x)), max(0, int(y))
    ix2, iy2 = min(w, int(x + width)), min(h, int(y + height))
    if ix2 > ix1 and iy2 > iy1:
        outer[iy1:iy2, ix1:ix2] = 0
    pixels = img[outer == 255]
    return pixels if len(pixels) else None


def dominant_color(img, mask_polygon=None, box=None, ring=False):
    """The colour a reader would name for this region, as `#rrggbb`.

    The mode of a coarsely quantised histogram, refined to the median of the winning bin. The
    median of the raw pixels is the wrong answer here: over sample10's shaded yellow blanket it
    lands on a muddy mid-tone halfway between the lit and shadowed halves, and a balloon painted in
    it reads as dirty. The mode picks the lit yellow, which is what the reference used.

    With `ring=True` the sample is the band just outside `box` rather than its interior -- see
    :func:`ring_pixels` for why that is what R2 wants.
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    pixels = None
    if ring and box:
        pixels = ring_pixels(img, *box)
    if pixels is None and mask_polygon:
        try:
            pts = json.loads(mask_polygon) if isinstance(mask_polygon, str) else mask_polygon
            if isinstance(pts, list) and len(pts) >= 3:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)  # type: ignore
                pixels = img[mask == 255]
        except Exception as e:
            logger.warning("dominant_color: unusable polygon (%s)", e)
    if pixels is None and box:
        x, y, bw, bh = (int(v) for v in box)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 > x1 and y2 > y1:
            pixels = img[y1:y2, x1:x2].reshape(-1, 3)
    if pixels is None or len(pixels) == 0:
        return None

    q = max(2, COVER_FILL_QUANT)
    binned = (pixels // q).astype(np.int32)
    keys = binned[:, 0] * 65536 + binned[:, 1] * 256 + binned[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    winner = values[int(np.argmax(counts))]
    median_bgr = np.median(pixels[keys == winner], axis=0)
    r, g, b = int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"


def cover_balloon_polygon(x, y, width, height, img_w, img_h, corner_steps=6):
    """A rounded-rectangle polygon covering the source text, plus a margin.

    The margin is proportional to the text extent rather than absolute, because these pages run
    from 832px to 6905px wide and every absolute pixel constant in this pipeline has eventually had
    to be replaced by a proportional one.
    """
    pad = max(2, round(min(width, height) * COVER_FILL_PAD_FRACTION))
    x1 = max(0, int(x) - pad)
    y1 = max(0, int(y) - pad)
    x2 = min(img_w, int(x + width) + pad)
    y2 = min(img_h, int(y + height) + pad)
    if x2 <= x1 or y2 <= y1:
        return None

    r = max(1.0, min(x2 - x1, y2 - y1) * 0.22)
    pts = []
    for cx, cy, start in (
        (x2 - r, y1 + r, -90.0),
        (x2 - r, y2 - r, 0.0),
        (x1 + r, y2 - r, 90.0),
        (x1 + r, y1 + r, 180.0),
    ):
        for i in range(corner_steps + 1):
            a = math.radians(start + 90.0 * i / corner_steps)
            pts.append([round(cx + r * math.cos(a)), round(cy + r * math.sin(a))])
    return pts


def cover_fill_for_region(img, mask_polygon, x, y, width, height):
    """How to erase this region: `(color, polygon)`.

    When the region is close enough to flat, this is the existing behaviour -- its own median
    colour over its own outline, which is honest erasure -- and `polygon` comes back as the mask
    that was passed in.

    Otherwise it is R2. There is no flat colour and often no real outline, so instead of drawing
    nothing and leaving English on top of unerased Japanese, we hand back a *new* balloon covering
    the source text, in the region's dominant colour. Nothing downstream needs to learn a new
    field: a synthesized balloon is just a mask polygon, and the renderer already fills one of
    those with `backgroundColor`.

    It is visibly an addition to the artwork rather than a repair of it. That is the trade the
    references make too, and it beats the alternative we currently ship, which is illegible.

    A region with **no** mask always gets a synthesized shape, whichever way the colour was found.
    Sampling the border of a text box that sits on a flat-enough area does give a usable colour --
    the yellow of sample10's burst comes back that way -- but with no polygon to paint it into, the
    renderer falls back to the element's own box, which after R1 rejected a balloon is the sliver
    the glyphs occupy. Right colour, wrong shape, and the source lettering still shows around it.
    """
    flat = detect_background_color_poly(img, mask_polygon) if mask_polygon else None
    if flat is not None:
        return flat, mask_polygon
    if img is None:
        return None, mask_polygon

    if not mask_polygon:
        flat = detect_background_color(img, x, y, width, height)
    covering = flat or dominant_color(img, mask_polygon, box=(x, y, width, height), ring=True)
    if covering is None or not COVER_FILL_ENABLED:
        return covering, mask_polygon
    img_h, img_w = img.shape[:2]
    poly = cover_balloon_polygon(x, y, width, height, img_w, img_h)
    if poly is None:
        return covering, mask_polygon
    return covering, poly


def bubble_covers_text(mask, fx1, fy1, fx2, fy2):
    """Is this mask plausibly the balloon that holds that text box? (R1)

    Assignment is by greatest overlap and nothing else, which cannot tell a balloon from two things
    that beat one on overlap. Both are rejected here, and they are separate tests because they are
    separate failures -- see `BUBBLE_MIN_TEXT_COVERAGE` in config.py for the corpus measurement that
    replaced a single coverage threshold, which had no separation in it at all.

    - *the balloon is inside its own text*: YOLO fires on the white outline drawn around unenclosed
      lettering, a text-shaped blob sitting exactly on the glyphs. A container cannot be contained
      by its contents.
    - *the balloon barely touches its text*: with no floor on overlap, a region can be handed a
      balloon that meets it by half a percent.

    Only meaningful for a *detected* contour. The raw OCR rectangle standing in for a balloon is
    engulfed by its own text by construction, and callers must not put one through this.
    """
    area = (fx2 - fx1) * (fy2 - fy1)
    if area <= 0:
        return False
    overlap = int(np.count_nonzero(mask[fy1:fy2, fx1:fx2]))
    if overlap / area < BUBBLE_MIN_TEXT_COVERAGE:
        return False
    balloon = int(np.count_nonzero(mask))
    return not (balloon > 0 and overlap / balloon >= BUBBLE_MAX_SELF_CONTAINMENT)


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

        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]
    except Exception as e:
        print(f"[OCR] Error splitting polygon: {e}", flush=True)
        raise e


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
    (ocr_x + ocr_w / 2) - x1  # type: ignore
    (ocr_y + ocr_h / 2) - y1  # type: ignore

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
            # The window the search ran in. A caller cannot tell an enclosed shape from the page
            # background without it: a blob that runs off the edge of the crop was clipped by the
            # crop, so its bounding box describes the window, not anything in the artwork.
            "searchWindow": (x1, y1, x2, y2),
        }

    return None


def contour_bubble_for_unmatched(img, rx, ry, rw, rh, img_w, img_h):
    """Last-chance bubble geometry for a text fragment YOLO matched to no bubble.

    YOLO is single-class and only confident on canonical enclosed balloons; irregular thought clouds
    score far below threshold, so their text ends up with the raw OCR bbox standing in for a bubble —
    the tight vertical Japanese column, which typesets English one word per line. The OpenCV contour
    search finds a containing shape for roughly half of these.

    Returns the contour dict from :func:`detect_bubble_contour`, or ``None`` when the flag is off,
    nothing was found, or the result fails a guard. Guards, in order: the blob must be enclosed
    inside the search window rather than clipped by it, it must fully contain the text (a contour
    that merely overlaps is tracing something else), it must not balloon past
    ``BUBBLE_CONTOUR_MAX_GROWTH`` per axis, and it must not cover more than
    ``BUBBLE_CONTOUR_MAX_PAGE_FRACTION`` of the page — that last one is what rejects a contour that
    has escaped into the panel border or the artwork.
    """
    if not BUBBLE_CONTOUR_FALLBACK:
        return None

    found = detect_bubble_contour(img, rx, ry, rw, rh)
    if not found:
        return None

    fx, fy, fw, fh = found["x"], found["y"], found["width"], found["height"]

    # Enclosed by the window, not clipped by it.
    #
    # The threshold that finds a balloon interior also finds the page background, and free-floating
    # text usually sits on it. That blob has no boundary inside the crop, so `boundingRect` returns
    # the crop, and every other guard here passes by construction: a search window contains the text
    # it was built around, sits within `2 * pad` of its size, and is a small part of the page. On
    # Openrouter ch. 11 p22 all three unmatched fragments came back as their own search window --
    # 129x1271 for a 49x489 caption -- which then reads downstream as a bubble a thought cloud's
    # worth of text can be laid into.
    #
    # A shape that is actually enclosed leaves a margin on every side. Edges where the window was
    # clamped by the page are exempt: there the crop boundary is the paper, and a balloon that runs
    # to the edge of the page is a real balloon.
    window = found.get("searchWindow")
    if window:
        wx1, wy1, wx2, wy2 = window
        if (
            (fx <= wx1 and wx1 > 0)
            or (fy <= wy1 and wy1 > 0)
            or (fx + fw >= wx2 and wx2 < img_w)
            or (fy + fh >= wy2 and wy2 < img_h)
        ):
            return None

    # Contains the text on every edge (2px of slack for contour simplification).
    if fx > rx + 2 or fy > ry + 2 or fx + fw < rx + rw - 2 or fy + fh < ry + rh - 2:
        return None

    if rw > 0 and fw > rw * BUBBLE_CONTOUR_MAX_GROWTH:
        return None
    if rh > 0 and fh > rh * BUBBLE_CONTOUR_MAX_GROWTH:
        return None

    page_area = float(img_w * img_h)
    if page_area > 0 and (fw * fh) / page_area > BUBBLE_CONTOUR_MAX_PAGE_FRACTION:
        return None

    return found


def process_ocr(job_data):
    from worker.utils.rate_limit import reset_job_costs

    reset_job_costs()

    image_id = job_data["imageId"]
    # The backend sets these from the series context when it enqueues the job.
    # Defaults preserve the original behaviour (Japanese RTL) when not supplied.
    source_language = (job_data.get("sourceLanguage") or "ja").strip().lower()
    reading_direction = (job_data.get("readingDirection") or "rtl").strip().lower()

    vlm_model_used = None
    transcriptions: dict = {}

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
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[OCR] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        panels = image_info.get("panels", [])
    except Exception as e:
        print(f"[OCR] Error fetching image details: {e}", flush=True)
        raise

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[OCR] Error downloading image: {e}", flush=True)
        raise

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

        provider = (job_data.get("ocrProvider") or OCR_CONFIG.provider or "local").lower().strip()
        use_paddle_ocr = (provider == "local") and not disable_local_ocr

        # WARNING: Even when using Cloud VLM OCR (where transcription is offloaded), local models
        # (PP-OCR-Det for text detection and YOLO for bubble detection) still execute locally on
        # this host. We must serialize these local predictions using the "ocr" lock to avoid CPU/GPU
        # overload and OOM crashes. This local bottleneck will be resolved when remote workers on
        # dedicated machines are supported, allowing parallel detection and full OCR job queues.
        # node_scoped: this lock guards *this host's* CPU/GPU, not a shared service, so it must stay
        # per-container — a deployment-wide "ocr" lock would serialise detection across every
        # worker. AUDIT-W4 changed the default the other way for locks like local-llm, which do
        # guard a shared endpoint.
        with acquire_lock("ocr", node_scoped=True):
            # Try PaddleOCR (PP-OCRv5) first — reader is lazily created per language
            paddle_ocr_reader = model_manager.get_paddle_ocr_reader(source_language) if use_paddle_ocr else None
            paddle_ocr_detector = model_manager.get_paddle_ocr_detector(source_language) if not use_paddle_ocr else None

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
                    det_model = os.environ.get("PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det").strip()
                    rec_model = os.environ.get("PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec").strip()
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

                    img_decoded, ocr_upscale = downscale_for_ocr(img_original, max_dim=1024)

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
                    det_model = os.environ.get("PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det").strip()
                    print(
                        f"[OCR] Running PaddleOCR Detector ({det_model}, lang={source_language}).",
                        flush=True,
                    )
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_decoded, ocr_upscale = downscale_for_ocr(img_original, max_dim=1024)
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
                cv2.fillPoly(mask, [poly], 255)  # type: ignore
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

                    # R1: winning the overlap contest is not the same as being this text's balloon.
                    # A balloon that does not cover its own text is not a balloon -- it is the
                    # white stroke around unenclosed lettering, which YOLO scores as a bubble and
                    # which beats every real balloon on overlap because it sits on the glyphs.
                    # Rejecting it here drops the fragment to the unmatched path below, where it
                    # gets covered geometry instead of a text-shaped slab painted over the artwork.
                    if best_b_idx >= 0 and not bubble_covers_text(bubble_masks[best_b_idx], fx1, fy1, fx2, fy2):
                        logger.info(
                            "[YOLO] Rejected bubble %d for fragment at (%d,%d,%dx%d): not a container for it",
                            best_b_idx,
                            fx1,
                            fy1,
                            fx2 - fx1,
                            fy2 - fy1,
                        )
                        best_b_idx = -1
                frag["bubble_idx"] = best_b_idx

            # 4. Group fragments for each bubble and merge them (or create default crop if empty and we are using Cloud VLM)
            candidate_regions = []  # regions we need to OCR/transcribe

            for b_idx, bubble in enumerate(detected_bubbles):
                bx, by, bw, bh = bubble["bbox"]
                bubble_mask = bubble_masks[b_idx]
                assigned_frags = [f for f in raw_fragments if f.get("bubble_idx", -1) == b_idx]

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

                # Run proximity merging inside the bubble to separate multiple semantic bubbles.
                # This is where BUG-2 lives: one YOLO blob routinely holds two touching balloons,
                # and joining them fuses two speakers into one translation unit and one flat fill,
                # which nothing downstream can undo. The mask is in scope here, so the grouping
                # gets the balloon's geometry as well as the text distance.
                merged_bubble_regions = merge_ocr_regions(
                    assigned_frags,
                    grouping=grouping_config(reading_direction),
                    context=bubble_grouping_context(bubble_mask, bubble["mask_polygon"]),
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
                        poly_pts = get_split_polygon(bubble_mask, r_box, img_w, img_h, margin=20)
                        if not poly_pts:
                            poly_pts = bubble["mask_polygon"]

                        # 2. Bounding box of the split polygon
                        sp_x, sp_y, sp_w, sp_h = cv2.boundingRect(np.array(poly_pts, dtype=np.int32))

                        # 3. Bounding box of the eroded mask (safe area)
                        split_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(split_mask, [np.array(poly_pts, dtype=np.int32)], 255)  # type: ignore
                        erosion_px = YOLO_MASK_EROSION
                        kernel_erode = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * erosion_px + 1, 2 * erosion_px + 1),
                        )
                        eroded_split_mask = cv2.erode(split_mask, kernel_erode, iterations=1)
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
            unmatched_frags = [f for f in raw_fragments if f.get("bubble_idx", -1) == -1]
            if unmatched_frags:
                # No bubble, so no mask and no clearance veto -- this path stays on distance alone,
                # and gets the orientation vote, which is the only lever that reaches a page where
                # YOLO found nothing at all.
                merged_unmatched = merge_ocr_regions(unmatched_frags, grouping=grouping_config(reading_direction))

                for idx, r_sub in enumerate(merged_unmatched):
                    rx, ry, rw, rh = (
                        r_sub["x"],
                        r_sub["y"],
                        r_sub["width"],
                        r_sub["height"],
                    )

                    # YOLO matched this fragment to no bubble. Before accepting the text bbox as the
                    # bubble — which typesets English into the vertical Japanese column — try the
                    # contour search; on irregular clouds it usually finds the shape YOLO missed.
                    contour = contour_bubble_for_unmatched(img, rx, ry, rw, rh, img_w, img_h)

                    if contour:
                        px1 = max(0, contour["x"])
                        py1 = max(0, contour["y"])
                        px2 = min(img_w, contour["x"] + contour["width"])
                        py2 = min(img_h, contour["y"] + contour["height"])
                        mask_polygon = contour["maskPolygon"]
                    else:
                        # Generate tight padded "virtual bubble" mask to allow typesetter inpainting / background cleaning
                        pad = 0
                        px1 = max(0, rx - pad)
                        py1 = max(0, ry - pad)
                        px2 = min(img_w, rx + rw + pad)
                        py2 = min(img_h, ry + rh + pad)
                        mask_polygon = [[px1, py1], [px2, py1], [px2, py2], [px1, py2]]

                    candidate_regions.append(
                        {
                            "type": "direct_text",
                            "direct_idx": idx,
                            # x/y/w/h stay the OCR text extent — that is what gets cropped and sent
                            # to the recognizer. Only the bubble geometry below comes from the contour.
                            "x": rx,
                            "y": ry,
                            "width": rw,
                            "height": rh,
                            "poly_pts": mask_polygon,
                            "safe_rect": [px1, py1, px2 - px1, py2 - py1],
                            "bubbleX": px1,
                            "bubbleY": py1,
                            "bubbleWidth": px2 - px1,
                            "bubbleHeight": py2 - py1,
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
                    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"
                    use_fallback_models = job_data.get("useFallbackModels", True)

                    # Generate base64 crops for all candidate regions
                    crops_payload = []
                    for cr_idx, r in enumerate(candidate_regions):
                        rx, ry, rw, rh = r["x"], r["y"], r["width"], r["height"]
                        rx1, ry1 = max(0, rx), max(0, ry)
                        rx2, ry2 = min(img_w, rx + rw), min(img_h, ry + rh)

                        crop = img[ry1:ry2, rx1:rx2]  # type: ignore
                        if crop.size > 0:
                            _, buffer = cv2.imencode(".jpg", crop)
                            base64_image = base64.b64encode(buffer).decode("utf-8")  # type: ignore
                            crops_payload.append({"id": f"region_{cr_idx}", "base64": base64_image})

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
                                        "confidence": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                        },
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

                        lang_name = LANG_MAP.get(source_language.lower(), source_language)
                        sys_prompt = (
                            f"You are an expert manga OCR system. Perform OCR on each of the provided image crops. "
                            f"The source language is {lang_name}. Return ONLY a valid JSON object matching the schema. "
                            "If the text is a sound effect (SFX), gibberish, an author handle, or already completely in English, return an empty string for the text field."
                        )

                        transcriptions = {}

                        def chunk_list(lst, n):
                            return [lst[i : i + n] for i in range(0, len(lst), n)]

                        crop_chunks = chunk_list(crops_payload, 10)

                        def process_crop_chunk(chunk_idx, chunk):
                            nonlocal vlm_model_used
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

                                user_model = vlm_model or OCR_CONFIG.vlm_model
                                try:
                                    chunk_res = try_cloud_ai_vision_batch(
                                        provider,
                                        api_key,
                                        user_model,
                                        chunk,
                                        schema,
                                        system_prompt=sys_prompt,
                                        routing_strategy=routing_strategy,
                                    )
                                    if chunk_res:
                                        parsed = json.loads(
                                            chunk_res.strip().removeprefix("```json").removesuffix("```").strip()
                                        )
                                        results_list = parsed.get("results", [])
                                        if results_list:
                                            print(
                                                f"[OCR] Successfully processed chunk {chunk_idx + 1} using model '{user_model}'",
                                                flush=True,
                                            )
                                            vlm_model_used = user_model

                                    if not chunk_res or not results_list:
                                        # Fallback to global default model (only if use_fallback_models is True)
                                        global_model = OCR_CONFIG.vlm_model
                                        global_provider = OCR_CONFIG.provider
                                        if (
                                            use_fallback_models
                                            and global_provider == provider
                                            and global_model
                                            and global_model != user_model
                                        ):
                                            print(
                                                f"[OCR] Falling back to global default VLM model '{global_model}'...",
                                                flush=True,
                                            )
                                            chunk_res = try_cloud_ai_vision_batch(
                                                provider,
                                                api_key,
                                                global_model,
                                                chunk,
                                                schema,
                                                system_prompt=sys_prompt,
                                                routing_strategy=routing_strategy,
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
                                                        f"[OCR] Successfully processed chunk {chunk_idx + 1} using global fallback model '{global_model}'",
                                                        flush=True,
                                                    )
                                                    vlm_model_used = global_model
                                        else:
                                            print(
                                                "[OCR] No fallback applied (global provider different or model identical).",
                                                flush=True,
                                            )

                                except Exception as parse_err:
                                    print(
                                        f"[OCR] Failed for model on chunk {chunk_idx + 1}: {parse_err}",
                                        flush=True,
                                    )
                            else:
                                local_model = job_data.get("ocrModel") or os.environ.get("LOCAL_VLM_MODEL", "").strip()
                                if local_model:
                                    user_prompt = "Extract the text from this speech bubble."
                                    crop_schema = {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "The extracted text",
                                            },
                                            "confidence": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
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
                                                            "text": parsed.get("text", ""),
                                                            "confidence": parsed.get("confidence", 0.99),
                                                        }
                                                    )
                                                    vlm_model_used = local_model
                                                except Exception:
                                                    results_list.append(
                                                        {
                                                            "id": crop_info["id"],
                                                            "text": crop_res,
                                                            "confidence": 0.99,
                                                        }
                                                    )
                                        except Exception as local_vlm_err:
                                            print(
                                                f"[OCR] Local VLM failed for crop {crop_info['id']}: {local_vlm_err}",
                                                flush=True,
                                            )
                            return results_list

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            futures = {
                                executor.submit(process_crop_chunk, idx, chunk): chunk
                                for idx, chunk in enumerate(crop_chunks)
                            }
                            for future in concurrent.futures.as_completed(futures):
                                results_list = future.result()
                                for item in results_list:
                                    item_id = item.get("id", "")
                                    item_text = item.get("text", "")
                                    item_conf = float(item.get("confidence", 0.99))
                                    transcriptions[item_id] = {
                                        "text": item_text,
                                        "confidence": min(max(item_conf, 0.0), 1.0),
                                    }

                    # Create regions list
                    for cr_idx, r in enumerate(candidate_regions):
                        entry = transcriptions.get(f"region_{cr_idx}", {})
                        final_text = entry.get("text", "").strip()
                        model_conf = entry.get("confidence", 0.99)
                        if final_text:
                            from worker.services.ocr import is_valid_ocr_text

                            if not is_valid_ocr_text(final_text):
                                print(
                                    f"[OCR] VLM result for region_{cr_idx} rejected by validation: '{final_text}'",
                                    flush=True,
                                )
                                continue
                            bg_color, r["poly_pts"] = cover_fill_for_region(
                                img, r["poly_pts"], r["x"], r["y"], r["width"], r["height"]
                            )
                            if r["type"] == "bubble":
                                regions.append(
                                    {
                                        "text": final_text,
                                        "detectedLanguage": detect_language(final_text),
                                        "confidence": model_conf,
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
                                # direct text / free-floating
                                regions.append(
                                    {
                                        "text": final_text,
                                        "detectedLanguage": detect_language(final_text),
                                        "confidence": model_conf,
                                        "rotation": 0.0,
                                        "x": r["x"],
                                        "y": r["y"],
                                        "width": r["width"],
                                        "height": r["height"],
                                        "panelId": None,
                                        "bubbleReadingOrder": 0,
                                        "backgroundColor": bg_color,
                                        # Contour geometry when the fallback found a shape, else the
                                        # text bbox — which the backend detects and grows rather
                                        # than insets, since it is not a container.
                                        "bubbleX": r.get("bubbleX", r["x"]),
                                        "bubbleY": r.get("bubbleY", r["y"]),
                                        "bubbleWidth": r.get("bubbleWidth", r["width"]),
                                        "bubbleHeight": r.get("bubbleHeight", r["height"]),
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
                        bg_color, r["poly_pts"] = cover_fill_for_region(
                            img, r["poly_pts"], r["x"], r["y"], r["width"], r["height"]
                        )
                        if r["type"] == "bubble":
                            regions.append(
                                {
                                    "text": final_text,
                                    "detectedLanguage": (detect_language(final_text) if final_text else "ja"),
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
                                    "detectedLanguage": (
                                        detect_language(final_text) if final_text else r["detectedLanguage"]
                                    ),
                                    "confidence": r["confidence"],
                                    "rotation": 0.0,
                                    "x": r["x"],
                                    "y": r["y"],
                                    "width": r["width"],
                                    "height": r["height"],
                                    "panelId": None,
                                    "bubbleReadingOrder": 0,
                                    "backgroundColor": bg_color,
                                    # Contour geometry when the fallback found a shape, else the text
                                    # bbox — see the cloud-mode branch above.
                                    "bubbleX": r.get("bubbleX", r["x"]),
                                    "bubbleY": r.get("bubbleY", r["y"]),
                                    "bubbleWidth": r.get("bubbleWidth", r["width"]),
                                    "bubbleHeight": r.get("bubbleHeight", r["height"]),
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
                    bubble_box and bubble_box["width"] <= width * 2.5 and bubble_box["height"] <= height * 2.5
                )
                if use_bubble_contour:
                    bx, by, bw, bh = (
                        bubble_box["x"],  # type: ignore
                        bubble_box["y"],  # type: ignore
                        bubble_box["width"],  # type: ignore
                        bubble_box["height"],  # type: ignore
                    )
                else:
                    bx, by, bw, bh = x, y, width, height

                mask_polygon = bubble_box.get("maskPolygon") if use_bubble_contour else None  # type: ignore
                bg_color, mask_polygon = cover_fill_for_region(img, mask_polygon, x, y, width, height)

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
                        "maskPolygon": (json.dumps(mask_polygon) if mask_polygon else None),
                        "safeTextX": bx,
                        "safeTextY": by,
                        "safeTextW": bw,
                        "safeTextH": bh,
                    }
                )

            regions = merge_ocr_regions(regions, grouping=grouping_config(reading_direction))

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
        sorted_panel_indices = sorted(panel_regions_map.keys(), key=lambda idx: panels[idx]["readingOrder"])

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

        avg_conf = sum(r["confidence"] for r in ordered_regions) / len(ordered_regions) if ordered_regions else 1.0

        rec_model = os.environ.get("PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec").strip()
        model_identifier = f"PaddleOCR({rec_model})"
        if vlm_model_used:
            model_identifier += f" + {vlm_model_used}"

        page_id = job_data.get("pageId")
        callback_payload = {
            "jobId": job_data.get("jobId"),
            "imageId": image_id,
            "pageId": page_id,
            "modelIdentifier": model_identifier,
            "confidence": avg_conf,
            "sourceLanguage": source_language,
            "readingDirection": reading_direction,
            "regionCount": len(ordered_regions),
            "regions": ordered_regions,
        }

        try:
            from worker.utils.rate_limit import get_job_costs

            costs = get_job_costs()
            if costs:
                cost_payload = {
                    "currency": "USD",
                    "breakdown": costs,
                    "prompt_tokens": sum((c.get("prompt_tokens") or 0) for c in costs),
                    "estimated_cost": sum((c.get("estimated_cost") or 0.0) for c in costs),
                    "completion_tokens": sum((c.get("completion_tokens") or 0) for c in costs),
                }
                callback_payload["cost"] = cost_payload
        except Exception as e:
            print(f"[OCR] Error fetching job costs: {e}", flush=True)

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
