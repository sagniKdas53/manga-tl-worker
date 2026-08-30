import uuid

import cv2
import numpy as np
import requests

from worker.config import CALLBACK_URL, backend_headers, logger
from worker.services.ocr import perform_redo_ocr
from worker.services.translation import translate_text
from worker.utils.image import download_image
from worker.utils.text import detect_language


def process_region_redo(job_data):
    image_id = job_data["imageId"]
    region_id = job_data["regionId"]
    redo_type = job_data["redoType"]  # 'ocr' or 'translation'

    # Generate request_id specifically for translation redo tracking
    request_id = str(uuid.uuid4())[:8] if redo_type == "translation" else None
    req_prefix = f"[{request_id}] " if request_id else ""

    if redo_type == "translation":
        logger.info(f"{req_prefix}Processing region redo: {region_id} on image {image_id} with type {redo_type}")
    else:
        logger.info(f"[Region Redo] Processing region: {region_id} on image {image_id} with type {redo_type}")

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
            if redo_type == "translation":
                logger.error(f"{req_prefix}Failed to get image info: {res.status_code}")
            else:
                logger.error(f"[Region Redo] Failed to get image info: {res.status_code}")
            return
        image_info = res.json()
        image_info["storagePath"]
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Error fetching image details: {e}")
        else:
            logger.error(f"[Region Redo] Error fetching image details: {e}")
        return

    region = None
    for r in ocr_regions:
        if r["id"] == region_id:
            region = r
            break

    if region is None:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Region {region_id} not found in image details")
        else:
            logger.warning(f"[Region Redo] Region {region_id} not found in image details")
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Error downloading image: {e}")
        else:
            logger.error(f"[Region Redo] Error downloading image: {e}")
        return

    callback_payload = {}

    if redo_type == "ocr":
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]

            x, y, width, height = (
                region["bboxX"],
                region["bboxY"],
                region["bboxW"],
                region["bboxH"],
            )
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + width), min(img_h, y + height)

            if (x2 - x1) > 0 and (y2 - y1) > 0:
                crop = img[y1:y2, x1:x2]
                _is_success, buffer = cv2.imencode(".jpg", crop)
                crop_bytes = buffer.tobytes()

                text, confidence = perform_redo_ocr(crop_bytes, region["detectedLanguage"], "user_rejected")
                detected_lang = detect_language(text)
                callback_payload["text"] = text
                callback_payload["confidence"] = confidence
                callback_payload["detectedLanguage"] = detected_lang
                logger.info(f"[Region Redo] Redo OCR success: '{text}' (conf={confidence}, lang={detected_lang})")
        except Exception as e:
            logger.error(f"[Region Redo] Redo OCR failed: {e}")
            raise

    elif redo_type == "translation":
        try:
            text = region["text"]
            lang = region["detectedLanguage"]
            translated = translate_text(
                text,
                source_lang=lang,
                request_id=request_id,
                provider=job_data.get("tlProvider"),
                model=job_data.get("tlModel"),
            )
            callback_payload["translatedText"] = translated
            callback_payload["translationFailed"] = translated is None
            logger.info(f"{req_prefix}Redo Translation result: '{translated}' (failed={translated is None})")
            from worker.utils.rate_limit import build_cost_payload, format_cost, get_job_costs

            # This block used to log a cost and then throw it away — the payload was never attached
            # to the callback, so a redo's spend never reached the database. It also inlined its own
            # formatter with thresholds that disagreed with format_cost.
            cost_payload = build_cost_payload(get_job_costs())
            if cost_payload:
                callback_payload["cost"] = cost_payload
                cost_str = format_cost(cost_payload.get("estimated_cost"))
                if cost_payload["unknown_calls"]:
                    cost_str += f" ({cost_payload['unknown_calls']} of {len(cost_payload['breakdown'])} calls unpriced)"
                logger.info(
                    f"{req_prefix}Redo translation estimated cost: {cost_str} "
                    f"(Tokens: in={cost_payload['prompt_tokens']}, out={cost_payload['completion_tokens']})"
                )
        except Exception as e:
            logger.error(f"{req_prefix}Redo Translation failed: {e}")
            raise

    try:
        callback_url = CALLBACK_URL.replace("/jobs/callback", f"/ocr-regions/{region_id}/callback")
        res = requests.post(callback_url, json=callback_payload, headers=backend_headers())
        if redo_type == "translation":
            logger.info(f"{req_prefix}Callback status code: {res.status_code}")
        else:
            logger.debug(f"[Region Redo] Callback status code: {res.status_code}")
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Failed to post callback: {e}")
        else:
            logger.error(f"[Region Redo] Failed to post callback: {e}")
