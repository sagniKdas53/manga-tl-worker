import uuid
import requests
import cv2
import numpy as np

from worker.config import logger, CALLBACK_URL, BACKEND_HEADERS
from worker.utils.image import download_image
from worker.utils.text import detect_language
from worker.services.ocr import perform_redo_ocr
from worker.services.translation import translate_text


def process_region_redo(job_data):
    from worker.utils.rate_limit import reset_job_costs

    reset_job_costs()
    image_id = job_data["imageId"]
    region_id = job_data["regionId"]
    redo_type = job_data["redoType"]  # 'ocr' or 'translation'

    # Generate request_id specifically for translation redo tracking
    request_id = str(uuid.uuid4())[:8] if redo_type == "translation" else None
    req_prefix = f"[{request_id}] " if request_id else ""

    if redo_type == "translation":
        logger.info(
            f"{req_prefix}Processing region redo: {region_id} on image {image_id} with type {redo_type}"
        )
    else:
        print(
            f"[Region Redo] Processing region: {region_id} on image {image_id} with type {redo_type}",
            flush=True,
        )

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            if redo_type == "translation":
                logger.error(f"{req_prefix}Failed to get image info: {res.status_code}")
            else:
                print(
                    f"[Region Redo] Failed to get image info: {res.status_code}",
                    flush=True,
                )
            return
        image_info = res.json()
        image_info["storagePath"]
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Error fetching image details: {e}")
        else:
            print(f"[Region Redo] Error fetching image details: {e}", flush=True)
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
            print(
                f"[Region Redo] Region {region_id} not found in image details",
                flush=True,
            )
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Error downloading image: {e}")
        else:
            print(f"[Region Redo] Error downloading image: {e}", flush=True)
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
                is_success, buffer = cv2.imencode(".jpg", crop)
                crop_bytes = buffer.tobytes()

                text, confidence = perform_redo_ocr(
                    crop_bytes, region["detectedLanguage"]
                )
                detected_lang = detect_language(text)
                callback_payload["text"] = text
                callback_payload["confidence"] = confidence
                callback_payload["detectedLanguage"] = detected_lang
                print(
                    f"[Region Redo] Redo OCR success: '{text}' (conf={confidence}, lang={detected_lang})",
                    flush=True,
                )
        except Exception as e:
            print(f"[Region Redo] Redo OCR failed: {e}", flush=True)
            raise

    elif redo_type == "translation":
        try:
            text = region["text"]
            lang = region["detectedLanguage"]
            translated = translate_text(text, source_lang=lang, request_id=request_id)
            callback_payload["translatedText"] = translated
            callback_payload["translationFailed"] = translated is None
            logger.info(
                f"{req_prefix}Redo Translation result: '{translated}' (failed={translated is None})"
            )
            from worker.utils.rate_limit import get_job_costs

            costs = get_job_costs()
            if costs:
                has_na = any(c.get("estimated_cost") is None for c in costs)
                if has_na:
                    total_estimated_cost = None
                else:
                    total_estimated_cost = sum(
                        c.get("estimated_cost", 0.0) or 0.0 for c in costs
                    )
                total_prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in costs)
                total_completion_tokens = sum(
                    c.get("completion_tokens", 0) or 0 for c in costs
                )

                if total_estimated_cost is None:
                    cost_str = "N/A"
                elif total_estimated_cost == 0.0:
                    cost_str = "$0.000"
                else:
                    cost_str = f"${total_estimated_cost:.5f}"

                logger.info(
                    f"{req_prefix}Redo translation estimated cost: {cost_str} "
                    f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
                )
        except Exception as e:
            logger.error(f"{req_prefix}Redo Translation failed: {e}")
            raise

    try:
        callback_url = CALLBACK_URL.replace(
            "/jobs/callback", f"/ocr-regions/{region_id}/callback"
        )
        res = requests.post(
            callback_url, json=callback_payload, headers=BACKEND_HEADERS
        )
        if redo_type == "translation":
            logger.info(f"{req_prefix}Callback status code: {res.status_code}")
        else:
            print(f"[Region Redo] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        if redo_type == "translation":
            logger.error(f"{req_prefix}Failed to post callback: {e}")
        else:
            print(f"[Region Redo] Failed to post callback: {e}", flush=True)
