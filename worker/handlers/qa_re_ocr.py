import requests
import cv2
import numpy as np

from worker.config import CALLBACK_URL, BACKEND_HEADERS
from worker.utils.image import download_image
from worker.utils.text import detect_language
from worker.services.ocr import perform_redo_ocr


def process_qa_re_ocr(job_data):
    image_id = job_data.get("imageId")
    region_ids = job_data.get("regionsToReOcr", [])

    print(
        f"[QA Re-OCR] Processing image {image_id} for regions: {region_ids}", flush=True
    )

    if not image_id or not region_ids:
        print("[QA Re-OCR] Missing imageId or regionsToReOcr", flush=True)
        return

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(
                f"[QA Re-OCR] Failed to get image info: {res.status_code}", flush=True
            )
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        print(f"[QA Re-OCR] Error fetching image details: {e}", flush=True)
        raise

    # Filter regions
    target_regions = [r for r in ocr_regions if r["id"] in region_ids]
    if not target_regions:
        print("[QA Re-OCR] No matching regions found in image details", flush=True)
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[QA Re-OCR] Error downloading image: {e}", flush=True)
        raise

    results = []

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]

        for region in target_regions:
            try:
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

                    results.append(
                        {
                            "regionId": region["id"],
                            "text": text,
                            "confidence": confidence,
                            "detectedLanguage": detected_lang,
                        }
                    )
                    print(
                        f"[QA Re-OCR] Region {region['id']} success: '{text}' (conf={confidence})",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[QA Re-OCR] Failed to OCR region {region['id']}: {e}", flush=True
                )

    except Exception as e:
        print(f"[QA Re-OCR] Error during batch OCR process: {e}", flush=True)
        raise

    callback_payload = {"imageId": image_id, "results": results}

    try:
        callback_url = f"{CALLBACK_URL}/qa-re-ocr"
        res = requests.post(
            callback_url, json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[QA Re-OCR] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA Re-OCR] Failed to post callback: {e}", flush=True)
