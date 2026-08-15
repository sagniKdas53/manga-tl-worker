import logging

import requests

from worker.config import CALLBACK_URL, backend_headers, redis_client
from worker.services.panel_detection import detect_panels
from worker.utils.image import download_image

logger = logging.getLogger(__name__)


def process_panel_detection(job_data):
    image_id = job_data["imageId"]
    reading_direction = (job_data.get("readingDirection") or "rtl").strip().lower()

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:panel-detection")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    logger.info(f"[Panel Detection] Processing image: {image_id} (direction={reading_direction}){progress_str}")

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
            raise Exception(f"Failed to get image info: {res.status_code}")
        image_info = res.json()
    except Exception as e:
        logger.error(f"[Panel Detection] Error fetching image details: {e}")
        raise

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        logger.error(f"[Panel Detection] Error downloading image: {e}")
        raise

    panels = detect_panels(img_bytes, reading_direction=reading_direction)
    logger.info(f"[Panel Detection] Detected {len(panels)} panels for image {image_id}")

    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": page_id,
        "panels": panels,
    }
    try:
        res = requests.post(f"{CALLBACK_URL}/panel", json=callback_payload, headers=backend_headers())
        logger.debug(f"[Panel Detection] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[Panel Detection] Failed to post callback to backend: {e}")
