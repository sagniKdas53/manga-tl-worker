import logging
import time

import requests

from worker.config import CALLBACK_URL, backend_headers

logger = logging.getLogger(__name__)


def process_stub(job_data, job_type):
    image_id = job_data["imageId"]
    logger.info(f"[Stub - {job_type}] Processing image: {image_id}")

    # Mimic work
    time.sleep(0.5)

    callback_payload = {"jobId": job_data.get("jobId"), "imageId": image_id}
    try:
        res = requests.post(f"{CALLBACK_URL}/{job_type}", json=callback_payload, headers=backend_headers())
        logger.debug(f"[Stub - {job_type}] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[Stub - {job_type}] Failed to post callback: {e}")
