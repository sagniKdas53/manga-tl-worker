import time

import requests

from worker.config import BACKEND_HEADERS, CALLBACK_URL


def process_stub(job_data, job_type):
    image_id = job_data["imageId"]
    print(f"[Stub - {job_type}] Processing image: {image_id}", flush=True)

    # Mimic work
    time.sleep(0.5)

    callback_payload = {"imageId": image_id}
    try:
        res = requests.post(
            f"{CALLBACK_URL}/{job_type}", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(
            f"[Stub - {job_type}] Callback status code: {res.status_code}", flush=True
        )
    except Exception as e:
        print(f"[Stub - {job_type}] Failed to post callback: {e}", flush=True)
