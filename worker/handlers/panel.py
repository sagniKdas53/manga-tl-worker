import requests
from worker.config import CALLBACK_URL, BACKEND_HEADERS, redis_client
from worker.services.panel_detection import detect_panels
from worker.utils.image import download_image


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

    print(
        f"[Panel Detection] Processing image: {image_id} (direction={reading_direction}){progress_str}",
        flush=True,
    )

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(
                f"[Panel Detection] Failed to get image info: {res.status_code}",
                flush=True,
            )
            return
        image_info = res.json()
    except Exception as e:
        print(f"[Panel Detection] Error fetching image details: {e}", flush=True)
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[Panel Detection] Error downloading image: {e}", flush=True)
        return

    panels = detect_panels(img_bytes, reading_direction=reading_direction)
    print(
        f"[Panel Detection] Detected {len(panels)} panels for image {image_id}",
        flush=True,
    )

    callback_payload = {"imageId": image_id, "panels": panels}
    try:
        res = requests.post(
            f"{CALLBACK_URL}/panel", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[Panel Detection] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[Panel Detection] Failed to post callback to backend: {e}", flush=True)
