import cv2


def downscale_for_ocr(img, max_dim=1024):
    """
    Reduce memory consumption before OCR.
    Returns (downscaled_img, scale_factor).
    scale_factor is the multiplier to convert downscaled coords back to original.
    """
    if img is None:
        return img, 1.0

    h, w = img.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return img, 1.0

    scale = max_dim / largest
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # inverse scale: multiply OCR coords by this to get original-image coords
    return resized, 1.0 / scale


def calculate_overlap_area(r, p):
    # r is ocr region dict, p is panel dict (from db)
    rx, ry, rw, rh = r["x"], r["y"], r["width"], r["height"]
    px, py, pw, ph = p["bboxX"], p["bboxY"], p["bboxW"], p["bboxH"]

    overlap_x = max(0, min(rx + rw, px + pw) - max(rx, px))
    overlap_y = max(0, min(ry + rh, py + ph) - max(ry, py))
    return overlap_x * overlap_y


def download_image(image_info):
    presigned_url = image_info.get("presignedUrl")
    if presigned_url:
        import requests

        from worker.config import logger

        logger.info("Downloading image via presigned GET URL")
        res = requests.get(presigned_url)
        res.raise_for_status()
        return res.content
    else:
        from worker.config import logger, minio_client

        storage_path = image_info["storagePath"]
        logger.info(f"Downloading image from local MinIO path: {storage_path}")
        response = minio_client.get_object("manga-library", storage_path)
        return response.read()
