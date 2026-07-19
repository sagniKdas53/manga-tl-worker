import hashlib
import os

import cv2
import numpy as np

from worker.config import (
    YOLO_CONF_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_MASK_EROSION,
    YOLO_MODEL_PATH,
    YOLO_PINNED_CHECKSUM,
    logger,
)

_ort_session = None


def get_sha256(file_path):
    if not os.path.exists(file_path):
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_ort_session():
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    if not YOLO_MODEL_PATH or not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"Required YOLO bubble detection model is not available at path: {YOLO_MODEL_PATH}. "
            "Cannot proceed in offline mode without the required model."
        )

    # Checksum verification
    current_checksum = get_sha256(YOLO_MODEL_PATH)
    if current_checksum != YOLO_PINNED_CHECKSUM:
        logger.warning(f"[YOLO] Pinned checksum mismatch! Expected: {YOLO_PINNED_CHECKSUM}, got: {current_checksum}")
    else:
        logger.info("[YOLO] Model checksum matches pinned checksum.")

    try:
        import onnxruntime as ort

        logger.info(f"[YOLO] Loading ONNX model from {YOLO_MODEL_PATH} via ONNX Runtime...")
        # Load ONNX session (CPU by default)
        _ort_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
        logger.info("[YOLO] ONNX Runtime session initialized successfully.")
        return _ort_session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model via ONNX Runtime: {e}") from e


def letterbox(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = (round(shape[1] * r), round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh), new_unpad


def detect_bubbles_yolo(img):
    """
    Detect speech bubbles once per page using YOLO11n-seg via ONNX Runtime.
    Returns:
        List of dicts, each containing:
            "bbox": [x, y, w, h] (original image scale)
            "confidence": float
            "mask_polygon": list of [x, y] coordinates (simplified, original scale)
            "safe_rect": [x, y, w, h] (eroded bounds, original scale)
    """
    session = get_ort_session()
    if session is None:
        raise RuntimeError("YOLO model session initialization failed. Cannot proceed without the required model.")

    if img is None:
        return []

    import time

    start_time = time.perf_counter()

    orig_h, orig_w = img.shape[:2]

    # Preprocess (Letterbox to YOLO_INPUT_SIZE)
    input_sz = YOLO_INPUT_SIZE
    padded_img, r, (dw, dh), (unpad_w, unpad_h) = letterbox(img, new_shape=(input_sz, input_sz))

    # Convert BGR to RGB, normalize, transpose to BCHW
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Inference
    try:
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)
    except Exception as e:
        logger.error(f"[YOLO] ONNX Inference run failed: {e}")
        return None

    # Postprocess
    # output0: detections [1, 37, 33600]
    # output1: prototype masks [1, 32, 320, 320]
    preds = outputs[0][0]  # [37, 33600]  # type: ignore
    proto = outputs[1][0]  # [32, 320, 320]  # type: ignore

    boxes = []
    scores = []
    coefficients = []
    class_ids = []

    for i in range(preds.shape[1]):
        score = np.max(preds[4:7, i])
        if score >= YOLO_CONF_THRESHOLD:
            class_id = int(np.argmax(preds[4:7, i]))
            cx, cy, w, h = preds[0:4, i]
            x = cx - w / 2
            y = cy - h / 2
            boxes.append([float(x), float(y), float(w), float(h)])
            scores.append(float(score))
            coefficients.append(preds[7:, i].tolist())
            class_ids.append(class_id)

    if not boxes:
        logger.info(f"[YOLO] No bubbles detected. Inference took {time.perf_counter() - start_time:.3f}s")
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONF_THRESHOLD, 0.45)
    if len(indices) == 0:
        logger.info(f"[YOLO] 0 bubbles survived NMS. Inference took {time.perf_counter() - start_time:.3f}s")
        return []

    indices = np.array(indices).flatten().tolist()

    bubbles = []
    for idx in indices:
        box = boxes[idx]  # [x, y, w, h] in 1280x1280 padded image space
        coeff = np.array(coefficients[idx])
        score = scores[idx]
        class_id = class_ids[idx]
        
        if class_id == 0:
            class_name = "frame"
        elif class_id == 1:
            class_name = "text"
        else:
            class_name = "balloon"

        # Generate mask on 320x320 grid
        mask_grid = (coeff @ proto.reshape(32, -1)).reshape(320, 320)
        # Sigmoid
        mask_grid = 1.0 / (1.0 + np.exp(-mask_grid))

        # Crop to box (converted to 320x320 mask coordinate system)
        x1_m = max(0, int(box[0] / 4))
        y1_m = max(0, int(box[1] / 4))
        x2_m = min(320, int((box[0] + box[2]) / 4))
        y2_m = min(320, int((box[1] + box[3]) / 4))

        cropped_mask = np.zeros_like(mask_grid)
        cropped_mask[y1_m:y2_m, x1_m:x2_m] = mask_grid[y1_m:y2_m, x1_m:x2_m]

        # Crop out padding region
        left_m = int(dw / 4)
        top_m = int(dh / 4)
        right_m = int((dw + unpad_w) / 4)
        bottom_m = int((dh + unpad_h) / 4)

        cropped_mask = cropped_mask[top_m:bottom_m, left_m:right_m]

        # Resize to original image dimensions
        resized_mask = cv2.resize(cropped_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary
        binary_mask = (resized_mask >= 0.5).astype(np.uint8) * 255

        # Morphological Closing to clean masks and seal gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        # Get Contour
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify Contour
        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Exact mask polygon
        mask_polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in simplified_contour]  # type: ignore

        # Derive eroded safe text area
        erosion_px = YOLO_MASK_EROSION
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1))
        eroded_mask = cv2.erode(cleaned_mask, kernel_erode, iterations=1)

        # If erosion completely ate the bubble, fallback to the cleaned mask
        if cv2.countNonZero(eroded_mask) == 0:
            eroded_mask = cleaned_mask

        # Get bounding box of the eroded mask (safe area)
        x_s, y_s, w_s, h_s = cv2.boundingRect(eroded_mask)

        # Rescale detection bounding box to original coordinates
        bx = (box[0] - dw) / r
        by = (box[1] - dh) / r
        bw = box[2] / r
        bh = box[3] / r

        # Ensure boxes remain inside image boundaries
        bx_clean = max(0, min(orig_w - 1, int(bx)))
        by_clean = max(0, min(orig_h - 1, int(by)))
        bw_clean = max(1, min(orig_w - bx_clean, int(bw)))
        bh_clean = max(1, min(orig_h - by_clean, int(bh)))

        bubbles.append(
            {
                "bbox": [bx_clean, by_clean, bw_clean, bh_clean],
                "confidence": float(score),
                "class_id": class_id,
                "class_name": class_name,
                "mask_polygon": mask_polygon,
                "safe_rect": [x_s, y_s, w_s, h_s],
            }
        )

    duration = time.perf_counter() - start_time
    logger.info(f"[YOLO] Bubble detection completed. Found {len(bubbles)} bubbles in {duration:.3f}s")
    return bubbles
