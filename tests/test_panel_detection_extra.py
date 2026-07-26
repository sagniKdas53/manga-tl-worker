import cv2
import numpy as np

from worker.services.panel_detection import detect_panels


def test_detect_panels_invalid_image():
    assert detect_panels(b"invalid_image_bytes") == []


def test_detect_panels_ttb():
    # Create black image with 2 white boxes (panels)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (900, 400), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 500), (900, 900), (255, 255, 255), -1)

    _, encoded = cv2.imencode(".jpg", img)
    panels = detect_panels(encoded.tobytes(), reading_direction="ttb")
    assert len(panels) >= 1
    assert panels[0]["readingOrder"] == 1


def test_detect_panels_rtl_and_ltr():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (450, 450), (255, 255, 255), -1)
    cv2.rectangle(img, (550, 50), (950, 450), (255, 255, 255), -1)

    _, encoded = cv2.imencode(".jpg", img)
    panels_rtl = detect_panels(encoded.tobytes(), reading_direction="rtl")
    panels_ltr = detect_panels(encoded.tobytes(), reading_direction="ltr")

    assert len(panels_rtl) >= 1
    assert len(panels_ltr) >= 1
