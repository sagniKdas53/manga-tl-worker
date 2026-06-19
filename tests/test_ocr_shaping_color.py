import numpy as np
import cv2
from worker.handlers.ocr import detect_background_color, detect_bubble_contour


def test_detect_background_color():
    # Create a 100x100 BGR image with light gray background (#e0e0e0)
    img = np.full((100, 100, 3), 224, dtype=np.uint8)  # 224 BGR -> #e0e0e0
    # Draw some black text strokes in the center (foreground)
    cv2.putText(img, "TEST", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Run background color detection
    color = detect_background_color(img, 20, 20, 60, 60)
    # The borders of the 60x60 region at (20,20) should be untouched by the text and have value #e0e0e0
    assert color.lower() == "#e0e0e0"


def test_detect_bubble_contour():
    # Create a 200x200 BGR image with gray background (#808080)
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Draw a white speech bubble (filled circle at 100,100 with radius 40)
    cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
    # Draw some black text in the center
    cv2.putText(img, "TXT", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # OCR bounding box of text is around (80, 95, 40, 20)
    bubble_box = detect_bubble_contour(img, 80, 95, 40, 20)

    assert bubble_box is not None
    # Bounding box of a circle centered at 100,100 with radius 40 should be close to (60, 60, 80, 80)
    assert abs(bubble_box["x"] - 60) <= 5
    assert abs(bubble_box["y"] - 60) <= 5
    assert abs(bubble_box["width"] - 80) <= 5
    assert abs(bubble_box["height"] - 80) <= 5
