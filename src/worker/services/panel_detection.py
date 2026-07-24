import cv2
import numpy as np


def detect_panels(image_bytes, reading_direction="rtl"):
    """Detect panels in a manga page and sort them by *reading_direction*.

    Supported reading directions:
      'rtl' — count row-grouped, rightmost panel first (manga default)
      'ltr' — count row-grouped, leftmost panel first
      'ttb' — webtoons / manhwa: pure top-to-bottom, no row grouping
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find white gutters/spaces between panels
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Perform morphology to close small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    min_area = (w * h) * 0.02  # Must be at least 2% of the image

    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        area = width * height
        if area >= min_area and width < w * 0.98 and height < h * 0.98:
            panels.append({"x": x, "y": y, "width": width, "height": height})

    # If no panels were detected, default to a single panel containing the whole image
    if not panels:
        panels.append({"x": 0, "y": 0, "width": w, "height": h})

    # --- TTB (top-to-bottom, webtoons) ---
    if reading_direction == "ttb":
        panels.sort(key=lambda p: p["y"])
        for idx, p in enumerate(panels, start=1):
            p["gridRow"] = idx - 1
            p["gridCol"] = 0
            p["readingOrder"] = idx
        return panels

    # --- RTL / LTR: row-grouped sorting ---
    panels.sort(key=lambda p: p["y"])
    rows = []
    for p in panels:
        added = False
        for row in rows:
            # Check if this panel y overlaps with the row's typical y
            avg_y = sum(item["y"] for item in row) / len(row)
            avg_h = sum(item["height"] for item in row) / len(row)
            # Overlap threshold: 25% of height
            if abs(p["y"] - avg_y) < avg_h * 0.25:
                row.append(p)
                added = True
                break
        if not added:
            rows.append([p])

    # Sort each row by x — RTL reverses, LTR does not
    reverse_x = reading_direction != "ltr"
    final_panels = []
    reading_order = 1
    for r_idx, row in enumerate(rows):
        row.sort(key=lambda p: p["x"], reverse=reverse_x)
        for c_idx, p in enumerate(row):
            p["gridRow"] = r_idx
            p["gridCol"] = c_idx
            p["readingOrder"] = reading_order
            reading_order += 1
            final_panels.append(p)

    return final_panels
