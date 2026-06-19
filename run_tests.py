import sys
from tests.test_ocr_shaping_color import (
    test_detect_background_color,
    test_detect_bubble_contour,
)

print("Running test_detect_background_color...")
try:
    test_detect_background_color()
    print("test_detect_background_color passed!")
except AssertionError as e:
    print("test_detect_background_color failed!", e)
    sys.exit(1)

print("Running test_detect_bubble_contour...")
try:
    test_detect_bubble_contour()
    print("test_detect_bubble_contour passed!")
except AssertionError as e:
    print("test_detect_bubble_contour failed!", e)
    sys.exit(1)

print("All tests passed successfully!")
