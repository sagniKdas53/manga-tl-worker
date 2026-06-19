import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if "tests" in sys.modules:
    del sys.modules["tests"]
from tests.test_ocr_shaping_color import (
    test_detect_background_color,
    test_detect_bubble_contour,
)
from tests.test_typesetting import (
    test_fit_text_rectangular,
    test_fit_text_polygon,
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

print("Running test_fit_text_rectangular...")
try:
    test_fit_text_rectangular()
    print("test_fit_text_rectangular passed!")
except AssertionError as e:
    print("test_fit_text_rectangular failed!", e)
    sys.exit(1)

print("Running test_fit_text_polygon...")
try:
    test_fit_text_polygon()
    print("test_fit_text_polygon passed!")
except AssertionError as e:
    print("test_fit_text_polygon failed!", e)
    sys.exit(1)

print("All tests passed successfully!")
