import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if "tests" in sys.modules:
    del sys.modules["tests"]
from tests.test_ocr_shaping_color import (
    test_detect_background_color,
    test_detect_bubble_contour,
    test_detect_background_color_poly,
    test_split_polygon_and_safe_area,
)
from tests.test_typesetting import (
    test_fit_text_rectangular,
    test_fit_text_polygon,
)
from tests.test_render_and_qa import (
    test_process_render_success,
    test_process_qa_llm_success,
    test_process_qa_vlm_cloud_success,
    test_process_qa_vlm_local_fallback,
    test_process_qa_vlm_empty_ocr_regions,
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

print("Running test_detect_background_color_poly...")
try:
    test_detect_background_color_poly()
    print("test_detect_background_color_poly passed!")
except AssertionError as e:
    print("test_detect_background_color_poly failed!", e)
    sys.exit(1)

print("Running test_split_polygon_and_safe_area...")
try:
    test_split_polygon_and_safe_area()
    print("test_split_polygon_and_safe_area passed!")
except AssertionError as e:
    print("test_split_polygon_and_safe_area failed!", e)
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

print("Running test_process_render_success...")
try:
    test_process_render_success()
    print("test_process_render_success passed!")
except Exception as e:
    print("test_process_render_success failed!", e)
    sys.exit(1)

print("Running test_process_qa_llm_success...")
try:
    test_process_qa_llm_success()
    print("test_process_qa_llm_success passed!")
except Exception as e:
    print("test_process_qa_llm_success failed!", e)
    sys.exit(1)

print("Running test_process_qa_vlm_cloud_success...")
try:
    test_process_qa_vlm_cloud_success()
    print("test_process_qa_vlm_cloud_success passed!")
except Exception as e:
    print("test_process_qa_vlm_cloud_success failed!", e)
    sys.exit(1)

print("Running test_process_qa_vlm_local_fallback...")
try:
    test_process_qa_vlm_local_fallback()
    print("test_process_qa_vlm_local_fallback passed!")
except Exception as e:
    print("test_process_qa_vlm_local_fallback failed!", e)
    sys.exit(1)

print("Running test_process_qa_vlm_empty_ocr_regions...")
try:
    test_process_qa_vlm_empty_ocr_regions()
    print("test_process_qa_vlm_empty_ocr_regions passed!")
except Exception as e:
    print("test_process_qa_vlm_empty_ocr_regions failed!", e)
    sys.exit(1)

print("All tests passed successfully!")
