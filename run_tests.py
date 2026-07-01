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
from tests.test_qa_feedback_loop import (
    test_translate_batch_llm_handles_qa_feedback,
)
from tests.test_translation_validation import (
    test_valid_translation,
    test_cjk_leak_translation,
    test_length_ratio_translation,
    test_excessive_repetition_translation,
)
from tests.test_ocr_vlm import (
    test_process_ocr_vlm_gemini,
    test_process_ocr_vlm_openrouter,
    test_process_ocr_vlm_nvidia,
    test_process_ocr_vlm_local_fallback,
)
from tests.test_translation_pipeline import (
    test_process_translation_gemini,
    test_process_translation_openrouter,
    test_process_translation_openai,
    test_process_translation_anthropic,
    test_process_translation_nvidia,
    test_process_translation_local_fallback,
    test_process_translation_retry_individual_fallback,
)
from tests.test_qa_pipeline import (
    test_process_qa_llm_gemini,
    test_process_qa_llm_nvidia,
    test_process_qa_vlm_openrouter,
    test_process_qa_vlm_nvidia,
)
from tests.test_redo_pipeline import (
    test_process_region_redo_ocr,
    test_process_region_redo_translation,
    test_process_qa_re_ocr,
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

print("Running test_translate_batch_llm_handles_qa_feedback...")
try:
    test_translate_batch_llm_handles_qa_feedback()
    print("test_translate_batch_llm_handles_qa_feedback passed!")
except Exception as e:
    print("test_translate_batch_llm_handles_qa_feedback failed!", e)
    sys.exit(1)

print("Running test_valid_translation...")
try:
    test_valid_translation()
    print("test_valid_translation passed!")
except Exception as e:
    print("test_valid_translation failed!", e)
    sys.exit(1)

print("Running test_cjk_leak_translation...")
try:
    test_cjk_leak_translation()
    print("test_cjk_leak_translation passed!")
except Exception as e:
    print("test_cjk_leak_translation failed!", e)
    sys.exit(1)

print("Running test_length_ratio_translation...")
try:
    test_length_ratio_translation()
    print("test_length_ratio_translation passed!")
except Exception as e:
    print("test_length_ratio_translation failed!", e)
    sys.exit(1)

print("Running test_excessive_repetition_translation...")
try:
    test_excessive_repetition_translation()
    print("test_excessive_repetition_translation passed!")
except Exception as e:
    print("test_excessive_repetition_translation failed!", e)
    sys.exit(1)

print("Running test_process_ocr_vlm_gemini...")
try:
    test_process_ocr_vlm_gemini()
    print("test_process_ocr_vlm_gemini passed!")
except Exception as e:
    print("test_process_ocr_vlm_gemini failed!", e)
    sys.exit(1)

print("Running test_process_ocr_vlm_openrouter...")
try:
    test_process_ocr_vlm_openrouter()
    print("test_process_ocr_vlm_openrouter passed!")
except Exception as e:
    print("test_process_ocr_vlm_openrouter failed!", e)
    sys.exit(1)

print("Running test_process_ocr_vlm_nvidia...")
try:
    test_process_ocr_vlm_nvidia()
    print("test_process_ocr_vlm_nvidia passed!")
except Exception as e:
    print("test_process_ocr_vlm_nvidia failed!", e)
    sys.exit(1)

print("Running test_process_ocr_vlm_local_fallback...")
try:
    test_process_ocr_vlm_local_fallback()
    print("test_process_ocr_vlm_local_fallback passed!")
except Exception as e:
    print("test_process_ocr_vlm_local_fallback failed!", e)
    sys.exit(1)

print("Running translation pipeline tests...")
try:
    test_process_translation_gemini()
    print("test_process_translation_gemini passed!")
    test_process_translation_openrouter()
    print("test_process_translation_openrouter passed!")
    test_process_translation_openai()
    print("test_process_translation_openai passed!")
    test_process_translation_anthropic()
    print("test_process_translation_anthropic passed!")
    test_process_translation_nvidia()
    print("test_process_translation_nvidia passed!")
    test_process_translation_local_fallback()
    print("test_process_translation_local_fallback passed!")
    test_process_translation_retry_individual_fallback()
    print("test_process_translation_retry_individual_fallback passed!")
except Exception as e:
    print("translation pipeline tests failed!", e)
    sys.exit(1)

print("Running QA pipeline tests...")
try:
    test_process_qa_llm_gemini()
    print("test_process_qa_llm_gemini passed!")
    test_process_qa_llm_nvidia()
    print("test_process_qa_llm_nvidia passed!")
    test_process_qa_vlm_openrouter()
    print("test_process_qa_vlm_openrouter passed!")
    test_process_qa_vlm_nvidia()
    print("test_process_qa_vlm_nvidia passed!")
except Exception as e:
    print("QA pipeline tests failed!", e)
    sys.exit(1)

print("Running Redo pipeline tests...")
try:
    test_process_region_redo_ocr()
    print("test_process_region_redo_ocr passed!")
    test_process_region_redo_translation()
    print("test_process_region_redo_translation passed!")
    test_process_qa_re_ocr()
    print("test_process_qa_re_ocr passed!")
except Exception as e:
    print("Redo pipeline tests failed!", e)
    sys.exit(1)

print("All tests passed successfully!")
