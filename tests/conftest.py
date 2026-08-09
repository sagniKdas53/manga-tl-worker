import contextlib
import sys
import unittest.mock as mock

# Try to import cv2, if it fails due to the gapi text error, mock it
with contextlib.suppress(Exception):
    pass

if "cv2" in sys.modules:
    try:
        # Prevent the text attribute error
        sys.modules["cv2"].gapi = mock.MagicMock()  # type: ignore
        sys.modules["cv2"].mat_wrapper = mock.MagicMock()  # type: ignore
    except Exception:
        pass

import os

import pytest

os.environ["PROVIDERS_CONFIG"] = os.path.join(os.path.dirname(__file__), "test_providers.json")


@pytest.fixture(autouse=True)
def _isolate_ocr_merge_threshold(monkeypatch):
    """Several merge tests call merge_ocr_regions with no threshold, so they read this from the
    environment. A developer with the deployed value exported (docker-compose.yml sets 1.0, double
    the code default) would silently test a different configuration than CI does."""
    monkeypatch.delenv("OCR_MERGE_THRESHOLD", raising=False)
