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
