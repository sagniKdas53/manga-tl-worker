import sys
import types
from enum import Enum

import numpy as np
import pytest

from worker.model_manager import ModelManager, get_local_ocr_backend
from worker.services.ocr import parse_rapid_ocr_results


@pytest.fixture(autouse=True)
def _reset_rapidocr_availability():
    ModelManager.rapidocr_available = True
    yield
    ModelManager.rapidocr_available = True


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    [
        ("linux", "aarch64", "rapidocr"),
        ("linux", "arm64", "rapidocr"),
        ("linux", "x86_64", "paddle"),
        # Apple Silicon / Windows ARM64 report "arm64" too, but get PaddleOCR — the
        # rapidocr wheel is Linux-aarch64 only (see requirements.txt markers).
        ("darwin", "arm64", "paddle"),
        ("win32", "arm64", "paddle"),
    ],
)
def test_auto_backend_selection(monkeypatch, platform_name, machine, expected):
    monkeypatch.setattr("worker.model_manager.sys.platform", platform_name)
    monkeypatch.setattr("worker.model_manager.platform.machine", lambda: machine)
    monkeypatch.delenv("LOCAL_OCR_BACKEND", raising=False)

    assert get_local_ocr_backend() == expected


def test_backend_override_accepted_when_package_present(monkeypatch):
    monkeypatch.setenv("LOCAL_OCR_BACKEND", "rapidocr")
    monkeypatch.setattr("worker.model_manager.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("worker.model_manager.importlib.util.find_spec", lambda name: object())

    assert get_local_ocr_backend() == "rapidocr"


def test_backend_override_rejected_when_package_missing(monkeypatch):
    monkeypatch.setenv("LOCAL_OCR_BACKEND", "rapidocr")
    monkeypatch.setattr("worker.model_manager.importlib.util.find_spec", lambda name: None)

    with pytest.raises(ValueError, match="rapidocr"):
        get_local_ocr_backend()


def test_paddle_override_rejected_when_package_missing(monkeypatch):
    """Symmetric to the RapidOCR guard: LOCAL_OCR_BACKEND=paddle on an image without
    paddleocr (the stock Linux ARM64 build) must fail loudly, not at the first job."""
    monkeypatch.setenv("LOCAL_OCR_BACKEND", "paddle")
    monkeypatch.setattr(
        "worker.model_manager.importlib.util.find_spec",
        lambda name: None if name == "paddleocr" else object(),
    )

    with pytest.raises(ValueError, match="paddleocr"):
        get_local_ocr_backend()


def test_perform_redo_ocr_routes_local_fallback_through_rapidocr(monkeypatch):
    """On the RapidOCR backend, the local redo fallback must not touch PaddleOCR."""
    from worker.services import ocr as ocr_service

    monkeypatch.setattr(ocr_service, "get_local_ocr_backend", lambda: "rapidocr")

    class _Cfg:
        provider = "paddleocr"
        vlm_model = ""

        def resolve_key(self):
            return ""

    monkeypatch.setattr("worker.config.OCR_CONFIG", _Cfg())

    calls = {"rapid": 0, "paddle": 0}

    def _rapid(lang, *args, **kwargs):
        calls["rapid"] += 1
        return None  # not initialised -> perform_redo_ocr returns ("", 0.0) without decoding

    def _paddle(*args, **kwargs):
        calls["paddle"] += 1
        raise AssertionError("PaddleOCR must not be used on the RapidOCR backend")

    monkeypatch.setattr(ocr_service.model_manager, "get_rapid_ocr_reader", _rapid)
    monkeypatch.setattr(ocr_service.model_manager, "get_paddle_ocr_reader", _paddle)

    assert ocr_service.perform_redo_ocr(b"not-an-image", "ja") == ("", 0.0)
    assert calls == {"rapid": 1, "paddle": 0}


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("LOCAL_OCR_BACKEND", "tesseract")

    with pytest.raises(ValueError, match="LOCAL_OCR_BACKEND"):
        get_local_ocr_backend()


def test_parse_rapidocr_full_output():
    result = types.SimpleNamespace(
        boxes=np.array([[[10, 20], [50, 20], [50, 40], [10, 40]]]),
        txts=("日本語",),
        scores=(0.93,),
    )

    assert parse_rapid_ocr_results(result) == [([[10, 20], [50, 20], [50, 40], [10, 40]], "日本語", 0.93)]


def test_parse_rapidocr_detection_only_output():
    result = {
        "boxes": np.array([[[1, 2], [3, 2], [3, 4], [1, 4]]]),
        "scores": np.array([0.81]),
    }

    assert parse_rapid_ocr_results(result) == [([[1, 2], [3, 2], [3, 4], [1, 4]], "", 0.81)]


def test_rapidocr_reader_uses_arm_safe_japanese_model(monkeypatch, tmp_path):
    class FakeModelType(Enum):
        MOBILE = "mobile"
        MEDIUM = "medium"

    class FakeOcrVersion(Enum):
        PPOCRV5 = "PP-OCRv5"
        PPOCRV6 = "PP-OCRv6"

    class FakeRapidOCR:
        def __init__(self, *, params):
            self.params = params

    rapidocr_module = types.ModuleType("rapidocr")
    rapidocr_module.__dict__["RapidOCR"] = FakeRapidOCR
    rapidocr_utils = types.ModuleType("rapidocr.utils")
    rapidocr_typings = types.ModuleType("rapidocr.utils.typings")
    rapidocr_typings.__dict__["ModelType"] = FakeModelType
    rapidocr_typings.__dict__["OCRVersion"] = FakeOcrVersion

    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", rapidocr_utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", rapidocr_typings)
    monkeypatch.setenv("RAPIDOCR_MODEL_ROOT", str(tmp_path))

    engine = ModelManager().get_rapid_ocr_reader("ja")
    assert engine is not None
    params = engine.params
    assert params["Det.ocr_version"] is FakeOcrVersion.PPOCRV6
    assert params["Rec.ocr_version"] is FakeOcrVersion.PPOCRV6
    assert params["Det.model_type"] is FakeModelType.MEDIUM
    assert params["Rec.model_type"] is FakeModelType.MEDIUM
    assert params["Det.lang_type"] == "multi"
    assert params["Rec.lang_type"] == "japan"


def test_rapidocr_reader_routes_korean_to_ppocrv5(monkeypatch, tmp_path):
    class FakeModelType(Enum):
        MOBILE = "mobile"
        MEDIUM = "medium"

    class FakeOcrVersion(Enum):
        PPOCRV5 = "PP-OCRv5"
        PPOCRV6 = "PP-OCRv6"

    class FakeRapidOCR:
        def __init__(self, *, params):
            self.params = params

    rapidocr_module = types.ModuleType("rapidocr")
    rapidocr_module.__dict__["RapidOCR"] = FakeRapidOCR
    rapidocr_utils = types.ModuleType("rapidocr.utils")
    rapidocr_typings = types.ModuleType("rapidocr.utils.typings")
    rapidocr_typings.__dict__["ModelType"] = FakeModelType
    rapidocr_typings.__dict__["OCRVersion"] = FakeOcrVersion

    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", rapidocr_utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", rapidocr_typings)
    monkeypatch.setenv("RAPIDOCR_MODEL_ROOT", str(tmp_path))

    engine = ModelManager().get_rapid_ocr_reader("ko")
    assert engine is not None
    params = engine.params
    assert params["Det.ocr_version"] is FakeOcrVersion.PPOCRV5
    assert params["Rec.ocr_version"] is FakeOcrVersion.PPOCRV5
    assert params["Det.model_type"] is FakeModelType.MOBILE
    assert params["Rec.model_type"] is FakeModelType.MOBILE
    assert params["Det.lang_type"] == "ch"
    assert params["Rec.lang_type"] == "korean"
