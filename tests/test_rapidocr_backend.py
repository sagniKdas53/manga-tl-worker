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
    ("machine", "expected"),
    [
        ("aarch64", "rapidocr"),
        ("arm64", "rapidocr"),
        ("x86_64", "paddle"),
    ],
)
def test_auto_backend_selection(monkeypatch, machine, expected):
    monkeypatch.setattr("worker.model_manager.platform.machine", lambda: machine)
    monkeypatch.delenv("LOCAL_OCR_BACKEND", raising=False)

    assert get_local_ocr_backend() == expected


def test_backend_override(monkeypatch):
    monkeypatch.setenv("LOCAL_OCR_BACKEND", "rapidocr")
    monkeypatch.setattr("worker.model_manager.platform.machine", lambda: "x86_64")

    assert get_local_ocr_backend() == "rapidocr"


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
    assert params["Rec.lang_type"] == "korean"
