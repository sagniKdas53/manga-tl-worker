import subprocess
import sys

import pytest

import worker
from worker import seed_models


def _fake_yolo(monkeypatch, tmp_path):
    path = tmp_path / "yolo.onnx"
    path.write_bytes(b"pinned")

    class Config:
        YOLO_MODEL_PATH = str(path)
        YOLO_PINNED_CHECKSUM = seed_models._sha256(str(path))

    monkeypatch.setattr(seed_models, "_verify_yolo", lambda: None)
    return Config


def test_languages_route_to_rapidocr(monkeypatch, tmp_path):
    _fake_yolo(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_models, "_disabled", lambda value: False)

    calls = []

    class Manager:
        def get_rapid_ocr_reader(self, language):
            calls.append(language)
            return object()

    monkeypatch.setattr("worker.model_manager.get_local_ocr_backend", lambda: "rapidocr")
    monkeypatch.setattr("worker.model_manager.model_manager", Manager())

    assert seed_models.seed_models(["ja", "ko", "zh"]) == [("ja", "rapidocr"), ("ko", "rapidocr"), ("zh", "rapidocr")]
    assert calls == ["ja", "ko", "zh"]


def test_none_reader_fails(monkeypatch, tmp_path):
    _fake_yolo(monkeypatch, tmp_path)
    monkeypatch.setattr("worker.model_manager.get_local_ocr_backend", lambda: "paddle")

    class Manager:
        def get_paddle_ocr_reader(self, language):
            return None

    monkeypatch.setattr("worker.model_manager.model_manager", Manager())
    with pytest.raises(seed_models.SeedModelsError, match="returned None"):
        seed_models.seed_models(["ja"])


def test_skip_local_ocr_does_not_import_or_initialize_reader(monkeypatch, tmp_path):
    _fake_yolo(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_models, "_disabled", lambda value: False)
    monkeypatch.setattr("worker.model_manager.get_local_ocr_backend", lambda: pytest.fail("not called"))
    assert seed_models.seed_models(["ja"], skip_local_ocr=True) == []


def test_yolo_missing_and_mismatched_are_failures(monkeypatch, tmp_path):
    missing = tmp_path / "missing.onnx"

    def verify_missing():
        class Config:
            YOLO_MODEL_PATH = str(missing)
            YOLO_PINNED_CHECKSUM = "0" * 64

        monkeypatch.setattr(worker, "config", Config)
        return seed_models._verify_yolo()

    with pytest.raises(seed_models.SeedModelsError, match="missing"):
        verify_missing()

    path = tmp_path / "wrong.onnx"
    path.write_bytes(b"wrong")

    class Config:
        YOLO_MODEL_PATH = str(path)
        YOLO_PINNED_CHECKSUM = "0" * 64

    monkeypatch.setattr(worker, "config", Config)
    with pytest.raises(seed_models.SeedModelsError, match="checksum mismatch"):
        seed_models._verify_yolo()


def test_help_is_available_without_ml_imports():
    result = subprocess.run(
        [sys.executable, seed_models.__file__, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--skip-local-ocr" in result.stdout
