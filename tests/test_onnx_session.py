"""Provider choice for the local ONNX models (services/onnx_session.py)."""

from unittest.mock import MagicMock

import pytest

import worker.services.onnx_session as onnx_session
from worker.services.onnx_session import CPU, OPENVINO, create_session, execution_providers


def _ort(available):
    ort = MagicMock()
    ort.get_available_providers.return_value = available
    return ort


@pytest.mark.parametrize(
    ("setting", "available", "expected"),
    [
        ("auto", [OPENVINO, CPU], [(OPENVINO, {"device_type": "CPU"}), CPU]),
        ("auto", [CPU], [CPU]),
        ("cpu", [OPENVINO, CPU], [CPU]),
        ("openvino", [CPU], [CPU]),
    ],
)
def test_execution_providers(monkeypatch, setting, available, expected):
    monkeypatch.setattr(onnx_session, "ONNX_EXECUTION_PROVIDER", setting)
    assert execution_providers(_ort(available)) == expected


def test_create_session_flushes_denormals_and_uses_openvino(monkeypatch):
    monkeypatch.setattr(onnx_session, "ONNX_EXECUTION_PROVIDER", "auto")
    ort = _ort([OPENVINO, CPU])
    session = create_session(ort, "/models/ctd.onnx", "CTD")
    options = ort.SessionOptions.return_value
    options.add_session_config_entry.assert_called_once_with("session.set_denormal_as_zero", "1")
    ort.InferenceSession.assert_called_once_with(
        "/models/ctd.onnx", options, providers=[(OPENVINO, {"device_type": "CPU"}), CPU]
    )
    assert session is ort.InferenceSession.return_value


def test_create_session_falls_back_to_cpu_when_openvino_fails(monkeypatch):
    monkeypatch.setattr(onnx_session, "ONNX_EXECUTION_PROVIDER", "auto")
    ort = _ort([OPENVINO, CPU])
    cpu_session = MagicMock()
    ort.InferenceSession.side_effect = [RuntimeError("compile failed"), cpu_session]
    assert create_session(ort, "/models/aot.onnx", "AOT") is cpu_session
    assert ort.InferenceSession.call_args.kwargs["providers"] == [CPU]


def test_create_session_raises_when_cpu_itself_fails(monkeypatch):
    monkeypatch.setattr(onnx_session, "ONNX_EXECUTION_PROVIDER", "cpu")
    ort = _ort([CPU])
    ort.InferenceSession.side_effect = RuntimeError("bad model")
    with pytest.raises(RuntimeError, match="bad model"):
        create_session(ort, "/models/yolo.onnx", "YOLO")
