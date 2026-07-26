import sys
from unittest.mock import MagicMock, patch

mock_ultralytics = MagicMock()
sys.modules["ultralytics"] = mock_ultralytics
from worker.utils.export_yolo import get_sha256, main  # noqa: E402


def test_get_sha256(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    assert get_sha256(str(f)) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_main(tmp_path):
    mock_model = MagicMock()
    mock_model.export.return_value = str(tmp_path / "temp.onnx")
    (tmp_path / "temp.onnx").write_bytes(b"onnx content")

    with (
        patch("worker.utils.export_yolo.hf_hub_download", return_value=str(tmp_path / "best.pt")),
        patch("worker.utils.export_yolo.YOLO", return_value=mock_model),
        patch("worker.utils.export_yolo.shutil.move") as mock_move,
        patch("worker.utils.export_yolo.get_sha256", return_value="dummy_sha"),
    ):
        main()
        mock_model.export.assert_called_once_with(format="onnx", imgsz=1280, simplify=True)
        mock_move.assert_called_once()
