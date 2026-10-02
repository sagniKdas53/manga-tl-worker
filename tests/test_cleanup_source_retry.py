import hashlib
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from worker.handlers.cleanup import _download_verified_source


@pytest.mark.parametrize("matches", [True, False])
def test_expired_source_url_uses_same_stored_object_and_checks_immutable_digest(matches):
    ok, encoded = cv2.imencode(".png", np.full((10, 10, 3), 200, np.uint8))
    assert ok
    source = encoded.tobytes()
    stored = MagicMock()
    stored.read.return_value = source
    job = {
        "imageUrl": "http://minio/manga-library/images/source.png?expired",
        "sourceSha256": hashlib.sha256(source if matches else b"changed").hexdigest(),
    }
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=MagicMock(status_code=403)),
        patch("worker.handlers.cleanup.minio_client.get_object", return_value=stored) as get,
    ):
        if matches:
            assert _download_verified_source(job).shape == (10, 10, 3)
        else:
            with pytest.raises(ValueError, match="sourceSha256"):
                _download_verified_source(job)
    get.assert_called_once_with("manga-library", "images/source.png")
    stored.close.assert_called_once()
    stored.release_conn.assert_called_once()
