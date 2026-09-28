"""The mask editor's repaint: tidy, underlay, methods, and the job's callback contract."""

import hashlib
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from worker.handlers.manual_cleanup import process_manual_cleanup
from worker.rq_tasks import process_job_rq
from worker.services.manual_cleanup import (
    UnderlayPatch,
    composite_underlay,
    parse_fill_colour,
    reconstruct_manual,
    tidy_mask,
)


def _png(array: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


def _decode(payload: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_tidy_mask_closes_gaps_and_fills_enclosed_holes():
    ring = np.zeros((60, 60), dtype=bool)
    cv2.circle(ring.view(np.uint8), (30, 30), 20, (1,), thickness=4)
    ring[28:32, 8:12] = False  # a brush gap in the ring
    tidied = tidy_mask(ring)
    assert tidied[30, 30], "the enclosed hole is filled"
    assert tidied[30, 10], "the gap is closed"
    assert not tidied[2, 2], "outside the mark stays unmarked"


def test_tidy_mask_keeps_a_mark_touching_the_frame_edge():
    mark = np.zeros((20, 20), dtype=bool)
    mark[:, :5] = True
    assert tidy_mask(mark)[:, :5].all()
    assert not tidy_mask(mark)[:, 12:].any()


def test_composite_underlay_stretches_and_blends_each_patch_in_order():
    page = np.full((20, 20, 3), 100, dtype=np.uint8)
    red = np.zeros((2, 2, 4), dtype=np.uint8)
    red[...] = (0, 0, 255, 255)
    faded_blue = np.zeros((1, 1, 4), dtype=np.uint8)
    faded_blue[...] = (255, 0, 0, 255)
    out = composite_underlay(
        page,
        [
            UnderlayPatch(red, x=4, y=4, width=6, height=6),
            UnderlayPatch(faded_blue, x=8, y=8, width=4, height=4, opacity=0.5),
        ],
    )
    assert tuple(out[5, 5]) == (0, 0, 255), "stretched to its 6x6 box"
    assert tuple(out[15, 15]) == (100, 100, 100), "untouched outside every box"
    assert tuple(out[11, 11]) == (178, 50, 50), "half-opacity blue over the page's grey"


def test_flat_repaint_is_cut_to_the_tidied_mark_and_transparent_outside_it():
    page = np.full((100, 100, 3), 30, dtype=np.uint8)
    mark = np.zeros((10, 20), dtype=bool)
    mark[2:8, 3:17] = True
    result = reconstruct_manual(page, mark, 40, 50, mode="flat", fill_bgr=parse_fill_colour("#ff8000"))
    assert result.bounds == {"x": 43, "y": 52, "width": 14, "height": 6}
    patch_image = _decode(result.patch_png)
    assert patch_image.shape == (6, 14, 4)
    assert tuple(patch_image[3, 7]) == (0, 128, 255, 255)
    assert result.diagnostics[0] == "manual repaint: flat (mode=flat)"


def test_auto_repaints_a_flat_area_with_telea_and_the_speck_is_gone():
    page = np.full((80, 80, 3), 220, dtype=np.uint8)
    page[38:42, 38:42] = 0  # a leftover speck of ink
    mark = np.ones((12, 12), dtype=bool)
    result = reconstruct_manual(page, mark, 34, 34, mode="auto")
    assert result.diagnostics[0] == "manual repaint: telea (mode=auto)"
    patch_image = _decode(result.patch_png)
    assert patch_image[..., :3][patch_image[..., 3] > 0].min() > 150


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "lama"}, "unknown manual cleanup method"),
        ({"mode": "flat"}, "needs a #rrggbb colour"),
    ],
)
def test_reconstruct_manual_refuses_what_it_cannot_do(kwargs, message):
    page = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        reconstruct_manual(page, np.ones((2, 2), dtype=bool), 1, 1, **kwargs)


def test_an_empty_mark_is_refused():
    with pytest.raises(ValueError, match="empty"):
        reconstruct_manual(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((2, 2), dtype=bool), 1, 1)


def _job(source: bytes, mask_sha: str, **extra) -> dict:
    return {
        "jobId": "job-m1",
        "imageId": "image-1",
        "pageId": "page-1",
        "attempt": 1,
        "inputGeneration": 3,
        "leaseToken": "lease-1",
        "imageUrl": "https://source.test/page.png",
        "sourceSha256": hashlib.sha256(source).hexdigest(),
        "manualMask": {"sha256": mask_sha, "x": 5, "y": 6, "width": 4, "height": 3},
        "method": "flat",
        "fillColor": "#00ff00",
        **extra,
    }


def _objects(store: dict):
    def get_object(bucket, path):
        assert bucket == "manga-library"
        stored = MagicMock()
        stored.read.return_value = store[path]
        return stored

    return get_object


def test_the_job_repaints_on_the_underlay_and_calls_back_complete():
    source = _png(np.full((20, 20, 3), 200, dtype=np.uint8))
    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[..., 3] = 255
    mask_png = _png(mask)
    mask_sha = hashlib.sha256(mask_png).hexdigest()
    underlay = np.zeros((1, 1, 4), dtype=np.uint8)
    underlay[...] = (0, 0, 0, 255)
    store = {f"scene-assets/page-1/{mask_sha}.png": mask_png, "scene-assets/page-1/under.png": _png(underlay)}
    job = _job(
        source,
        mask_sha,
        underlay=[{"path": "scene-assets/page-1/under.png", "x": 0, "y": 0, "width": 20, "height": 20}],
    )
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=MagicMock(content=source, status_code=200)),
        patch("worker.handlers.manual_cleanup.requests.post") as post,
        patch("worker.handlers.manual_cleanup.minio_client") as reads,
        patch("worker.handlers.cleanup.minio_client") as writes,
        patch("worker.handlers.manual_cleanup.composite_underlay", wraps=composite_underlay) as composite,
        patch("worker.handlers.manual_cleanup.acquire_lock"),
    ):
        reads.get_object.side_effect = _objects(store)
        process_manual_cleanup(job)

    assert composite.call_args.args[1][0].width == 20, "the underlay was composited before repainting"
    assert writes.put_object.call_count == 2, "patch and mask uploaded under the page's prefix"
    assert all(call.args[1].startswith("scene-assets/page-1/") for call in writes.put_object.call_args_list)
    assert post.call_args.args[0].endswith("/manual-cleanup")
    payload = post.call_args.kwargs["json"]
    assert payload["status"] == "complete"
    assert payload["cleanupBounds"] == {"x": 5, "y": 6, "width": 4, "height": 3}
    assert (payload["jobId"], payload["attempt"], payload["inputGeneration"], payload["leaseToken"]) == (
        "job-m1",
        1,
        3,
        "lease-1",
    )


def test_a_mask_that_does_not_match_its_digest_calls_back_failed():
    source = _png(np.full((20, 20, 3), 200, dtype=np.uint8))
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=MagicMock(content=source, status_code=200)),
        patch("worker.handlers.manual_cleanup.requests.post") as post,
        patch("worker.handlers.manual_cleanup.minio_client") as reads,
    ):
        reads.get_object.side_effect = _objects({f"scene-assets/page-1/{'a' * 64}.png": b"not this"})
        process_manual_cleanup(_job(source, "a" * 64))

    payload = post.call_args.kwargs["json"]
    assert payload["status"] == "failed"
    assert "does not match its sha256" in payload["diagnostics"][0]


def test_an_accepted_manual_cleanup_callback_leaves_the_terminal_state_to_the_backend():
    pending = MagicMock(status_code=200)
    pending.json.return_value = {"status": "PENDING"}
    statuses = []
    job = {"jobId": "job-m1", "imageId": "image-1", "attempt": 1, "maxAttempts": 3}
    with (
        patch("worker.rq_tasks.check_stale_job", return_value=False),
        patch("worker.rq_tasks.requests.get", return_value=pending),
        patch(
            "worker.rq_tasks.update_job_status",
            side_effect=lambda _id, status, *a, **k: statuses.append(status) or True,
        ),
        patch("worker.rq_tasks.JobHeartbeat"),
        patch("worker.handlers.manual_cleanup.process_manual_cleanup") as handler,
    ):
        process_job_rq("queue:manual-cleanup", job)

    handler.assert_called_once_with(job)
    assert statuses == ["PROCESSING"]
