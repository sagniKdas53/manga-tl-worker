import hashlib
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from worker.handlers.cleanup import process_cleanup
from worker.services.cleanup_reconstruct import CleanupResult


def _source_bytes():
    ok, encoded = cv2.imencode(".png", np.full((20, 20, 3), 200, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _job(source, regions):
    return {
        "jobId": "job-1",
        "imageId": "image-1",
        "pageId": "page-1",
        "attempt": 2,
        "inputGeneration": 4,
        "leaseToken": "lease-1",
        "imageUrl": "https://source.test/page.png",
        "sourceSha256": hashlib.sha256(source).hexdigest(),
        "cleanupInputDigest": "page-digest",
        "cleanupRegions": regions,
    }


def test_cleanup_is_sequential_and_reports_complete_excluded_and_failed():
    source = _source_bytes()
    response = MagicMock(content=source)
    result = CleanupResult(mask_png=b"mask", patch_png=b"patch", bounds={"x": 1, "y": 2}, diagnostics=["telea"])
    regions = [
        {
            "regionId": "a",
            "inputDigest": "a-digest",
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "policyAction": "replace",
        },
        {
            "regionId": "b",
            "inputDigest": "b-digest",
            "x": 2,
            "y": 2,
            "width": 3,
            "height": 4,
            "policyAction": "exclude",
        },
        {
            "regionId": "c",
            "inputDigest": "c-digest",
            "x": 3,
            "y": 2,
            "width": 3,
            "height": 4,
            "policyAction": "replace",
        },
    ]
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=response),
        patch("worker.handlers.cleanup.requests.post") as post,
        patch("worker.handlers.cleanup.reconstruct_region", side_effect=[result, None]) as reconstruct,
        patch("worker.handlers.cleanup.minio_client") as minio,
    ):
        process_cleanup(_job(source, regions))

    assert reconstruct.call_count == 2
    assert minio.put_object.call_count == 2
    payload = post.call_args.kwargs["json"]
    assert [item["regionId"] for item in payload["regions"]] == ["a", "b", "c"]
    assert [item["status"] for item in payload["regions"]] == ["complete", "excluded", "failed"]
    assert payload["attempt"] == 2
    assert payload["inputGeneration"] == 4
    assert payload["leaseToken"] == "lease-1"
    assert payload["cleanupInputDigest"] == "page-digest"


def test_cleanup_source_digest_mismatch_reports_each_region_failed():
    response = MagicMock(content=_source_bytes())
    regions = [
        {"regionId": "a", "inputDigest": "a-digest", "x": 1, "y": 2, "width": 3, "height": 4, "policyAction": "replace"}
    ]
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=response),
        patch("worker.handlers.cleanup.requests.post") as post,
        patch("worker.handlers.cleanup.reconstruct_region") as reconstruct,
    ):
        process_cleanup(_job(b"different-source", regions))

    reconstruct.assert_not_called()
    outcome = post.call_args.kwargs["json"]["regions"][0]
    assert outcome["regionId"] == "a"
    assert outcome["status"] == "failed"
    assert "sourceSha256" in outcome["diagnostics"][0]


def test_cleanup_assets_go_to_the_content_addressed_scene_asset_path():
    """The same seam `page_scene_builder.rs` already reads: `scene-assets/{pageId}/{sha}.png`,
    with the digest of the bytes as the name. The callback carries refs, never pixels."""
    source = _source_bytes()
    response = MagicMock(content=source)
    result = CleanupResult(mask_png=b"mask", patch_png=b"patch", bounds={"x": 1, "y": 2}, diagnostics=["telea"])
    regions = [
        {"regionId": "a", "inputDigest": "a-digest", "x": 1, "y": 2, "width": 3, "height": 4, "policyAction": "replace"}
    ]
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=response),
        patch("worker.handlers.cleanup.requests.post") as post,
        patch("worker.handlers.cleanup.reconstruct_region", return_value=result),
        patch("worker.handlers.cleanup.minio_client") as minio,
    ):
        process_cleanup(_job(source, regions))

    paths = [call.args[1] for call in minio.put_object.call_args_list]
    assert all(path.startswith("scene-assets/page-1/") for path in paths)
    assert {path.removeprefix("scene-assets/page-1/").removesuffix(".png") for path in paths} == {
        hashlib.sha256(b"mask").hexdigest(),
        hashlib.sha256(b"patch").hexdigest(),
    }
    outcome = post.call_args.kwargs["json"]["regions"][0]
    assert outcome["cleanupMaskAssetId"] == f"mask-{hashlib.sha256(b'mask').hexdigest()}"
    assert outcome["cleanupPatchAssetId"] == f"patch-{hashlib.sha256(b'patch').hexdigest()}"
    assert outcome["cleanupPatchByteLength"] == len(b"patch")


def test_an_asset_upload_failure_is_a_failed_region_not_a_quiet_degrade():
    """Deliberate change from the inline-in-OCR behaviour, where a MinIO outage silently fell
    back to R2's flat plate and the page carried on to translation. Cleanup is a stage now: a
    region whose patch never reached storage did not get cleaned, so it says so, and the backend
    withholds translation rather than typesetting over Japanese that is still there."""
    source = _source_bytes()
    response = MagicMock(content=source)
    result = CleanupResult(mask_png=b"mask", patch_png=b"patch", bounds={"x": 1, "y": 2})
    regions = [
        {"regionId": "a", "inputDigest": "a-digest", "x": 1, "y": 2, "width": 3, "height": 4, "policyAction": "replace"}
    ]
    minio = MagicMock()
    minio.put_object.side_effect = RuntimeError("minio unavailable")
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=response),
        patch("worker.handlers.cleanup.requests.post") as post,
        patch("worker.handlers.cleanup.reconstruct_region", return_value=result),
        patch("worker.handlers.cleanup.minio_client", minio),
    ):
        process_cleanup(_job(source, regions))

    outcome = post.call_args.kwargs["json"]["regions"][0]
    assert outcome["status"] == "failed"
    assert "minio unavailable" in outcome["diagnostics"][0]
    assert "cleanupPatchAssetId" not in outcome


def test_a_region_reconstruct_declines_and_the_page_still_reports_it():
    """`reconstruct_region` returning None is its own rejection path -- degenerate crop, no
    glyphs found, model failure. Every dispatched region gets an outcome either way, because the
    backend accounts the response against the list it dispatched."""
    source = _source_bytes()
    response = MagicMock(content=source)
    regions = [
        {"regionId": "a", "inputDigest": "a-digest", "x": 1, "y": 2, "width": 3, "height": 4, "policyAction": "replace"}
    ]
    with (
        patch("worker.handlers.cleanup.requests.get", return_value=response),
        patch("worker.handlers.cleanup.requests.post") as post,
        patch("worker.handlers.cleanup.reconstruct_region", return_value=None),
        patch("worker.handlers.cleanup.minio_client") as minio,
    ):
        process_cleanup(_job(source, regions))

    minio.put_object.assert_not_called()
    outcome = post.call_args.kwargs["json"]["regions"][0]
    assert (outcome["regionId"], outcome["status"]) == ("a", "failed")


def test_ocr_has_no_inline_cleanup_execution():
    from pathlib import Path

    source = Path(__file__).parents[1] / "src" / "worker" / "handlers" / "ocr.py"
    assert "reconstruct_region(" not in source.read_text()
