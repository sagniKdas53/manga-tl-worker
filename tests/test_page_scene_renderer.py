import base64
import hashlib
import os
from unittest.mock import patch

from worker.page_scene_renderer import render_page_scene


def scene():
    sha = "a" * 64
    return {
        "contract_version": "page-scene/v1",
        "scene_kind": "logical",
        "page": {
            "page_id": "p",
            "revision": 3,
            "source": {"sha256": sha, "width": 1, "height": 1, "mime_type": "image/png"},
        },
        "provenance": {
            "app_commit": "b" * 64,
            "worker_commit": "c" * 64,
            "renderer_commit": "d" * 64,
            "models": [],
            "configuration_sha256": "e" * 64,
            "fonts": [],
            "runtime": {"os": "linux", "architecture": "amd64", "execution_provider": "cpu"},
            "timings_ms": {},
            "warnings": [],
        },
        "fragments": [],
        "owners": [],
        "policies": [],
        "assets": [],
        "cleanup_artifacts": [],
        "objects": [],
    }


@patch("worker.page_scene_renderer.minio_client")
@patch("worker.page_scene_renderer.download_image")
@patch("worker.page_scene_renderer.requests")
def test_browser_adapter_uses_only_queued_scene_and_checks_source(mock_requests, mock_download, mock_minio):
    source = b"source"
    document = scene()
    document["page"]["source"]["sha256"] = hashlib.sha256(source).hexdigest()
    digest = hashlib.sha256(
        __import__("json").dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mock_requests.get.return_value.json.return_value = {"presignedUrl": "source"}
    mock_requests.get.return_value.raise_for_status.return_value = None
    mock_download.return_value = source
    png = b"png"
    mock_requests.post.return_value.status_code = 200
    mock_requests.post.return_value.json.return_value = {
        "logicalSceneSha256": digest,
        "pageRevision": 3,
        "pngSha256": hashlib.sha256(png).hexdigest(),
        "pngBase64": base64.b64encode(png).decode(),
    }
    mock_requests.RequestException = Exception

    previous_url = os.environ.get("PAGE_RENDERER_URL")
    os.environ["PAGE_RENDERER_URL"] = "http://renderer"
    try:
        result = render_page_scene(
            {
                "imageId": "image",
                "jobId": "job-1",
                "attempt": 2,
                "pageRevision": 3,
                "logicalSceneSha256": digest,
                "logicalScene": document,
            }
        )
    finally:
        if previous_url is None:
            os.environ.pop("PAGE_RENDERER_URL")
        else:
            os.environ["PAGE_RENDERER_URL"] = previous_url

    payload = mock_requests.post.call_args.kwargs["json"]
    assert payload["scene"]["textObjects"] == []
    assert payload["scene"]["cleanupAssets"] == []
    assert payload["logicalSceneSha256"] == digest
    mock_minio.put_object.assert_called_once()
    # OQ-01: the PNG lands under this attempt's own content-addressed key, never a page-global one
    # another render could overwrite before the callback is read.
    png_sha = hashlib.sha256(png).hexdigest()
    expected_key = f"rendered/image/jobs/job-1/attempts/2/{png_sha}.png"
    assert mock_minio.put_object.call_args.args[1] == expected_key
    assert result["artifact"] == {
        "storagePath": expected_key,
        "sha256": png_sha,
        "byteLength": len(png),
        "contentType": "image/png",
    }


def test_render_artifact_key_requires_attempt_identity():
    import pytest

    from worker.page_scene_renderer import _render_artifact_path

    with pytest.raises(ValueError):
        _render_artifact_path({"imageId": "image", "jobId": "job"}, "a" * 64)
    with pytest.raises(ValueError):
        _render_artifact_path({"imageId": "image", "jobId": "../x", "attempt": 1}, "a" * 64)


@patch("worker.handlers.render.requests.post")
@patch("worker.page_scene_renderer.render_page_scene")
@patch("worker.handlers.render.redis_client")
def test_render_dispatcher_routes_immutable_scene_to_browser_then_callback(
    mock_redis, mock_browser_render, mock_callback
):
    from worker.handlers.render import process_render

    mock_redis.llen.return_value = 0
    mock_browser_render.return_value = {
        "pageRevision": 3,
        "logicalSceneSha256": "a" * 64,
        "pngSha256": "b" * 64,
        "artifact": {"storagePath": "k", "sha256": "b" * 64, "byteLength": 1, "contentType": "image/png"},
        "diagnostics": [{"code": "text-overflow", "objectId": "text-1"}],
        "layout": [{"object_id": "text-1", "font_size": 28.5, "lines": ["Hello", "there"]}],
    }
    process_render({"jobId": "job", "imageId": "image", "pageId": "page", "logicalScene": {}})

    mock_browser_render.assert_called_once()
    assert mock_callback.call_args.args[0].endswith("/render")
    assert mock_callback.call_args.kwargs["json"] == {
        "jobId": "job",
        "imageId": "image",
        "pageId": "page",
        "pageRevision": 3,
        "logicalSceneSha256": "a" * 64,
        "renderedPngSha256": "b" * 64,
        "artifact": {"storagePath": "k", "sha256": "b" * 64, "byteLength": 1, "contentType": "image/png"},
        "diagnostics": [{"code": "text-overflow", "objectId": "text-1"}],
        "layout": [{"object_id": "text-1", "font_size": 28.5, "lines": ["Hello", "there"]}],
    }


@patch("worker.handlers.render.requests.post")
@patch("worker.handlers.render.redis_client")
def test_render_job_without_scene_fails_instead_of_falling_back(mock_redis, mock_callback):
    import pytest

    from worker.handlers.render import RenderJobError, process_render

    mock_redis.llen.return_value = 0
    with pytest.raises(RenderJobError, match="no logicalScene"):
        process_render({"jobId": "job", "imageId": "image", "pageId": "page"})
    mock_callback.assert_not_called()


def _scene_with_cleanup(source: bytes, patch: bytes):
    document = scene()
    document["page"]["source"]["sha256"] = hashlib.sha256(source).hexdigest()
    document["assets"] = [
        {
            "asset_id": "patch-1",
            "kind": "cleanup_patch",
            "sha256": hashlib.sha256(patch).hexdigest(),
            "byte_length": len(patch),
            "mime_type": "image/png",
        },
        {"asset_id": "mask-1", "kind": "glyph_mask", "sha256": "c" * 64, "byte_length": 1, "mime_type": "image/png"},
    ]
    document["fragments"] = [
        {
            "fragment_id": "f1",
            "quad": [{"x": 0, "y": 0}] * 4,
            "text": "x",
            "confidence": 0,
            "detector_to_source": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "glyph_evidence": None,
        }
    ]
    document["owners"] = [
        {
            "owner_id": "o1",
            "fragment_ids": ["f1"],
            "container_id": None,
            "panel_id": None,
            "grouping_evidence": [],
            "vetoes": [],
        }
    ]
    document["policies"] = [
        {
            "owner_id": "o1",
            "kind": "dialogue",
            "confidence": 0,
            "reason": "t",
            "action": "replace",
            "user_override": None,
        }
    ]
    document["cleanup_artifacts"] = [
        {
            "cleanup_id": "c1",
            "owner_ids": ["o1"],
            "source_sha256": document["page"]["source"]["sha256"],
            "mask_asset_id": "mask-1",
            "patch_asset_id": "patch-1",
            "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
            "generator_sha256": "d" * 64,
            "active_set_dependency": "independent",
            "diagnostics": [],
        }
    ]
    return document


def _run_with_renderer(mock_requests, mock_download, source, document, asset_urls, patch_bytes):
    digest = hashlib.sha256(
        __import__("json").dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_response = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    source_response.json.return_value = {"presignedUrl": "source"}
    asset_response = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    asset_response.content = patch_bytes
    mock_requests.get.side_effect = [source_response, asset_response]
    mock_requests.RequestException = Exception
    mock_download.return_value = source
    png = b"png"
    mock_requests.post.return_value.status_code = 200
    mock_requests.post.return_value.json.return_value = {
        "logicalSceneSha256": digest,
        "pageRevision": 3,
        "pngSha256": hashlib.sha256(png).hexdigest(),
        "pngBase64": base64.b64encode(png).decode(),
    }
    previous_url = os.environ.get("PAGE_RENDERER_URL")
    os.environ["PAGE_RENDERER_URL"] = "http://renderer"
    try:
        return render_page_scene(
            {
                "imageId": "image",
                "jobId": "job-1",
                "attempt": 1,
                "pageRevision": 3,
                "logicalSceneSha256": digest,
                "logicalScene": document,
                "renderAssetUrls": asset_urls,
            }
        )
    finally:
        if previous_url is None:
            os.environ.pop("PAGE_RENDERER_URL")
        else:
            os.environ["PAGE_RENDERER_URL"] = previous_url


@patch("worker.page_scene_renderer.minio_client")
@patch("worker.page_scene_renderer.download_image")
@patch("worker.page_scene_renderer.requests")
def test_cleanup_patch_is_fetched_verified_and_embedded(mock_requests, mock_download, mock_minio):
    source, patch = b"source", b"patch-png-bytes"
    document = _scene_with_cleanup(source, patch)
    result = _run_with_renderer(
        mock_requests, mock_download, source, document, {"patch-1": "http://assets/patch"}, patch
    )

    payload = mock_requests.post.call_args.kwargs["json"]
    assert payload["scene"]["cleanupAssets"] == [
        {
            "cleanupId": "c1",
            "href": "data:image/png;base64," + base64.b64encode(patch).decode(),
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "zIndex": 0,
            "visible": True,
        }
    ]
    assert result["pngSha256"] == hashlib.sha256(b"png").hexdigest()
    mock_minio.put_object.assert_called_once()


@patch("worker.page_scene_renderer.minio_client")
@patch("worker.page_scene_renderer.download_image")
@patch("worker.page_scene_renderer.requests")
def test_cleanup_patch_with_wrong_bytes_fails_the_job(mock_requests, mock_download, mock_minio):
    import pytest

    source, patch = b"source", b"patch-png-bytes"
    document = _scene_with_cleanup(source, patch)
    with pytest.raises(ValueError, match="does not match the digest"):
        _run_with_renderer(mock_requests, mock_download, source, document, {"patch-1": "http://assets/patch"}, b"other")
    mock_requests.post.assert_not_called()
    mock_minio.put_object.assert_not_called()


def test_a_busy_renderer_is_waited_for_not_failed():
    from unittest.mock import MagicMock

    from worker import page_scene_renderer

    busy = MagicMock(status_code=503, headers={"Retry-After": "2"})
    done = MagicMock(status_code=200, headers={})
    slept = []
    with patch.object(page_scene_renderer.requests, "post", side_effect=[busy, busy, done]) as post:
        result = page_scene_renderer._post_render("http://renderer", {}, sleep=slept.append, clock=lambda: 0.0)
    assert result is done
    assert post.call_count == 3
    assert len(slept) == 2 and all(s >= 2 for s in slept), "the renderer's Retry-After is honoured"


def test_a_renderer_busy_past_the_deadline_returns_its_answer():
    from unittest.mock import MagicMock

    from worker import page_scene_renderer

    busy = MagicMock(status_code=503, headers={})
    now = iter([0.0, 10_000.0])
    with patch.object(page_scene_renderer.requests, "post", return_value=busy) as post:
        result = page_scene_renderer._post_render("http://renderer", {}, sleep=lambda _s: None, clock=lambda: next(now))
    assert result is busy, "the caller then reports it, as for any other refusal"
    assert post.call_count == 1


def test_a_bad_scene_is_not_retried():
    from unittest.mock import MagicMock

    from worker import page_scene_renderer

    bad = MagicMock(status_code=400, headers={})
    with patch.object(page_scene_renderer.requests, "post", return_value=bad) as post:
        assert page_scene_renderer._post_render("http://renderer", {}, sleep=lambda _s: None) is bad
    assert post.call_count == 1


def test_the_safety_share_is_part_of_the_render_input_identity():
    from worker.page_scene_renderer import _render_input_digest

    a = _render_input_digest("s" * 64, "t" * 64, ["u" * 64], 100)
    b = _render_input_digest("s" * 64, "t" * 64, ["u" * 64], 90)
    assert a != b, "a different safety share draws a different page"
