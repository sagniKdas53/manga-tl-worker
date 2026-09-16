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
    mock_requests.post.return_value.json.return_value = {
        "logicalSceneSha256": digest,
        "pageRevision": 3,
        "pngSha256": hashlib.sha256(png).hexdigest(),
        "pngBase64": base64.b64encode(png).decode(),
    }
    mock_requests.post.return_value.raise_for_status.return_value = None

    previous_url = os.environ.get("PAGE_RENDERER_URL")
    os.environ["PAGE_RENDERER_URL"] = "http://renderer"
    try:
        render_page_scene(
            {"imageId": "image", "pageRevision": 3, "logicalSceneSha256": digest, "logicalScene": document}
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


@patch("worker.handlers.render.requests.post")
@patch("worker.page_scene_renderer.render_page_scene")
@patch("worker.handlers.render.redis_client")
def test_render_dispatcher_routes_immutable_scene_to_browser_then_callback(
    mock_redis, mock_browser_render, mock_callback
):
    from worker.handlers.render import process_render

    mock_redis.llen.return_value = 0
    process_render({"jobId": "job", "imageId": "image", "pageId": "page", "logicalScene": {}})

    mock_browser_render.assert_called_once()
    assert mock_callback.call_args.args[0].endswith("/render")
    assert mock_callback.call_args.kwargs["json"] == {"jobId": "job", "imageId": "image", "pageId": "page"}
