"""Immutable page-scene/v1 transport to the pinned browser renderer.

The renderer accepts only embedded (`data:`) images — it never fetches a URL — so every cleanup
patch the scene references is downloaded here by its presigned URL, checked against the digest the
scene records for that asset, and embedded. A patch whose bytes do not match the scene is a wrong
patch, not a transport hiccup, and fails the job.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from typing import Any

import requests

from worker.config import CALLBACK_URL, backend_headers, minio_client
from worker.page_scene import validate_page_scene
from worker.utils.image import download_image


def _data_url(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _render_input_digest(scene_digest: str, source_sha256: str, asset_sha256s: list[str]) -> str:
    canonical = json.dumps(
        {
            "logicalSceneSha256": scene_digest,
            "sourceSha256": source_sha256,
            "assetSha256s": sorted(asset_sha256s),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _fetch_asset(url: str, record: dict[str, Any], asset_id: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.content
    if hashlib.sha256(payload).hexdigest() != record["sha256"] or len(payload) != record["byte_length"]:
        raise ValueError(f"cleanup asset {asset_id} does not match the digest recorded in the scene")
    return payload


def render_page_scene(job_data: dict[str, Any]) -> dict[str, Any]:
    """Render the queued immutable scene; never read mutable geometry or call Pillow typography."""
    scene = validate_page_scene(job_data.get("logicalScene"))
    if scene.document["scene_kind"] != "logical":
        raise ValueError("render jobs require a logical scene")
    if job_data.get("logicalSceneSha256") != scene.logical_scene_sha256:
        raise ValueError("queued logical scene digest mismatch")
    if job_data.get("pageRevision") != scene.document["page"]["revision"]:
        raise ValueError("queued page revision mismatch")

    image_id = job_data.get("imageId")
    source_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
    response = requests.get(source_url, headers=backend_headers(), timeout=30)
    response.raise_for_status()
    renderer_url = os.environ.get("PAGE_RENDERER_URL")
    source_bytes = download_image(response.json())
    source = scene.document["page"]["source"]
    if hashlib.sha256(source_bytes).hexdigest() != source["sha256"]:
        raise ValueError("immutable scene source digest mismatch")

    asset_urls = job_data.get("renderAssetUrls") or {}
    assets = {asset["asset_id"]: asset for asset in scene.document["assets"]}
    cleanup_assets = []
    for index, cleanup in enumerate(scene.document["cleanup_artifacts"]):
        patch_id = cleanup["patch_asset_id"]
        patch_url = asset_urls.get(patch_id)
        if not isinstance(patch_url, str):
            raise ValueError(f"missing immutable cleanup asset {patch_id}")
        patch_bytes = _fetch_asset(patch_url, assets[patch_id], patch_id)
        bounds = cleanup["bounds"]
        cleanup_assets.append(
            {
                "cleanupId": cleanup["cleanup_id"],
                "href": _data_url(assets[patch_id]["mime_type"], patch_bytes),
                "x": bounds["x"],
                "y": bounds["y"],
                "width": bounds["width"],
                "height": bounds["height"],
                "zIndex": index,
                "visible": True,
            }
        )

    text_objects = []
    font_ids = set()
    for item in scene.document["objects"]:
        if item["kind"] == "manual_cleanup":
            continue
        style = item["style"]
        font_ids.add(style["font_id"])
        transform = item["transform"]
        text_objects.append(
            {
                "objectId": item["object_id"],
                "text": item["text"],
                "transform": {
                    "x": transform["x"],
                    "y": transform["y"],
                    "width": transform["width"],
                    "height": transform["height"],
                    "rotationDegrees": transform["rotation_degrees"],
                },
                "writingMode": item["writing_mode"],
                "alignment": item["alignment"],
                "style": {
                    "fontFamily": style["font_id"],
                    "fill": style["fill"],
                    "stroke": style["stroke"],
                    "weight": style["weight"],
                    "padding": style["padding"],
                },
                "visible": item["visible"],
                "zIndex": item["z_index"],
            }
        )
    if not renderer_url:
        raise ValueError("PAGE_RENDERER_URL is required for page-scene render jobs")
    payload = {
        "contractVersion": "page-scene/v1",
        "pageRevision": job_data["pageRevision"],
        "logicalSceneSha256": scene.logical_scene_sha256,
        "renderInputSha256": _render_input_digest(
            scene.logical_scene_sha256, source["sha256"], [asset["sha256"] for asset in assets.values()]
        ),
        "requiredFontIds": sorted(font_ids),
        "scene": {
            "source": {
                "href": _data_url(source["mime_type"], source_bytes),
                "width": source["width"],
                "height": source["height"],
            },
            "cleanupAssets": cleanup_assets,
            "textObjects": text_objects,
        },
    }
    try:
        result = requests.post(renderer_url.rstrip("/") + "/render", json=payload, timeout=180)
    except requests.RequestException as err:
        # The worker no longer hard-depends on the renderer at startup (Compose), so an absent or
        # crashed renderer surfaces here, per job, with the reason in the job table.
        raise RuntimeError(f"page renderer at {renderer_url} is unreachable: {err}") from err
    if result.status_code >= 400:
        detail = ""
        try:
            detail = result.json().get("error") or ""
        except ValueError:
            detail = result.text[:300]
        raise RuntimeError(f"page renderer rejected the scene ({result.status_code}): {detail}")
    rendered = result.json()
    if (
        rendered.get("logicalSceneSha256") != scene.logical_scene_sha256
        or rendered.get("pageRevision") != job_data["pageRevision"]
    ):
        raise ValueError("renderer returned a mismatched immutable identity")
    png = base64.b64decode(rendered["pngBase64"], validate=True)
    if hashlib.sha256(png).hexdigest() != rendered.get("pngSha256"):
        raise ValueError("renderer PNG digest mismatch")
    minio_client.put_object(
        "manga-library", f"rendered/{image_id}.png", io.BytesIO(png), len(png), content_type="image/png"
    )
    return {
        "pageRevision": rendered["pageRevision"],
        "logicalSceneSha256": rendered["logicalSceneSha256"],
        "pngSha256": rendered["pngSha256"],
        "diagnostics": rendered.get("diagnostics") or [],
    }
