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
import random
import re
import time
from typing import Any

import requests

from worker.config import CALLBACK_URL, backend_headers, logger, minio_client
from worker.page_scene import validate_page_scene
from worker.utils.image import download_image


def _data_url(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _render_input_digest(
    scene_digest: str, source_sha256: str, asset_sha256s: list[str], safety_percent: float = 100
) -> str:
    canonical = json.dumps(
        {
            "logicalSceneSha256": scene_digest,
            "sourceSha256": source_sha256,
            "assetSha256s": sorted(asset_sha256s),
            # Part of what the renderer draws, and not in the logical scene, so part of the input.
            "textBoxSafetyPercent": safety_percent,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _render_artifact_path(job_data: dict[str, Any], png_sha256: str) -> str:
    """Return an attempt-scoped, immutable object key for a completed render."""
    image_id = job_data.get("imageId")
    job_id = job_data.get("jobId")
    attempt = job_data.get("attempt")
    if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]+", value) for value in (image_id, job_id)):
        raise ValueError("render artifacts require safe imageId and jobId values")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("render artifacts require a positive integer attempt")
    if not isinstance(png_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", png_sha256):
        raise ValueError("renderer returned an invalid PNG digest")
    return f"rendered/{image_id}/jobs/{job_id}/attempts/{attempt}/{png_sha256}.png"


def _fetch_asset(url: str, record: dict[str, Any], asset_id: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.content
    if hashlib.sha256(payload).hexdigest() != record["sha256"] or len(payload) != record["byte_length"]:
        raise ValueError(f"cleanup asset {asset_id} does not match the digest recorded in the scene")
    return payload


# The renderer holds one browser context (UR02); a render or QA pass arriving while another page
# renders is answered 503 "busy". That is a wait, not a failure: resend within the job instead of
# spending one of its attempts. Three immediate attempts used to burn out in seconds and leave a
# FAILED render in the queue until the sweeper's five-minute cooldown rendered the page anyway.
RENDER_BUSY_WAIT_SECONDS = float(os.environ.get("RENDER_BUSY_WAIT_SECONDS", "300"))


def _post_render(renderer_url, payload, *, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + RENDER_BUSY_WAIT_SECONDS
    delay = 1.0
    waited = 0
    while True:
        try:
            result = requests.post(renderer_url.rstrip("/") + "/render", json=payload, timeout=180)
        except requests.RequestException as err:
            # The worker no longer hard-depends on the renderer at startup (Compose), so an absent or
            # crashed renderer surfaces here, per job, with the reason in the job table.
            raise RuntimeError(f"page renderer at {renderer_url} is unreachable: {err}") from err
        if result.status_code != 503 or clock() + delay > deadline:
            if waited:
                logger.info(f"[Render] Renderer was busy; waited {waited} time(s) before this answer")
            return result
        try:
            hinted = float(result.headers.get("Retry-After", ""))
        except ValueError:
            hinted = 0.0
        sleep(max(delay, hinted) + random.uniform(0, 0.5))
        waited += 1
        delay = min(delay * 2, 10.0)


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

    safety_percent = job_data.get("textBoxSafetyPercent")
    if not isinstance(safety_percent, (int, float)) or isinstance(safety_percent, bool):
        safety_percent = 100
    safety_percent = min(100, max(1, safety_percent))
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
                    # System Settings' safety share. The frozen scene contract has no field for
                    # it, so it rides on the render job (the backend sends it on every job).
                    "safetyPercent": safety_percent,
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
            scene.logical_scene_sha256,
            source["sha256"],
            [asset["sha256"] for asset in assets.values()],
            safety_percent,
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
    result = _post_render(renderer_url, payload)
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
    artifact = {
        "storagePath": _render_artifact_path(job_data, rendered["pngSha256"]),
        "sha256": rendered["pngSha256"],
        "byteLength": len(png),
        "contentType": "image/png",
    }
    minio_client.put_object(
        "manga-library", artifact["storagePath"], io.BytesIO(png), len(png), content_type=artifact["contentType"]
    )
    return {
        "pageRevision": rendered["pageRevision"],
        "logicalSceneSha256": rendered["logicalSceneSha256"],
        "pngSha256": rendered["pngSha256"],
        "artifact": artifact,
        "diagnostics": rendered.get("diagnostics") or [],
        # Resolved font px + line breaks per text object (tracker R2 (c)); the backend writes
        # the size back onto the element and keeps the whole layout in the render ledger.
        "layout": rendered.get("layout") or [],
    }
