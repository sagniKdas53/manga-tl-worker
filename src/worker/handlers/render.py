"""Render job handler: hands the queued immutable scene to the browser renderer.

Tracker R1 (2026-09-17 realignment). Until then this module was ~1,400 lines of Pillow typography
(fonts, wrapping, hyphenation, halo strokes, fitting) that drew every pipeline render, while the
Chromium service built for M4 received no jobs. The typography is gone, not kept as a fallback: a
render job without a `logicalScene` is a backend bug and fails loudly so it shows up in the job
table, instead of silently producing a second, diverging set of pixels. Masks, crops, image IO and
thumbnails stay Pillow work elsewhere in the worker (tracker I05 boundary).
"""

import logging

import requests

from worker.config import CALLBACK_URL, backend_headers, redis_client

logger = logging.getLogger(__name__)


class RenderJobError(RuntimeError):
    """A render job that cannot be drawn; the message is what the job table should show."""


def process_render(job_data):
    image_id = job_data.get("imageId")
    page_id = job_data.get("pageId")

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:render")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    logger.info(f"[Render] Processing page: {page_id or image_id}{progress_str}")

    if "logicalScene" not in job_data:
        raise RenderJobError(
            "render job carries no logicalScene: the backend must snapshot the page before queuing a "
            "render (page_scene_builder); there is no Pillow typography fallback"
        )

    from worker.page_scene_renderer import render_page_scene

    result = render_page_scene(job_data)
    logger.info(
        f"[Render] Browser render complete for page {page_id or image_id}: revision "
        f"{result['pageRevision']}, png {result['pngSha256'][:12]}…, "
        f"{len(result.get('diagnostics') or [])} layout diagnostic(s)"
    )

    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": page_id,
        "pageRevision": result["pageRevision"],
        "logicalSceneSha256": result["logicalSceneSha256"],
        "renderedPngSha256": result["pngSha256"],
        "diagnostics": result.get("diagnostics") or [],
        "layout": result.get("layout") or [],
    }
    try:
        res = requests.post(f"{CALLBACK_URL}/render", json=callback_payload, headers=backend_headers(), timeout=(5, 30))
        res.raise_for_status()
        logger.debug(f"[Render] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[Render] Failed to post callback: {e}")
        raise
