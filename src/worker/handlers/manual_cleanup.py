"""The mask editor's job: repaint one hand-marked area of a page (2026-09-28).

Payload, beside the usual job identity:
- `imageUrl`, `sourceSha256`: the immutable source, verified as the cleanup job does;
- `manualMask`: `{sha256, x, y, width, height}`, a PNG under the page's `scene-assets/` prefix whose
  alpha marks the area, sized exactly `width` x `height` at page (x, y);
- `method`: auto | aot | telea | flat | restore, and `fillColor` (#rrggbb) for flat;
- `underlay`: the visible patches the export draws, in paint order, as
  `{path, x, y, width, height, opacity}`. The repaint runs on the page they make.

Always calls back, with `status` complete or failed: the backend owns the job's outcome, as for
cleanup, so a failure shows up as a failed job instead of a silent no-op.
"""

import hashlib
import logging
import re

import cv2
import numpy as np
import requests

from worker.config import CALLBACK_URL, backend_headers, minio_client
from worker.handlers.cleanup import _asset_fields, _download_verified_source
from worker.job_attempt import record_progress
from worker.services.manual_cleanup import (
    UnderlayPatch,
    composite_underlay,
    parse_fill_colour,
    reconstruct_manual,
)
from worker.utils.lock import acquire_lock

logger = logging.getLogger(__name__)

_SHA256 = re.compile(r"[0-9a-f]{64}")

BUCKET = "manga-library"


def _read_object(path: str) -> bytes:
    stored = minio_client.get_object(BUCKET, path)
    try:
        return stored.read()
    finally:
        stored.close()
        stored.release_conn()


def _decode_png(payload: bytes, what: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{what} is not a decodable PNG")
    return image


def _number(value: object, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"manual cleanup {what} is not a number")
    return float(value)


def _page_asset_key(page_id: str, sha: object, what: str) -> str:
    """`scene-assets/{page_id}/{sha}.png`, refusing any sha that is not a lowercase hex digest.

    The job names objects by digest; anything else (`../other-page/...`) would reach outside this
    page's prefix, and the digest check after the read would not stop it.
    """
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise ValueError(f"{what} is not a sha256 hex digest")
    return f"scene-assets/{page_id}/{sha}.png"


def _load_mark(page_id: str, spec: object) -> tuple[np.ndarray, int, int]:
    """The marked area as a boolean array, and where it sits on the page."""
    if not isinstance(spec, dict) or not isinstance(spec.get("sha256"), str):
        raise ValueError("manual cleanup job is missing manualMask")
    sha = spec["sha256"]
    payload = _read_object(_page_asset_key(page_id, sha, "manual mask sha256"))
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError("manual mask does not match its sha256")
    image = _decode_png(payload, "manual mask")
    width, height = int(_number(spec.get("width"), "mask width")), int(_number(spec.get("height"), "mask height"))
    if image.shape[1] != width or image.shape[0] != height:
        raise ValueError(f"manual mask is {image.shape[1]}x{image.shape[0]}, not {width}x{height}")
    if image.ndim == 3 and image.shape[2] == 4:
        marked = image[..., 3] > 0
    elif image.ndim == 3:
        marked = image.max(axis=2) > 0
    else:
        marked = image > 0
    return marked, int(_number(spec.get("x"), "mask x")), int(_number(spec.get("y"), "mask y"))


def _load_underlay(page_id: str, entries: object) -> list[UnderlayPatch]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("manual cleanup underlay is not a list")
    patches = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("manual cleanup underlay entry has no path")
        # The backend names each visible patch by its page-scoped path; a clone copies its
        # source's patches under its own page id, so nothing legitimate lies outside it.
        prefix, path = f"scene-assets/{page_id}/", entry["path"]
        digest = path.removeprefix(prefix).removesuffix(".png") if path.startswith(prefix) else ""
        if not _SHA256.fullmatch(digest) or path != f"{prefix}{digest}.png":
            raise ValueError(f"underlay path {path!r} is not one of this page's patches")
        image = _decode_png(_read_object(entry["path"]), f"underlay patch {entry['path']}")
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"underlay patch {entry['path']} has no alpha channel")
        opacity = entry.get("opacity", 1.0)
        patches.append(
            UnderlayPatch(
                bgra=image,
                x=_number(entry.get("x"), "underlay x"),
                y=_number(entry.get("y"), "underlay y"),
                width=_number(entry.get("width"), "underlay width"),
                height=_number(entry.get("height"), "underlay height"),
                opacity=_number(opacity, "underlay opacity") if opacity is not None else 1.0,
            )
        )
    return patches


def _repaint(job_data: dict, page_id: str) -> dict:
    """Everything that can fail, returning the callback's asset fields."""
    image = _download_verified_source(job_data)
    method = str(job_data.get("method") or "auto").strip().lower()
    # A restore puts the original back, so it starts from the bare source, not the drawn page.
    if method != "restore":
        image = composite_underlay(image, _load_underlay(page_id, job_data.get("underlay")))
    mark, x, y = _load_mark(page_id, job_data.get("manualMask"))
    record_progress()
    # The same node-wide lock as OCR and cleanup: AOT on this host must not run twice at once.
    with acquire_lock("ocr", node_scoped=True):
        result = reconstruct_manual(
            image, mark, x, y, mode=method, fill_bgr=parse_fill_colour(job_data.get("fillColor"))
        )
    return _asset_fields(result, page_id)


def process_manual_cleanup(job_data: dict) -> None:
    record_progress()
    page_id = job_data.get("pageId")
    if not isinstance(page_id, str) or not page_id:
        raise ValueError("manual cleanup job is missing pageId")
    try:
        fields = _repaint(job_data, page_id)
        outcome = {"status": "complete", "diagnostics": fields.get("cleanupDiagnostics", []), **fields}
    except Exception as exc:
        logger.warning("[ManualCleanup] page %s failed: %s", page_id, exc)
        outcome = {"status": "failed", "diagnostics": [str(exc)]}

    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": job_data.get("imageId"),
        "pageId": page_id,
        "attempt": job_data.get("attempt"),
        "inputGeneration": job_data.get("inputGeneration"),
        "leaseToken": job_data.get("leaseToken"),
        **outcome,
    }
    response = requests.post(
        f"{CALLBACK_URL}/manual-cleanup", json=callback_payload, headers=backend_headers(), timeout=(5, 30)
    )
    response.raise_for_status()
