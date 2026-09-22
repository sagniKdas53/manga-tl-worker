"""Sequential R3 cleanup job handler.

OCR records cleanup eligibility only.  This handler owns the single CTD pass, asset upload, and
explicit per-region outcome for the immutable cleanup payload.
"""

import hashlib
import io
import logging

import cv2
import numpy as np
import requests

from worker.config import CALLBACK_URL, backend_headers, minio_client
from worker.job_attempt import record_progress
from worker.services.cleanup_reconstruct import CleanupResult, reconstruct_region
from worker.utils.lock import acquire_lock

logger = logging.getLogger(__name__)


def _asset_fields(result: CleanupResult, page_id: str) -> dict:
    """Use the established content-addressed scene-asset upload seam."""
    mask_sha = hashlib.sha256(result.mask_png).hexdigest()
    patch_sha = hashlib.sha256(result.patch_png).hexdigest()
    for digest, payload in ((mask_sha, result.mask_png), (patch_sha, result.patch_png)):
        minio_client.put_object(
            "manga-library",
            f"scene-assets/{page_id}/{digest}.png",
            io.BytesIO(payload),
            len(payload),
            content_type="image/png",
        )
    return {
        "cleanupMaskAssetId": f"mask-{mask_sha}",
        "cleanupMaskSha256": mask_sha,
        "cleanupMaskByteLength": len(result.mask_png),
        "cleanupPatchAssetId": f"patch-{patch_sha}",
        "cleanupPatchSha256": patch_sha,
        "cleanupPatchByteLength": len(result.patch_png),
        "cleanupBounds": result.bounds,
        "cleanupGeneratorSha256": result.generator_sha256,
        "cleanupDiagnostics": result.diagnostics,
    }


def _download_verified_source(job_data: dict) -> np.ndarray:
    image_url = job_data.get("imageUrl")
    source_sha256 = job_data.get("sourceSha256")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("cleanup job is missing immutable imageUrl")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("cleanup job is missing immutable sourceSha256")
    response = requests.get(image_url, timeout=(5, 60))
    response.raise_for_status()
    actual = hashlib.sha256(response.content).hexdigest()
    if actual.lower() != source_sha256.lower():
        raise ValueError("downloaded source does not match sourceSha256")
    image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("downloaded source is not a decodable image")
    return image


def _region_result(region: object, status: str, diagnostics: list[str], **fields) -> dict:
    """One outcome, always echoing the identity it was dispatched with.

    Every dispatched region gets exactly one of these, whatever happened to it. The backend
    accounts the response against its own list and treats a missing region as a failed cleanup,
    so silently dropping one here would only turn a visible failure into a withheld translation.
    """
    if not isinstance(region, dict):
        return {"regionId": None, "inputDigest": None, "status": "failed", "diagnostics": diagnostics}
    return {
        "regionId": region.get("regionId"),
        "inputDigest": region.get("inputDigest"),
        "status": status,
        "diagnostics": diagnostics,
        **fields,
    }


def _geometry(region: dict) -> tuple[float, float, float, float] | None:
    """The region's box as four real numbers, or None if the payload did not carry one."""
    values = []
    for key in ("x", "y", "width", "height"):
        value = region.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return values[0], values[1], values[2], values[3]


def _process_region(image: np.ndarray, page_id: str, region: object) -> dict:
    if not isinstance(region, dict):
        return _region_result(region, "failed", ["invalid cleanup region payload"])
    if region.get("policyAction") == "exclude":
        # SFX and anything else policy keeps are never typeset, so their source lettering must
        # stay on the page. An exclusion is a complete outcome, not a skipped one.
        return _region_result(region, "excluded", ["cleanup excluded by immutable policy"])
    if region.get("policyAction") != "replace":
        return _region_result(region, "failed", ["unknown cleanup policyAction"])
    if not isinstance(region.get("regionId"), str) or not isinstance(region.get("inputDigest"), str):
        return _region_result(region, "failed", ["cleanup region lacks immutable identity"])
    geometry = _geometry(region)
    if geometry is None:
        return _region_result(region, "failed", ["cleanup region lacks usable geometry"])
    try:
        result = reconstruct_region(image, *geometry)
        if result is None:
            # reconstruct_region returns None for every rejection it makes — degenerate crop,
            # CTD found no glyphs, model failure. R2's flat-plate/halo behaviour for this region
            # still renders, but it is not cleanup, so it is not reported as cleanup.
            return _region_result(region, "failed", ["CTD/reconstruction produced no cleanup artifact"])
        fields = _asset_fields(result, page_id)
        status = "degraded" if any("fallback" in diagnostic for diagnostic in result.diagnostics) else "complete"
        return _region_result(region, status, result.diagnostics, **fields)
    except Exception as exc:
        logger.warning("[Cleanup] region %s failed: %s", region.get("regionId"), exc)
        return _region_result(region, "failed", [str(exc)])


def process_cleanup(job_data: dict) -> None:
    """Process regions in immutable payload order, without cache reuse or parallel CTD calls."""
    # Cheapest possible place to notice this attempt is void: the source download is a whole page
    # and the region loop is minutes of CTD. `record_progress` and `backend_headers` both check
    # again later; this one just avoids paying for a page nobody will accept.
    record_progress()
    page_id = job_data.get("pageId")
    regions = job_data.get("cleanupRegions")
    if not isinstance(page_id, str) or not page_id:
        raise ValueError("cleanup job is missing pageId")
    if not isinstance(regions, list):
        raise ValueError("cleanup job is missing immutable cleanupRegions list")

    try:
        image = _download_verified_source(job_data)
    except Exception as exc:
        logger.warning("[Cleanup] source verification failed: %s", exc)
        outcomes = [_region_result(region, "failed", [str(exc)]) for region in regions]
    else:
        outcomes = []
        # node_scoped, and the same lock name process_ocr uses. Before R3 this CTD/AOT work ran
        # *inside* the OCR job and was covered by that lock; moving it to its own job would
        # otherwise let a cleanup page and an OCR page run their local models concurrently on one
        # host, which is the CPU/RAM overload the lock exists to prevent. MAX_HEAVY_SLOTS bounds
        # one container; this bounds the machine.
        with acquire_lock("ocr", node_scoped=True):
            for region in regions:
                outcomes.append(_process_region(image, page_id, region))
                # Cleanup is the longest stage on the page and makes no network calls while it
                # runs, so the per-region tick is the only thing that distinguishes "working" from
                # "hung" in the backend's progress counter.
                record_progress()

    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": job_data.get("imageId"),
        "pageId": page_id,
        "attempt": job_data.get("attempt"),
        "inputGeneration": job_data.get("inputGeneration"),
        "leaseToken": job_data.get("leaseToken"),
        "cleanupInputDigest": job_data.get("cleanupInputDigest"),
        "regions": outcomes,
    }
    response = requests.post(
        f"{CALLBACK_URL}/cleanup", json=callback_payload, headers=backend_headers(), timeout=(5, 30)
    )
    response.raise_for_status()
