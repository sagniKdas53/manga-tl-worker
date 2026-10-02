"""QA jobs as the backend's render callback enqueues them: bound to one immutable artifact."""

import hashlib


def artifact_for(rendered: bytes, key: str = "rendered/revisions/page/1/scene") -> dict:
    sha = hashlib.sha256(rendered).hexdigest()
    return {
        "storagePath": f"{key}/{sha}.png",
        "sha256": sha,
        "byteLength": len(rendered),
        "contentType": "image/png",
    }


def bound_qa_job(job: dict, rendered: bytes = b"rendered") -> dict:
    return {
        "jobId": "qa-job",
        "attempt": 1,
        "pageRevision": 1,
        "logicalSceneSha256": "f" * 64,
        "renderArtifact": artifact_for(rendered),
        **job,
    }


def render_result(rendered: bytes, revision: int = 2) -> dict:
    artifact = artifact_for(rendered, key="rendered/image/jobs/qa-job/attempts/1")
    return {
        "pageRevision": revision,
        "logicalSceneSha256": "a" * 64,
        "pngSha256": artifact["sha256"],
        "artifact": artifact,
        "diagnostics": [],
    }
