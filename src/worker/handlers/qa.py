import base64
import copy
import hashlib
import io
import json
import logging
import os

import requests
from PIL import Image

from worker.config import (
    CALLBACK_URL,
    QA_CONFIG,
    QA_MODE,
    QA_VLM_FALLBACK_MODELS,
    backend_headers,
    is_usable_model,
    log_payload,
    minio_client,
    redis_client,
)
from worker.provider_config import get_config_loader
from worker.services.region_policy import select_region_action
from worker.services.translation import (
    try_cloud_ai,
    try_cloud_ai_vision,
    try_local_ai,
    try_local_vlm_vision,
)
from worker.utils.image import download_image

logger = logging.getLogger(__name__)

# `directFix` and `escalation` used to be optional, and the model simply never emitted them: the
# 20260803-084755 run produced qaStatus "direct_fix" 10 times with zero directFix payloads and
# "failed" 10 times with zero escalation blocks. Both consuming branches in JobCoordinatorService
# are keyed on the object being present, so direct fixes were never applied and needsReOcr never
# routed — every failure fell through to a blind re-translation of the same bad OCR.
#
# Everything is required now, which is also what OpenAI-style `strict` structured output demands.
# "Not applicable" is expressed as false / "" / 0 rather than by omitting the key. If a provider
# rejects the schema, LLMClient degrades to plain json_object and retries.
QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "regionId": {"type": "string"},
                    "qaStatus": {
                        "type": "string",
                        "enum": ["passed", "failed", "direct_fix", "reject_sfx"],
                    },
                    "qaScore": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "qaFeedback": {"type": "string"},
                    "directFix": {
                        "type": "object",
                        "properties": {
                            "correctedText": {"type": "string"},
                            "suggestedFontSize": {"type": "number"},
                        },
                        "required": ["correctedText", "suggestedFontSize"],
                        "additionalProperties": False,
                    },
                    "escalation": {
                        "type": "object",
                        "properties": {
                            "ocrBad": {"type": "boolean"},
                            "correctedSourceText": {"type": "string"},
                            "needsReOcr": {"type": "boolean"},
                            "needsManualIntervention": {"type": "boolean"},
                            "orderBad": {"type": "boolean"},
                            "suggestedReadingOrderIndex": {"type": "number"},
                        },
                        "required": [
                            "ocrBad",
                            "correctedSourceText",
                            "needsReOcr",
                            "needsManualIntervention",
                            "orderBad",
                            "suggestedReadingOrderIndex",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "regionId",
                    "qaStatus",
                    "qaScore",
                    "qaFeedback",
                    "directFix",
                    "escalation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


VALID_QA_STATUSES = {"passed", "failed", "direct_fix", "reject_sfx"}

# Verdicts for regions whose cleanup found no glyphs (qaStatus "cleanup_review"). The VLM decides
# what OCR actually found there, instead of the user having to.
UNCERTAIN_KINDS = {"dialogue", "sfx", "background_text", "not_text"}

# Vision QA also answers for uncertain regions. A separate schema, so the text-only QA modes are
# not forced to emit a key they have no image to fill.
QA_VLM_JSON_SCHEMA = copy.deepcopy(QA_JSON_SCHEMA)
QA_VLM_JSON_SCHEMA["properties"]["uncertainChecks"] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "regionId": {"type": "string"},
            "kind": {"type": "string", "enum": sorted(UNCERTAIN_KINDS)},
            "reason": {"type": "string"},
        },
        "required": ["regionId", "kind", "reason"],
        "additionalProperties": False,
    },
}
QA_VLM_JSON_SCHEMA["required"] = ["results", "uncertainChecks"]

# Key spellings models use when no schema holds them to ours. qwen3.7-flash's only OpenRouter host
# does not support structured outputs, so its json_schema request 400s, degrades to json_object,
# and it then writes "status" for "qaStatus" -- every verdict on page 4 was discarded that way.
_QA_KEY_ALIASES = {
    "status": "qaStatus",
    "qa_status": "qaStatus",
    "verdict": "qaStatus",
    "score": "qaScore",
    "qa_score": "qaScore",
    "feedback": "qaFeedback",
    "qa_feedback": "qaFeedback",
    "region_id": "regionId",
    "id": "regionId",
    "region": "regionId",
    "regionNumber": "regionId",
    "direct_fix": "directFix",
    "type": "kind",
    "classification": "kind",
}


def _region_labels(regions):
    """Map the number the Reader shows for each region (#1, #2, ...) to its UUID.

    The prompt names regions by these numbers rather than UUIDs: models copy a short number
    reliably, and a UUID they re-type can come back one character off (GLM on page 13 invented
    one). The reading order is used when it is a clean 1..n numbering; otherwise the regions are
    numbered by reading order, then position.
    """
    orders = [region.get("bubbleReadingOrder") for region in regions]
    if all(isinstance(order, int) and order > 0 for order in orders) and len(set(orders)) == len(orders):
        return {str(order): str(region["id"]) for order, region in zip(orders, regions, strict=True)}
    ordered = sorted(
        regions,
        key=lambda r: (r.get("bubbleReadingOrder") or 0, r.get("bboxY") or 0, r.get("bboxX") or 0),
    )
    return {str(index): str(region["id"]) for index, region in enumerate(ordered, 1)}


def _normalize_qa_items(items, uuid_by_label):
    """Return model output items with our key names and region UUIDs.

    Only spelling is repaired: a status still has to be one of ours, and an unknown region stays
    unknown so the integrity check reports it. Nothing is invented.
    """
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        item = dict(item)
        for alias, key in _QA_KEY_ALIASES.items():
            if alias in item and key not in item:
                item[key] = item.pop(alias)
        region_id = item.get("regionId")
        if isinstance(region_id, (int, float)) and not isinstance(region_id, bool):
            region_id = str(int(region_id))
        if isinstance(region_id, str):
            label = region_id.strip().lstrip("#").strip()
            item["regionId"] = uuid_by_label.get(label, label)
        for key in ("qaStatus", "kind"):
            if isinstance(item.get(key), str):
                item[key] = item[key].strip().lower()
        normalized.append(item)
    return normalized


def _parse_qa_response(qa_response):
    """Parse a raw model reply into (results, uncertainChecks); ``None`` when it is not JSON."""
    if not qa_response:
        return None
    cleaned = qa_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        logger.error(f"[QA] Failed to parse VLM response: {e}. Raw response: {log_payload(qa_response)}")
        return None
    if isinstance(parsed, list):
        return parsed, []
    if not isinstance(parsed, dict):
        return None
    return parsed.get("results") or [], parsed.get("uncertainChecks") or []


def _unique_by_region(items):
    """Keep items that name a region exactly once; a region named twice gets asked again."""
    counts = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("regionId"), str):
            counts[item["regionId"]] = counts.get(item["regionId"], 0) + 1
    return [item for item in items if isinstance(item, dict) and counts.get(item.get("regionId")) == 1]


def _qa_vlm_model_chain(job_data, provider):
    """Models to try, in order: the page's configured model, then the deployment's fallbacks.

    A later model is asked only about what the earlier ones did not answer: a refusal (Alibaba's
    content filter refuses explicit pages at random), an empty reply, or missing verdicts.
    """
    # "" stands for the provider's own qaVLM default, which _resolve_qa_model looks up.
    primary = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model or ""
    chain = [primary]
    if job_data.get("useFallbackModels", True) and QA_CONFIG.provider == provider:
        chain += [m for m in (QA_CONFIG.vlm_model, *QA_VLM_FALLBACK_MODELS) if m]
    seen = set()
    return [m for m in chain if not (m in seen or seen.add(m))]


def _qa_response_integrity(results, ocr_regions):
    """Describe every completeness failure without discarding the provider's raw result."""
    expected_ids = {str(region.get("id")) for region in ocr_regions if region.get("id")}
    if not isinstance(results, list):
        return {"complete": False, "errors": ["results is not an array"]}

    errors = []
    returned_ids = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append(f"result[{index}] is not an object")
            continue
        region_id = result.get("regionId")
        if not isinstance(region_id, str) or not region_id.strip():
            errors.append(f"result[{index}] has no regionId")
            continue
        if region_id not in expected_ids:
            errors.append(f"result[{index}] has foreign regionId {region_id}")
            continue
        if result.get("qaStatus") not in VALID_QA_STATUSES:
            errors.append(f"result[{index}] has invalid qaStatus for {region_id}")
            continue
        returned_ids.append(region_id)

    duplicate_ids = sorted({region_id for region_id in returned_ids if returned_ids.count(region_id) > 1})
    errors.extend(f"duplicate verdict for {region_id}" for region_id in duplicate_ids)
    missing_ids = sorted(expected_ids - set(returned_ids))
    errors.extend(f"missing verdict for {region_id}" for region_id in missing_ids)
    return {"complete": not errors, "errors": errors}


def _artifact_fields(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("QA job has no immutable render artifact")
    required = {"storagePath", "sha256", "byteLength", "contentType"}
    if set(artifact) != required:
        raise ValueError("QA render artifact has an invalid shape")
    if not isinstance(artifact["storagePath"], str) or not artifact["storagePath"]:
        raise ValueError("QA render artifact has no storage path")
    if not isinstance(artifact["sha256"], str) or len(artifact["sha256"]) != 64:
        raise ValueError("QA render artifact has an invalid digest")
    if not isinstance(artifact["byteLength"], int) or artifact["byteLength"] < 0:
        raise ValueError("QA render artifact has an invalid byte length")
    if artifact["contentType"] != "image/png":
        raise ValueError("QA render artifact is not a PNG")
    return artifact


def _read_render_artifact(artifact):
    """Read and verify exactly the immutable PNG a VLM is about to judge."""
    artifact = _artifact_fields(artifact)
    response = minio_client.get_object("manga-library", artifact["storagePath"])
    rendered_bytes = response.read()
    if len(rendered_bytes) != artifact["byteLength"]:
        raise ValueError("QA render artifact byte length mismatch")
    if hashlib.sha256(rendered_bytes).hexdigest() != artifact["sha256"]:
        raise ValueError("QA render artifact digest mismatch")
    return rendered_bytes


def _judged_artifact(job_data, artifact=None, render_result=None):
    """Build the identity the backend compares with the persisted QA-job binding."""
    if render_result is not None:
        artifact = render_result["artifact"]
        page_revision = render_result["pageRevision"]
        logical_scene_sha256 = render_result["logicalSceneSha256"]
    else:
        artifact = artifact if artifact is not None else job_data.get("renderArtifact")
        page_revision = job_data.get("pageRevision")
        logical_scene_sha256 = job_data.get("logicalSceneSha256")
    _artifact_fields(artifact)
    if not isinstance(page_revision, int) or isinstance(page_revision, bool):
        raise ValueError("QA artifact has no page revision")
    if not isinstance(logical_scene_sha256, str) or len(logical_scene_sha256) != 64:
        raise ValueError("QA artifact has no logical scene digest")
    return {
        "artifact": artifact,
        "pageRevision": page_revision,
        "logicalSceneSha256": logical_scene_sha256,
    }


def _qa_accounting(job_data, raw_results, qa_regions, *, judged=True, render_result=None):
    """Fields the backend needs to tell a complete verdict set from a truncated or stale one.

    The backend, not this worker, decides whether QA passed: it compares the verdicts against
    ``qaTargetIds`` and against every region the page actually displays text for, and it only
    accepts verdicts about the revision it asked to have judged.
    """
    accounting = {
        "qaTargetIds": sorted(str(region["id"]) for region in qa_regions if region.get("id")),
        "qaResponseIntegrity": _qa_response_integrity(raw_results, qa_regions),
    }
    if judged:
        accounting["judgedArtifact"] = _judged_artifact(job_data, render_result=render_result)
    return accounting


def _sanitize_qa_results(results, ocr_regions, label="LLM"):
    """
    Keep only results that actually identify a region we asked about.

    A response truncated at the output token limit still parses: OpenRouter's response-healing
    plugin closes the JSON, so `json.loads` succeeds and hands back an object that has lost its
    trailing fields. In the 20260803-084755 run that produced a single `{"qaFeedback": "..."}`
    entry with no regionId, which the backend then failed to apply — and, because nothing was
    scored, recorded as a clean QA pass.

    Anything unusable is dropped here rather than forwarded. Returning fewer results than regions
    is fine; the backend treats an empty verdict as "QA did not run" instead of as a pass.
    """
    known_ids = {str(r.get("id")) for r in ocr_regions if r.get("id")}
    kept, discarded = [], []

    for r in results or []:
        if not isinstance(r, dict):
            discarded.append("not-an-object")
            continue
        region_id = r.get("regionId")
        if not isinstance(region_id, str) or not region_id.strip():
            discarded.append("missing regionId")
            continue
        if known_ids and region_id not in known_ids:
            discarded.append(f"unknown regionId {region_id}")
            continue
        if r.get("qaStatus") not in VALID_QA_STATUSES:
            discarded.append(f"bad qaStatus {r.get('qaStatus')!r} for {region_id}")
            continue
        kept.append(r)

    if discarded:
        logger.info(
            f"[QA] Discarded {len(discarded)} unusable {label} result(s): {'; '.join(discarded[:5])}"
            f"{' ...' if len(discarded) > 5 else ''}"
        )

    missing = len(known_ids) - len(kept) if known_ids else 0
    if kept and missing > 0:
        logger.info(
            f"[QA] {label} returned a verdict for {len(kept)}/{len(known_ids)} regions — "
            "the response was probably truncated."
        )

    # A `failed` verdict with no escalation gives the backend nothing to act on, so it falls back
    # to re-translating the same source text. Surface it; the schema now requires the object.
    unactionable = [r["regionId"] for r in kept if r.get("qaStatus") == "failed" and not r.get("escalation")]
    if unactionable:
        logger.error(
            f"[QA] {len(unactionable)} failed region(s) carry no escalation block; re-OCR cannot be routed for them."
        )

    return kept


def _translation_qa_regions(ocr_regions):
    """Return successful translation-layer elements for per-region QA.

    Unreviewed policy is a canonical-scene cleanup authorization state, not a reason to
    bypass the live translation/visual-QA pipeline.  QA must receive those elements so it
    can reject SFX or bad OCR and the backend can hide only the rejected element. Explicit
    ``preserve`` and ``explain`` overrides remain source-preserving even if stale translation
    data exists.

    Uncertain regions (``cleanup_review``: cleanup found no glyphs) are translated but kept
    hidden, so they are not translation targets; vision QA judges them separately through
    ``uncertainChecks``. Rejected regions are hidden for good and are not judged again.
    """
    return [
        region
        for region in ocr_regions
        if region.get("translatedText")
        and not region.get("translationFailed")
        and region.get("qaStatus") not in {"cleanup_review", "rejected"}
        and select_region_action(
            region.get("regionType") or region.get("region_type"),
            region.get("user_override"),
        ).user_override
        not in {"preserve", "explain"}
    ]


def _qa_default_model(prov: str, task: str) -> str | None:
    """The provider's own QA default, read from config/providers.json.

    AUDIT-W1: this used to be two tables in this file listing openrouter/gemini/nvidia, so
    neurometric — selectable in the UI and in providers.json — had no default at all, while
    `gemini` had one but is not a configured provider. providers.json is
    already the single source of truth for every other default (`defaultTLModel`,
    `defaultOCRModel`); QA now reads `defaultQALLMModel` / `defaultQAVLMModel` from the same place.
    `task` is the providers.json key: "qaLLM" or "qaVLM".

    Reloads on an edited file for the same reason LLMClient does — one stat against a call that is
    about to spend seconds in HTTP — so adding a provider does not need a worker restart.
    """
    loader = get_config_loader()
    loader.reload_if_changed()
    pconfig = loader.providers.get(prov)
    if pconfig is None:
        return None
    return pconfig.defaults.get(task)


def _resolve_qa_model(prov: str, api_key: str | None, user_model: str | None, task: str) -> str | None:
    """Resolve the model for a QA call, logging the reason when the call cannot be made."""
    if not prov:
        return None
    if not api_key:
        logger.warning(f"[QA] No API key configured for provider '{prov}' — skipping.")
        return None
    model = user_model or _qa_default_model(prov, task)
    if not model:
        logger.warning(
            f"[QA] Provider '{prov}' has no model configured and no '{task}' default in "
            "providers.json — set one on the chapter, series, or global settings."
        )
        return None
    return model


def _qa_cloud_llm(prov, api_key, user_model, prompt, routing_strategy):
    """Text QA against any provider in config/providers.json."""
    model = _resolve_qa_model(prov, api_key, user_model, "qaLLM")
    if not model:
        return None
    try:
        return try_cloud_ai(
            prov,
            api_key,
            model,
            prompt,
            QA_JSON_SCHEMA,
            routing_strategy=routing_strategy,
        )
    except Exception as e:
        logger.error(f"[QA] LLM QA via '{prov}' with model '{model}' failed: {e}")
        return None


def _qa_cloud_vlm(prov, api_key, user_model, prompt, base64_image, routing_strategy, schema=None):
    """Vision QA against any provider in config/providers.json."""
    model = _resolve_qa_model(prov, api_key, user_model, "qaVLM")
    if not model:
        return None
    try:
        return try_cloud_ai_vision(
            prov,
            api_key,
            model,
            prompt,
            base64_image,
            schema or QA_JSON_SCHEMA,
            routing_strategy=routing_strategy,
        )
    except Exception as e:
        logger.error(f"[QA] VLM QA via '{prov}' with model '{model}' failed: {e}")
        return None


def process_qa(job_data):
    image_id = job_data["imageId"]
    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:qa")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    qa_mode_resolved = job_data.get("qaMode") or QA_MODE

    if qa_mode_resolved == "auto":
        provider = job_data.get("qaProvider") or getattr(QA_CONFIG, "provider", None)
        has_vlm = is_usable_model(job_data.get("qaVlmModel")) or is_usable_model(getattr(QA_CONFIG, "vlm_model", None))
        has_llm = is_usable_model(job_data.get("qaLlmModel")) or is_usable_model(getattr(QA_CONFIG, "llm_model", None))
        if has_vlm and provider:
            qa_mode_resolved = "vlm"
        elif has_llm and provider:
            qa_mode_resolved = "llm"
        else:
            qa_mode_resolved = "none"
        logger.info(
            f"[QA] AUTO mode resolved to '{qa_mode_resolved}' (provider={provider}, "
            f"vlm={'yes' if has_vlm else 'no'}, llm={'yes' if has_llm else 'no'})"
        )

    logger.info(f"[QA] Processing image: {image_id}{progress_str} (mode={qa_mode_resolved})")

    if job_data.get("qaAttempt", 0) > 0:
        logger.warning("[QA] Skipping QA because qaAttempt > 0 (One pass only to prevent loops)")
        _auto_pass_all(job_data)
        return

    if qa_mode_resolved == "none":
        _auto_pass_all(job_data)
    elif qa_mode_resolved == "llm":
        _process_qa_llm(job_data)
    elif qa_mode_resolved == "vlm":
        _process_qa_vlm(job_data)
    elif qa_mode_resolved == "hybrid":
        _process_qa_hybrid(job_data)
    else:
        logger.warning(f"[QA] Unknown QA_MODE={qa_mode_resolved}, falling back to auto-pass")
        _auto_pass_all(job_data)


def _process_qa_hybrid(job_data):
    image_id = job_data.get("imageId")
    page_id = job_data.get("pageId")
    logger.info(f"[QA] Processing Hybrid QA check for page: {page_id or image_id}")

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[QA] Failed to get page/image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            logger.warning("[QA] No OCR regions found. Skipping Hybrid QA.")
            _auto_pass_all(job_data)
            return
    except Exception as e:
        logger.error(f"[QA] Error fetching image details: {e}")
        raise

    qa_regions = _translation_qa_regions(ocr_regions)
    # Build metadata only for replacement-authorized translation QA targets.
    regions_metadata = []
    for r in qa_regions:
        regions_metadata.append(
            {
                "regionId": r["id"],
                "ocrText": r["text"],
                "ocrScore": r.get("ocrScore") or r.get("confidence") or 1.0,
                "translatedText": r.get("translatedText") or "",
                "translationScore": r.get("translationScore") or 1.0,
                "readingOrder": r.get("bubbleReadingOrder") or 0,
            }
        )

    logger.debug(f"[QA] LLM QA input metadata (regions_metadata) for Hybrid pass:\n{log_payload(regions_metadata)}")

    prompt = f"""You are an expert bilingual Japanese-to-English manga translator and QA reviewer.
Your job is to evaluate translation quality and conversation flow based on text-only metadata.

For each region in the provided metadata, evaluate and check if:
1. The English translation is accurate, natural, and contextually appropriate compared to the original Japanese OCR text.
2. The conversation flow between dialogue regions feels coherent.
3. The original Japanese OCR transcription was bad/inaccurate:
   - If you can deduce the correct text, flag with ocrBad=true and provide correctedSourceText.
   - If the OCR text is garbage (like misread sound effects) and you CANNOT deduce it, flag needsReOcr=true.
   - If the region is completely unfixable or obscured, flag needsManualIntervention=true.
4. The reading order/bubble sequence is incorrect (flag with orderBad=true and provide suggestedReadingOrderIndex).

Status categories:
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed.
- "direct_fix": If you have a better translation, output it directly. You must supply "directFix" object with correctedText. You MUST also provide detailed reasoning in "qaFeedback".
- "reject_sfx": If the region is a sound effect (SFX) or gibberish that shouldn't be translated, set this status (downstream will hide the element).
- "failed": Translation error requiring a translation re-run. Specify "qaFeedback" with detailed correction notes/feedback to guide the re-translation. Your output must be strictly better. Do not send back the exact same text if flagging an error.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

IMPORTANT: Every result MUST include both a "directFix" object and an "escalation" object. They are
never omitted. When a field does not apply, send its empty value rather than leaving it out —
empty string for text fields, false for flags, 0 for numbers.
  - "directFix" always carries: correctedText, suggestedFontSize
  - "escalation" always carries: ocrBad, correctedSourceText, needsReOcr, needsManualIntervention,
    orderBad, suggestedReadingOrderIndex
The ocrBad / needsReOcr / needsManualIntervention / orderBad flags described above live inside
"escalation". Describing a problem only in "qaFeedback" prose has no effect — the flags are what
route the fix. In particular, if the OCR text is unreadable, set escalation.needsReOcr to true;
asking for a re-OCR in prose alone will instead re-run the translation over the same bad text.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = (job_data.get("qaProvider") or QA_CONFIG.provider) if regions_metadata else ""
    api_key = QA_CONFIG.resolve_key(provider) if provider else None
    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"

    qa_response = None

    def attempt_llm(prov, model_override=None):
        user_model = model_override or job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        # AUDIT-Q3: dropped a `cache_key` that was built, logged with a hardcoded (hit=False) and
        # then thrown away. There is no QA cache, so the line reported a 0% hit rate on nothing.
        return _qa_cloud_llm(prov, api_key, user_model, prompt, routing_strategy)

    # Try preferred provider/models
    if provider:
        user_model = job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        qa_response = attempt_llm(provider, user_model)

        if not qa_response:
            use_fallback_models = job_data.get("useFallbackModels", True)
            if use_fallback_models:
                # Fallback to global default model
                global_model = QA_CONFIG.llm_model
                global_provider = QA_CONFIG.provider
                if global_provider == provider and global_model and global_model != user_model:
                    logger.warning(f"[QA] Falling back to global default model '{global_model}'...")
                    qa_response = attempt_llm(provider, global_model)
                else:
                    logger.warning("[QA] No fallback applied (global provider different or model identical).")

    local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    is_explicit_local = provider in ("ollama", "lmstudio")

    if not qa_response and regions_metadata and local_llm_model and (is_explicit_local or not disable_local):
        try:
            qa_response = try_local_ai(prompt, json.dumps(regions_metadata), QA_JSON_SCHEMA)
        except Exception as e:
            logger.error(f"[QA] LLM QA via Local LLM failed: {e}")

    results = []
    if qa_response:
        try:
            cleaned = qa_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            results = parsed.get("results") or []
        except Exception as e:
            logger.error(f"[QA] Failed to parse LLM response: {e}. Raw response: {log_payload(qa_response)}")
    llm_integrity = _qa_response_integrity(results, qa_regions)
    results = _sanitize_qa_results(results, qa_regions, label="LLM")
    if not llm_integrity["complete"]:
        # OQ-02: a partial first pass is untrustworthy as a whole, so none of its fixes are applied.
        # The VLM pass still judges the page as it stands, under the same accounting rules.
        logger.warning(
            f"[QA] Hybrid LLM pass incomplete ({len(llm_integrity['errors'])} problem(s)); "
            "applying none of its fixes before the VLM pass."
        )
        results = []

    # Call backend prepare endpoint to apply fixes and set visibility
    prepare_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}/qa-hybrid-prepare")
    try:
        prep_res = requests.post(
            prepare_url,
            json={
                "pageId": job_data.get("pageId"),
                "qaResults": results,
            },
            headers=backend_headers(),
        )
        logger.info(f"[QA] Hybrid QA preparation status code: {prep_res.status_code}")
    except Exception as e:
        logger.error(f"[QA] Failed to post Hybrid QA preparation: {e}")
        raise

    # The prepare endpoint applied the first pass's fixes, snapshotted the page and returned the
    # immutable render payload; draw it through the browser renderer so the VLM judges the same
    # pixels the export will carry (tracker R1). 204 means the image has no page — nothing to render.
    if prep_res.status_code == 204:
        logger.warning("[QA] Hybrid QA: image has no page; skipping the VLM pass.")
        return
    prep_res.raise_for_status()
    from worker.page_scene_renderer import render_page_scene

    # The prepared scene is rendered under this QA job's own attempt identity, so its artifact key
    # cannot collide with a render job's, or with another QA attempt's.
    render_result = render_page_scene(
        {**prep_res.json(), "jobId": job_data.get("jobId"), "attempt": job_data.get("attempt", 1)}
    )

    # Now run VLM check on updated render
    try:
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[QA] Failed to get updated image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            logger.warning("[QA] No OCR regions found. Skipping VLM QA.")
            _auto_pass_all(job_data)
            return
    except Exception as e:
        logger.error(f"[QA] Error fetching image details: {e}")
        raise
    qa_regions = _translation_qa_regions(ocr_regions)

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        logger.error(f"[QA] Error downloading original image: {e}")
        raise

    # Judge the exact immutable PNG rendered from the prepared scene, never a page-global key.
    rendered_bytes = _read_render_artifact(render_result["artifact"])

    try:
        img1 = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(rendered_bytes)).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size
        combined_width = w1 + w2
        combined_height = max(h1, h2)

        combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (w1, 0))

        combined_buf = io.BytesIO()
        combined_img.save(combined_buf, format="JPEG", quality=85)
        combined_base64 = base64.b64encode(combined_buf.getvalue()).decode("utf-8")

        import time

        from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR

        if ENABLE_QA_AUDIT_CACHE:
            try:
                os.makedirs(QA_AUDIT_CACHE_DIR, exist_ok=True)
                audit_path = os.path.join(QA_AUDIT_CACHE_DIR, f"{image_id}_{int(time.time())}.jpg")
                combined_img.save(audit_path, format="JPEG", quality=85)
            except Exception as e:
                logger.error(f"[QA] Failed to write QA audit cache image: {e}")
    except Exception as e:
        logger.error(f"[QA] Error combining images: {e}")
        raise

    # The VLM sees the complete page and every successful translation-layer element. It decides
    # which candidate SFX/gibberish must be hidden after visual inspection.
    regions_metadata_vlm = []
    for r in qa_regions:
        regions_metadata_vlm.append(
            {
                "regionId": r["id"],
                "ocrText": r["text"],
                "ocrScore": r.get("ocrScore") or r.get("confidence") or 1.0,
                "translatedText": r.get("translatedText") or "",
                "translationScore": r.get("translationScore") or 1.0,
                "x": r["bboxX"],
                "y": r["bboxY"],
                "w": r["bboxW"],
                "h": r["bboxH"],
                "readingOrder": r.get("bubbleReadingOrder") or 0,
            }
        )

    prompt_vlm = f"""You are an expert Japanese-to-English manga translator and typesetting reviewer. Given the original Japanese manga page (left) and the English typeset page (right), verify: (1) OCR accuracy by comparing visible Japanese text against transcription, (2) Translation quality and natural English, (3) Typesetting quality — text fitting, overflow, readability.

We have seeded each text region with its OCR confidence (ocrScore) and translation confidence (translationScore). Keep these previous scores in mind when evaluating the overall results.

For each region in the provided metadata, evaluate and check if:
1. Text overflows the speech bubble/mask boundaries.
2. Text overlaps with panel borders or other text.
3. Translation flow is awkward, or the English translation does not match the original Japanese text.
4. The OCR transcription was bad/inaccurate:
   - If you can deduce the correct text from the image, flag with ocrBad=true and provide correctedSourceText.
   - If the OCR text is garbage and you CANNOT deduce it or read it, flag needsReOcr=true.
   - If the region is completely unfixable or obscured, flag needsManualIntervention=true.
5. The reading order/bubble sequence is incorrect (flag with orderBad=true and provide suggestedReadingOrderIndex).

Status categories:
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed.
- "direct_fix": If you have a better translation, output it directly. You must supply "directFix" object with correctedText or suggestedFontSize. You MUST also provide detailed reasoning in "qaFeedback".
- "reject_sfx": If the region is a sound effect (SFX) or gibberish that shouldn't be translated, set this status (downstream will hide the element).
- "failed": Major translation error or layout issue requiring a translation/typesetting re-run. Specify "qaFeedback" with detailed correction notes. Your output must be strictly better. Do not send back the exact same text if flagging an error.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

IMPORTANT: Every result MUST include both a "directFix" object and an "escalation" object. They are
never omitted. When a field does not apply, send its empty value rather than leaving it out —
empty string for text fields, false for flags, 0 for numbers.
  - "directFix" always carries: correctedText, suggestedFontSize
  - "escalation" always carries: ocrBad, correctedSourceText, needsReOcr, needsManualIntervention,
    orderBad, suggestedReadingOrderIndex
The ocrBad / needsReOcr / needsManualIntervention / orderBad flags described above live inside
"escalation". Describing a problem only in "qaFeedback" prose has no effect — the flags are what
route the fix. In particular, if the OCR text is unreadable, set escalation.needsReOcr to true;
asking for a re-OCR in prose alone will instead re-run the translation over the same bad text.

Region Metadata:
{json.dumps(regions_metadata_vlm, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    vlm_api_key = QA_CONFIG.resolve_key(provider)
    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"
    qa_response_vlm = None

    def attempt_vlm(prov, model_override=None):
        user_model = model_override or job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        # AUDIT-Q3: see attempt_llm — same phantom cache key, same hardcoded (hit=False).
        return _qa_cloud_vlm(prov, vlm_api_key, user_model, prompt_vlm, combined_base64, routing_strategy)

    if provider:
        user_model = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        qa_response_vlm = attempt_vlm(provider, user_model)

        if not qa_response_vlm:
            use_fallback_models = job_data.get("useFallbackModels", True)
            if use_fallback_models:
                global_model = QA_CONFIG.vlm_model
                global_provider = QA_CONFIG.provider
                if global_provider == provider and global_model and global_model != user_model:
                    logger.warning(f"[QA] Falling back to global default VLM model '{global_model}'...")
                    qa_response_vlm = attempt_vlm(provider, global_model)
                else:
                    logger.warning("[QA] No fallback applied (global provider different or model identical).")

    local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()

    if not qa_response_vlm and local_vlm_model and (is_explicit_local or not disable_local):
        try:
            qa_response_vlm = try_local_vlm_vision(local_vlm_model, prompt_vlm, combined_base64, QA_JSON_SCHEMA)
        except Exception as e:
            logger.error(f"[QA] VLM QA via Local VLM failed: {e}")

    results_vlm = []
    if qa_response_vlm:
        try:
            cleaned = qa_response_vlm.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            results_vlm = parsed.get("results") or []
        except Exception as e:
            logger.error(f"[QA] Failed to parse VLM response: {e}. Raw response: {log_payload(qa_response_vlm)}")

    accounting = _qa_accounting(job_data, results_vlm, qa_regions, render_result=render_result)
    results_vlm = _sanitize_qa_results(results_vlm, qa_regions, label="VLM")

    if not results_vlm:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        logger.warning("[QA] No usable VLM results — reporting no verdict rather than auto-passing.")

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results_vlm,
        **accounting,
    }
    from worker.utils.rate_limit import build_cost_payload, format_cost, get_job_costs

    cost_payload = build_cost_payload(get_job_costs())
    if cost_payload:
        callback_payload["cost"] = cost_payload
        total_estimated_cost = cost_payload.get("estimated_cost")
        total_prompt_tokens = cost_payload["prompt_tokens"]
        total_completion_tokens = cost_payload["completion_tokens"]

        cost_str = format_cost(total_estimated_cost)
        if cost_payload["unknown_calls"]:
            cost_str += f" ({cost_payload['unknown_calls']} of {len(cost_payload['breakdown'])} calls unpriced)"

        logger.info(
            f"[QA] Hybrid QA VLM pass estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=backend_headers(), timeout=(5, 30))
        res.raise_for_status()
        logger.debug(f"[QA] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[QA] Failed to post QA callback to backend: {e}")
        raise


def _auto_pass_all(job_data):
    image_id = job_data["imageId"]
    logger.warning(f"[QA] Skipping QA (QA_MODE=none) for image: {image_id}")

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[QA] Failed to get image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        logger.error(f"[QA] Error fetching image details: {e}")
        raise

    # Settled (rejected) and unjudged-uncertain regions keep their state: a bypass is not a verdict
    # that the ornament OCR misread is now fine, or that a hidden uncertain region was checked.
    ocr_regions = [r for r in ocr_regions if r.get("qaStatus") not in {"rejected", "cleanup_review"}]
    results = []
    for r in ocr_regions:
        results.append(
            {
                "regionId": r["id"],
                "qaStatus": "passed",
                "qaScore": 1.0,
                "qaFeedback": "Auto-passed (QA bypassed)",
            }
        )

    # QA is bypassed, not judged: the targets are every region, and each gets a stated pass.
    accounting = _qa_accounting(
        job_data,
        results,
        [{"id": r["id"]} for r in ocr_regions],
        judged=bool(job_data.get("renderArtifact")),
    )

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
        **accounting,
    }
    from worker.utils.rate_limit import build_cost_payload, format_cost, get_job_costs

    cost_payload = build_cost_payload(get_job_costs())
    if cost_payload:
        callback_payload["cost"] = cost_payload
        total_estimated_cost = cost_payload.get("estimated_cost")
        total_prompt_tokens = cost_payload["prompt_tokens"]
        total_completion_tokens = cost_payload["completion_tokens"]

        cost_str = format_cost(total_estimated_cost)
        if cost_payload["unknown_calls"]:
            cost_str += f" ({cost_payload['unknown_calls']} of {len(cost_payload['breakdown'])} calls unpriced)"

        logger.info(
            f"[QA] Auto-pass QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=backend_headers(), timeout=(5, 30))
        res.raise_for_status()
        logger.debug(f"[QA] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[QA] Failed to post QA callback to backend: {e}")
        raise


def _process_qa_llm(job_data):
    image_id = job_data["imageId"]
    logger.info(f"[QA] Processing text-only LLM QA check for image: {image_id}")

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[QA] Failed to get image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            logger.warning("[QA] No OCR regions found. Skipping LLM QA.")
            _auto_pass_all(job_data)
            return
    except Exception as e:
        logger.error(f"[QA] Error fetching image details: {e}")
        raise

    # Per-region QA follows actual translation-layer elements, not canonical-scene cleanup
    # authorization. It can therefore reject/hide SFX after inspecting the translation.
    qa_regions = _translation_qa_regions(ocr_regions)
    regions_metadata = []
    for r in qa_regions:
        regions_metadata.append(
            {
                "regionId": r["id"],
                "ocrText": r["text"],
                "ocrScore": r.get("ocrScore") or r.get("confidence") or 1.0,
                "translatedText": r.get("translatedText") or "",
                "translationScore": r.get("translationScore") or 1.0,
                "readingOrder": r.get("bubbleReadingOrder") or 0,
            }
        )

    logger.debug(f"[QA] LLM QA input metadata (regions_metadata):\n{log_payload(regions_metadata)}")

    prompt = f"""You are an expert bilingual Japanese-to-English manga translator and QA reviewer.
Your job is to evaluate translation quality and conversation flow based on text-only metadata.

For each region in the provided metadata, evaluate and check if:
1. The English translation is accurate, natural, and contextually appropriate compared to the original Japanese OCR text.
2. The conversation flow between dialogue regions feels coherent.
3. The original Japanese OCR transcription was bad/inaccurate:
   - If you can deduce the correct text, flag with ocrBad=true and provide correctedSourceText.
   - If the OCR text is garbage (like misread sound effects) and you CANNOT deduce it, flag needsReOcr=true.
   - If the region is completely unfixable or obscured, flag needsManualIntervention=true.
4. The reading order/bubble sequence is incorrect (flag with orderBad=true and provide suggestedReadingOrderIndex).

Status categories:
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed.
- "direct_fix": If you have a better translation, output it directly. You must supply "directFix" object with correctedText. You MUST also provide detailed reasoning in "qaFeedback".
- "reject_sfx": If the region is a sound effect (SFX) or gibberish that shouldn't be translated, set this status (downstream will hide the element).
- "failed": Translation error requiring a translation re-run. Specify "qaFeedback" with detailed correction notes/feedback to guide the re-translation. Your output must be strictly better. Do not send back the exact same text if flagging an error.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

IMPORTANT: Every result MUST include both a "directFix" object and an "escalation" object. They are
never omitted. When a field does not apply, send its empty value rather than leaving it out —
empty string for text fields, false for flags, 0 for numbers.
  - "directFix" always carries: correctedText, suggestedFontSize
  - "escalation" always carries: ocrBad, correctedSourceText, needsReOcr, needsManualIntervention,
    orderBad, suggestedReadingOrderIndex
The ocrBad / needsReOcr / needsManualIntervention / orderBad flags described above live inside
"escalation". Describing a problem only in "qaFeedback" prose has no effect — the flags are what
route the fix. In particular, if the OCR text is unreadable, set escalation.needsReOcr to true;
asking for a re-OCR in prose alone will instead re-run the translation over the same bad text.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = (job_data.get("qaProvider") or QA_CONFIG.provider) if regions_metadata else ""
    api_key = QA_CONFIG.resolve_key(provider) if provider else None
    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"

    qa_response = None

    def attempt_llm(prov, model_override=None):
        user_model = model_override or job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        return _qa_cloud_llm(prov, api_key, user_model, prompt, routing_strategy)

    local_only = provider in ("ollama", "lmstudio")
    if local_only:
        local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
        if local_llm_model:
            try:
                qa_response = try_local_ai(prompt, json.dumps(regions_metadata), QA_JSON_SCHEMA)
            except Exception as e:
                logger.error(f"[QA] LLM QA via Local LLM failed: {e}")
    else:
        # Try the preferred provider first
        if provider:
            user_model = job_data.get("qaLlmModel") or QA_CONFIG.llm_model
            qa_response = attempt_llm(provider, user_model)

            if not qa_response:
                use_fallback_models = job_data.get("useFallbackModels", True)
                if use_fallback_models:
                    global_model = QA_CONFIG.llm_model
                    global_provider = QA_CONFIG.provider
                    if global_provider == provider and global_model and global_model != user_model:
                        logger.warning(f"[QA] Falling back to global default LLM model '{global_model}'...")
                        qa_response = attempt_llm(provider, global_model)
                    else:
                        logger.warning("[QA] No fallback applied (global provider different or model identical).")

    results = []
    if logger.isEnabledFor(logging.DEBUG) and qa_response:
        logger.debug(f"[QA] Raw LLM Response: {qa_response}")

    if qa_response:
        try:
            cleaned = qa_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            results = parsed.get("results") or []
        except Exception as e:
            logger.error(f"[QA] Failed to parse LLM response: {e}. Raw response: {log_payload(qa_response)}")

    accounting = _qa_accounting(job_data, results, qa_regions)
    results = _sanitize_qa_results(results, qa_regions, label="LLM")

    if not results:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        logger.warning("[QA] No usable LLM results — reporting no verdict rather than auto-passing.")

    logger.debug(f"[QA] LLM QA results output:\n{log_payload(results)}")

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
        **accounting,
    }
    from worker.utils.rate_limit import build_cost_payload, format_cost, get_job_costs

    cost_payload = build_cost_payload(get_job_costs())
    if cost_payload:
        callback_payload["cost"] = cost_payload
        total_estimated_cost = cost_payload.get("estimated_cost")
        total_prompt_tokens = cost_payload["prompt_tokens"]
        total_completion_tokens = cost_payload["completion_tokens"]

        cost_str = format_cost(total_estimated_cost)
        if cost_payload["unknown_calls"]:
            cost_str += f" ({cost_payload['unknown_calls']} of {len(cost_payload['breakdown'])} calls unpriced)"

        logger.info(
            f"[QA] LLM QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=backend_headers(), timeout=(5, 30))
        res.raise_for_status()
        logger.debug(f"[QA] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[QA] Failed to post QA callback to backend: {e}")
        raise


def _vlm_qa_prompt(targets, uncertain, label_by_uuid):
    """The vision-QA prompt for translated ``targets`` and cleanup-``uncertain`` regions."""
    regions_metadata = [
        {
            "regionId": label_by_uuid[str(r["id"])],
            "ocrText": r["text"],
            "ocrScore": r.get("ocrScore") or r.get("confidence") or 1.0,
            "translatedText": r.get("translatedText") or "",
            "translationScore": r.get("translationScore") or 1.0,
            "x": r["bboxX"],
            "y": r["bboxY"],
            "w": r["bboxW"],
            "h": r["bboxH"],
        }
        for r in targets
    ]
    uncertain_metadata = [
        {
            "regionId": label_by_uuid[str(r["id"])],
            "ocrText": r.get("text") or "",
            "x": r["bboxX"],
            "y": r["bboxY"],
            "w": r["bboxW"],
            "h": r["bboxH"],
        }
        for r in uncertain
    ]
    logger.debug(f"[QA] VLM QA input metadata (regions_metadata):\n{log_payload(regions_metadata)}")

    uncertain_section = ""
    if uncertain_metadata:
        uncertain_section = f"""

UNCERTAIN REGIONS. OCR reported text at these boxes, but the text detector found no lettering
inside them. Look at each box on the ORIGINAL page (left) and classify what is really there:
- "dialogue": lettering meant for the reader -- speech, thoughts, narration, captions.
- "sfx": a sound effect or drawn onomatopoeia -- it stays as the artist drew it.
- "background_text": text that is part of the scenery -- signs, book spines, posters, labels,
  logos, screens.
- "not_text": no lettering at all -- artwork, ornaments, patterns, hair, clothing or noise that
  OCR misread as text.
Give one entry per uncertain region in "uncertainChecks", with a one-sentence "reason".

Uncertain regions:
{json.dumps(uncertain_metadata, ensure_ascii=False, indent=2)}"""

    return f"""You are an expert Japanese-to-English manga translator and typesetting reviewer. Given the original Japanese manga page (left) and the English typeset page (right), verify: (1) OCR accuracy by comparing visible Japanese text against transcription, (2) Translation quality and natural English, (3) Typesetting quality — text fitting, overflow, readability.

Each region is identified by its number ("regionId"). Copy that number exactly; do not invent,
renumber or skip regions.

We have seeded each text region with its OCR confidence (ocrScore) and translation confidence (translationScore). Keep these previous scores in mind when evaluating the overall results.

For each region in the provided metadata, evaluate and check if:
1. Text overflows the speech bubble/mask boundaries.
2. Text overlaps with panel borders or other text.
3. Translation flow is awkward, or the English translation does not match the original Japanese text.
4. The OCR transcription was bad/inaccurate:
   - If you can deduce the correct text from the image, flag with ocrBad=true and provide correctedSourceText.
   - If the OCR text is garbage and you CANNOT deduce it or read it, flag needsReOcr=true.
   - If the region is completely unfixable or obscured, flag needsManualIntervention=true.
5. The reading order/bubble sequence is incorrect (flag with orderBad=true and provide suggestedReadingOrderIndex).

Status categories ("qaStatus"):
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed.
- "direct_fix": If you have a better translation, output it directly. You must supply "directFix" object with correctedText or suggestedFontSize. You MUST also provide detailed reasoning in "qaFeedback".
- "reject_sfx": If the region is a sound effect (SFX) or gibberish that shouldn't be translated, set this status (downstream will hide the element).
- "failed": Major translation error or layout issue requiring a translation/typesetting re-run. Specify "qaFeedback" with detailed correction notes. Your output must be strictly better. Do not send back the exact same text if flagging an error.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

IMPORTANT: Every result MUST include both a "directFix" object and an "escalation" object. They are
never omitted. When a field does not apply, send its empty value rather than leaving it out —
empty string for text fields, false for flags, 0 for numbers.
  - "directFix" always carries: correctedText, suggestedFontSize
  - "escalation" always carries: ocrBad, correctedSourceText, needsReOcr, needsManualIntervention,
    orderBad, suggestedReadingOrderIndex
The ocrBad / needsReOcr / needsManualIntervention / orderBad flags described above live inside
"escalation". Describing a problem only in "qaFeedback" prose has no effect — the flags are what
route the fix. In particular, if the OCR text is unreadable, set escalation.needsReOcr to true;
asking for a re-OCR in prose alone will instead re-run the translation over the same bad text.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}{uncertain_section}

Return ONLY a JSON object of exactly this shape, using exactly these key names:
{{"results": [{{"regionId": "<region number>", "qaStatus": "passed", "qaScore": 0.9, "qaFeedback": "...",
  "directFix": {{"correctedText": "", "suggestedFontSize": 0}},
  "escalation": {{"ocrBad": false, "correctedSourceText": "", "needsReOcr": false,
    "needsManualIntervention": false, "orderBad": false, "suggestedReadingOrderIndex": 0}}}}],
 "uncertainChecks": [{{"regionId": "<region number>", "kind": "not_text", "reason": "..."}}]}}
One "results" entry per region in Region Metadata and one "uncertainChecks" entry per uncertain
region (an empty list when there are none). No other text."""


def _process_qa_vlm(job_data):
    image_id = job_data["imageId"]
    logger.info(f"[QA] Processing VLM vision QA check for image: {image_id}")

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        chapter_id = job_data.get("chapterId")
        page_id = job_data.get("pageId")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[QA] Failed to get image info: {res.status_code}")
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            logger.warning("[QA] No OCR regions found. Skipping VLM QA.")
            _auto_pass_all(job_data)
            return
    except Exception as e:
        logger.error(f"[QA] Error fetching image details: {e}")
        raise

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        logger.error(f"[QA] Error downloading original image: {e}")
        raise

    # Judge exactly the immutable PNG the render callback bound to this QA job. A page-global
    # key could hold another revision's pixels; a job without a binding is a backend bug.
    rendered_bytes = _read_render_artifact(job_data.get("renderArtifact"))

    try:
        # Create side-by-side combined image for VLM comparison
        img1 = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(rendered_bytes)).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size
        combined_width = w1 + w2
        combined_height = max(h1, h2)

        combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (w1, 0))

        # Save to base64
        combined_buf = io.BytesIO()
        combined_img.save(combined_buf, format="JPEG", quality=85)
        combined_base64 = base64.b64encode(combined_buf.getvalue()).decode("utf-8")

        import time

        from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR

        if ENABLE_QA_AUDIT_CACHE:
            try:
                os.makedirs(QA_AUDIT_CACHE_DIR, exist_ok=True)
                audit_path = os.path.join(QA_AUDIT_CACHE_DIR, f"{image_id}_{int(time.time())}.jpg")
                combined_img.save(audit_path, format="JPEG", quality=85)
            except Exception as e:
                logger.error(f"[QA] Failed to write QA audit cache image: {e}")
    except Exception as e:
        logger.error(f"[QA] Error combining images: {e}")
        raise

    qa_regions = _translation_qa_regions(ocr_regions)
    uncertain_regions = [r for r in ocr_regions if r.get("qaStatus") == "cleanup_review"]
    # Every region is named by the number the Reader shows for it, never by UUID.
    uuid_by_label = _region_labels(ocr_regions)
    label_by_uuid = {uuid: label for label, uuid in uuid_by_label.items()}

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)
    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"

    judged = {}
    checked = {}
    call_notes = []

    def ask(model, targets, uncertain):
        prompt = _vlm_qa_prompt(targets, uncertain, label_by_uuid)
        if model is None:
            local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
            if not local_vlm_model:
                return None
            try:
                return try_local_vlm_vision(local_vlm_model, prompt, combined_base64, QA_VLM_JSON_SCHEMA)
            except Exception as e:
                logger.error(f"[QA] VLM QA via Local VLM failed: {e}")
                return None
        return _qa_cloud_vlm(provider, api_key, model, prompt, combined_base64, routing_strategy, QA_VLM_JSON_SCHEMA)

    local_only = provider in ("ollama", "lmstudio")
    chain: list[str | None] = [None] if local_only else []
    if provider and not local_only:
        chain = list(_qa_vlm_model_chain(job_data, provider))
    for index, model in enumerate(chain):
        targets = [r for r in qa_regions if str(r["id"]) not in judged]
        uncertain = [r for r in uncertain_regions if str(r["id"]) not in checked]
        if not targets and not uncertain:
            break
        if index:
            logger.warning(
                f"[QA] Asking fallback VLM '{model}' about {len(targets)} unjudged region(s) "
                f"and {len(uncertain)} uncertain region(s)"
            )
        qa_response = ask(model, targets, uncertain)
        if logger.isEnabledFor(logging.DEBUG) and qa_response:
            logger.debug(f"[QA] Raw VLM Response ({model}): {qa_response}")
        parsed = _parse_qa_response(qa_response)
        if parsed is None:
            call_notes.append(f"{model if model is not None else 'local'}: no usable reply")
            continue
        raw_results, raw_checks = parsed
        answers = _unique_by_region(_normalize_qa_items(raw_results, uuid_by_label))
        for verdict in _sanitize_qa_results(answers, targets, label=f"VLM {model if model is not None else 'local'}"):
            judged.setdefault(verdict["regionId"], verdict)
        uncertain_ids = {str(r["id"]) for r in uncertain}
        for check in _unique_by_region(_normalize_qa_items(raw_checks, uuid_by_label)):
            if check.get("regionId") in uncertain_ids and check.get("kind") in UNCERTAIN_KINDS:
                checked.setdefault(check["regionId"], {**check, "model": model if model is not None else "local"})
        missing = len([r for r in targets if str(r["id"]) not in judged])
        if missing:
            call_notes.append(f"{model if model is not None else 'local'}: {missing} region(s) left unjudged")

    results = [judged[str(r["id"])] for r in qa_regions if str(r["id"]) in judged]
    uncertain_checks = [checked[str(r["id"])] for r in uncertain_regions if str(r["id"]) in checked]
    if call_notes:
        logger.info(f"[QA] VLM calls: {'; '.join(call_notes)}")

    accounting = _qa_accounting(job_data, results, qa_regions)

    if not results:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        logger.warning("[QA] No usable VLM results — reporting no verdict rather than auto-passing.")

    logger.debug(f"[QA] VLM QA results output:\n{log_payload(results)}")

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
        "uncertainChecks": uncertain_checks,
        **accounting,
    }
    from worker.utils.rate_limit import build_cost_payload, format_cost, get_job_costs

    cost_payload = build_cost_payload(get_job_costs())
    if cost_payload:
        callback_payload["cost"] = cost_payload
        total_estimated_cost = cost_payload.get("estimated_cost")
        total_prompt_tokens = cost_payload["prompt_tokens"]
        total_completion_tokens = cost_payload["completion_tokens"]

        cost_str = format_cost(total_estimated_cost)
        if cost_payload["unknown_calls"]:
            cost_str += f" ({cost_payload['unknown_calls']} of {len(cost_payload['breakdown'])} calls unpriced)"

        logger.info(
            f"[QA] VLM QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=backend_headers(), timeout=(5, 30))
        res.raise_for_status()
        logger.debug(f"[QA] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[QA] Failed to post QA callback to backend: {e}")
        raise
