import base64
import io
import json
import logging
import os

import requests
from PIL import Image

from worker.config import (
    BACKEND_HEADERS,
    CALLBACK_URL,
    QA_CONFIG,
    QA_MODE,
    is_usable_model,
    logger,
    minio_client,
    redis_client,
)
from worker.provider_config import get_config_loader
from worker.services.translation import (
    try_cloud_ai,
    try_cloud_ai_vision,
    try_local_ai,
    try_local_vlm_vision,
)
from worker.utils.image import download_image

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
        print(
            f"[QA] Discarded {len(discarded)} unusable {label} result(s): {'; '.join(discarded[:5])}"
            f"{' ...' if len(discarded) > 5 else ''}",
            flush=True,
        )

    missing = len(known_ids) - len(kept) if known_ids else 0
    if kept and missing > 0:
        print(
            f"[QA] {label} returned a verdict for {len(kept)}/{len(known_ids)} regions — "
            "the response was probably truncated.",
            flush=True,
        )

    # A `failed` verdict with no escalation gives the backend nothing to act on, so it falls back
    # to re-translating the same source text. Surface it; the schema now requires the object.
    unactionable = [r["regionId"] for r in kept if r.get("qaStatus") == "failed" and not r.get("escalation")]
    if unactionable:
        print(
            f"[QA] {len(unactionable)} failed region(s) carry no escalation block; re-OCR cannot be routed for them.",
            flush=True,
        )

    return kept


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
        print(f"[QA] LLM QA via '{prov}' with model '{model}' failed: {e}", flush=True)
        return None


def _qa_cloud_vlm(prov, api_key, user_model, prompt, base64_image, routing_strategy):
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
            QA_JSON_SCHEMA,
            routing_strategy=routing_strategy,
        )
    except Exception as e:
        print(f"[QA] VLM QA via '{prov}' with model '{model}' failed: {e}", flush=True)
        return None


def process_qa(job_data):
    from worker.utils.rate_limit import reset_job_costs

    reset_job_costs()
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
        print(
            f"[QA] AUTO mode resolved to '{qa_mode_resolved}' (provider={provider}, "
            f"vlm={'yes' if has_vlm else 'no'}, llm={'yes' if has_llm else 'no'})",
            flush=True,
        )

    print(
        f"[QA] Processing image: {image_id}{progress_str} (mode={qa_mode_resolved})",
        flush=True,
    )

    if job_data.get("qaAttempt", 0) > 0:
        print(
            "[QA] Skipping QA because qaAttempt > 0 (One pass only to prevent loops)",
            flush=True,
        )
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
    print(f"[QA] Processing Hybrid QA check for page: {page_id or image_id}", flush=True)

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
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get page/image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            print("[QA] No OCR regions found. Skipping Hybrid QA.", flush=True)
            _auto_pass_all(job_data)
            return
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        raise

    # Build region metadata list to seed the LLM
    regions_metadata = []
    for r in ocr_regions:
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

    logger.debug(
        f"[QA] LLM QA input metadata (regions_metadata) for Hybrid pass:\n{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}"
    )

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

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)
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
                    print(
                        f"[QA] Falling back to global default model '{global_model}'...",
                        flush=True,
                    )
                    qa_response = attempt_llm(provider, global_model)
                else:
                    print(
                        "[QA] No fallback applied (global provider different or model identical).",
                        flush=True,
                    )

    local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    is_explicit_local = provider in ("ollama", "lmstudio")

    if not qa_response and local_llm_model and (is_explicit_local or not disable_local):
        try:
            qa_response = try_local_ai(prompt, json.dumps(regions_metadata), QA_JSON_SCHEMA)
        except Exception as e:
            print(f"[QA] LLM QA via Local LLM failed: {e}", flush=True)

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
            print(
                f"[QA] Failed to parse LLM response: {e}. Raw response: {qa_response}",
                flush=True,
            )

    # Call backend prepare endpoint to apply fixes and set visibility
    prepare_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}/qa-hybrid-prepare")
    try:
        prep_res = requests.post(
            prepare_url,
            json={"pageId": job_data.get("pageId"), "qaResults": results},
            headers=BACKEND_HEADERS,
        )
        print(
            f"[QA] Hybrid QA preparation status code: {prep_res.status_code}",
            flush=True,
        )
    except Exception as e:
        print(f"[QA] Failed to post Hybrid QA preparation: {e}", flush=True)
        raise

    # Trigger render inline
    from worker.handlers.render import render_image_core

    render_ok = render_image_core(image_id)
    if not render_ok:
        print("[QA] Rendering failed during Hybrid QA. Aborting.", flush=True)
        return

    # Now run VLM check on updated render
    try:
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get updated image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            print("[QA] No OCR regions found. Skipping VLM QA.", flush=True)
            _auto_pass_all(job_data)
            return
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        raise

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        print(f"[QA] Error downloading original image: {e}", flush=True)
        raise

    # Download rendered typeset image from MinIO
    try:
        response = minio_client.get_object("manga-library", f"rendered/{image_id}.png")
        rendered_bytes = response.read()
    except Exception as e:
        print(f"[QA] Error downloading rendered image: {e}", flush=True)
        raise

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
                print(f"[QA] Failed to write QA audit cache image: {e}", flush=True)
    except Exception as e:
        print(f"[QA] Error combining images: {e}", flush=True)
        raise

    # Build region metadata list to seed the VLM
    regions_metadata_vlm = []
    for r in ocr_regions:
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
                    print(
                        f"[QA] Falling back to global default VLM model '{global_model}'...",
                        flush=True,
                    )
                    qa_response_vlm = attempt_vlm(provider, global_model)
                else:
                    print(
                        "[QA] No fallback applied (global provider different or model identical).",
                        flush=True,
                    )

    local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()

    if not qa_response_vlm and local_vlm_model and (is_explicit_local or not disable_local):
        try:
            qa_response_vlm = try_local_vlm_vision(local_vlm_model, prompt_vlm, combined_base64, QA_JSON_SCHEMA)
        except Exception as e:
            print(f"[QA] VLM QA via Local VLM failed: {e}", flush=True)

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
            print(
                f"[QA] Failed to parse VLM response: {e}. Raw response: {qa_response_vlm}",
                flush=True,
            )

    results_vlm = _sanitize_qa_results(results_vlm, ocr_regions, label="VLM")

    if not results_vlm:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        print("[QA] No usable VLM results — reporting no verdict rather than auto-passing.", flush=True)

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results_vlm,
    }
    from worker.utils.rate_limit import format_cost, get_job_costs

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        total_estimated_cost = None if has_na else sum(c.get("estimated_cost", 0.0) or 0.0 for c in costs)
        total_prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in costs)
        total_completion_tokens = sum(c.get("completion_tokens", 0) or 0 for c in costs)

        cost_payload = {
            "currency": "USD",
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "breakdown": costs,
        }
        if total_estimated_cost is not None:
            cost_payload["estimated_cost"] = total_estimated_cost
        callback_payload["cost"] = cost_payload

        cost_str = format_cost(total_estimated_cost)

        logger.info(
            f"[QA] Hybrid QA VLM pass estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS)
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)


def _auto_pass_all(job_data):
    image_id = job_data["imageId"]
    print(f"[QA] Skipping QA (QA_MODE=none) for image: {image_id}", flush=True)

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
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        raise

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

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
    }
    from worker.utils.rate_limit import format_cost, get_job_costs

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        total_estimated_cost = None if has_na else sum(c.get("estimated_cost", 0.0) or 0.0 for c in costs)
        total_prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in costs)
        total_completion_tokens = sum(c.get("completion_tokens", 0) or 0 for c in costs)

        cost_payload = {
            "currency": "USD",
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "breakdown": costs,
        }
        if total_estimated_cost is not None:
            cost_payload["estimated_cost"] = total_estimated_cost
        callback_payload["cost"] = cost_payload

        cost_str = format_cost(total_estimated_cost)

        logger.info(
            f"[QA] Auto-pass QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS)
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)


def _process_qa_llm(job_data):
    image_id = job_data["imageId"]
    print(f"[QA] Processing text-only LLM QA check for image: {image_id}", flush=True)

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
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            print("[QA] No OCR regions found. Skipping LLM QA.", flush=True)
            _auto_pass_all(job_data)
            return
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        raise

    # Build region metadata list to seed the LLM
    regions_metadata = []
    for r in ocr_regions:
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

    logger.debug(
        f"[QA] LLM QA input metadata (regions_metadata):\n{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}"
    )

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

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)
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
                print(f"[QA] LLM QA via Local LLM failed: {e}", flush=True)
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
                        print(
                            f"[QA] Falling back to global default LLM model '{global_model}'...",
                            flush=True,
                        )
                        qa_response = attempt_llm(provider, global_model)
                    else:
                        print(
                            "[QA] No fallback applied (global provider different or model identical).",
                            flush=True,
                        )

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
            print(
                f"[QA] Failed to parse LLM response: {e}. Raw response: {qa_response}",
                flush=True,
            )

    results = _sanitize_qa_results(results, ocr_regions, label="LLM")

    if not results:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        print("[QA] No usable LLM results — reporting no verdict rather than auto-passing.", flush=True)

    logger.debug(f"[QA] LLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}")

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
    }
    from worker.utils.rate_limit import format_cost, get_job_costs

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        total_estimated_cost = None if has_na else sum(c.get("estimated_cost", 0.0) or 0.0 for c in costs)
        total_prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in costs)
        total_completion_tokens = sum(c.get("completion_tokens", 0) or 0 for c in costs)

        cost_payload = {
            "currency": "USD",
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "breakdown": costs,
        }
        if total_estimated_cost is not None:
            cost_payload["estimated_cost"] = total_estimated_cost
        callback_payload["cost"] = cost_payload

        cost_str = format_cost(total_estimated_cost)

        logger.info(
            f"[QA] LLM QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS)
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)


def _process_qa_vlm(job_data):
    image_id = job_data["imageId"]
    print(f"[QA] Processing VLM vision QA check for image: {image_id}", flush=True)

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
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        if not ocr_regions:
            print("[QA] No OCR regions found. Skipping VLM QA.", flush=True)
            _auto_pass_all(job_data)
            return
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        raise

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        print(f"[QA] Error downloading original image: {e}", flush=True)
        raise

    # Download rendered typeset image from MinIO
    try:
        response = minio_client.get_object("manga-library", f"rendered/{image_id}.png")
        rendered_bytes = response.read()
    except Exception as e:
        print(f"[QA] Error downloading rendered image: {e}", flush=True)
        raise

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
                print(f"[QA] Failed to write QA audit cache image: {e}", flush=True)
    except Exception as e:
        print(f"[QA] Error combining images: {e}", flush=True)
        raise

    # Build region metadata list to seed the VLM
    regions_metadata = []
    for r in ocr_regions:
        regions_metadata.append(
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

    logger.debug(
        f"[QA] VLM QA input metadata (regions_metadata):\n{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}"
    )

    prompt = f"""You are an expert Japanese-to-English manga translator and typesetting reviewer. Given the original Japanese manga page (left) and the English typeset page (right), verify: (1) OCR accuracy by comparing visible Japanese text against transcription, (2) Translation quality and natural English, (3) Typesetting quality — text fitting, overflow, readability.

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
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)
    routing_strategy = job_data.get("routingStrategy") or "lowest-cost"

    qa_response = None

    def attempt_vlm(prov, model_override=None):
        user_model = model_override or job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        return _qa_cloud_vlm(prov, api_key, user_model, prompt, combined_base64, routing_strategy)

    local_only = provider in ("ollama", "lmstudio")
    if local_only:
        local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
        if local_vlm_model:
            try:
                qa_response = try_local_vlm_vision(local_vlm_model, prompt, combined_base64, QA_JSON_SCHEMA)
            except Exception as e:
                print(f"[QA] VLM QA via Local VLM failed: {e}", flush=True)
    else:
        # Try the preferred provider first
        if provider:
            user_model = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
            qa_response = attempt_vlm(provider, user_model)

            if not qa_response:
                use_fallback_models = job_data.get("useFallbackModels", True)
                if use_fallback_models:
                    global_model = QA_CONFIG.vlm_model
                    global_provider = QA_CONFIG.provider
                    if global_provider == provider and global_model and global_model != user_model:
                        print(
                            f"[QA] Falling back to global default VLM model '{global_model}'...",
                            flush=True,
                        )
                        qa_response = attempt_vlm(provider, global_model)
                    else:
                        print(
                            "[QA] No fallback applied (global provider different or model identical).",
                            flush=True,
                        )

    # VLM Evaluation Fail-Safe Fallback:
    # If all configured/active VLM options fail to return a parseable response,
    # rather than crashing the worker, we construct a default "passed" result
    # for all regions so the typesetting/translation pipeline can successfully complete.
    results = []
    if logger.isEnabledFor(logging.DEBUG) and qa_response:
        logger.debug(f"[QA] Raw VLM Response: {qa_response}")

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
            print(
                f"[QA] Failed to parse VLM response: {e}. Raw response: {qa_response}",
                flush=True,
            )

    results = _sanitize_qa_results(results, ocr_regions, label="VLM")

    if not results:
        # Deliberately not auto-passing. Fabricating a pass for every region is what made a failed
        # QA call indistinguishable from a clean page; an empty list tells the backend QA did not
        # run, and it records that instead of a verdict.
        print("[QA] No usable VLM results — reporting no verdict rather than auto-passing.", flush=True)

    logger.debug(f"[QA] VLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}")

    # Call backend
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": job_data.get("pageId"),
        "qaResults": results,
    }
    from worker.utils.rate_limit import format_cost, get_job_costs

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        total_estimated_cost = None if has_na else sum(c.get("estimated_cost", 0.0) or 0.0 for c in costs)
        total_prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in costs)
        total_completion_tokens = sum(c.get("completion_tokens", 0) or 0 for c in costs)

        cost_payload = {
            "currency": "USD",
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "breakdown": costs,
        }
        if total_estimated_cost is not None:
            cost_payload["estimated_cost"] = total_estimated_cost
        callback_payload["cost"] = cost_payload

        cost_str = format_cost(total_estimated_cost)

        logger.info(
            f"[QA] VLM QA job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    try:
        res = requests.post(f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS)
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)
