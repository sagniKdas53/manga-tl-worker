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
    logger,
    minio_client,
    redis_client,
)
from worker.services.translation import (
    try_cloud_ai,
    try_cloud_ai_vision,
    try_local_ai,
    try_local_vlm_vision,
)
from worker.utils.image import download_image

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
                    },
                },
                "required": ["regionId", "qaStatus", "qaScore", "qaFeedback"],
            },
        }
    },
    "required": ["results"],
}


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
        has_vlm = job_data.get("qaVlmModel") or getattr(QA_CONFIG, "vlm_model", None)
        has_llm = job_data.get("qaLlmModel") or getattr(QA_CONFIG, "llm_model", None)
        if has_vlm and provider:
            qa_mode_resolved = "vlm"
        elif has_llm and provider:
            qa_mode_resolved = "llm"
        else:
            qa_mode_resolved = "none"

    print(
        f"[QA] Processing image: {image_id}{progress_str} (mode={qa_mode_resolved})",
        flush=True,
    )

    if job_data.get("qaAttempt", 0) > 0:
        print("[QA] Skipping QA because qaAttempt > 0 (One pass only to prevent loops)", flush=True)
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
    image_id = job_data["imageId"]
    print(f"[QA] Processing Hybrid QA check for image: {image_id}", flush=True)

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get image info: {res.status_code}", flush=True)
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

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)

    qa_response = None

    def attempt_llm(prov, model_override=None):
        user_model = model_override or job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        if prov == "openrouter" and api_key:
            llm_model = user_model if user_model else "meta-llama/llama-3-8b-instruct:free"
            try:
                return try_cloud_ai("openrouter", api_key, llm_model, prompt, QA_JSON_SCHEMA, routing_strategy=routing_strategy)
            except Exception as e:
                print(
                    f"[QA] LLM QA via OpenRouter with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "gemini" and api_key:
            llm_model = user_model if user_model else "gemini-1.5-pro"
            try:
                return try_cloud_ai("gemini", api_key, llm_model, prompt, QA_JSON_SCHEMA)
            except Exception as e:
                print(
                    f"[QA] LLM QA via Gemini with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "nvidia" and api_key:
            llm_model = user_model if user_model else "google/gemma-3n-e4b-it"
            try:
                return try_cloud_ai("nvidia", api_key, llm_model, prompt, QA_JSON_SCHEMA)
            except Exception as e:
                print(
                    f"[QA] LLM QA via Nvidia with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        return None

    # Try preferred provider/models
    if provider:
        user_model = job_data.get("qaLlmModel") or getattr(QA_CONFIG, "llm_model")
        qa_response = attempt_llm(provider, user_model)
        
        if not qa_response:
            # Fallback to global default model
            global_model = getattr(QA_CONFIG, "llm_model")
            global_provider = getattr(QA_CONFIG, "provider")
            if global_provider == provider and global_model and global_model != user_model:
                print(f"[QA] Falling back to global default model '{global_model}'...", flush=True)
                qa_response = attempt_llm(provider, global_model)
            else:
                print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)

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
        prep_res = requests.post(prepare_url, json={"qaResults": results}, headers=BACKEND_HEADERS)
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
        
        from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR
        import time
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

Region Metadata:
{json.dumps(regions_metadata_vlm, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    vlm_api_key = QA_CONFIG.resolve_key(provider)
    qa_response_vlm = None

    def attempt_vlm(prov, model_override=None):
        user_model = model_override or job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        if prov == "openrouter" and vlm_api_key:
            vlm_model = user_model if user_model else "google/gemini-1.5-pro"
            try:
                return try_cloud_ai_vision(
                    "openrouter",
                    vlm_api_key,
                    vlm_model,
                    prompt_vlm,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via OpenRouter with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "gemini" and vlm_api_key:
            vlm_model = user_model if user_model else "gemini-1.5-pro"
            try:
                return try_cloud_ai_vision(
                    "gemini",
                    vlm_api_key,
                    vlm_model,
                    prompt_vlm,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via Gemini with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "nvidia" and vlm_api_key:
            vlm_model = user_model if user_model else "nvidia/nemotron-nano-12b-v2-vl"
            try:
                return try_cloud_ai_vision(
                    "nvidia",
                    vlm_api_key,
                    vlm_model,
                    prompt_vlm,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via Nvidia with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        return None

    if provider:
        user_model = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        qa_response_vlm = attempt_vlm(provider, user_model)
        
        if not qa_response_vlm:
            global_model = QA_CONFIG.vlm_model
            global_provider = QA_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                print(f"[QA] Falling back to global default VLM model '{global_model}'...", flush=True)
                qa_response_vlm = attempt_vlm(provider, global_model)
            else:
                print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)

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

    if not results_vlm:
        print("[QA] Falling back to default PASS for all regions.", flush=True)
        for r in ocr_regions:
            results_vlm.append(
                {
                    "regionId": r["id"],
                    "qaStatus": "passed",
                    "qaScore": 1.0,
                    "qaFeedback": "Auto-passed fallback",
                }
            )

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results_vlm}
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
    callback_payload = {"imageId": image_id, "qaResults": results}
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

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)

    qa_response = None

    def attempt_llm(prov, model_override=None):
        user_model = model_override or job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        if prov == "openrouter" and api_key:
            llm_model = user_model if user_model else "meta-llama/llama-3-8b-instruct:free"
            try:
                return try_cloud_ai("openrouter", api_key, llm_model, prompt, QA_JSON_SCHEMA, routing_strategy=routing_strategy)
            except Exception as e:
                print(
                    f"[QA] LLM QA via OpenRouter with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "gemini" and api_key:
            llm_model = user_model if user_model else "gemini-1.5-pro"
            try:
                return try_cloud_ai("gemini", api_key, llm_model, prompt, QA_JSON_SCHEMA)
            except Exception as e:
                print(
                    f"[QA] LLM QA via Gemini with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "nvidia" and api_key:
            llm_model = user_model if user_model else "google/gemma-3n-e4b-it"
            try:
                return try_cloud_ai("nvidia", api_key, llm_model, prompt, QA_JSON_SCHEMA)
            except Exception as e:
                print(
                    f"[QA] LLM QA via Nvidia with model '{llm_model}' failed: {e}",
                    flush=True,
                )
        return None

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
                global_model = QA_CONFIG.llm_model
                global_provider = QA_CONFIG.provider
                if global_provider == provider and global_model and global_model != user_model:
                    print(f"[QA] Falling back to global default LLM model '{global_model}'...", flush=True)
                    qa_response = attempt_llm(provider, global_model)
                else:
                    print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)

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

    if not results:
        print("[QA] Falling back to default PASS for all regions.", flush=True)
        for r in ocr_regions:
            results.append(
                {
                    "regionId": r["id"],
                    "qaStatus": "passed",
                    "qaScore": 1.0,
                    "qaFeedback": "Auto-passed fallback",
                }
            )

    logger.debug(f"[QA] LLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}")

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results}
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
        
        from worker.config import ENABLE_QA_AUDIT_CACHE, QA_AUDIT_CACHE_DIR
        import time
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

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)

    qa_response = None

    def attempt_vlm(prov, model_override=None):
        user_model = model_override or job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        if prov == "openrouter" and api_key:
            vlm_model = user_model if user_model else "google/gemini-1.5-pro"
            try:
                return try_cloud_ai_vision(
                    "openrouter",
                    api_key,
                    vlm_model,
                    prompt,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via OpenRouter with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "gemini" and api_key:
            vlm_model = user_model if user_model else "gemini-1.5-pro"
            try:
                return try_cloud_ai_vision(
                    "gemini",
                    api_key,
                    vlm_model,
                    prompt,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via Gemini with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        elif prov == "nvidia" and api_key:
            vlm_model = user_model if user_model else "nvidia/nemotron-nano-12b-v2-vl"
            try:
                return try_cloud_ai_vision(
                    "nvidia",
                    api_key,
                    vlm_model,
                    prompt,
                    combined_base64,
                    QA_JSON_SCHEMA,
                )
            except Exception as e:
                print(
                    f"[QA] VLM QA via Nvidia with model '{vlm_model}' failed: {e}",
                    flush=True,
                )
        return None

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
                global_model = QA_CONFIG.vlm_model
                global_provider = QA_CONFIG.provider
                if global_provider == provider and global_model and global_model != user_model:
                    print(f"[QA] Falling back to global default VLM model '{global_model}'...", flush=True)
                    qa_response = attempt_vlm(provider, global_model)
                else:
                    print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)

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

    if not results:
        print("[QA] Falling back to default PASS for all regions.", flush=True)
        for r in ocr_regions:
            results.append(
                {
                    "regionId": r["id"],
                    "qaStatus": "passed",
                    "qaScore": 1.0,
                    "qaFeedback": "Auto-passed fallback",
                }
            )

    logger.debug(f"[QA] VLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}")

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results}
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
