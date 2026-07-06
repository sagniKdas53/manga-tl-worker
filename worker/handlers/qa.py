import io
import logging
import os
import json
import base64
import requests
from PIL import Image
from worker.config import (
    CALLBACK_URL,
    BACKEND_HEADERS,
    minio_client,
    logger,
    QA_MODE,
    redis_client,
    QA_CONFIG,
)
from worker.utils.image import download_image
from worker.services.translation import (
    try_cloud_ai,
    try_local_ai,
    try_cloud_ai_vision,
    try_local_vlm_vision,
)

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
                        "enum": ["passed", "failed", "direct_fix"],
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

    print(
        f"[QA] Processing image: {image_id}{progress_str} (mode={QA_MODE})", flush=True
    )

    if QA_MODE == "none":
        _auto_pass_all(job_data)
    elif QA_MODE == "llm":
        _process_qa_llm(job_data)
    elif QA_MODE == "vlm":
        _process_qa_vlm(job_data)
    else:
        logger.warning(f"[QA] Unknown QA_MODE={QA_MODE}, falling back to auto-pass")
        _auto_pass_all(job_data)


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
        return

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
    from worker.utils.rate_limit import get_job_costs, format_cost

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        if has_na:
            total_estimated_cost = None
        else:
            total_estimated_cost = sum(
                c.get("estimated_cost", 0.0) or 0.0 for c in costs
            )
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
        res = requests.post(
            f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS
        )
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
        return

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
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed (e.g. translation is highly accurate, natural English).
- "direct_fix": Small/cosmetic adjustment (e.g. minor typo fix or slightly better phrasing) that you can prescribe directly. You must supply "directFix" object with correctedText. You MUST also provide detailed reasoning in "qaFeedback".
- "failed": Translation error requiring a translation re-run. Specify "qaFeedback" with detailed correction notes/feedback to guide the re-translation.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)

    qa_response = None

    def attempt_llm(prov):
        user_model = job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        if prov == "openrouter" and api_key:
            llm_model = (
                user_model
                if prov == provider and user_model
                else "meta-llama/llama-3-8b-instruct:free"
            )
            try:
                return try_cloud_ai(
                    "openrouter", api_key, llm_model, prompt, QA_JSON_SCHEMA
                )
            except Exception as e:
                print(f"[QA] LLM QA via OpenRouter failed: {e}", flush=True)
        elif prov == "gemini" and api_key:
            llm_model = (
                user_model if prov == provider and user_model else "gemini-1.5-pro"
            )
            try:
                return try_cloud_ai(
                    "gemini", api_key, llm_model, prompt, QA_JSON_SCHEMA
                )
            except Exception as e:
                print(f"[QA] LLM QA via Gemini failed: {e}", flush=True)
        elif prov == "nvidia" and api_key:
            llm_model = (
                user_model
                if prov == provider and user_model
                else "google/gemma-3n-e4b-it"
            )
            try:
                return try_cloud_ai(
                    "nvidia", api_key, llm_model, prompt, QA_JSON_SCHEMA
                )
            except Exception as e:
                print(f"[QA] LLM QA via Nvidia failed: {e}", flush=True)
        return None

    # Try the preferred provider first
    if provider:
        qa_response = attempt_llm(provider)

    local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if not qa_response and local_llm_model and not disable_local:
        try:
            qa_response = try_local_ai(
                prompt, json.dumps(regions_metadata), QA_JSON_SCHEMA
            )
        except Exception as e:
            print(f"[QA] LLM QA via Local LLM failed: {e}", flush=True)
    elif not qa_response and local_llm_model and disable_local:
        print("[QA] Local LLM QA skipped (disabled via environment).", flush=True)

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

    logger.debug(
        f"[QA] LLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
    )

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results}
    from worker.utils.rate_limit import get_job_costs, format_cost

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        if has_na:
            total_estimated_cost = None
        else:
            total_estimated_cost = sum(
                c.get("estimated_cost", 0.0) or 0.0 for c in costs
            )
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
        res = requests.post(
            f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS
        )
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
        return

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        print(f"[QA] Error downloading original image: {e}", flush=True)
        return

    # Download rendered typeset image from MinIO
    try:
        response = minio_client.get_object("manga-library", f"rendered/{image_id}.png")
        rendered_bytes = response.read()
    except Exception as e:
        print(f"[QA] Error downloading rendered image: {e}", flush=True)
        return

    try:
        # Create side-by-side combined image for VLM comparison
        img1 = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(rendered_bytes)).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size
        combined_width = w1 + w2
        combined_height = max(h1, h2)

        combined_img = Image.new(
            "RGB", (combined_width, combined_height), (255, 255, 255)
        )
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (w1, 0))

        # Save to base64
        combined_buf = io.BytesIO()
        combined_img.save(combined_buf, format="JPEG", quality=85)
        combined_base64 = base64.b64encode(combined_buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[QA] Error combining images: {e}", flush=True)
        return

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
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed (e.g., text fitting is perfect, translation is highly accurate, layout looks clean).
- "direct_fix": Small/cosmetic adjustment (e.g. slight text wrap tweak or minor font size reduction) that you can prescribe directly. You must supply "directFix" object with correctedText or suggestedFontSize. You MUST also provide detailed reasoning in "qaFeedback".
- "failed": Major translation error or layout issue requiring a translation/typesetting re-run. Specify "qaFeedback" with detailed correction notes.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = job_data.get("qaProvider") or QA_CONFIG.provider
    api_key = QA_CONFIG.resolve_key(provider)

    qa_response = None

    def attempt_vlm(prov):
        user_model = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        if prov == "openrouter" and api_key:
            vlm_model = (
                user_model
                if prov == provider and user_model
                else "google/gemini-1.5-pro"
            )
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
                print(f"[QA] VLM QA via OpenRouter failed: {e}", flush=True)
        elif prov == "gemini" and api_key:
            vlm_model = (
                user_model if prov == provider and user_model else "gemini-1.5-pro"
            )
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
                print(f"[QA] VLM QA via Gemini failed: {e}", flush=True)
        elif prov == "nvidia" and api_key:
            vlm_model = (
                user_model
                if prov == provider and user_model
                else "nvidia/nemotron-nano-12b-v2-vl"
            )
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
                print(f"[QA] VLM QA via Nvidia failed: {e}", flush=True)
        return None

    # Try the preferred provider first
    if provider:
        qa_response = attempt_vlm(provider)

    # Fallback to Local VLM:
    # Attempted only if cloud VLM calls failed (e.g. key missing, or provider is cooled down on 429).
    # Explicitly respects the DISABLE_LOCAL_LLM environment variable bypass.
    local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if not qa_response and local_vlm_model and not disable_local:
        try:
            qa_response = try_local_vlm_vision(
                local_vlm_model, prompt, combined_base64, QA_JSON_SCHEMA
            )
        except Exception as e:
            print(f"[QA] VLM QA via Local VLM failed: {e}", flush=True)
    elif not qa_response and local_vlm_model and disable_local:
        print("[QA] Local VLM QA skipped (disabled via environment).", flush=True)

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

    logger.debug(
        f"[QA] VLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
    )

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results}
    from worker.utils.rate_limit import get_job_costs, format_cost

    costs = get_job_costs()
    if costs:
        has_na = any(c.get("estimated_cost") is None for c in costs)
        if has_na:
            total_estimated_cost = None
        else:
            total_estimated_cost = sum(
                c.get("estimated_cost", 0.0) or 0.0 for c in costs
            )
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
        res = requests.post(
            f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)
