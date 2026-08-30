import gc
import json
import logging

import cv2
import numpy as np
import requests

from worker.config import redact
from worker.model_manager import model_manager
from worker.utils.image import downscale_for_ocr

logger = logging.getLogger(__name__)

OCR_REFUSAL_PATTERNS = [
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "as a language model",
    "unable to",
    "not able to",
    "cannot process",
    "cannot fulfill",
    "not capable",
]


def is_valid_ocr_text(text):
    if not text or not text.strip():
        return False
    text_lower = text.strip().lower()
    for pattern in OCR_REFUSAL_PATTERNS:
        if pattern in text_lower:
            logger.warning(f"[OCR] Rejected OCR response: matches refusal pattern '{pattern}'")
            return False
    return True


OCR_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["text"],
}


def parse_paddle_ocr_results(raw_results):
    results = []
    if raw_results is None:
        return results

    try:
        if not isinstance(raw_results, list):
            raw_results = [raw_results]

        for result in raw_results:
            dt_polys = result.get("dt_polys", [])
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])

            # Support detection-only mode
            if dt_polys and not rec_texts:
                rec_texts = [""] * len(dt_polys)
                rec_scores = [1.0] * len(dt_polys)

            count = min(len(dt_polys), len(rec_texts), len(rec_scores))
            for i in range(count):
                bbox = dt_polys[i]
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                results.append((bbox, rec_texts[i], float(rec_scores[i])))

    except Exception as e:
        logger.error(f"[OCR] Failed parsing PaddleOCR results: {e}")

    return results


def _record_cloud_ocr_cost(res_json, provider, model):
    """Record a cloud OCR call against the current job.

    This path posts to the provider directly instead of going through LLMClient, so nothing else
    records it — every cloud QA re-OCR and region-redo OCR call was invisible to cost accounting,
    and the callback simply omitted the cost object because the job's list was empty.
    """
    from worker.utils.rate_limit import record_llm_call

    try:
        cache_write_tokens = 0
        authoritative_cost = None

        if provider == "gemini":
            usage = res_json.get("usageMetadata") or {}
            cached_tokens = usage.get("cachedContentTokenCount") or 0
            prompt_tokens = usage.get("promptTokenCount") or 0
            completion_tokens = usage.get("candidatesTokenCount") or 0
        elif provider == "anthropic":
            usage = res_json.get("usage") or {}
            cached_tokens = usage.get("cache_read_input_tokens") or 0
            cache_write_tokens = usage.get("cache_creation_input_tokens") or 0
            # Normalised onto the OpenAI convention, as in llm_client._parse_response.
            prompt_tokens = (usage.get("input_tokens") or 0) + cached_tokens + cache_write_tokens
            completion_tokens = usage.get("output_tokens") or 0
        else:
            usage = res_json.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            authoritative_cost = usage.get("cost")

        record_llm_call(
            model,
            prompt_tokens,
            completion_tokens,
            provider=provider,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            generation_id=res_json.get("id") or "",
            upstream_provider=res_json.get("provider") or "",
            model_resolved=res_json.get("model") or "",
            stage="ocr",
            authoritative_cost=authoritative_cost,
        )
    except Exception as e:
        # Never let accounting break OCR itself.
        logger.warning(f"[OCR] Could not record cloud OCR cost: {redact(e)}")


def try_cloud_ocr(img_crop_bytes, provider, api_key, model, qa_feedback=None, routing_strategy=None):
    import base64

    base64_image = base64.b64encode(img_crop_bytes).decode("utf-8")

    feedback_instruction = ""
    if qa_feedback:
        if qa_feedback.lower() == "user_rejected":
            feedback_instruction = (
                " The user rejected the previous OCR result. Please provide a clean, accurate extraction."
            )
        else:
            feedback_instruction = f" The QA reviewer rejected the previous extraction with this feedback: '{qa_feedback}'. Please fix the issue."

    prompt = (
        "Respond with a JSON object containing the text shown in this image "
        "and your confidence score. Use the format: "
        '{"text": "<extracted text>", "confidence": <0.0-1.0>}. '
        "If the text is a sound effect (SFX), gibberish, an author handle, or already completely in English, return an empty string for text. "
        'If there is no valid text to extract, use {"text": "", "confidence": 0.0}. '
        "Do not add any explanations or notes outside the JSON."
        f"{feedback_instruction}"
    )

    url = ""
    headers = {}
    payload = {}

    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ocr_result",
                    "schema": OCR_SINGLE_SCHEMA,
                    "strict": True,
                },
            },
        }
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "google/gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ocr_result",
                    "schema": OCR_SINGLE_SCHEMA,
                    "strict": True,
                },
            },
            "plugins": [{"id": "response-healing"}],
            # Same reason as llm_client: ask OpenRouter what the call actually cost rather than
            # inferring it from a local rate table.
            "usage": {"include": True},
        }
    elif provider == "gemini":
        gemini_model = model or "gemini-1.5-flash"
        if "/" not in gemini_model:
            gemini_model = f"models/{gemini_model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{gemini_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image,
                            }
                        },
                    ]
                }
            ]
        }
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
    else:
        return None

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=(10, 45))
        if res.status_code == 200:
            res_json = res.json()
            _record_cloud_ocr_cost(res_json, provider, model)
            if provider == "gemini":
                raw = res_json["candidates"][0]["content"]["parts"][0]["text"]
            elif provider == "anthropic":
                raw = res_json["content"][0]["text"]
            else:
                raw = res_json["choices"][0]["message"]["content"]

            try:
                parsed = json.loads(raw.strip())
                text = parsed.get("text", "")
                confidence = float(parsed.get("confidence", 1.0))
                return text.strip(), min(max(confidence, 0.0), 1.0)
            except (json.JSONDecodeError, ValueError, TypeError):
                return raw.strip(), 1.0
        else:
            logger.error(f"[OCR Redo] Cloud OCR error {res.status_code} from provider={provider}")
    except Exception as e:
        logger.error(f"[OCR Redo] Cloud OCR HTTP post failed: {redact(e)}")
    return None


def perform_redo_ocr(img_crop_bytes, lang, qa_feedback=None):
    from worker.config import OCR_CONFIG

    provider = OCR_CONFIG.provider
    api_key = OCR_CONFIG.resolve_key()
    model = OCR_CONFIG.vlm_model

    # Try Cloud AI OCR if configured
    if api_key and provider in ("openai", "openrouter", "gemini", "anthropic"):
        models_to_try = []
        if model:
            models_to_try.append(model)
        for m in getattr(OCR_CONFIG, "vlm_model_list", []):
            if m not in models_to_try:
                models_to_try.append(m)
        if not models_to_try:
            models_to_try.append("")

        for current_model in models_to_try:
            try:
                logger.debug(
                    f"[OCR Redo] Trying Cloud AI OCR with provider '{provider}' and model '{current_model}'..."
                )
                result = try_cloud_ocr(img_crop_bytes, provider, api_key, current_model, qa_feedback)
                if result:
                    text, confidence = result
                    if text and is_valid_ocr_text(text):
                        logger.debug(
                            f"[OCR Redo] Cloud AI OCR Success using '{current_model}': '{text}' (conf={confidence})"
                        )
                        return text, confidence
            except Exception as e:
                logger.error(f"[OCR Redo] Cloud AI OCR with model '{current_model}' failed: {redact(e)}")

    # Try local PaddleOCR first — the lazy-init reader resolves a det/rec pair for the region's
    # language, so a Korean region redone here gets PP-OCRv5 rather than the PP-OCRv6 pair that has
    # no Hangul charset. No model id is threaded through: redo has no per-job model choice, and the
    # catalog default already picks something that can read the script.
    _redo_paddle_reader = None
    if not api_key or provider == "paddleocr":
        _redo_paddle_reader = model_manager.get_paddle_ocr_reader(lang)

    if _redo_paddle_reader is not None:
        try:
            logger.debug("[OCR Redo] Trying local PaddleOCR...")
            nparr = np.frombuffer(img_crop_bytes, np.uint8)
            img_crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            del nparr
            if img_crop is not None:
                img_crop, _ = downscale_for_ocr(img_crop, max_dim=1024)
                crop_results = _redo_paddle_reader.predict(img_crop)
                del img_crop
                gc.collect()
                parsed_crop_results = parse_paddle_ocr_results(crop_results)
                if parsed_crop_results:
                    text = " ".join(line[1] for line in parsed_crop_results if line[1].strip())
                    if not is_valid_ocr_text(text):
                        logger.warning(f"[OCR Redo] PaddleOCR result rejected by validation: '{text}'")
                        text = ""
                    confidence = float(np.mean([line[2] for line in parsed_crop_results]))
                    logger.debug(f"[OCR Redo] PaddleOCR Success: '{text}' (conf={confidence})")
                    return text.strip(), confidence
        except Exception as e:
            logger.error(f"[OCR Redo] PaddleOCR failed: {e}")

    return "", 0.0
