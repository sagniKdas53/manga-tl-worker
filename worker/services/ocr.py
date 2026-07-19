import gc
import json

import cv2
import numpy as np
import requests

from worker.model_manager import model_manager
from worker.utils.image import downscale_for_ocr

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
            print(
                f"[OCR] Rejected OCR response: matches refusal pattern '{pattern}'",
                flush=True,
            )
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
        print(f"[OCR] Failed parsing PaddleOCR results: {e}", flush=True)

    return results


def try_cloud_ocr(img_crop_bytes, provider, api_key, model, qa_feedback=None):
    import base64

    base64_image = base64.b64encode(img_crop_bytes).decode("utf-8")
    prompt = (
        "Respond with a JSON object containing the text shown in this image "
        "and your confidence score. Use the format: "
        '{"text": "<extracted text>", "confidence": <0.0-1.0>}. '
        'If there is no text, or if the text is a sound effect (SFX), gibberish, an author handle, or already completely in English, return an empty string for text: {"text": "", "confidence": 0.0}. '
        "Do not add any explanations or notes outside the JSON."
    )

    if qa_feedback:
        if qa_feedback == "user_rejected":
            prompt += "\nNote: The user rejected the previous OCR result. Please provide a clean, accurate extraction."
        else:
            prompt += f"\nNote: The QA reviewer rejected the previous extraction with this feedback: '{qa_feedback}'. Please fix the issue."

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
        res = requests.post(url, json=payload, headers=headers, timeout=12)
        if res.status_code == 200:
            res_json = res.json()
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
            print(
                f"[OCR Redo] Cloud OCR error {res.status_code} from provider={provider}",
                flush=True,
            )
    except Exception as e:
        print(f"[OCR Redo] Cloud OCR HTTP post failed: {e}", flush=True)
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
                print(
                    f"[OCR Redo] Trying Cloud AI OCR with provider '{provider}' and model '{current_model}'...",
                    flush=True,
                )
                result = try_cloud_ocr(img_crop_bytes, provider, api_key, current_model, qa_feedback=qa_feedback)
                if result:
                    text, confidence = result
                    if text and is_valid_ocr_text(text):
                        print(
                            f"[OCR Redo] Cloud AI OCR Success using '{current_model}': '{text}' (conf={confidence})",
                            flush=True,
                        )
                        return text, confidence
            except Exception as e:
                print(
                    f"[OCR Redo] Cloud AI OCR with model '{current_model}' failed: {e}",
                    flush=True,
                )

    # Try local PaddleOCR first — use the lazy-init reader for the region's language
    _redo_paddle_reader = model_manager.get_paddle_ocr_reader(lang)
    if _redo_paddle_reader is not None:
        try:
            print("[OCR Redo] Trying local PaddleOCR...", flush=True)
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
                        print(
                            f"[OCR Redo] PaddleOCR result rejected by validation: '{text}'",
                            flush=True,
                        )
                        text = ""
                    confidence = float(np.mean([line[2] for line in parsed_crop_results]))
                    print(
                        f"[OCR Redo] PaddleOCR Success: '{text}' (conf={confidence})",
                        flush=True,
                    )
                    return text.strip(), confidence
        except Exception as e:
            print(f"[OCR Redo] PaddleOCR failed: {e}", flush=True)

    return "", 0.0
