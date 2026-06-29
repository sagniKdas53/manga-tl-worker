import os
import gc
import cv2
import numpy as np
import requests
from PIL import Image

from worker.model_manager import model_manager
from worker.utils.image import downscale_for_ocr


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

            count = min(len(dt_polys), len(rec_texts), len(rec_scores))
            for i in range(count):
                bbox = dt_polys[i]
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                results.append((bbox, rec_texts[i], float(rec_scores[i])))

    except Exception as e:
        print(f"[OCR] Failed parsing PaddleOCR results: {e}", flush=True)

    return results


def try_cloud_ocr(img_crop_bytes, provider, api_key, model):
    import base64

    base64_image = base64.b64encode(img_crop_bytes).decode("utf-8")
    prompt = "Respond ONLY with the text shown in this image. Do not add any explanations, notes, or markdown. If there is no text, respond with empty string."

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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
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
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            elif provider == "anthropic":
                return res_json["content"][0]["text"]
            else:
                return res_json["choices"][0]["message"]["content"]
        else:
            print(
                f"[OCR Redo] Cloud OCR error {res.status_code} from provider={provider}",
                flush=True,
            )
    except Exception as e:
        print(f"[OCR Redo] Cloud OCR HTTP post failed: {e}", flush=True)
    return None


def perform_redo_ocr(img_crop_bytes, lang):
    provider = (
        os.environ.get("MODEL_PROVIDER", os.environ.get("LLM_PROVIDER", "none"))
        .lower()
        .strip()
    )
    api_key = os.environ.get("API_KEY", os.environ.get("LLM_API_KEY", ""))
    model = (
        os.environ.get("PREFERRED_LLM_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
    )

    # Try Cloud AI OCR if configured
    if api_key and provider in ("openai", "openrouter", "gemini", "anthropic"):
        try:
            print(
                f"[OCR Redo] Trying Cloud AI OCR with provider '{provider}'...",
                flush=True,
            )
            text = try_cloud_ocr(img_crop_bytes, provider, api_key, model)
            if text and len(text.strip()) > 0:
                print(
                    f"[OCR Redo] Cloud AI OCR Success (chars={len(text.strip())})",
                    flush=True,
                )
                return text.strip(), 1.0
        except Exception as e:
            print(f"[OCR Redo] Cloud AI OCR failed: {e}", flush=True)

    # Try PP-OCRv5 first — use the lazy-init reader for the region's language
    _redo_paddle_reader = model_manager.get_paddle_ocr_reader(lang)
    if _redo_paddle_reader is not None:
        try:
            print("[OCR Redo] Trying local PP-OCRv5...", flush=True)
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
                    text = " ".join([line[1] for line in parsed_crop_results])
                    confidence = float(
                        np.mean([line[2] for line in parsed_crop_results])
                    )
                    print(
                        f"[OCR Redo] PP-OCRv5 Success: '{text}' (conf={confidence})",
                        flush=True,
                    )
                    return text.strip(), confidence
        except Exception as e:
            print(f"[OCR Redo] PP-OCRv5 failed: {e}", flush=True)

    # Fallback to local MangaOCR if initialized
    manga_reader = model_manager.get_manga_ocr_reader()
    if manga_reader is not None:
        try:
            print("[OCR Redo] Trying local MangaOCR...", flush=True)
            nparr = np.frombuffer(img_crop_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            del nparr

            crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            manga_text = manga_reader(pil_img)
            if manga_text and len(manga_text.strip()) > 0:
                print(f"[OCR Redo] Local MangaOCR Success: '{manga_text}'", flush=True)
                return manga_text.strip(), 1.0
        except Exception as e:
            print(f"[OCR Redo] Local MangaOCR failed: {e}", flush=True)

    # Fallback to EasyOCR
    easy_reader = model_manager.get_easy_ocr_reader()
    if easy_reader is not None:
        try:
            print("[OCR Redo] Trying local EasyOCR...", flush=True)
            crop_results = easy_reader.readtext(img_crop_bytes)
            if crop_results:
                text = " ".join([res[1] for res in crop_results])
                confidence = float(np.mean([res[2] for res in crop_results]))
                print(
                    f"[OCR Redo] Local EasyOCR Success: '{text}' (conf={confidence})",
                    flush=True,
                )
                return text, confidence
        except Exception as e:
            print(f"[OCR Redo] Local EasyOCR failed: {e}", flush=True)

    return "", 0.0
