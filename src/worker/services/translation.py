import json
import logging
import os
import re
import time
import uuid

import requests

from worker.config import logger
from worker.services.llm_client import PROVIDER_COOLDOWNS, LLMClient, wait_for_cooldown  # noqa: F401
from worker.utils.rate_limit import enforce_rate_limit, estimate_cost  # noqa: F401
from worker.utils.text import clean_translated_text, contains_japanese

LANG_MAP = {
    "ja": "Japanese",
    "en": "English",
    "ko": "Korean",
    "zh-tw": "Traditional Chinese",
    "zh-cn": "Simplified Chinese",
    "zh": "Chinese",
    "auto": "Auto-Detect",
}

TRANSLATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "translation": {"type": "string"},
                    "translationNotes": {"type": "string"},
                    "emotion": {"type": "string"},
                    "tone": {"type": "string"},
                    "translationScore": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["id", "translation"],
            },
        }
    },
    "required": ["translations"],
}

MANGA_TRANSLATION_JSON_SYSTEM_PROMPT = """You are an expert manga translator.
Translate the list of manga text regions into natural English.
These regions appear in reading order. Maintain context, tone, emotion, and relationships between speakers.
Ensure consistent character names across the chapter. Maintain a consistent tone and emotion, matching the context of previous dialogue.
Inject necessary context naturally into the translation when handling ambiguous Japanese pronouns or subjects.

Region type handling:
- "speech": Translate as natural dialogue.
- "narration": Translate as third-person narrative prose.
- "sfx": Transliterate the sound effect AND provide an English equivalent in parentheses (e.g. "DOKAA (WHAM)").
- "caption": Translate as editorial/scene-setting text.
- "sign": Translate literally, noting it's environmental text.

If multiple regions share the same conversationGroup, treat them as a continuous dialogue exchange and ensure coherent flow.

NEVER include romanized text, pinyin, romaji, or pronunciation guides. Return ONLY the target-language translation.
BAD: "Yào chūfā le o (About to depart!)"
GOOD: "About to depart!"
BAD: "ERUFU (ELF!)"
GOOD: "ELF!"

Return ONLY valid JSON format conforming to the requested schema. No conversational prefix, suffix, or markdown formatting."""

MANGA_TRANSLATION_SYSTEM_PROMPT = """You are an expert manga translator.

Translate Japanese manga dialogue into natural English.

Rules:
- Keep names unchanged and ensure consistent character names across the chapter.
- Preserve tone and emotion, matching the context of previous dialogue.
- Inject necessary context naturally into the translation when handling ambiguous Japanese pronouns or subjects.
- Do not explain.
- Do not explain.
- Do not add notes.
- Do not add quotation marks.
- NEVER include romanized text, pinyin, romaji, or pronunciation guides.
- Return only the translated text.

BAD: "Yào chūfā le o (About to depart!)"
GOOD: "About to depart!"
BAD: "ERUFU (ELF!)"
GOOD: "ELF!"
"""

PROMPT_VERSION = "batch-v3"


def is_valid_translation(source, translated, request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    if not translated:
        logger.warning(f"{req_prefix}Validation failed reason=empty_translation source={source}")
        return False

    translated_stripped = translated.strip()
    source_stripped = source.strip()

    # Check forbidden phrases / boilerplate
    forbidden_substrings = ["translate the following text", "text:", "output:", "json"]
    for pattern in forbidden_substrings:
        if pattern in translated_stripped.lower():
            logger.warning(
                f"{req_prefix}Validation failed "
                f"reason=contains_boilerplate "
                f"boilerplate='{pattern}' "
                f"source={source} "
                f"translation={translated}"
            )
            return False

    # Check if translated == source for Japanese
    if contains_japanese(source_stripped) and translated_stripped == source_stripped:
        logger.warning(f"{req_prefix}Validation failed reason=identical_to_source source={source}")
        return False

    # Check if translated is pathologically longer than source
    if len(source_stripped) <= 5 and len(translated_stripped) > len(source_stripped) * 20:
        logger.warning(
            f"{req_prefix}Validation failed reason=pathologically_long source={source} translation={translated}"
        )
        return False

    # 1. CJK leak detection — Japanese/Chinese in "English" translation
    if contains_japanese(source_stripped):
        import re

        cjk_chars = re.findall(r"[\u3040-\u9FFF\uF900-\uFAFF]", translated_stripped)
        cjk_ratio = len(cjk_chars) / max(len(translated_stripped), 1)
        if cjk_ratio > 0.15:
            logger.warning(
                f"{req_prefix}Validation failed "
                f"reason=cjk_leak "
                f"cjk_ratio={cjk_ratio:.2f} "
                f"source={source} "
                f"translation={translated}"
            )
            return False

    # 2. Length ratio check — flag translations that are excessively long
    if len(source_stripped) > 5:
        ratio = len(translated_stripped) / len(source_stripped)
        if ratio > 10.0:
            logger.warning(
                f"{req_prefix}Validation failed "
                f"reason=length_ratio_exceeded "
                f"ratio={ratio:.1f} "
                f"source={source} "
                f"translation={translated}"
            )
            return False

    # 3. Duplicate word detection — filter out repetition anomalies
    words = translated_stripped.split()
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            logger.warning(
                f"{req_prefix}Validation failed "
                f"reason=excessive_repetition "
                f"unique_ratio={unique_ratio:.2f} "
                f"source={source} "
                f"translation={translated}"
            )
            return False

    return True


def should_translate_region(region):
    text = region.get("text", "")
    stripped = text.strip()
    confidence = region.get("confidence")
    if confidence is None:
        confidence = 1.0
    width = region.get("width") or region.get("bboxW") or 0
    height = region.get("height") or region.get("bboxH") or 0
    region_type = region.get("regionType") or region.get("region_type") or "speech"

    # SFX regions identified by layout analysis are always kept — even if small
    if region_type == "sfx":
        return True

    # Reject regions smaller than 10x10
    if width < 10 or height < 10:
        print(
            f"[Quality Filter] Rejecting region: too small ({width}x{height}) - text: '{text}'",
            flush=True,
        )
        return False

    # Special handling for SFX and Japanese kana-only text
    sfx_whitelist = {"ドン", "ガッ", "ぱんッ", "ズキュン"}

    # Check if text is in whitelist
    if stripped in sfx_whitelist:
        return True

    # Check if text is Japanese kana-only
    cleaned_for_kana = re.sub(r"[\s！？\?!\.\,\-\_\"]", "", stripped)
    is_kana_only = False
    if cleaned_for_kana:
        is_kana_only = bool(re.match(r"^[\u3040-\u309F\u30A0-\u30FF\u30FC\uFF66-\uFF9F]+$", cleaned_for_kana))

    if is_kana_only:
        return True

    # Reject low confidence regions (< 0.30)
    if confidence < 0.30:
        print(
            f"[Quality Filter] Rejecting region: low confidence ({confidence:.2f}) - text: '{text}'",
            flush=True,
        )
        return False

    # Otherwise, reject obvious garbage / non-Japanese low quality texts
    if len(stripped) < 2:
        print(
            f"[Quality Filter] Rejecting region: too short (len={len(stripped)}) - text: '{text}'",
            flush=True,
        )
        return False

    # Reject alphanumeric-only when confidence is low
    if re.match(r"^[A-Za-z0-9._-]+$", stripped) and confidence < 0.50:
        print(
            f"[Quality Filter] Rejecting region: alphanumeric-only with low confidence ({confidence:.2f}) - text: '{text}'",
            flush=True,
        )
        return False

    return True


def validate_translation_response(parsed_json):
    items = []
    if isinstance(parsed_json, dict):
        if "translations" in parsed_json:
            items = parsed_json["translations"]
        elif "items" in parsed_json:
            items = parsed_json["items"]
        else:
            if all(isinstance(k, str) and isinstance(v, str) for k, v in parsed_json.items()):
                return {
                    k: {
                        "translatedText": v,
                        "translationNotes": "",
                        "emotion": "",
                        "tone": "",
                    }
                    for k, v in parsed_json.items()
                }
    elif isinstance(parsed_json, list):
        items = parsed_json

    if not isinstance(items, list):
        return None

    validated = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        translation = item.get("translation")
        if rid and translation and isinstance(rid, str) and isinstance(translation, str) and translation.strip():
            validated[rid] = {
                "translatedText": translation.strip(),
                "translationNotes": item.get("translationNotes", ""),
                "emotion": item.get("emotion", ""),
                "tone": item.get("tone", ""),
                "translationScore": float(item.get("translationScore", 1.0)),
            }

    return validated if validated else None


def parse_and_validate_batch(response_text, unmatched_regions):
    if not response_text:
        return None

    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned_text)
        validated = validate_translation_response(parsed)
        if validated:
            return validated
    except Exception as e:
        print(
            f"[Translation] Failed to parse batch translation JSON response: {e}. Raw response: {response_text}",
            flush=True,
        )

    return None


def _inject_openrouter_routing(provider, routing_strategy, payload):
    if provider == "openrouter":
        if routing_strategy == "lowest-cost":
            provider_block = {
                "allow_fallbacks": False,
                "sort": "price",
                "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"],
            }
            payload["provider"] = provider_block
            logger.info(f"Routing: strategy=lowest-cost provider_order={provider_block['order']} allow_fallbacks=False")
        elif routing_strategy == "highest-throughput":
            payload["provider"] = {"allow_fallbacks": True, "sort": "throughput"}
            logger.info("Routing: strategy=highest-throughput allow_fallbacks=True")


def _get_api_url_and_headers(provider, api_key, model):
    client = LLMClient(provider, api_key, model)
    return client.url, client.headers, client.model


def try_cloud_ai(
    provider,
    api_key,
    model,
    prompt,
    response_schema=None,
    request_id=None,
    routing_strategy=None,
):
    """Cloud LLM text completion."""
    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        request_id=request_id or "",
        routing_strategy=routing_strategy,
    )
    system_prompt = MANGA_TRANSLATION_JSON_SYSTEM_PROMPT if response_schema else None
    messages = [{"role": "user", "content": prompt}]
    result = client.complete(messages, system_prompt=system_prompt, response_schema=response_schema)
    return result.content if result else None


def try_cloud_ai_vision(
    provider,
    api_key,
    model,
    prompt,
    base64_image,
    response_schema=None,
    system_prompt=None,
    request_id=None,
    routing_strategy=None,
):
    """Cloud VLM single-image completion."""
    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        request_id=request_id or "",
        routing_strategy=routing_strategy,
    )

    if client.is_anthropic:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ]

    sys_prompt = system_prompt or (MANGA_TRANSLATION_JSON_SYSTEM_PROMPT if response_schema else None)
    result = client.complete(messages, system_prompt=sys_prompt, response_schema=response_schema)
    return result.content if result else None


def try_cloud_ai_vision_batch(
    provider,
    api_key,
    model,
    crops,
    response_schema,
    system_prompt=None,
    request_id=None,
    routing_strategy=None,
):
    """Cloud VLM multi-image batch completion."""
    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        request_id=request_id or "",
        routing_strategy=routing_strategy,
    )

    ocr_prompt = (
        "You are an expert manga OCR system. Perform OCR on each of the provided image crops. "
        "Each crop is labeled with a Region ID header (e.g., 'Region ID: crop_0'). "
        "Extract the text and map it back to the ID exactly as specified in the JSON schema."
    )

    if client.is_anthropic:
        content_parts: list[dict] = [{"type": "text", "text": ocr_prompt}]
        for crop in crops:
            content_parts.append({"type": "text", "text": f"Region ID: {crop['id']}"})
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": crop["base64"],
                    },
                }
            )
        messages = [{"role": "user", "content": content_parts}]
    else:
        content_parts: list[dict] = [{"type": "text", "text": ocr_prompt}]
        for crop in crops:
            content_parts.append({"type": "text", "text": f"Region ID: {crop['id']}"})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop['base64']}"}}
            )
        messages = [{"role": "user", "content": content_parts}]

    sys = system_prompt or (
        "Respond with a valid JSON object matching the requested schema." if response_schema else None
    )
    result = client.complete(messages, system_prompt=sys, response_schema=response_schema)
    return result.content if result else None


def try_local_ai(prompt, text, response_schema=None, request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    enforce_rate_limit()

    local_provider = os.environ.get("LOCAL_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio")).lower().strip()
    local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", os.environ.get("LLM_ENDPOINT", "")).strip()
    model = os.environ.get("LOCAL_LLM_MODEL", "gemma3:4b")

    if not local_endpoint:
        if local_provider == "ollama":
            local_endpoint = "http://ollama:11434/v1/chat/completions"
        else:
            local_endpoint = "http://host.docker.internal:1234/v1/chat/completions"
    else:
        if not local_endpoint.endswith("/v1/chat/completions") and not local_endpoint.endswith("/api/v1/chat"):
            if local_endpoint.endswith("/"):
                local_endpoint += "v1/chat/completions"
            else:
                local_endpoint += "/v1/chat/completions"

    endpoints_to_try = [local_endpoint]
    if "localhost" in local_endpoint:
        endpoints_to_try.append(local_endpoint.replace("localhost", "host.docker.internal"))
    elif "host.docker.internal" in local_endpoint:
        endpoints_to_try.append(local_endpoint.replace("host.docker.internal", "localhost"))

    system_pr = MANGA_TRANSLATION_JSON_SYSTEM_PROMPT if response_schema else MANGA_TRANSLATION_SYSTEM_PROMPT

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_pr},
            {"role": "user", "content": text},
        ],
    }

    if response_schema:
        if local_provider == "ollama":
            payload["format"] = "json"
        else:
            payload["response_format"] = {"type": "json_object"}

    response = None
    for endpoint in endpoints_to_try:
        try:
            logger.info(f"{req_prefix}Trying Local AI endpoint '{endpoint}' using model '{model}'...")

            from worker.utils.lock import acquire_lock

            with acquire_lock("local-llm"):
                start = time.perf_counter()
                response = requests.post(endpoint, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - start

            logger.info(f"{req_prefix}Provider={local_provider} Model={model} Time={elapsed:.2f}s")
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"{req_prefix}Local AI connection failed for '{endpoint}': {e}")
            if "response" in locals() and response is not None and hasattr(response, "text"):
                logger.error(f"Response text: {response.text}")

    return None


def try_deepl(text, target_lang="en", request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    deepl_key = os.environ.get("DEEPL_API_KEY", os.environ.get("DEEPL_KEY", "")).strip()
    if not deepl_key:
        return None

    if deepl_key.endswith(":fx"):
        url = "https://api-free.deepl.com/v2/translate"
    else:
        url = "https://api.deepl.com/v2/translate"

    try:
        logger.info(f"{req_prefix}Sending request to DeepL API...")
        headers = {
            "Authorization": f"DeepL-Auth-Key {deepl_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": [text], "target_lang": target_lang.upper()}
        if logger.isEnabledFor(logging.TRACE):  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] DeepL Request URL: {url}")  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] DeepL Request Headers: {headers}")  # type: ignore

        start = time.perf_counter()
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        elapsed = time.perf_counter() - start
        logger.info(f"{req_prefix}Provider=deepl Model=deepl Time={elapsed:.2f}s")
        if logger.isEnabledFor(logging.TRACE):  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] DeepL Response Status: {res.status_code}")  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] DeepL Response Headers: {dict(res.headers)}")  # type: ignore

        if res.status_code == 200:
            res_json = res.json()
            translated = res_json["translations"][0]["text"]
            logger.info(f"{req_prefix}DeepL Translation Success: '{translated}'")
            return translated
        else:
            logger.error(f"{req_prefix}DeepL API returned error: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"{req_prefix}DeepL Translation failed: {e}")
    return None


def try_google_translate(text, source_lang="auto", target_lang="en", request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    try:
        logger.info(f"{req_prefix}Falling back to free Google Translate API...")
        import urllib.parse

        url = (
            f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q="
            + urllib.parse.quote(text)
        )

        start = time.perf_counter()
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        elapsed = time.perf_counter() - start
        if logger.isEnabledFor(logging.TRACE):  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] Google Translate Request URL: {url}")  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] Google Translate Response Status: {res.status_code}")  # type: ignore
            logger.trace(f"{req_prefix}[TRACE] Google Translate Response Headers: {dict(res.headers)}")  # type: ignore
        logger.info(f"{req_prefix}Provider=google_translate Model=free_api Time={elapsed:.2f}s")

        if res.status_code == 200:
            data = res.json()
            translated = "".join([part[0] for part in data[0] if part[0]])
            logger.info(f"{req_prefix}Google Translate Success: '{translated}'")
            return translated
    except Exception as e:
        logger.error(f"{req_prefix}Google Translate fallback failed: {e}")
    return None


def translate_text(
    text,
    source_lang="auto",
    target_lang="en",
    request_id=None,
    use_fallback_models=True,
):
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    req_prefix = f"[{request_id}] "

    from worker.config import TL_CONFIG

    provider = TL_CONFIG.provider
    api_key = TL_CONFIG.resolve_key()

    # LOCAL_ONLY mode: when provider is a local runtime, skip all cloud tiers
    local_only = provider in ("ollama", "lmstudio")

    deepl_key = os.environ.get("DEEPL_API_KEY", os.environ.get("DEEPL_KEY", "")).strip()

    LANG_MAP.get(source_lang.lower(), source_lang)
    tgt_name = LANG_MAP.get(target_lang.lower(), target_lang)
    prompt = f"Translate the following text to natural {tgt_name}, maintaining its tone and context. Respond ONLY with the translated text. Do not include any tags, notes, or explanations. NEVER include romanized text, pinyin, romaji, or pronunciation guides. (e.g. BAD: 'ERUFU (ELF!)', GOOD: 'ELF!').\n\nText: {text}"

    # Log Strategy
    logger.info(f"{req_prefix}Translation Strategy:")
    strategy_idx = 1
    if not local_only:
        if provider == "openrouter":
            preferred = TL_CONFIG.llm_model or "meta-llama/llama-3-8b-instruct:free"
            logger.info(f"{req_prefix}{strategy_idx}. {preferred} (OpenRouter)")
            strategy_idx += 1
        elif provider == "gemini":
            logger.info(f"{req_prefix}{strategy_idx}. Gemini 2.5 Flash (Direct)")
            strategy_idx += 1
        elif provider == "openai":
            openai_model = TL_CONFIG.llm_model or "gpt-4o-mini"
            logger.info(f"{req_prefix}{strategy_idx}. {openai_model} (Direct OpenAI)")
            strategy_idx += 1

        if provider == "nvidia":
            nvidia_model = TL_CONFIG.llm_model or "google/gemma-3n-e4b-it"
            logger.info(f"{req_prefix}{strategy_idx}. {nvidia_model} (Nvidia)")
            strategy_idx += 1
        if provider == "anthropic":
            logger.info(f"{req_prefix}{strategy_idx}. Claude 3.5 Sonnet (Direct)")
            strategy_idx += 1

    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    disable_deepl = os.environ.get("DISABLE_DEEPL_TRANSLATE", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    disable_gt = os.environ.get("DISABLE_GOOGLE_TRANSLATE", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )

    if not disable_local:
        logger.info(f"{req_prefix}{strategy_idx}. Local LLM")
        strategy_idx += 1
    if not local_only:
        if deepl_key and not disable_deepl:
            logger.info(f"{req_prefix}{strategy_idx}. DeepL")
            strategy_idx += 1
        if not disable_gt:
            logger.info(f"{req_prefix}{strategy_idx}. Google Translate")

    if local_only:
        translated = try_local_ai(prompt, text, request_id=request_id)
        if translated:
            cleaned = clean_translated_text(translated)
            if is_valid_translation(text, cleaned, request_id=request_id):
                return cleaned
    else:
        # 1. Cloud LLM Layer
        if api_key:
            user_model = TL_CONFIG.llm_model
            logger.info(f"{req_prefix}Trying provider '{provider}' with model '{user_model}'...")
            translated = try_cloud_ai(provider, api_key, user_model, prompt, request_id=request_id)
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned

            # Fallback to global default model
            global_model = TL_CONFIG.llm_model
            global_provider = TL_CONFIG.provider
            if use_fallback_models and global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Falling back to global default model '{global_model}'...")
                translated = try_cloud_ai(provider, api_key, global_model, prompt, request_id=request_id)
                if translated:
                    cleaned = clean_translated_text(translated)
                    if is_valid_translation(text, cleaned, request_id=request_id):
                        return cleaned
                else:
                    logger.error(f"{req_prefix}Translation with global fallback model '{global_model}' failed.")
            else:
                logger.info(
                    f"{req_prefix}No fallback applied (global provider different, model identical, or fallback disabled)."
                )

    logger.error(f"{req_prefix}All translation tiers failed for text: '{text}'")
    return None


def translate_batch_llm(
    regions,
    context_str="",
    response_schema=None,
    request_id=None,
    source_lang="ja",
    target_lang="en",
    provider=None,
    llm_model=None,
    routing_strategy=None,
    use_fallback_models=True,
):
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    req_prefix = f"[{request_id}] "

    bubbles_input = []
    for r in regions:
        entry = {
            "id": r["id"],
            "panel": r.get("panelReadingOrder") or r.get("panelId") or 0,
            "bubble": r.get("bubbleReadingOrder") or 0,
            "speaker": r.get("speakerLabel") or None,
            "regionType": r.get("regionType") or "speech",
            "conversationGroup": r.get("conversationId") or None,
            "text": r["text"],
        }
        if r.get("qaStatus") == "failed" and r.get("qaFeedback"):
            entry["previousTranslation"] = r.get("translatedText")
            entry["qaFeedback"] = r.get("qaFeedback")
        bubbles_input.append(entry)
    bubbles_json = json.dumps(bubbles_input, ensure_ascii=False, indent=2)

    logger.debug(f"{req_prefix}Batch Input:\n{bubbles_json}")
    logger.info(f"{req_prefix}Prompt={PROMPT_VERSION}")

    src_name = LANG_MAP.get(source_lang.lower(), source_lang)
    tgt_name = LANG_MAP.get(target_lang.lower(), target_lang)

    prompt = f"""{context_str}These text regions appear in reading order.
Each region has a "regionType" field indicating its category (speech/narration/sfx/caption/sign).
Regions with the same "conversationGroup" are part of the same dialogue exchange.
Translate each region from {src_name} to natural {tgt_name} according to its type and maintain conversational coherence within groups.

Preserve:
- tone
- emotional state
- relationships
- ongoing conversation

Region type handling:
- "speech": Translate as natural dialogue.
- "narration": Translate as third-person narrative prose.
- "sfx": Transliterate the sound effect AND provide a {tgt_name} equivalent in parentheses (e.g. "DOKAA (WHAM)").
- "caption": Translate as editorial/scene-setting text.
- "sign": Translate literally, noting it's environmental text.

If multiple regions share the same conversationGroup, treat them as a continuous dialogue exchange and ensure coherent flow.

NEVER include romanized text, pinyin, romaji, or pronunciation guides. Return ONLY the target-language translation.
BAD: "Yào chūfā le o (About to depart!)"
GOOD: "About to depart!"
BAD: "ERUFU (ELF!)"
GOOD: "ELF!"

If a region has "previousTranslation" and "qaFeedback" fields, it means the previous translation attempt failed QA (e.g. text overflow, poor phrasing, or formatting issues). Adjust your translation accordingly based on the feedback (for example, shortening the text to fit if it overflowed).

You MUST return a JSON object containing a "translations" key with an array of objects.
Each object in the array MUST have the following keys:
- "id" (the original string ID)
- "translation" (your {tgt_name} translation)
- "translationNotes" (brief explanation of translation decisions or register choices)
- "emotion" (detected speaker emotion, e.g. "earnest", "angry", "playful")
- "tone" (detected tone, e.g. "formal", "sarcastic", "casual")
- "translationScore" (a self-evaluation score from 0.0 to 1.0 representing translation quality/confidence)

Example structure:
{{
  "translations": [
    {{
      "id": "some-id-1",
      "translation": "Translated text here",
      "translationNotes": "Preserved informal/teasing tone",
      "emotion": "playful",
      "tone": "casual",
      "translationScore": 0.95
    }}
  ]
}}

Return ONLY valid JSON format conforming to the requested schema. No conversational prefix, suffix, or markdown formatting.

Input:
{bubbles_json}
"""
    from worker.config import TL_CONFIG

    provider = provider or TL_CONFIG.provider
    api_key = TL_CONFIG.resolve_key(provider)
    user_model = llm_model or TL_CONFIG.llm_model

    # LOCAL_ONLY mode: when provider is a local runtime, skip all cloud tiers
    local_only = provider in ("ollama", "lmstudio")

    if local_only:
        logger.info(f"{req_prefix}Batch: Trying Local LLM ({provider})...")
        try:
            res = try_local_ai(prompt, bubbles_json, response_schema, request_id=request_id)
            if res:
                return res
        except Exception as e:
            logger.error(f"{req_prefix}Local LLM batch translation failed: {e}")
    else:
        if api_key:
            logger.info(f"{req_prefix}Batch: Trying provider '{provider}' with model '{user_model}'...")
            res = try_cloud_ai(
                provider,
                api_key,
                user_model,
                prompt,
                response_schema,
                request_id=request_id,
                routing_strategy=routing_strategy,
            )
            if res:
                return res

            # Fallback to global default model (only if use_fallback_models is True)
            global_model = TL_CONFIG.llm_model
            global_provider = TL_CONFIG.provider
            if use_fallback_models and global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Batch: Falling back to global default model '{global_model}'...")

                res = try_cloud_ai(
                    provider,
                    api_key,
                    global_model,
                    prompt,
                    response_schema,
                    request_id=request_id,
                    routing_strategy=routing_strategy,
                )
                if res:
                    return res
                else:
                    logger.error(f"{req_prefix}Batch translation with global fallback model '{global_model}' failed.")
            else:
                logger.info(f"{req_prefix}Batch: No fallback applied (global provider different or model identical).")
    return None


def try_local_vlm_vision(
    model,
    prompt,
    base64_image,
    response_schema=None,
    system_prompt=None,
    request_id=None,
):
    req_prefix = f"[{request_id}] " if request_id else ""
    local_provider = os.environ.get("LOCAL_LLM_PROVIDER", "ollama").lower().strip()
    local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
    if not local_endpoint:
        if local_provider == "ollama":
            local_endpoint = "http://ollama:11434/v1/chat/completions"
        else:
            local_endpoint = "http://host.docker.internal:1234/v1/chat/completions"

    if not local_endpoint.endswith("/v1/chat/completions") and not local_endpoint.endswith("/api/v1/chat"):
        if local_endpoint.endswith("/"):
            local_endpoint += "v1/chat/completions"
        else:
            local_endpoint += "/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
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
    )

    payload = {
        "model": model,
        "messages": messages,
    }

    if response_schema:
        if local_provider == "ollama":
            payload["format"] = "json"
        else:
            payload["response_format"] = {"type": "json_object"}

    from worker.utils.lock import acquire_lock

    try:
        with acquire_lock("local-llm"):
            logger.info(f"{req_prefix}Sending local VLM request to {local_endpoint} using model {model}...")
            start = time.perf_counter()
            response = requests.post(local_endpoint, json=payload, timeout=(10, 45))
            elapsed = time.perf_counter() - start
            logger.info(f"{req_prefix}Local VLM query completed in {elapsed:.2f}s")

            if response.status_code == 200:
                res_json = response.json()
                choice = res_json["choices"][0]["message"]["content"]
                return choice
            else:
                logger.error(f"{req_prefix}Local VLM API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"{req_prefix}Error during local VLM query: {e}")
    return None


def translate_batch_deepl(unmatched_regions, target_lang="en", request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    deepl_key = os.environ.get("DEEPL_API_KEY", os.environ.get("DEEPL_KEY", "")).strip()
    if not deepl_key:
        return None

    if deepl_key.endswith(":fx"):
        url = "https://api-free.deepl.com/v2/translate"
    else:
        url = "https://api.deepl.com/v2/translate"

    try:
        logger.info(f"{req_prefix}Sending batch request of {len(unmatched_regions)} bubbles to DeepL API...")
        headers = {
            "Authorization": f"DeepL-Auth-Key {deepl_key}",
            "Content-Type": "application/json",
        }
        texts = [r["text"] for r in unmatched_regions]
        payload = {"text": texts, "target_lang": target_lang.upper()}

        start = time.perf_counter()
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        elapsed = time.perf_counter() - start
        logger.info(f"{req_prefix}Provider=deepl Model=deepl_batch Time={elapsed:.2f}s")

        if res.status_code == 200:
            res_json = res.json()
            translations = res_json["translations"]
            mapping = {}
            for i, r in enumerate(unmatched_regions):
                mapping[r["id"]] = translations[i]["text"]
            return mapping
        else:
            logger.error(f"{req_prefix}DeepL API returned error: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"{req_prefix}DeepL batch translation failed: {e}")
    return None


def build_context_string(image_info):
    context_str = ""
    if not image_info:
        return context_str

    series_meta = image_info.get("seriesMetadata")
    if series_meta:
        context_str += f"Series Title: {series_meta.get('title')}\n"
        context_str += f"Original Language: {series_meta.get('originalLanguage')}\n"
        if series_meta.get("metadataJson"):
            try:
                meta = series_meta.get("metadataJson")
                if isinstance(meta, str):
                    meta = json.loads(meta)
                context_str += (
                    f"Roster & Editorial Style Guidelines:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
                )
            except Exception:
                context_str += f"Roster & Editorial Style Guidelines:\n{series_meta.get('metadataJson')}\n"

    chapter_sum = image_info.get("chapterSummary")
    if chapter_sum:
        context_str += f"Previous Chapter Summary:\n{chapter_sum}\n"

    prev_text = image_info.get("previousPageText")
    if prev_text:
        if isinstance(prev_text, str) and "|" in prev_text:
            lines = [line.strip() for line in prev_text.split("|") if line.strip()]
            formatted = "\n".join(f"  - {line}" for line in lines)
            context_str += f"Previous Page Dialogue (in reading order):\n{formatted}\n"
        else:
            context_str += f"Previous Page Text/Dialogue Context:\n{prev_text}\n"

    if context_str:
        return f"Narrative and Style Context:\n{context_str}\n---\n"
    return ""
