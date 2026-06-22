import os
import re
import json
import time
import uuid
import logging
import requests

from worker.config import logger
from worker.utils.text import contains_japanese, clean_translated_text
from worker.utils.rate_limit import enforce_rate_limit, estimate_cost

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

Region type handling:
- "speech": Translate as natural dialogue.
- "narration": Translate as third-person narrative prose.
- "sfx": Transliterate the sound effect AND provide an English equivalent in parentheses (e.g. "DOKAA (WHAM)").
- "caption": Translate as editorial/scene-setting text.
- "sign": Translate literally, noting it's environmental text.

If multiple regions share the same conversationGroup, treat them as a continuous dialogue exchange and ensure coherent flow.

Return ONLY valid JSON format conforming to the requested schema. No conversational prefix, suffix, or markdown formatting."""

MANGA_TRANSLATION_SYSTEM_PROMPT = """You are an expert manga translator.

Translate Japanese manga dialogue into natural English.

Rules:
- Keep names unchanged.
- Preserve tone and emotion.
- Do not explain.
- Do not add notes.
- Do not add quotation marks.
- Return only the translated text."""

PROMPT_VERSION = "batch-v3"


def is_valid_translation(source, translated, request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    if not translated:
        logger.warning(
            f"{req_prefix}Validation failed reason=empty_translation source={source}"
        )
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
        logger.warning(
            f"{req_prefix}Validation failed "
            f"reason=identical_to_source "
            f"source={source}"
        )
        return False

    # Check if translated is pathologically longer than source
    if (
        len(source_stripped) <= 5
        and len(translated_stripped) > len(source_stripped) * 20
    ):
        logger.warning(
            f"{req_prefix}Validation failed "
            f"reason=pathologically_long "
            f"source={source} "
            f"translation={translated}"
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

    # Reject low confidence regions (< 0.30)
    if confidence < 0.30:
        print(
            f"[Quality Filter] Rejecting region: low confidence ({confidence:.2f}) - text: '{text}'",
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
        is_kana_only = bool(
            re.match(
                r"^[\u3040-\u309F\u30A0-\u30FF\u30FC\uFF66-\uFF9F]+$", cleaned_for_kana
            )
        )

    if is_kana_only:
        return True

    # Otherwise, reject obvious garbage / non-Japanese low quality texts
    if len(stripped) < 2:
        print(
            f"[Quality Filter] Rejecting region: too short (len={len(stripped)}) - text: '{text}'",
            flush=True,
        )
        return False

    # Reject alphanumeric-only when confidence is low
    if re.match(r"^[A-Za-z0-9._-]+$", stripped):
        if confidence < 0.50:
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
            if all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in parsed_json.items()
            ):
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
        if (
            rid
            and translation
            and isinstance(rid, str)
            and isinstance(translation, str)
            and translation.strip()
        ):
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


PROVIDER_COOLDOWNS = {}


def try_cloud_ai(
    provider, api_key, model, prompt, response_schema=None, request_id=None
):
    req_prefix = f"[{request_id}] " if request_id else ""
    global PROVIDER_COOLDOWNS
    cooldown_until = PROVIDER_COOLDOWNS.get(provider, 0.0)
    if time.time() < cooldown_until:
        logger.warning(
            f"{req_prefix}Skipping provider '{provider}' because it is in cooldown for another {int(cooldown_until - time.time())} seconds."
        )
        return None

    enforce_rate_limit()
    url = ""
    headers = {}
    payload = {}

    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "meta-llama/llama-3-8b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "schema": response_schema},
            }
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "schema": response_schema},
            }
    elif provider == "nvidia":
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        system_pr = (
            MANGA_TRANSLATION_JSON_SYSTEM_PROMPT
            if response_schema
            else MANGA_TRANSLATION_SYSTEM_PROMPT
        )
        payload = {
            "model": model or "nvidia/riva-translate-4b-instruct-v1.1",
            "messages": [
                {"role": "system", "content": system_pr},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
        if response_schema:
            payload["response_format"] = {"type": "json_object"}
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
            "messages": [{"role": "user", "content": prompt}],
        }
    elif provider == "gemini":
        gemini_model = model or "gemini-1.5-flash"
        if "/" not in gemini_model:
            gemini_model = f"models/{gemini_model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{gemini_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if response_schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            }
    else:
        return None

    try:
        logger.info(
            f"{req_prefix}Sending request to Cloud LLM provider '{provider}' using model '{model}'..."
        )
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"{req_prefix}[TRACE] Request URL: {url}")
            logger.trace(f"{req_prefix}[TRACE] Request Headers: {headers}")
        start = time.perf_counter()
        res = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=45 if provider == "nvidia" else 30,
        )
        elapsed = time.perf_counter() - start
        logger.info(
            f"{req_prefix}Provider={provider} " f"Model={model} " f"Time={elapsed:.2f}s"
        )
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"{req_prefix}[TRACE] Response Status: {res.status_code}")
            logger.trace(f"{req_prefix}[TRACE] Response Headers: {dict(res.headers)}")

        response_text = res.text
        logger.debug(f"{req_prefix}Raw Model Output:\n{response_text}")

        if res.status_code == 200:
            res_json = res.json()

            # Extract and log token usage
            usage = res_json.get("usage")
            usage_meta = res_json.get("usageMetadata")
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion_tokens = usage.get("completion_tokens") or usage.get(
                    "output_tokens"
                )
                total_tokens = usage.get("total_tokens") or (
                    (prompt_tokens + completion_tokens)
                    if prompt_tokens and completion_tokens
                    else None
                )
            elif usage_meta:
                prompt_tokens = usage_meta.get("promptTokenCount")
                completion_tokens = usage_meta.get("candidatesTokenCount")
                total_tokens = usage_meta.get("totalTokenCount")

            if prompt_tokens is not None:
                logger.info(
                    f"{req_prefix}Tokens "
                    f"in={prompt_tokens} "
                    f"out={completion_tokens} "
                    f"total={total_tokens}"
                )
                cost = estimate_cost(model, prompt_tokens, completion_tokens, provider)
                logger.info(f"{req_prefix}Estimated cost: ${cost:.5f}")

            if provider == "gemini":
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            elif provider == "anthropic":
                return res_json["content"][0]["text"]
            else:
                return res_json["choices"][0]["message"]["content"]
        else:
            if res.status_code == 429:
                logger.warning(
                    f"{req_prefix}Cloud LLM provider '{provider}' returned 429 (Too Many Requests). Initiating a 60-second cooldown."
                )
                PROVIDER_COOLDOWNS[provider] = time.time() + 60.0
            logger.error(
                f"{req_prefix}Cloud LLM provider '{provider}' returned error: {res.status_code} - {res.text}"
            )
    except Exception as e:
        logger.error(f"{req_prefix}Cloud LLM Translation failed: {e}")
    return None


def try_cloud_ai_vision(
    provider,
    api_key,
    model,
    prompt,
    base64_image,
    response_schema=None,
    request_id=None,
):
    req_prefix = f"[{request_id}] " if request_id else ""
    global PROVIDER_COOLDOWNS
    cooldown_until = PROVIDER_COOLDOWNS.get(provider, 0.0)
    if time.time() < cooldown_until:
        logger.warning(
            f"{req_prefix}Skipping vision provider '{provider}' because it is in cooldown for another {int(cooldown_until - time.time())} seconds."
        )
        return None

    enforce_rate_limit()
    url = ""
    headers = {}
    payload = {}

    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
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
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "schema": response_schema},
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
        if response_schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            }
    elif provider == "nvidia":
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "nvidia/nemotron-nano-12b-v2-vl",
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
        if response_schema:
            payload["response_format"] = {"type": "json_object"}
    else:
        return None

    try:
        logger.info(
            f"{req_prefix}Sending vision request to provider '{provider}' using model '{model}'..."
        )
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"{req_prefix}[TRACE] Vision Request URL: {url}")
            logger.trace(f"{req_prefix}[TRACE] Vision Request Headers: {headers}")
        start = time.perf_counter()
        res = requests.post(url, json=payload, headers=headers, timeout=45)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{req_prefix}Provider={provider} " f"Model={model} " f"Time={elapsed:.2f}s"
        )
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"{req_prefix}[TRACE] Vision Response Status: {res.status_code}"
            )
            logger.trace(
                f"{req_prefix}[TRACE] Vision Response Headers: {dict(res.headers)}"
            )

        response_text = res.text
        logger.debug(f"{req_prefix}Raw Model Output:\n{response_text}")

        if res.status_code == 200:
            res_json = res.json()

            # Extract and log token usage
            usage = res_json.get("usage")
            usage_meta = res_json.get("usageMetadata")
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion_tokens = usage.get("completion_tokens") or usage.get(
                    "output_tokens"
                )
                total_tokens = usage.get("total_tokens") or (
                    (prompt_tokens + completion_tokens)
                    if prompt_tokens and completion_tokens
                    else None
                )
            elif usage_meta:
                prompt_tokens = usage_meta.get("promptTokenCount")
                completion_tokens = usage_meta.get("candidatesTokenCount")
                total_tokens = usage_meta.get("totalTokenCount")

            if prompt_tokens is not None:
                logger.info(
                    f"{req_prefix}Tokens "
                    f"in={prompt_tokens} "
                    f"out={completion_tokens} "
                    f"total={total_tokens}"
                )
                cost = estimate_cost(model, prompt_tokens, completion_tokens, provider)
                logger.info(f"{req_prefix}Estimated cost: ${cost:.5f}")

            if provider == "gemini":
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return res_json["choices"][0]["message"]["content"]
        else:
            if res.status_code == 429:
                logger.warning(
                    f"{req_prefix}Vision provider '{provider}' returned 429 (Too Many Requests). Initiating a 60-second cooldown."
                )
                PROVIDER_COOLDOWNS[provider] = time.time() + 60.0
            logger.error(
                f"{req_prefix}Provider '{provider}' returned error: {res.status_code} - {res.text}"
            )
    except Exception as e:
        logger.error(f"{req_prefix}Vision Translation failed: {e}")
    return None


def try_local_ai(prompt, text, response_schema=None, request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    enforce_rate_limit()
    local_provider = (
        os.environ.get("LOCAL_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio"))
        .lower()
        .strip()
    )
    local_endpoint = os.environ.get(
        "LOCAL_LLM_ENDPOINT", os.environ.get("LLM_ENDPOINT", "")
    ).strip()
    # Keep gemma3:4b as fallback as requested by user
    model = os.environ.get("LOCAL_LLM_MODEL", "gemma3:4b")

    if local_endpoint:
        if not local_endpoint.endswith(
            "/v1/chat/completions"
        ) and not local_endpoint.endswith("/api/v1/chat"):
            if local_endpoint.endswith("/"):
                local_endpoint += "v1/chat/completions"
            else:
                local_endpoint += "/v1/chat/completions"

    if not local_endpoint:
        if local_provider == "ollama":
            local_endpoint = "http://ollama:11434/v1/chat/completions"
        else:
            local_endpoint = "http://host.docker.internal:1234/v1/chat/completions"

    endpoints_to_try = [local_endpoint]
    if "localhost" in local_endpoint:
        endpoints_to_try.append(
            local_endpoint.replace("localhost", "host.docker.internal")
        )
    elif "host.docker.internal" in local_endpoint:
        endpoints_to_try.append(
            local_endpoint.replace("host.docker.internal", "localhost")
        )

    system_pr = (
        MANGA_TRANSLATION_JSON_SYSTEM_PROMPT
        if response_schema
        else MANGA_TRANSLATION_SYSTEM_PROMPT
    )

    for endpoint in endpoints_to_try:
        try:
            logger.info(
                f"{req_prefix}Trying Local AI endpoint '{endpoint}' using model '{model}'..."
            )

            if "/api/v1/chat" in endpoint:
                payload = {"model": model, "system_prompt": system_pr, "input": text}
            else:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_pr},
                        {"role": "user", "content": text},
                    ],
                }
                if response_schema:
                    if "ollama" in endpoint or local_provider == "ollama":
                        payload["format"] = "json"
                    else:
                        payload["response_format"] = {"type": "json_object"}

            from worker.utils.lock import acquire_lock

            with acquire_lock("local-llm"):
                if logger.isEnabledFor(logging.TRACE):
                    logger.trace(f"{req_prefix}[TRACE] Local Request URL: {endpoint}")
                    logger.trace(
                        f"{req_prefix}[TRACE] Local Request Headers: {payload}"
                    )
                start = time.perf_counter()
                res = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                elapsed = time.perf_counter() - start
            logger.info(
                f"{req_prefix}Provider={local_provider} "
                f"Model={model} "
                f"Time={elapsed:.2f}s"
            )
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(
                    f"{req_prefix}[TRACE] Local Response Status: {res.status_code}"
                )
                logger.trace(
                    f"{req_prefix}[TRACE] Local Response Headers: {dict(res.headers)}"
                )

            response_text = res.text
            logger.debug(f"{req_prefix}Raw Model Output:\n{response_text}")

            if res.status_code == 200:
                res_json = res.json()
                translated = None
                if "/api/v1/chat" in endpoint:
                    if "choices" in res_json:
                        choice = res_json["choices"][0]
                        if "message" in choice:
                            translated = choice["message"]["content"]
                        elif "text" in choice:
                            translated = choice["text"]
                    elif "output" in res_json:
                        translated = res_json["output"]
                    elif "response" in res_json:
                        translated = res_json["response"]
                else:
                    if "choices" in res_json:
                        translated = res_json["choices"][0]["message"]["content"]
                    elif "response" in res_json:
                        translated = res_json["response"]

                if translated:
                    return translated
        except Exception as e:
            logger.error(
                f"{req_prefix}Local AI connection failed for '{endpoint}': {e}"
            )

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
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"{req_prefix}[TRACE] DeepL Request URL: {url}")
            logger.trace(f"{req_prefix}[TRACE] DeepL Request Headers: {headers}")

        start = time.perf_counter()
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{req_prefix}Provider=deepl " f"Model=deepl " f"Time={elapsed:.2f}s"
        )
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"{req_prefix}[TRACE] DeepL Response Status: {res.status_code}"
            )
            logger.trace(
                f"{req_prefix}[TRACE] DeepL Response Headers: {dict(res.headers)}"
            )

        if res.status_code == 200:
            res_json = res.json()
            translated = res_json["translations"][0]["text"]
            logger.info(f"{req_prefix}DeepL Translation Success: '{translated}'")
            return translated
        else:
            logger.error(
                f"{req_prefix}DeepL API returned error: {res.status_code} - {res.text}"
            )
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
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f"{req_prefix}[TRACE] Google Translate Request URL: {url}")
            logger.trace(
                f"{req_prefix}[TRACE] Google Translate Response Status: {res.status_code}"
            )
            logger.trace(
                f"{req_prefix}[TRACE] Google Translate Response Headers: {dict(res.headers)}"
            )
        logger.info(
            f"{req_prefix}Provider=google_translate "
            f"Model=free_api "
            f"Time={elapsed:.2f}s"
        )

        if res.status_code == 200:
            data = res.json()
            translated = "".join([part[0] for part in data[0] if part[0]])
            logger.info(f"{req_prefix}Google Translate Success: '{translated}'")
            return translated
    except Exception as e:
        logger.error(f"{req_prefix}Google Translate fallback failed: {e}")
    return None


def translate_text(text, source_lang="auto", target_lang="en", request_id=None):
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    req_prefix = f"[{request_id}] "

    provider = os.environ.get("MODEL_PROVIDER", "").lower().strip()
    api_key = os.environ.get("API_KEY", "").strip()

    # LOCAL_ONLY mode: when provider is a local runtime, skip all cloud tiers
    local_only = provider in ("ollama", "lmstudio")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or (
        api_key if provider == "openrouter" else ""
    )
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip() or (
        api_key if provider == "nvidia" else ""
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or (
        api_key if provider == "gemini" else ""
    )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or (
        api_key if provider == "anthropic" else ""
    )
    deepl_key = os.environ.get("DEEPL_API_KEY", os.environ.get("DEEPL_KEY", "")).strip()

    src_name = LANG_MAP.get(source_lang.lower(), source_lang)
    tgt_name = LANG_MAP.get(target_lang.lower(), target_lang)
    prompt = f"Translate the following text to natural {tgt_name}, maintaining its tone and context. Respond ONLY with the translated text. Do not include any tags, notes, or explanations.\n\nText: {text}"

    # Log Strategy
    logger.info(f"{req_prefix}Translation Strategy:")
    strategy_idx = 1
    if not local_only:
        if provider == "openrouter":
            logger.info(f"{req_prefix}{strategy_idx}. DeepSeek V4 Pro (OpenRouter)")
            strategy_idx += 1
            logger.info(f"{req_prefix}{strategy_idx}. Gemini 2.5 Flash (OpenRouter)")
            strategy_idx += 1
        elif provider == "gemini":
            logger.info(f"{req_prefix}{strategy_idx}. Gemini 2.5 Flash (Direct)")
            strategy_idx += 1
        elif provider == "openai":
            openai_model = os.environ.get("PREFERRED_MODEL", "gpt-4o-mini")
            logger.info(f"{req_prefix}{strategy_idx}. {openai_model} (Direct OpenAI)")
            strategy_idx += 1

        if provider == "nvidia":
            nvidia_model = os.environ.get("PREFERRED_MODEL", "google/gemma-3n-e4b-it")
            logger.info(f"{req_prefix}{strategy_idx}. {nvidia_model} (Nvidia)")
            strategy_idx += 1
        if provider == "anthropic":
            logger.info(f"{req_prefix}{strategy_idx}. Claude 3.5 Sonnet (Direct)")
            strategy_idx += 1

    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in ("true", "1", "yes")
    disable_deepl = os.environ.get("DISABLE_DEEPL_TRANSLATE", "").strip().lower() in ("true", "1", "yes")
    disable_gt = os.environ.get("DISABLE_GOOGLE_TRANSLATE", "").strip().lower() in ("true", "1", "yes")

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
        logger.info(
            f"{req_prefix}LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers."
        )
    else:
        # 1. Cloud LLM Layer (DeepSeek V4 Pro, then Gemini 2.5 Flash / Claude Sonnet fallback)
        if provider == "openrouter":
            translated = try_cloud_ai(
                "openrouter",
                openrouter_key,
                "deepseek-ai/deepseek-v4-pro",
                prompt,
                request_id=request_id,
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned

            # Fallback to Gemini 2.5 Flash via OpenRouter
            translated = try_cloud_ai(
                "openrouter",
                openrouter_key,
                "google/gemini-2.5-flash",
                prompt,
                request_id=request_id,
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned

        elif provider == "gemini":
            # Direct Gemini API fallback
            preferred = os.environ.get("PREFERRED_MODEL", "gemini-2.5-flash")
            translated = try_cloud_ai(
                "gemini", gemini_key, preferred, prompt, request_id=request_id
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned
        elif provider == "openai":
            openai_model = os.environ.get("PREFERRED_MODEL", "gpt-4o-mini")
            translated = try_cloud_ai(
                "openai",
                api_key,
                openai_model,
                prompt,
                request_id=request_id,
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned
        if provider == "nvidia":
            nvidia_model = os.environ.get("PREFERRED_MODEL", "google/gemma-3n-e4b-it")
            translated = try_cloud_ai(
                "nvidia",
                nvidia_key,
                nvidia_model,
                prompt,
                request_id=request_id,
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned

        if provider == "anthropic":
            translated = try_cloud_ai(
                "anthropic",
                anthropic_key,
                "claude-3-5-sonnet-20241022",
                prompt,
                request_id=request_id,
            )
            if translated:
                cleaned = clean_translated_text(translated)
                if is_valid_translation(text, cleaned, request_id=request_id):
                    return cleaned

    # 2. Local Ollama/LMStudio Layer
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in ("true", "1", "yes")
    if not disable_local:
        translated = try_local_ai(prompt, text, request_id=request_id)
        if translated:
            cleaned = clean_translated_text(translated)
            if is_valid_translation(text, cleaned, request_id=request_id):
                return cleaned
    else:
        logger.info(f"{req_prefix}Local LLM layer skipped (disabled via environment).")

    if local_only:
        logger.info(
            f"{req_prefix}LOCAL_ONLY mode — not falling back to DeepL/Google Translate."
        )
        logger.error(f"{req_prefix}All translation tiers failed for text: '{text}'")
        return None

    # 3. DeepL Layer
    disable_deepl = os.environ.get("DISABLE_DEEPL_TRANSLATE", "").strip().lower() in ("true", "1", "yes")
    if not disable_deepl:
        translated = try_deepl(text, target_lang, request_id=request_id)
        if translated:
            cleaned = clean_translated_text(translated)
            if is_valid_translation(text, cleaned, request_id=request_id):
                return cleaned
    else:
        logger.info(f"{req_prefix}DeepL fallback skipped (disabled via environment).")

    # 4. Google Translate Layer
    disable_gt = os.environ.get("DISABLE_GOOGLE_TRANSLATE", "").strip().lower() in ("true", "1", "yes")
    if not disable_gt:
        translated = try_google_translate(
            text, source_lang, target_lang, request_id=request_id
        )
        if translated:
            cleaned = clean_translated_text(translated)
            if is_valid_translation(text, cleaned, request_id=request_id):
                return cleaned
    else:
        logger.info(f"{req_prefix}Google Translate fallback skipped (disabled via environment).")

    logger.error(f"{req_prefix}All translation tiers failed for text: '{text}'")
    return None


def translate_batch_llm(
    unmatched_regions,
    context_str="",
    response_schema=None,
    request_id=None,
    source_lang="ja",
    target_lang="en",
):
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    req_prefix = f"[{request_id}] "

    bubbles_input = []
    for r in unmatched_regions:
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

Return ONLY valid JSON.

Input:
{bubbles_json}
"""
    provider = os.environ.get("MODEL_PROVIDER", "").lower().strip()
    api_key = os.environ.get("API_KEY", "").strip()

    # LOCAL_ONLY mode: when provider is a local runtime, skip all cloud tiers
    local_only = provider in ("ollama", "lmstudio")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or (
        api_key if provider == "openrouter" else ""
    )
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip() or (
        api_key if provider == "nvidia" else ""
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or (
        api_key if provider == "gemini" else ""
    )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or (
        api_key if provider == "anthropic" else ""
    )

    if local_only:
        logger.info(
            f"{req_prefix}Batch: LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers."
        )
    else:
        # Try DeepSeek V4 Pro
        if provider == "openrouter":
            logger.info(f"{req_prefix}Batch: Trying DeepSeek V4 Pro...")
            try:
                res = try_cloud_ai(
                    "openrouter",
                    openrouter_key,
                    "deepseek-ai/deepseek-v4-pro",
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(f"{req_prefix}DeepSeek batch translation failed: {e}")

            # Try Gemini 2.5 Flash via OpenRouter
            logger.info(f"{req_prefix}Batch: Trying Gemini 2.5 Flash (OpenRouter)...")
            try:
                res = try_cloud_ai(
                    "openrouter",
                    openrouter_key,
                    "google/gemini-2.5-flash",
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(
                    f"{req_prefix}Gemini OpenRouter batch translation failed: {e}"
                )

        elif provider == "gemini":
            # Try Direct Gemini API
            preferred = os.environ.get("PREFERRED_MODEL", "gemini-2.5-flash")
            logger.info(f"{req_prefix}Batch: Trying Gemini ({preferred}) Direct...")
            try:
                res = try_cloud_ai(
                    "gemini",
                    gemini_key,
                    preferred,
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(f"{req_prefix}Gemini Direct batch translation failed: {e}")

        elif provider == "openai":
            preferred = os.environ.get("PREFERRED_MODEL", "gpt-4o-mini")
            logger.info(f"{req_prefix}Batch: Trying OpenAI ({preferred}) Direct...")
            try:
                res = try_cloud_ai(
                    "openai",
                    api_key,
                    preferred,
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(f"{req_prefix}OpenAI Direct batch translation failed: {e}")

        elif provider == "anthropic":
            preferred = os.environ.get("PREFERRED_MODEL", "claude-3-5-sonnet-20241022")
            logger.info(f"{req_prefix}Batch: Trying Anthropic ({preferred}) Direct...")
            try:
                res = try_cloud_ai(
                    "anthropic",
                    anthropic_key,
                    preferred,
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(
                    f"{req_prefix}Anthropic Direct batch translation failed: {e}"
                )

        # Try Nvidia NIM
        if provider == "nvidia":
            nvidia_model = os.environ.get("PREFERRED_MODEL", "google/gemma-3n-e4b-it")
            logger.info(f"{req_prefix}Batch: Trying Nvidia model {nvidia_model}...")
            try:
                res = try_cloud_ai(
                    "nvidia",
                    nvidia_key,
                    nvidia_model,
                    prompt,
                    response_schema,
                    request_id=request_id,
                )
                if res:
                    return res
            except Exception as e:
                logger.error(f"{req_prefix}Nvidia batch translation failed: {e}")

    # Try Local LLM (Ollama/LMStudio)
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in ("true", "1", "yes")
    if not disable_local:
        local_provider = (
            os.environ.get("LOCAL_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio"))
            .lower()
            .strip()
        )
        logger.info(f"{req_prefix}Batch: Trying Local LLM ({local_provider})...")
        try:
            res = try_local_ai(prompt, bubbles_json, response_schema, request_id=request_id)
            if res:
                return res
        except Exception as e:
            logger.error(f"{req_prefix}Local LLM batch translation failed: {e}")
    else:
        logger.info(f"{req_prefix}Batch: Local LLM layer skipped (disabled via environment).")


def try_local_vlm_vision(
    model, prompt, base64_image, response_schema=None, request_id=None
):
    req_prefix = f"[{request_id}] " if request_id else ""
    local_provider = os.environ.get("LOCAL_LLM_PROVIDER", "ollama").lower().strip()
    local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
    if not local_endpoint:
        if local_provider == "ollama":
            local_endpoint = "http://ollama:11434/v1/chat/completions"
        else:
            local_endpoint = "http://host.docker.internal:1234/v1/chat/completions"

    if not local_endpoint.endswith(
        "/v1/chat/completions"
    ) and not local_endpoint.endswith("/api/v1/chat"):
        if local_endpoint.endswith("/"):
            local_endpoint += "v1/chat/completions"
        else:
            local_endpoint += "/v1/chat/completions"

    payload = {
        "model": model,
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
    }

    if response_schema:
        if local_provider == "ollama":
            payload["format"] = "json"
        else:
            payload["response_format"] = {"type": "json_object"}

    from worker.utils.lock import acquire_lock

    try:
        with acquire_lock("local-llm"):
            logger.info(
                f"{req_prefix}Sending local VLM request to {local_endpoint} using model {model}..."
            )
            start = time.perf_counter()
            response = requests.post(local_endpoint, json=payload, timeout=90)
            elapsed = time.perf_counter() - start
            logger.info(f"{req_prefix}Local VLM query completed in {elapsed:.2f}s")

            if response.status_code == 200:
                res_json = response.json()
                choice = res_json["choices"][0]["message"]["content"]
                return choice
            else:
                logger.error(
                    f"{req_prefix}Local VLM API returned status {response.status_code}: {response.text}"
                )
    except Exception as e:
        logger.error(f"{req_prefix}Error during local VLM query: {e}")
    return None


def translate_vlm_vision(
    img_bytes,
    unmatched_regions,
    context_str="",
    response_schema=None,
    request_id=None,
    source_lang="ja",
    target_lang="en",
):
    if not img_bytes:
        return None
    req_prefix = f"[{request_id}] " if request_id else ""

    import base64

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    bubbles_input = []
    for r in unmatched_regions:
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

    src_name = LANG_MAP.get(source_lang.lower(), source_lang)
    tgt_name = LANG_MAP.get(target_lang.lower(), target_lang)

    prompt = f"""{context_str}These OCR regions were extracted from this manga page using automated OCR.

IMPORTANT — Before translating:
1. Verify each region's OCR text against the visible text in the image. If the OCR text
   appears incorrect (garbled, truncated, or mis-recognized), use the text you actually
   see in the image instead.
2. For each bubble, identify the speaker based on visual cues (speech bubble tails,
   character positions, expressions, panel context).
3. If a region's "regionType" is "sfx", look at the visual style of the text (bold,
   angular, wavy) to inform your transliteration style.

Translate each region from {src_name} into natural manga {tgt_name}.
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

Return ONLY valid JSON.

Input:
{json.dumps(bubbles_input, ensure_ascii=False, indent=2)}
"""
    provider = os.environ.get("MODEL_PROVIDER", "").lower().strip()
    api_key = os.environ.get("API_KEY", "").strip()

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or (
        api_key if provider == "openrouter" else ""
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or (
        api_key if provider == "gemini" else ""
    )
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip() or (
        api_key if provider == "nvidia" else ""
    )

    if openrouter_key:
        logger.info(f"{req_prefix}VLM: Trying vision model via OpenRouter...")
        vlm_model = os.environ.get("PREFERRED_VLM_MODEL", "google/gemini-2.5-flash")
        try:
            res = try_cloud_ai_vision(
                "openrouter",
                openrouter_key,
                vlm_model,
                prompt,
                base64_image,
                response_schema,
                request_id=request_id,
            )
            if res:
                return res
        except Exception as e:
            logger.error(
                f"{req_prefix}VLM vision translation via OpenRouter failed: {e}"
            )

    if (provider == "gemini" or gemini_key) and gemini_key:
        logger.info(f"{req_prefix}VLM: Trying vision model via Gemini...")
        vlm_model = os.environ.get("PREFERRED_MODEL", "gemini-1.5-flash")
        try:
            res = try_cloud_ai_vision(
                "gemini",
                gemini_key or api_key,
                vlm_model,
                prompt,
                base64_image,
                response_schema,
                request_id=request_id,
            )
            if res:
                return res
        except Exception as e:
            logger.error(f"{req_prefix}VLM vision translation via Gemini failed: {e}")

    if (provider == "nvidia" or nvidia_key) and nvidia_key:
        logger.info(f"{req_prefix}VLM: Trying vision model via Nvidia...")
        nvidia_vlm_model = (
            os.environ.get("NVIDIA_VLM_MODEL", "").strip()
            or os.environ.get("PREFERRED_VLM_MODEL", "").strip()
        )
        if not nvidia_vlm_model:
            nvidia_vlm_model = "nvidia/nemotron-nano-12b-v2-vl"
        try:
            res = try_cloud_ai_vision(
                "nvidia",
                nvidia_key,
                nvidia_vlm_model,
                prompt,
                base64_image,
                response_schema,
                request_id=request_id,
            )
            if res:
                return res
        except Exception as e:
            logger.error(f"{req_prefix}VLM vision translation via Nvidia failed: {e}")

    # Fallback to local VLM if cloud failed or skipped, and LOCAL_VLM_MODEL is configured
    local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
    disable_local = os.environ.get("DISABLE_LOCAL_LLM", "").strip().lower() in ("true", "1", "yes")
    if local_vlm_model and not disable_local:
        logger.info(f"{req_prefix}VLM: Trying local VLM model '{local_vlm_model}'...")
        try:
            res = try_local_vlm_vision(
                local_vlm_model,
                prompt,
                base64_image,
                response_schema,
                request_id=request_id,
            )
            if res:
                return res
        except Exception as e:
            logger.error(f"{req_prefix}Local VLM vision translation failed: {e}")
    elif local_vlm_model and disable_local:
        logger.info(f"{req_prefix}VLM: Local VLM model '{local_vlm_model}' skipped (disabled via environment).")

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
        logger.info(
            f"{req_prefix}Sending batch request of {len(unmatched_regions)} bubbles to DeepL API..."
        )
        headers = {
            "Authorization": f"DeepL-Auth-Key {deepl_key}",
            "Content-Type": "application/json",
        }
        texts = [r["text"] for r in unmatched_regions]
        payload = {"text": texts, "target_lang": target_lang.upper()}

        start = time.perf_counter()
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{req_prefix}Provider=deepl " f"Model=deepl_batch " f"Time={elapsed:.2f}s"
        )

        if res.status_code == 200:
            res_json = res.json()
            translations = res_json["translations"]
            mapping = {}
            for i, r in enumerate(unmatched_regions):
                mapping[r["id"]] = translations[i]["text"]
            return mapping
        else:
            logger.error(
                f"{req_prefix}DeepL API returned error: {res.status_code} - {res.text}"
            )
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
                context_str += f"Roster & Editorial Style Guidelines:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
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
