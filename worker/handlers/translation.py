import concurrent.futures
import logging
import uuid

import requests

from worker.config import (
    BACKEND_HEADERS,
    CALLBACK_URL,
    TL_CONFIG,
    logger,
    redis_client,
)
from worker.services.layout import chunk_regions_by_conversation
from worker.services.translation import (
    TRANSLATION_JSON_SCHEMA,
    build_context_string,
    is_valid_translation,
    parse_and_validate_batch,
    should_translate_region,
    translate_batch_llm,
    translate_text,
)


def process_translation(job_data):
    from worker.utils.rate_limit import reset_job_costs

    reset_job_costs()
    image_id = job_data.get("imageId")
    page_id = job_data.get("pageId")
    request_id = str(uuid.uuid4())[:8]
    req_prefix = f"[{request_id}] "

    source_lang = job_data.get("sourceLanguage", "ja").strip().lower()
    target_lang = job_data.get("targetLanguage", "en").strip().lower()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{req_prefix}Translation Inputs: job_data={job_data}")

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:translation")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    logger.info(
        f"{req_prefix}Processing translation for page: {page_id or image_id} ({source_lang} -> {target_lang}){progress_str}"
    )

    try:
        if page_id:
            backend_url = CALLBACK_URL.replace("/jobs/callback", f"/pages/{page_id}/details")
        else:
            backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            raise Exception(f"Failed to get page/image info: {res.status_code}")
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
        conversations = image_info.get("conversations", [])
    except Exception as e:
        logger.error(f"{req_prefix}Error fetching image details: {e}")
        raise

    # OCR Quality Filter & Separation
    resolved_translations = {}
    unmatched_regions = []

    for r in ocr_regions:
        if not should_translate_region(r):
            # Bypass translation for garbage, keep original text
            resolved_translations[r["id"]] = {"translatedText": r["text"]}
        else:
            unmatched_regions.append(r)

    # Translate unmatched regions
    if unmatched_regions:
        batch_mapping = {}

        provider = job_data.get("tlProvider") or TL_CONFIG.provider
        tl_model = job_data.get("tlModel")
        routing_strategy = job_data.get("routingStrategy") or "lowest-cost"
        use_fallback_models = job_data.get("useFallbackModels", True)
        local_only = provider in ("ollama", "lmstudio")
        max_batch_size = 5 if local_only else 8

        logger.info(f"{req_prefix}Batch size set to {max_batch_size} (local_only={local_only})")

        # Build context string
        import json

        context_str = build_context_string(image_info)

        # Compile all page regions/bubbles into a single page manifest to pass as translation context
        page_manifest_entries = []
        for r in ocr_regions:
            page_manifest_entries.append(
                {
                    "id": r["id"],
                    "regionType": r.get("regionType") or r.get("region_type") or "speech",
                    "readingOrder": r.get("bubbleReadingOrder") or 0,
                    "conversationGroup": r.get("conversationId") or None,
                    "text": r["text"],
                }
            )
        page_manifest_str = json.dumps(page_manifest_entries, ensure_ascii=False, indent=2)
        manifest_context = (
            f"Full Page Region Manifest (for conversational flow and context):\n{page_manifest_str}\n---\n"
        )
        context_str = manifest_context + context_str

        # Chunk regions respecting conversation grouping
        unmatched_chunks = chunk_regions_by_conversation(unmatched_regions, conversations, max_batch_size)

        def process_chunk(idx, chunk):
            logger.info(
                f"{req_prefix}Processing batch chunk {idx + 1}/{len(unmatched_chunks)} ({len(chunk)} regions)..."
            )
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{req_prefix}translate_batch_llm input chunk: {chunk}")
                    logger.debug(f"{req_prefix}translate_batch_llm prompt context: {context_str}")

                batch_res = translate_batch_llm(
                    chunk,
                    context_str,
                    TRANSLATION_JSON_SCHEMA,
                    request_id=request_id,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    provider=provider,
                    llm_model=tl_model,
                    routing_strategy=routing_strategy,
                    use_fallback_models=use_fallback_models,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{req_prefix}translate_batch_llm output: {batch_res}")

                return parse_and_validate_batch(batch_res, chunk)
            except Exception as e:
                logger.error(f"{req_prefix}Standard batch translation failed for chunk {idx + 1}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(process_chunk, idx, chunk): chunk for idx, chunk in enumerate(unmatched_chunks)}
            for future in concurrent.futures.as_completed(futures):
                chunk_mapping = future.result()
                if chunk_mapping:
                    for rid, trans in chunk_mapping.items():
                        batch_mapping[rid] = trans

        failed_batch_regions = []
        # Validate output for each unmatched region
        for r in unmatched_regions:
            rid = r["id"]
            translated = batch_mapping.get(rid)

            translated_text = None
            if isinstance(translated, dict):
                translated_text = translated.get("translatedText")
            elif isinstance(translated, str):
                translated_text = translated

            # Run sanity check
            if translated_text and is_valid_translation(r["text"], translated_text, request_id=request_id):
                resolved_translations[rid] = translated
            else:
                failed_batch_regions.append(r)

        # 3. Retry failed items
        LOCAL_AI_MAX_BATCH_RETRIES = 1
        if failed_batch_regions:
            from worker.services.translation import wait_for_cooldown

            wait_for_cooldown(provider)
            logger.info(f"{req_prefix}Retry pass 1")
            logger.info(
                f"{req_prefix}Retrying {len(failed_batch_regions)} failed items in batch (max {LOCAL_AI_MAX_BATCH_RETRIES} retry pass)..."
            )
            retry_chunks = chunk_regions_by_conversation(failed_batch_regions, conversations, max_batch_size)

            def process_retry_chunk(idx, r_chunk):
                logger.info(
                    f"{req_prefix}Processing retry batch chunk {idx + 1}/{len(retry_chunks)} ({len(r_chunk)} regions)..."
                )
                try:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req_prefix}Retry translate_batch_llm input chunk: {r_chunk}")

                    retry_res = translate_batch_llm(
                        r_chunk,
                        context_str,
                        TRANSLATION_JSON_SCHEMA,
                        request_id=request_id,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        provider=provider,
                        llm_model=tl_model,
                        routing_strategy=routing_strategy,
                        use_fallback_models=use_fallback_models,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req_prefix}Retry translate_batch_llm output: {retry_res}")

                    return parse_and_validate_batch(retry_res, r_chunk)
                except Exception as e:
                    logger.error(f"{req_prefix}Retry batch chunk {idx + 1} translation failed: {e}")
                    return None

            retry_mapping = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = {
                    executor.submit(process_retry_chunk, idx, r_chunk): r_chunk
                    for idx, r_chunk in enumerate(retry_chunks)
                }
                for future in concurrent.futures.as_completed(futures):
                    r_chunk_mapping = future.result()
                    if r_chunk_mapping:
                        for rid, trans in r_chunk_mapping.items():
                            retry_mapping[rid] = trans

            still_failed_regions = []
            for r in failed_batch_regions:
                rid = r["id"]
                translated = retry_mapping.get(rid)

                translated_text = None
                if isinstance(translated, dict):
                    translated_text = translated.get("translatedText")
                elif isinstance(translated, str):
                    translated_text = translated

                if translated_text and is_valid_translation(r["text"], translated_text, request_id=request_id):
                    resolved_translations[rid] = translated
                else:
                    still_failed_regions.append(r)

            # 4. Individual fallback for still-failed regions
            if still_failed_regions and use_fallback_models:
                from worker.services.translation import wait_for_cooldown

                wait_for_cooldown(provider)
                logger.info(f"{req_prefix}Individual fallback")
                logger.info(
                    f"{req_prefix}Falling back to individual translation for {len(still_failed_regions)} regions (attempt 3/3)..."
                )
                for r in still_failed_regions:
                    rid = r["id"]
                    text = r["text"]
                    lang = r["detectedLanguage"]

                    translated = translate_text(
                        text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        request_id=request_id,
                        use_fallback_models=use_fallback_models,
                    )
                    if translated and is_valid_translation(text, translated, request_id=request_id):
                        resolved_translations[rid] = {
                            "translatedText": translated,
                            "translationNotes": "Individual translation fallback",
                            "emotion": "",
                            "tone": "",
                        }
                    else:
                        logger.warning(f"{req_prefix}Giving up on '{text}' after 3 attempts.")
                        resolved_translations[rid] = None

    # Format the final callback response
    translations = []
    for r in ocr_regions:
        rid = r["id"]
        text = r["text"]
        lang = r["detectedLanguage"]

        translated = resolved_translations.get(rid)

        translated_text = None
        notes = ""
        emotion = ""
        tone = ""
        translation_score = 1.0
        if isinstance(translated, dict):
            translated_text = translated.get("translatedText")
            notes = translated.get("translationNotes", "")
            emotion = translated.get("emotion", "")
            tone = translated.get("tone", "")
            translation_score = float(translated.get("translationScore", 1.0))
        elif isinstance(translated, str):
            translated_text = translated

        translations.append(
            {
                "regionId": rid,
                "translatedText": translated_text,
                "translationFailed": (translated_text is None),
                "translationNotes": notes,
                "emotion": emotion,
                "tone": tone,
                "translationScore": translation_score,
                "modelIdentifier": f"{TL_CONFIG.provider}/{TL_CONFIG.llm_model}",
                "confidence": translation_score,
            }
        )
        logger.info(f"{req_prefix}Final: '{text}' ({lang}) -> '{translated_text}' (failed={translated_text is None})")

    # Check if all translations failed
    failed_count = sum(1 for t in translations if t.get("translationFailed"))
    all_failed = failed_count > 0 and failed_count == len(translations)
    if all_failed:
        logger.error(f"{req_prefix}All {failed_count} translation(s) failed — reporting error to backend")

    callback_payload = {
        "imageId": image_id,
        "pageId": page_id,
        "translations": translations,
        "allFailed": all_failed,
        "failedCount": failed_count,
        "totalCount": len(translations),
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
            f"{req_prefix}Translation job estimated cost: {cost_str} "
            f"(Tokens: in={total_prompt_tokens}, out={total_completion_tokens})"
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{req_prefix}Translation Outputs: callback_payload={callback_payload}")
    try:
        res = requests.post(
            f"{CALLBACK_URL}/translation",
            json=callback_payload,
            headers=BACKEND_HEADERS,
        )
        logger.info(f"{req_prefix}Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"{req_prefix}Failed to post callback to backend: {e}")
