import sys

with open("worker/services/translation.py", "r") as f:
    lines = f.readlines()

new_content = """import litellm

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

    litellm_model = model
    kwargs = {}

    if provider == "openrouter":
        litellm_model = f"openrouter/{model}" if model else "openrouter/meta-llama/llama-3-8b-instruct:free"
        kwargs["api_key"] = api_key
    elif provider == "openai":
        litellm_model = model or "gpt-4o-mini"
        kwargs["api_key"] = api_key
    elif provider == "anthropic":
        litellm_model = f"anthropic/{model}" if model else "anthropic/claude-3-5-sonnet-20241022"
        kwargs["api_key"] = api_key
    elif provider == "gemini":
        litellm_model = f"gemini/{model}" if model else "gemini/gemini-1.5-flash"
        kwargs["api_key"] = api_key
    elif provider == "nvidia":
        litellm_model = f"openai/{model}" if model else "openai/nvidia/riva-translate-4b-instruct-v1.1"
        kwargs["api_key"] = api_key
        kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
    else:
        return None

    messages = [{"role": "user", "content": prompt}]
    if response_schema:
        if provider == "nvidia":
            kwargs["response_format"] = {"type": "json_object"}
            messages.insert(0, {"role": "system", "content": MANGA_TRANSLATION_JSON_SYSTEM_PROMPT})
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "schema": response_schema},
            }

    try:
        logger.info(f"{req_prefix}Sending request to '{provider}' using model '{litellm_model}'...")
        start = time.perf_counter()
        
        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            timeout=45 if provider == "nvidia" else 30,
            **kwargs
        )
        
        elapsed = time.perf_counter() - start
        logger.info(f"{req_prefix}Provider={provider} Model={litellm_model} Time={elapsed:.2f}s")
        
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            logger.info(
                f"{req_prefix}Tokens in={prompt_tokens} out={completion_tokens} total={total_tokens}"
            )
            cost = estimate_cost(model, prompt_tokens, completion_tokens, provider)
            logger.info(f"{req_prefix}Estimated cost: ${cost:.5f}")

        return response.choices[0].message.content
    except litellm.RateLimitError as e:
        logger.warning(
            f"{req_prefix}Provider '{provider}' returned 429 (Too Many Requests). Initiating a 60-second cooldown."
        )
        PROVIDER_COOLDOWNS[provider] = time.time() + 60.0
        return None
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

    litellm_model = model
    kwargs = {}

    if provider == "openrouter":
        litellm_model = f"openrouter/{model}" if model else "openrouter/meta-llama/llama-3-8b-instruct:free"
        kwargs["api_key"] = api_key
    elif provider == "gemini":
        litellm_model = f"gemini/{model}" if model else "gemini/gemini-1.5-flash"
        kwargs["api_key"] = api_key
    elif provider == "nvidia":
        litellm_model = f"openai/{model}" if model else "openai/nvidia/nemotron-nano-12b-v2-vl"
        kwargs["api_key"] = api_key
        kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
    else:
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]

    if response_schema:
        if provider == "nvidia":
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "schema": response_schema},
            }

    try:
        logger.info(f"{req_prefix}Sending vision request to '{provider}' using model '{litellm_model}'...")
        start = time.perf_counter()
        
        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            timeout=45,
            **kwargs
        )
        
        elapsed = time.perf_counter() - start
        logger.info(f"{req_prefix}Provider={provider} Model={litellm_model} Time={elapsed:.2f}s")

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            logger.info(
                f"{req_prefix}Tokens in={prompt_tokens} out={completion_tokens} total={total_tokens}"
            )
            cost = estimate_cost(model, prompt_tokens, completion_tokens, provider)
            logger.info(f"{req_prefix}Estimated cost: ${cost:.5f}")

        return response.choices[0].message.content
    except litellm.RateLimitError as e:
        logger.warning(
            f"{req_prefix}Vision provider '{provider}' returned 429 (Too Many Requests). Initiating a 60-second cooldown."
        )
        PROVIDER_COOLDOWNS[provider] = time.time() + 60.0
        return None
    except Exception as e:
        logger.error(f"{req_prefix}Vision Translation failed: {e}")
        return None


def try_local_ai(prompt, text, response_schema=None, request_id=None):
    req_prefix = f"[{request_id}] " if request_id else ""
    enforce_rate_limit()
    
    local_provider = os.environ.get("LOCAL_LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "lmstudio")).lower().strip()
    local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", os.environ.get("LLM_ENDPOINT", "")).strip()
    model = os.environ.get("LOCAL_LLM_MODEL", "gemma3:4b")

    if not local_endpoint:
        if local_provider == "ollama":
            local_endpoint = "http://ollama:11434"
        else:
            local_endpoint = "http://host.docker.internal:1234/v1"
    else:
        local_endpoint = local_endpoint.replace("/chat/completions", "").replace("/api/v1/chat", "")

    endpoints_to_try = [local_endpoint]
    if "localhost" in local_endpoint:
        endpoints_to_try.append(local_endpoint.replace("localhost", "host.docker.internal"))
    elif "host.docker.internal" in local_endpoint:
        endpoints_to_try.append(local_endpoint.replace("host.docker.internal", "localhost"))

    system_pr = MANGA_TRANSLATION_JSON_SYSTEM_PROMPT if response_schema else MANGA_TRANSLATION_SYSTEM_PROMPT
    
    kwargs = {}
    if response_schema:
        kwargs["response_format"] = {"type": "json_object"}

    for endpoint in endpoints_to_try:
        try:
            logger.info(f"{req_prefix}Trying Local AI endpoint '{endpoint}' using model '{model}'...")
            
            litellm_model = f"ollama/{model}" if local_provider == "ollama" else f"openai/{model}"
            
            from worker.utils.lock import acquire_lock
            with acquire_lock("local-llm"):
                start = time.perf_counter()
                response = litellm.completion(
                    model=litellm_model,
                    api_base=endpoint,
                    messages=[
                        {"role": "system", "content": system_pr},
                        {"role": "user", "content": text},
                    ],
                    timeout=300,
                    **kwargs
                )
                elapsed = time.perf_counter() - start
                
            logger.info(f"{req_prefix}Provider={local_provider} Model={model} Time={elapsed:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{req_prefix}Local AI connection failed for '{endpoint}': {e}")

    return None
"""

lines = lines[:321] + [new_content + "\n"] + lines[806:]

with open("worker/services/translation.py", "w") as f:
    f.writelines(lines)
