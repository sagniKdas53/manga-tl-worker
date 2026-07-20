import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

def replace_block(content, start_str, end_str, new_block):
    start_idx = content.find(start_str)
    if start_idx == -1:
        print("Start not found:", start_str[:20])
        return content
    end_idx = content.find(end_str, start_idx)
    if end_idx == -1:
        print("End not found:", end_str[:20])
        return content
    
    return content[:start_idx] + new_block + content[end_idx:]


text_start = """    if local_only:
        logger.info(f"{req_prefix}LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers.")
    else:
        # 1. Cloud LLM Layer"""

text_end = """    # 2. Local Ollama/LMStudio Layer"""

text_new = """    if local_only:
        logger.info(f"{req_prefix}LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers.")
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
            if global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Falling back to global default model '{global_model}'...")
                translated = try_cloud_ai(provider, api_key, global_model, prompt, request_id=request_id)
                if translated:
                    cleaned = clean_translated_text(translated)
                    if is_valid_translation(text, cleaned, request_id=request_id):
                        return cleaned
                else:
                    logger.error(f"{req_prefix}Translation with global fallback model '{global_model}' failed.")
            else:
                logger.info(f"{req_prefix}No fallback applied (global provider different or model identical).")

"""

content = replace_block(content, text_start, text_end, text_new)

batch_start = """    if local_only:
        logger.info(f"{req_prefix}Batch: LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers.")
    else:
        if provider == "openrouter" and api_key:"""

batch_end = """    # Try Local LLM (Ollama/LMStudio)"""

batch_new = """    if local_only:
        logger.info(f"{req_prefix}Batch: LOCAL_ONLY mode (provider='{provider}') — skipping cloud AI tiers.")
    else:
        if api_key:
            logger.info(f"{req_prefix}Batch: Trying provider '{provider}' with model '{user_model}'...")
            res = try_cloud_ai(provider, api_key, user_model, prompt, response_schema, request_id=request_id)
            if res:
                return res
            
            # Fallback to global default model
            global_model = TL_CONFIG.llm_model
            global_provider = TL_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Batch: Falling back to global default model '{global_model}'...")
                res = try_cloud_ai(provider, api_key, global_model, prompt, response_schema, request_id=request_id)
                if res:
                    return res
                else:
                    logger.error(f"{req_prefix}Batch translation with global fallback model '{global_model}' failed.")
            else:
                logger.info(f"{req_prefix}Batch: No fallback applied (global provider different or model identical).")

"""

content = replace_block(content, batch_start, batch_end, batch_new)

with open("worker/services/translation.py", "w") as f:
    f.write(content)

