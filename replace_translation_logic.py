import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

# For translate_batch_llm:
# We need to replace from:
#         if provider == "openrouter" and api_key:
# to
#     # Try Local LLM (Ollama/LMStudio)

new_batch_logic = """
        if api_key:
            logger.info(f"{req_prefix}Batch: Trying provider '{provider}' with model '{user_model}'...")
            try:
                res = try_cloud_ai(provider, api_key, user_model, prompt, response_schema, request_id=request_id)
                if res:
                    return res
            except Exception as e:
                logger.error(f"{req_prefix}Batch translation with model '{user_model}' failed: {e}")
            
            # Fallback to global default model
            global_model = TL_CONFIG.llm_model
            global_provider = TL_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Batch: Falling back to global default model '{global_model}'...")
                try:
                    res = try_cloud_ai(provider, api_key, global_model, prompt, response_schema, request_id=request_id)
                    if res:
                        return res
                except Exception as e:
                    logger.error(f"{req_prefix}Batch translation with global fallback model '{global_model}' failed: {e}")
"""

start_str = '        if provider == "openrouter" and api_key:'
end_str = '    # Try Local LLM (Ollama/LMStudio)'

start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_batch_logic.strip("\n") + "\n\n" + content[end_idx:]

# Now for translate_text:
# It starts around:
#         if provider == "openrouter" and api_key:
# inside translate_text. We need to find the correct one.

start_str_text = '        # 1. Cloud LLM Layer\n        if provider == "openrouter" and api_key:'
end_str_text = '    # 2. Local Ollama/LMStudio Layer'

new_text_logic = """        # 1. Cloud LLM Layer
        if api_key:
            logger.info(f"{req_prefix}Trying provider '{provider}' with model '{user_model}'...")
            try:
                translated = try_cloud_ai(provider, api_key, user_model, prompt, request_id=request_id)
                if translated:
                    cleaned = clean_translated_text(translated)
                    if is_valid_translation(text, cleaned, request_id=request_id):
                        return cleaned
            except Exception as e:
                logger.error(f"{req_prefix}Translation with model '{user_model}' failed: {e}")
            
            # Fallback to global default model
            global_model = TL_CONFIG.llm_model
            global_provider = TL_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                logger.info(f"{req_prefix}Falling back to global default model '{global_model}'...")
                try:
                    translated = try_cloud_ai(provider, api_key, global_model, prompt, request_id=request_id)
                    if translated:
                        cleaned = clean_translated_text(translated)
                        if is_valid_translation(text, cleaned, request_id=request_id):
                            return cleaned
                except Exception as e:
                    logger.error(f"{req_prefix}Translation with global fallback model '{global_model}' failed: {e}")
"""

start_idx_t = content.find(start_str_text)
end_idx_t = content.find(end_str_text, start_idx_t)

if start_idx_t != -1 and end_idx_t != -1:
    content = content[:start_idx_t] + new_text_logic.strip("\n") + "\n\n" + content[end_idx_t:]

with open("worker/services/translation.py", "w") as f:
    f.write(content)
