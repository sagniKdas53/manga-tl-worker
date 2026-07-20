import re

with open("worker/handlers/qa.py", "r") as f:
    content = f.read()

def replace_model_iteration(content, is_vlm=False):
    # Regex to find the model iteration block and replace it with the new fallback logic
    # Find:
    #     if provider:
    #         if provider in ("openai", "openrouter", "gemini", "nvidia", "anthropic"):
    #             ...
    #         else:
    #             qa_response = attempt_llm(provider)
    
    # We will search for:
    #     if provider:
    #         if provider in ("openai", "openrouter", "gemini", "nvidia", "anthropic"):
    
    model_field = "qaVlmModel" if is_vlm else "qaLlmModel"
    config_field = "vlm_model" if is_vlm else "llm_model"
    attempt_func = "attempt_vlm" if is_vlm else "attempt_llm"
    response_var = "qa_response_vlm" if is_vlm else "qa_response"
    
    pattern = r'    # Try preferred provider/models\n    if provider:\n        if provider in \("openai", "openrouter", "gemini", "nvidia", "anthropic"\):.*?else:\n            ' + response_var + ' = ' + attempt_func + r'\(provider\)'
    
    new_logic = f"""    # Try preferred provider/models
    if provider:
        user_model = job_data.get("{model_field}") or getattr(QA_CONFIG, "{config_field}")
        {response_var} = {attempt_func}(provider, user_model)
        
        if not {response_var}:
            # Fallback to global default model
            global_model = getattr(QA_CONFIG, "{config_field}")
            global_provider = getattr(QA_CONFIG, "provider")
            if global_provider == provider and global_model and global_model != user_model:
                print(f"[QA] Falling back to global default model '{{global_model}}'...", flush=True)
                {response_var} = {attempt_func}(provider, global_model)
            else:
                print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)"""
                
    content, count = re.subn(pattern, new_logic, content, flags=re.DOTALL)
    print(f"Replaced {count} instances for {model_field}")
    return content

content = replace_model_iteration(content, is_vlm=False)
content = replace_model_iteration(content, is_vlm=True)

with open("worker/handlers/qa.py", "w") as f:
    f.write(content)
