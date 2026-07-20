import re

with open("worker/handlers/qa.py", "r") as f:
    content = f.read()

# Replace VLM logic
pattern_vlm = re.compile(
    r'        if provider in \("openai", "openrouter", "gemini", "nvidia", "anthropic"\):\n'
    r'            user_model = job_data\.get\("qaVlmModel"\) or QA_CONFIG\.vlm_model\n'
    r'.*?'
    r'        else:\n'
    r'            qa_response_vlm = attempt_vlm\(provider\)',
    re.DOTALL
)

new_vlm = """        user_model = job_data.get("qaVlmModel") or QA_CONFIG.vlm_model
        qa_response_vlm = attempt_vlm(provider, user_model)
        
        if not qa_response_vlm:
            global_model = QA_CONFIG.vlm_model
            global_provider = QA_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                print(f"[QA] Falling back to global default VLM model '{global_model}'...", flush=True)
                qa_response_vlm = attempt_vlm(provider, global_model)
            else:
                print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)"""

content, count_vlm = pattern_vlm.subn(new_vlm, content)
print(f"Replaced {count_vlm} VLM blocks.")

# Replace LLM logic
pattern_llm = re.compile(
    r'        if provider in \("openai", "openrouter", "gemini", "nvidia", "anthropic"\):\n'
    r'            user_model = job_data\.get\("qaLlmModel"\) or QA_CONFIG\.llm_model\n'
    r'.*?'
    r'        else:\n'
    r'            qa_response = attempt_llm\(provider\)',
    re.DOTALL
)

new_llm = """        user_model = job_data.get("qaLlmModel") or QA_CONFIG.llm_model
        qa_response = attempt_llm(provider, user_model)
        
        if not qa_response:
            global_model = QA_CONFIG.llm_model
            global_provider = QA_CONFIG.provider
            if global_provider == provider and global_model and global_model != user_model:
                print(f"[QA] Falling back to global default LLM model '{global_model}'...", flush=True)
                qa_response = attempt_llm(provider, global_model)
            else:
                print(f"[QA] No fallback applied (global provider different or model identical).", flush=True)"""

content, count_llm = pattern_llm.subn(new_llm, content)
print(f"Replaced {count_llm} LLM blocks.")

with open("worker/handlers/qa.py", "w") as f:
    f.write(content)
