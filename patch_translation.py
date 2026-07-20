import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

# Add routing_strategy parameter
def patch_signature(func_name, args):
    global content
    content = content.replace(f"def {func_name}({args}):", f"def {func_name}({args}, routing_strategy=None):")

patch_signature("try_cloud_ai", "provider, api_key, model, prompt, response_schema=None, request_id=None")
patch_signature("try_cloud_ai_vision", "provider, api_key, model, prompt, img_bytes, mime_type=\"image/jpeg\", response_schema=None, request_id=None")
patch_signature("try_cloud_ai_vision_batch", "provider, api_key, model, prompt, images_data, response_schema=None, request_id=None")

# Inject OpenRouter payload
inject_code = """
                if provider == "openrouter":
                    payload["plugins"] = [{"id": "response-healing"}]

        if provider == "openrouter" and routing_strategy == "lowest-cost":
            payload["provider"] = {
                "allow_fallbacks": False,
                "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"]
            }
"""

content = content.replace(
    "                if provider == \"openrouter\":\n                    payload[\"plugins\"] = [{\"id\": \"response-healing\"}]",
    inject_code
)

with open("worker/services/translation.py", "w") as f:
    f.write(content)

