import re

with open("worker/services/ocr.py", "r") as f:
    content = f.read()

content = content.replace(
    "def try_cloud_ocr(img_crop_bytes, provider, api_key, model, qa_feedback=None):",
    "def try_cloud_ocr(img_crop_bytes, provider, api_key, model, qa_feedback=None, routing_strategy=None):"
)

# Also need to inject into OpenRouter payload for OCR
inject_code = """
        if provider == "openrouter" and routing_strategy == "lowest-cost":
            payload["provider"] = {
                "allow_fallbacks": False,
                "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"]
            }
"""

content = content.replace(
    "    max_retries = 3",
    inject_code + "\n    max_retries = 3"
)

with open("worker/services/ocr.py", "w") as f:
    f.write(content)
