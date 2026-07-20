import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

# Remove the old hardcoded block
old_block = """
        if provider == "openrouter" and routing_strategy == "lowest-cost":
            payload["provider"] = {
                "allow_fallbacks": False,
                "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"]
            }
"""
content = content.replace(old_block, "")

# Add helper call to try_cloud_ai_vision
content = content.replace(
    "            if provider == \"openrouter\":\n                payload[\"plugins\"] = [{\"id\": \"response-healing\"}]",
    "            if provider == \"openrouter\":\n                payload[\"plugins\"] = [{\"id\": \"response-healing\"}]\n\n    _inject_openrouter_routing(provider, routing_strategy, payload)"
)

with open("worker/services/translation.py", "w") as f:
    f.write(content)
