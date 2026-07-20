import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

# Helper function
helper_func = """
def _inject_openrouter_routing(provider, routing_strategy, payload):
    if provider == "openrouter" and routing_strategy == "lowest-cost":
        payload["provider"] = {
            "allow_fallbacks": False,
            "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"]
        }
"""
idx = content.find("def wait_for_cooldown(")
content = content[:idx] + helper_func + "\n" + content[idx:]

# Inject into try_cloud_ai
content = content.replace(
    "    max_retries = 3",
    "    _inject_openrouter_routing(provider, routing_strategy, payload)\n    max_retries = 3"
)

with open("worker/services/translation.py", "w") as f:
    f.write(content)
