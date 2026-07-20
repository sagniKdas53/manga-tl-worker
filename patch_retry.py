import re

with open("worker/services/translation.py", "r") as f:
    content = f.read()

for func_name, failure_msg in [
    ("try_cloud_ai", "Cloud LLM Translation failed: {e}"),
    ("try_cloud_ai_vision", "Vision Translation failed: {e}"),
    ("try_cloud_ai_vision_batch", "Vision Batch OCR failed: {e}")
]:
    target = f"""        except Exception as e:
            logger.error(f"{{req_prefix}}{failure_msg}")
            if "response" in locals() and hasattr(response, "text"):
                logger.error(f"Response text: {{response.text}}")
            return None"""
            
    replacement = f"""        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            is_transient = isinstance(e, requests.exceptions.Timeout) or isinstance(e, requests.exceptions.ConnectionError) or status_code in (429, 500, 502, 503, 504)
            if is_transient and attempt < max_retries:
                sleep_time = base_backoff * (2**attempt)
                logger.warning(f"{{req_prefix}}Provider '{{provider}}' transient error: {{e}}. Retrying in {{sleep_time:.2f}}s...")
                time.sleep(sleep_time)
                continue
            logger.error(f"{{req_prefix}}{failure_msg}")
            if "response" in locals() and hasattr(response, "text"):
                logger.error(f"Response text: {{response.text}}")
            return None
        except Exception as e:
            logger.error(f"{{req_prefix}}{failure_msg}")
            return None"""
            
    content = content.replace(target, replacement)

content = re.sub(r'timeout=90 if provider == "nvidia" else 60', 'timeout=(10, 45)', content)
content = re.sub(r'timeout=90\)', 'timeout=(10, 45))', content)
content = re.sub(r'timeout=120\)', 'timeout=(10, 60))', content) # batch might need 60

with open("worker/services/translation.py", "w") as f:
    f.write(content)
