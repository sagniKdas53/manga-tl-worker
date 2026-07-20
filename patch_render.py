import re

with open("worker/handlers/render.py", "r") as f:
    content = f.read()

# Pattern to replace the cache logic
old_cache_logic = """        from worker.config import RENDER_CACHE_DIR

        os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(RENDER_CACHE_DIR, f"{image_id}.png")
        with open(cache_path, "wb") as f:
            f.write(out_bytes)
        logger.info(f"[Render] Cached rendered image to {cache_path}")"""

new_cache_logic = """        from worker.config import RENDER_CACHE_DIR
        
        if os.environ.get("ENABLE_QA_AUDIT_CACHE", "false").lower() in ("true", "1", "yes"):
            os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(RENDER_CACHE_DIR, f"{image_id}.png")
            with open(cache_path, "wb") as f:
                f.write(out_bytes)
            logger.info(f"[Render] Cached rendered image to {cache_path}")"""

content = content.replace(old_cache_logic, new_cache_logic)

with open("worker/handlers/render.py", "w") as f:
    f.write(content)
