import re

with open("app.py", "r") as f:
    content = f.read()

cleanup_routine = """
def cleanup_audit_cache():
    import glob
    import os
    import time
    from worker.config import RENDER_CACHE_DIR

    if os.environ.get("ENABLE_QA_AUDIT_CACHE", "false").lower() in ("true", "1", "yes"):
        print("[Unified Worker] Cleaning up old QA audit cache files...", flush=True)
        try:
            now = time.time()
            max_age = 7 * 24 * 3600  # 7 days
            if os.path.exists(RENDER_CACHE_DIR):
                files = glob.glob(os.path.join(RENDER_CACHE_DIR, "*.png"))
                count = 0
                for f in files:
                    if os.path.isfile(f):
                        if (now - os.path.getmtime(f)) > max_age:
                            os.remove(f)
                            count += 1
                print(f"[Unified Worker] Cleaned up {count} old files in {RENDER_CACHE_DIR}.", flush=True)
        except Exception as e:
            print(f"[Unified Worker] Error cleaning up QA audit cache: {e}", flush=True)

def main():"""

content = content.replace("def main():", cleanup_routine)
content = content.replace("    # Seed models on startup", "    # Cleanup old audit cache if enabled\n    cleanup_audit_cache()\n\n    # Seed models on startup")

with open("app.py", "w") as f:
    f.write(content)
