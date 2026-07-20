import time
import requests
from worker.config import logger

def retry_with_backoff(func, max_retries=3, base_delay=2.0):
    """
    Executes a function and retries it on transient requests.exceptions.
    Transients include Timeout and 5xx errors (e.g. 502, 503, 504), or 429 Rate Limits.
    Uses exponential backoff: delay = base_delay * (2 ^ attempt).
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            # Determine if it's transient
            is_transient = False
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            
            if isinstance(e, requests.exceptions.Timeout):
                is_transient = True
            elif isinstance(e, requests.exceptions.ConnectionError):
                is_transient = True
            elif status_code in (429, 500, 502, 503, 504):
                is_transient = True
                
            if not is_transient or attempt == max_retries:
                logger.error(f"[Retry] Action failed after {attempt} retries: {e}")
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"[Retry] Transient error: {e}. Retrying in {delay} seconds (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)
