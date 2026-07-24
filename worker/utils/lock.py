import platform
import time
from contextlib import contextmanager

from worker.config import logger, redis_client


@contextmanager
def acquire_lock(lock_name, timeout=600, expire=600):
    """
    Acquires an exclusive lock in Valkey/Redis to coordinate sequential tasks.
    - timeout: max time to block/wait for the lock to become free.
    - expire: TTL of the lock key in Valkey.
    """
    node_id = platform.node()
    lock_key = f"lock:{lock_name}:{node_id}"
    start_time = time.time()
    acquired = False

    logger.info(f"Attempting to acquire Valkey lock: {lock_name}")
    while time.time() - start_time < timeout:
        # Try to set the lock key. nx=True sets only if it does not exist.
        if redis_client.set(lock_key, "1", nx=True, ex=expire):
            acquired = True
            break
        time.sleep(0.5)

    if not acquired:
        logger.error(
            f"Failed to acquire Valkey lock: {lock_name} within {timeout}s timeout"
        )
        raise TimeoutError(f"Could not acquire Valkey lock: {lock_name}")

    logger.info(f"Acquired Valkey lock: {lock_name}")
    try:
        yield
    finally:
        try:
            redis_client.delete(lock_key)
            logger.info(f"Released Valkey lock: {lock_name}")
        except Exception as e:
            logger.error(f"Error releasing Valkey lock {lock_name}: {e}")
