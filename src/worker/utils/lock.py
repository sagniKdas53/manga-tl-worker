import platform
import time
import uuid
from contextlib import contextmanager

from worker.config import logger, redis_client

# Release the lock only if we still hold it. AUDIT-W4: the release used to be an unconditional
# DELETE, so a holder that overran its own TTL would delete whatever lock had been acquired in the
# meantime — by a different worker — and two holders would then run concurrently. Comparing the
# token to the stored value and deleting in the same Lua call makes check-and-release atomic.
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


@contextmanager
def acquire_lock(lock_name: str, timeout: float = 600, expire: int = 600, node_scoped: bool = False):
    """
    Acquires an exclusive lock in Valkey/Redis to coordinate sequential tasks.
    - timeout: max time to block/wait for the lock to become free.
    - expire: TTL of the lock key in Valkey.
    - node_scoped: scope the lock to this container instead of the whole deployment.

    AUDIT-W4: the key used to embed ``platform.node()`` unconditionally, so every lock was
    per-container. For ``local-llm`` that defeated the lock entirely — ``LOCAL_LLM_ENDPOINT``
    resolves to a *shared* address (the ``ollama`` compose service, or LM Studio on the host), so N
    workers each took their own lock and then hammered the one instance concurrently, which is the
    overload the lock exists to prevent. ``WORKER_URLS`` is explicitly a comma-separated list, so
    multi-worker is a supported topology.

    Per-container is still right for locks that guard *this host's* CPU/GPU rather than a shared
    service — that is what ``node_scoped=True`` is for, and the ``ocr`` lock uses it.
    """
    lock_key = f"lock:{lock_name}"
    if node_scoped:
        lock_key = f"{lock_key}:{platform.node()}"
    # Identifies this holder, so the release can tell "my lock" from "someone else's lock that
    # happens to share the key".
    token = uuid.uuid4().hex
    start_time = time.time()
    acquired = False

    logger.info(f"Attempting to acquire Valkey lock: {lock_name}")
    while time.time() - start_time < timeout:
        # Try to set the lock key. nx=True sets only if it does not exist.
        if redis_client.set(lock_key, token, nx=True, ex=expire):
            acquired = True
            break
        time.sleep(0.5)

    if not acquired:
        logger.error(f"Failed to acquire Valkey lock: {lock_name} within {timeout}s timeout")
        raise TimeoutError(f"Could not acquire Valkey lock: {lock_name}")

    logger.info(f"Acquired Valkey lock: {lock_name}")
    try:
        yield
    finally:
        try:
            released = redis_client.eval(_RELEASE_SCRIPT, 1, lock_key, token)
            if released:
                logger.info(f"Released Valkey lock: {lock_name}")
            else:
                # The TTL fired while we were still working, and someone else may hold it now.
                logger.warning(
                    f"Valkey lock {lock_name} was no longer ours to release — it expired after "
                    f"{expire}s while the work was still running"
                )
        except Exception as e:
            logger.error(f"Error releasing Valkey lock {lock_name}: {e}")
