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


def release_stale_node_locks() -> int:
    """Drop node-scoped locks left behind by a previous incarnation of this container.

    ``acquire_lock`` releases in a ``finally`` block, which SIGKILL does not run. The key then
    survives until its TTL fires — 600s during which every job needing that lock blocks on a
    holder that no longer exists.

    An OOM kill produces exactly that: the kernel kills the process, Docker restarts the
    container under the *same* hostname, and the new process inherits its predecessor's lock.
    Observed in production as a 9m38s stall on the page after a PP-OCRv5 detection OOM, which
    also pushed that job past the backend's staleness reaper so its result was discarded.

    Sweeping at startup is safe precisely because these keys are node-scoped: they are suffixed
    with this container's hostname, and this process has just started, so nothing it spawned can
    be mid-work. Any key still bearing this node's name is therefore an orphan by definition.
    Locks belonging to other nodes — and every non-node-scoped lock, which may be legitimately
    held by a *different* container — are left strictly alone.

    Returns the number of locks cleared.
    """
    pattern = f"lock:*:{platform.node()}"
    released: list[str] = []
    try:
        for raw in redis_client.scan_iter(match=pattern, count=100):
            key = raw.decode() if isinstance(raw, bytes) else str(raw)
            if redis_client.delete(key):
                released.append(key)
    except Exception as e:
        # Never block startup on this: a worker that cannot sweep is strictly better than a
        # worker that will not boot, and the TTL still clears the key eventually.
        logger.error(f"Could not sweep stale node-scoped Valkey locks: {e}")
        return 0

    if released:
        logger.warning(
            f"Cleared {len(released)} stale node-scoped Valkey lock(s) held by a process that no "
            f"longer exists: {', '.join(released)}. This node's previous worker was killed "
            "(OOM or SIGKILL) mid-task; without this sweep the next job would block until the "
            "lock's TTL expired."
        )
    return len(released)


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
