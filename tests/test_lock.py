"""Tests for the Valkey lock — AUDIT-W4.

Two defects, both of which this file pins:

1. The key embedded ``platform.node()`` unconditionally, so the ``local-llm`` lock did not
   serialise across workers even though ``LOCAL_LLM_ENDPOINT`` is a shared address.
2. The release was an unconditional DELETE, so a holder that overran its TTL deleted whatever
   lock had been acquired since — by someone else.
"""

from unittest.mock import patch

import pytest

from worker.utils.lock import acquire_lock


class FakeRedis:
    """Enough of the redis-py surface for the lock: SET NX EX, and the release Lua script."""

    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def eval(self, _script, _numkeys, *args):
        key, token = args[0], args[1]
        # Mirror the compare-and-delete the real script performs.
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


def test_lock_key_is_global_by_default():
    fake = FakeRedis()
    seen = {}
    with patch("worker.utils.lock.redis_client", fake), acquire_lock("local-llm"):
        seen.update(fake.store)

    assert list(seen) == ["lock:local-llm"], (
        "the local-llm endpoint is shared across workers, so the lock must not be per-container"
    )


def test_lock_key_is_node_scoped_when_asked():
    fake = FakeRedis()
    seen = {}
    with (
        patch("worker.utils.lock.redis_client", fake),
        patch("platform.node", return_value="worker-7"),
        acquire_lock("ocr", node_scoped=True),
    ):
        seen.update(fake.store)

    assert list(seen) == ["lock:ocr:worker-7"], "the ocr lock guards this host's CPU/GPU, so it stays per-container"


def test_release_removes_our_own_lock():
    fake = FakeRedis()
    with patch("worker.utils.lock.redis_client", fake), acquire_lock("local-llm"):
        pass

    assert fake.store == {}, "a normal release must free the lock"


def test_release_does_not_delete_another_holders_lock():
    """The AUDIT-W4 defect: our TTL fires, someone else acquires, and our finally deletes theirs."""
    fake = FakeRedis()
    with patch("worker.utils.lock.redis_client", fake), acquire_lock("local-llm", expire=1):
        # Simulate the TTL firing and a different worker taking the lock while we still run.
        fake.store["lock:local-llm"] = "a-different-workers-token"

    assert fake.store.get("lock:local-llm") == "a-different-workers-token", (
        "releasing must not delete a lock this holder no longer owns"
    )


def test_acquire_times_out_when_the_lock_is_held():
    fake = FakeRedis()
    fake.store["lock:local-llm"] = "someone-elses-token"
    with (
        patch("worker.utils.lock.redis_client", fake),
        pytest.raises(TimeoutError),
        acquire_lock("local-llm", timeout=0.1),
    ):
        pass
