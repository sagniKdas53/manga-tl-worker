"""Tests for the Valkey lock — AUDIT-W4.

Two defects, both of which this file pins:

1. The key embedded ``platform.node()`` unconditionally, so the ``local-llm`` lock did not
   serialise across workers even though ``LOCAL_LLM_ENDPOINT`` is a shared address.
2. The release was an unconditional DELETE, so a holder that overran its TTL deleted whatever
   lock had been acquired since — by someone else.
"""

from unittest.mock import patch

import pytest

from worker.utils.lock import acquire_lock, release_stale_node_locks


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

    def scan_iter(self, match=None, count=None):
        # Real redis-py returns bytes when decode_responses is off, which the worker's client is.
        prefix, _, suffix = (match or "*").partition("*")
        for key in list(self.store):
            if key.startswith(prefix) and key.endswith(suffix):
                yield key.encode()


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


class TestStaleNodeLockSweep:
    """A SIGKILLed holder never runs the release, so its key outlives it by the whole TTL.

    That is what an OOM kill leaves behind: the container restarts under the same hostname and
    the new process blocks on a lock belonging to a process that no longer exists.
    """

    def test_orphaned_lock_for_this_node_is_cleared(self):
        fake = FakeRedis()
        fake.store["lock:ocr:worker-7"] = "token-of-a-dead-process"
        with (
            patch("worker.utils.lock.redis_client", fake),
            patch("platform.node", return_value="worker-7"),
        ):
            assert release_stale_node_locks() == 1

        assert fake.store == {}, "the dead process's lock must not survive our startup"

    def test_locks_belonging_to_other_nodes_are_left_alone(self):
        """Only this node's keys are provably orphaned — another container may still be running."""
        fake = FakeRedis()
        fake.store["lock:ocr:worker-7"] = "ours"
        fake.store["lock:ocr:worker-9"] = "another-live-containers-token"
        with (
            patch("worker.utils.lock.redis_client", fake),
            patch("platform.node", return_value="worker-7"),
        ):
            assert release_stale_node_locks() == 1

        assert fake.store == {"lock:ocr:worker-9": "another-live-containers-token"}

    def test_global_locks_are_never_swept(self):
        """A non-node-scoped lock guards a shared service and may be held by a different worker."""
        fake = FakeRedis()
        fake.store["lock:local-llm"] = "held-by-some-worker"
        with (
            patch("worker.utils.lock.redis_client", fake),
            patch("platform.node", return_value="worker-7"),
        ):
            assert release_stale_node_locks() == 0

        assert fake.store == {"lock:local-llm": "held-by-some-worker"}

    def test_a_sweep_failure_does_not_block_startup(self):
        """Booting without the sweep beats not booting; the TTL still clears the key eventually."""

        class ExplodingRedis(FakeRedis):
            def scan_iter(self, match=None, count=None):
                raise ConnectionError("valkey is not up yet")

        with (
            patch("worker.utils.lock.redis_client", ExplodingRedis()),
            patch("platform.node", return_value="worker-7"),
        ):
            assert release_stale_node_locks() == 0

    def test_the_sweep_unblocks_a_waiter(self):
        """End to end: the exact production sequence, minus the ten minute wait."""
        fake = FakeRedis()
        fake.store["lock:ocr:worker-7"] = "token-of-the-oom-killed-process"
        with (
            patch("worker.utils.lock.redis_client", fake),
            patch("platform.node", return_value="worker-7"),
        ):
            release_stale_node_locks()
            # Previously this blocked for the orphan's full TTL before raising or proceeding.
            with acquire_lock("ocr", timeout=0.1, node_scoped=True):
                assert fake.store["lock:ocr:worker-7"] != "token-of-the-oom-killed-process"
