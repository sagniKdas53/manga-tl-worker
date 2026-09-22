"""Per-attempt authority, shared by job and heartbeat threads (never by unrelated jobs)."""

import contextvars
import threading
import time
from dataclasses import dataclass, field


class AttemptExpired(RuntimeError):
    """This attempt must stop making requests; the backend remains authoritative."""


@dataclass
class Attempt:
    headers: dict[str, str]
    deadline: float
    revoked: threading.Event = field(default_factory=threading.Event)
    progress: int = 0
    progress_lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self):
        if self.revoked.is_set() or time.monotonic() >= self.deadline:
            raise AttemptExpired("Job attempt expired or was superseded")


current_attempt: contextvars.ContextVar[Attempt | None] = contextvars.ContextVar("job_attempt", default=None)


def bind_attempt(payload):
    headers = {
        header: str(payload[key])
        for key, header in (
            ("jobId", "X-Job-Id"),
            ("attempt", "X-Job-Attempt"),
            ("inputGeneration", "X-Input-Generation"),
            ("leaseToken", "X-Lease-Token"),
        )
        if payload.get(key) is not None
    }
    runtime = max(1, min(3600, int(payload.get("maxRuntimeSeconds", 3600))))
    return current_attempt.set(Attempt(headers, time.monotonic() + runtime))


def attempt_headers():
    attempt = current_attempt.get()
    if attempt is None:
        return {}
    attempt.check()
    return attempt.headers


def record_progress():
    attempt = current_attempt.get()
    if attempt is not None:
        attempt.check()
        with attempt.progress_lock:
            attempt.progress += 1
