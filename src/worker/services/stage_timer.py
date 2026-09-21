"""Wall-clock accounting for the stages inside one job.

Added 2026-09-21 after a page spent 15 minutes in the OCR job with nothing in the log between
"Merged N regions" and "Completed OCR" -- the cleanup step that cost 80-95% of the page emitted
no line of its own on the success path, and the only trace it left was ONNX Runtime's shape
warning on stderr (docs/output-quality-implementation-tracker.md, R3 addenda 2026-09-21). One
timer per job; `mark` closes the stage that has been running since the previous mark, `add`
accumulates time measured elsewhere (a per-region step summed across regions), `summary`
renders one greppable line.
"""

import time


class StageTimer:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._start = time.perf_counter()
        self._last = self._start
        self._stages: list[tuple[str, float]] = []
        self._extra: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def mark(self, stage: str) -> float:
        """Close the stage running since the previous mark (or since construction)."""
        now = time.perf_counter()
        elapsed = now - self._last
        self._stages.append((stage, elapsed))
        self._last = now
        return elapsed

    def add(self, stage: str, seconds: float, count: int = 1) -> None:
        """Accumulate time measured by a callee, for steps that repeat inside one stage."""
        self._extra[stage] = self._extra.get(stage, 0.0) + seconds
        self._counts[stage] = self._counts.get(stage, 0) + count

    def total(self) -> float:
        return time.perf_counter() - self._start

    def summary(self) -> str:
        parts = [f"{stage}={elapsed:.1f}s" for stage, elapsed in self._stages]
        for stage, seconds in self._extra.items():
            parts.append(f"{stage}={seconds:.1f}s/{self._counts[stage]}")
        return f"{self.tag} stage timings: total={self.total():.1f}s | " + " ".join(parts)
