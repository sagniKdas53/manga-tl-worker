from worker.rq_tasks import _seconds_since
from worker.services.stage_timer import StageTimer


def test_stage_timer_marks_and_accumulates():
    timer = StageTimer("[T]")
    assert timer.mark("fetch") >= 0.0
    timer.add("cleanup", 1.5)
    timer.add("cleanup", 2.5, count=1)
    timer.mark("regions")
    summary = timer.summary()
    assert summary.startswith("[T] stage timings: total=")
    assert "fetch=" in summary and "regions=" in summary
    assert "cleanup=4.0s/2" in summary


def test_seconds_since_accepts_backend_nanosecond_timestamps():
    # chrono writes nine fractional digits; fromisoformat alone rejects more than six.
    assert _seconds_since("2026-09-21T05:30:15.232694225+00:00") is not None
    assert _seconds_since("2026-09-21T05:30:15Z") is not None
    assert _seconds_since("not a timestamp") is None
    assert _seconds_since(None) is None
