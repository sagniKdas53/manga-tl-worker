"""AUDIT-Q3: cleanup_audit_cache's stale-file sweep.

The sweep used to perform its deletions as a side effect inside a generator
expression — `sum(1 for f in files if ... and not os.remove(f))`, relying on
os.remove returning None. That is unreadable, and it also made the sweep
all-or-nothing: a single raising unlink propagated out of the generator to the
enclosing `except Exception`, so every file after it was left on disk and the
count was never printed.
"""

import os
from unittest.mock import patch

import app


def _touch(path, age_seconds):
    with open(path, "wb") as f:
        f.write(b"x")
    stamp = os.path.getmtime(path) - age_seconds
    os.utime(path, (stamp, stamp))


def test_cleanup_removes_only_files_older_than_a_day(tmp_path, capsys):
    old_a = tmp_path / "old_a.jpg"
    old_b = tmp_path / "old_b.jpg"
    fresh = tmp_path / "fresh.jpg"
    _touch(old_a, 48 * 3600)
    _touch(old_b, 48 * 3600)
    _touch(fresh, 60)

    with patch.dict(
        "worker.config.__dict__",
        {"ENABLE_QA_AUDIT_CACHE": True, "QA_AUDIT_CACHE_DIR": str(tmp_path)},
    ):
        app.cleanup_audit_cache()

    assert not old_a.exists()
    assert not old_b.exists()
    assert fresh.exists()
    assert "Cleaned up 2 old files" in capsys.readouterr().out


def test_cleanup_continues_past_a_file_that_cannot_be_removed(tmp_path, capsys):
    """Red before the fix: the generator aborted on the first raising unlink."""
    first = tmp_path / "a_locked.jpg"
    second = tmp_path / "b_removable.jpg"
    third = tmp_path / "c_removable.jpg"
    for p in (first, second, third):
        _touch(p, 48 * 3600)

    real_remove = os.remove

    def remove_but_fail_on_first(path):
        if os.path.basename(path) == "a_locked.jpg":
            raise PermissionError("in use")
        real_remove(path)

    with (
        patch.dict(
            "worker.config.__dict__",
            {"ENABLE_QA_AUDIT_CACHE": True, "QA_AUDIT_CACHE_DIR": str(tmp_path)},
        ),
        patch("app.os.remove", side_effect=remove_but_fail_on_first),
    ):
        app.cleanup_audit_cache()

    out = capsys.readouterr().out
    assert first.exists(), "the unremovable file should still be there"
    assert not second.exists(), "the sweep must not abort on the first failure"
    assert not third.exists(), "the sweep must not abort on the first failure"
    assert "Cleaned up 2 old files" in out
    assert "Could not remove QA audit cache file" in out
