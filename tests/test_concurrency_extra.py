from unittest.mock import patch

from worker.concurrency import (
    _parse_env_int,
    run_job_async,
    set_seeding_complete,
)


def test_parse_env_int():
    with patch.dict("os.environ", {"TEST_KEY": "10"}):
        assert _parse_env_int("TEST_KEY", 5) == 10

    with patch.dict("os.environ", {"TEST_KEY": "invalid"}):
        assert _parse_env_int("TEST_KEY", 5) == 5

    with patch.dict("os.environ", {}):
        assert _parse_env_int("TEST_KEY", 5) == 5


def test_set_seeding_complete():
    set_seeding_complete(True)
    import worker.concurrency as conc

    assert conc.SEEDING_COMPLETE is True


def test_run_job_async_heavy():
    with patch("worker.concurrency.process_job_rq") as mock_proc:
        run_job_async("queue:ocr", {"jobId": "123"})
        mock_proc.assert_called_once_with("queue:ocr", {"jobId": "123"})


def test_run_job_async_light_exception():
    with patch("worker.concurrency.process_job_rq", side_effect=Exception("Async fail")):
        run_job_async("queue:translation", {"jobId": "123"})
