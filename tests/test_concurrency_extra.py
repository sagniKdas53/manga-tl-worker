from unittest.mock import patch

from worker.concurrency import (
    _parse_env_int,
    resolve_slot_config,
    run_job_async,
    set_seeding_complete,
)


def test_resolve_slot_config_leaves_a_valid_triple_alone():
    concurrent, heavy, light, warnings = resolve_slot_config(5, 1, 4)
    assert (concurrent, heavy, light) == (5, 1, 4)
    assert warnings == []


def test_resolve_slot_config_rescues_zero_light_slots():
    """CONCURRENT_JOBS=1 with the default MAX_HEAVY_SLOTS=1 subtracts to 0 light slots.

    Unclamped that is a permanent 429 on every light queue — a hard pipeline deadlock from a
    plausible config, with nothing logged to explain it (AUDIT-W6).
    """
    concurrent, heavy, light, warnings = resolve_slot_config(1, 1, 0)
    assert light == 1
    assert (concurrent, heavy) == (1, 1)
    assert any("MAX_LIGHT_SLOTS" in w for w in warnings)


def test_resolve_slot_config_rescues_negative_light_slots():
    _, _, light, warnings = resolve_slot_config(2, 3, -1)
    assert light == 1
    assert any("MAX_LIGHT_SLOTS" in w for w in warnings)


def test_resolve_slot_config_rescues_nonpositive_global_and_heavy():
    concurrent, heavy, light, warnings = resolve_slot_config(0, 0, 2)
    assert (concurrent, heavy, light) == (1, 1, 2)
    assert any("CONCURRENT_JOBS" in w for w in warnings)
    assert any("MAX_HEAVY_SLOTS" in w for w in warnings)


def test_resolve_slot_config_warns_when_tiers_exceed_the_global_cap():
    concurrent, heavy, light, warnings = resolve_slot_config(2, 1, 4)
    # The global cap is a deliberate ceiling, so the tiers are left as reservations within it.
    assert (concurrent, heavy, light) == (2, 1, 4)
    assert any("exceeds CONCURRENT_JOBS" in w for w in warnings)


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
