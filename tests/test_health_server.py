import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import worker.concurrency as conc
from worker.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_concurrency_state():
    conc.ACTIVE_JOBS = 0
    conc.ACTIVE_HEAVY_JOBS = 0
    conc.ACTIVE_LIGHT_JOBS = 0
    conc.SEEDING_COMPLETE = True
    conc.WORKER_API_SECRET = "test_secret"


def test_check_auth_failure():
    response = client.get("/capabilities", headers={"WORKER_API_SECRET": "wrong_secret"})
    assert response.status_code == 401

    response_no_header = client.get("/capabilities")
    assert response_no_header.status_code == 401


def test_check_auth_success():
    response = client.get("/capabilities", headers={"WORKER_API_SECRET": "test_secret"})
    assert response.status_code == 200
    data = response.json()
    assert "worker_id" in data
    assert "supported_tasks" in data


@patch("worker.concurrency.run_job_async")
def test_job_submission_success(mock_run_async):
    body = {
        "queue_name": "queue:ocr",
        "job_data": {"jobId": "test_job_id", "imageId": "123"},
    }
    response = client.post(
        "/api/v1/jobs/submit",
        json=body,
        headers={"WORKER_API_SECRET": "test_secret"},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


def test_job_submission_rate_limit():
    conc.ACTIVE_JOBS = 2
    conc.MAX_CONCURRENT_JOBS = 2

    body = {
        "queue_name": "queue:ocr",
        "job_data": {"jobId": "test_job_id", "imageId": "123"},
    }
    response = client.post(
        "/api/v1/jobs/submit",
        json=body,
        headers={"WORKER_API_SECRET": "test_secret"},
    )
    assert response.status_code == 429


@patch("worker.concurrency.process_job_rq")
def test_job_execution_wrapper(mock_process_job):
    conc.ACTIVE_JOBS = 1
    conc.ACTIVE_HEAVY_JOBS = 1

    conc.run_job_async("queue:ocr", {"id": "123"})

    mock_process_job.assert_called_once_with("queue:ocr", {"id": "123"})
    assert conc.ACTIVE_JOBS == 0
    assert conc.ACTIVE_HEAVY_JOBS == 0


def test_heavy_light_concurrency_slots():
    conc.MAX_CONCURRENT_JOBS = 2
    conc.MAX_HEAVY_SLOTS = 1
    conc.MAX_LIGHT_SLOTS = 1

    headers = {"WORKER_API_SECRET": "test_secret"}

    # Step 1: Submit Heavy Job (should succeed)
    res1 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j1", "imageId": "1"}},
        headers=headers,
    )
    assert res1.status_code == 202
    assert conc.ACTIVE_HEAVY_JOBS == 1
    assert conc.ACTIVE_LIGHT_JOBS == 0

    # Step 2: Submit another Heavy Job (should fail with 429)
    res2 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j2", "imageId": "2"}},
        headers=headers,
    )
    assert res2.status_code == 429
    assert conc.ACTIVE_HEAVY_JOBS == 1

    # Step 3: Submit Light Job (should succeed)
    res3 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j3", "imageId": "3"}},
        headers=headers,
    )
    assert res3.status_code == 202
    assert conc.ACTIVE_HEAVY_JOBS == 1
    assert conc.ACTIVE_LIGHT_JOBS == 1

    # Step 4: Submit another Light Job (should fail with 429)
    res4 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j4", "imageId": "4"}},
        headers=headers,
    )
    assert res4.status_code == 429


def test_scenario_three_jobs_concurrency():
    conc.MAX_CONCURRENT_JOBS = 2
    conc.MAX_HEAVY_SLOTS = 1
    conc.MAX_LIGHT_SLOTS = 1
    conc.REUSE_IDLE_SLOTS = False

    headers = {"WORKER_API_SECRET": "test_secret"}

    # 1. Job 1 submitted for OCR
    res1 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j1", "imageId": "job1"}},
        headers=headers,
    )
    assert res1.status_code == 202
    assert conc.ACTIVE_HEAVY_JOBS == 1

    # 2. Job 2 submitted for OCR -> 429
    res2 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j2", "imageId": "job2"}},
        headers=headers,
    )
    assert res2.status_code == 429

    # 3. Job 1 finishes OCR
    conc.ACTIVE_HEAVY_JOBS = 0
    conc.ACTIVE_JOBS = 0

    # 4. Job 1 submitted for Translation
    res3 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j1", "imageId": "job1"}},
        headers=headers,
    )
    assert res3.status_code == 202
    assert conc.ACTIVE_LIGHT_JOBS == 1

    # 5. Job 2 starts OCR now
    res4 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j2", "imageId": "job2"}},
        headers=headers,
    )
    assert res4.status_code == 202
    assert conc.ACTIVE_HEAVY_JOBS == 1
    assert conc.ACTIVE_LIGHT_JOBS == 1

    # 6. Job 2 finishes OCR
    conc.ACTIVE_HEAVY_JOBS = 0
    conc.ACTIVE_JOBS = 1

    # 7. Job 2 submitted for Translation -> 429
    res5 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j2", "imageId": "job2"}},
        headers=headers,
    )
    assert res5.status_code == 429


def test_configurable_heavy_slots():
    conc.MAX_CONCURRENT_JOBS = 3
    conc.MAX_HEAVY_SLOTS = 2
    conc.MAX_LIGHT_SLOTS = 1

    headers = {"WORKER_API_SECRET": "test_secret"}

    # Heavy job 1
    res1 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j1", "imageId": "h1"}},
        headers=headers,
    )
    assert res1.status_code == 202

    # Heavy job 2
    res2 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:panel-detection", "job_data": {"jobId": "j2", "imageId": "h2"}},
        headers=headers,
    )
    assert res2.status_code == 202
    assert conc.ACTIVE_HEAVY_JOBS == 2

    # Heavy job 3 -> 429
    res3 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:ocr", "job_data": {"jobId": "j3", "imageId": "h3"}},
        headers=headers,
    )
    assert res3.status_code == 429


def test_configurable_light_slots():
    conc.MAX_CONCURRENT_JOBS = 3
    conc.MAX_HEAVY_SLOTS = 1
    conc.MAX_LIGHT_SLOTS = 2
    conc.REUSE_IDLE_SLOTS = False

    headers = {"WORKER_API_SECRET": "test_secret"}

    res1 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j1", "imageId": "l1"}},
        headers=headers,
    )
    assert res1.status_code == 202

    res2 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:layout", "job_data": {"jobId": "j2", "imageId": "l2"}},
        headers=headers,
    )
    assert res2.status_code == 202
    assert conc.ACTIVE_LIGHT_JOBS == 2

    res3 = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:render", "job_data": {"jobId": "j3", "imageId": "l3"}},
        headers=headers,
    )
    assert res3.status_code == 429


def test_region_redo_removed_from_heavy():
    assert "queue:region-redo" not in conc.HEAVY_QUEUES
    assert "queue:region-redo" not in conc.LIGHT_QUEUES


def test_light_overflow_when_heavy_idle():
    conc.MAX_CONCURRENT_JOBS = 2
    conc.MAX_HEAVY_SLOTS = 1
    conc.MAX_LIGHT_SLOTS = 1
    conc.REUSE_IDLE_SLOTS = True
    conc.ACTIVE_JOBS = 1
    conc.ACTIVE_HEAVY_JOBS = 0
    conc.ACTIVE_LIGHT_JOBS = 1

    headers = {"WORKER_API_SECRET": "test_secret"}

    res = client.post(
        "/api/v1/jobs/submit",
        json={"queue_name": "queue:translation", "job_data": {"jobId": "j2", "imageId": "light2"}},
        headers=headers,
    )
    assert res.status_code == 202
    assert conc.ACTIVE_LIGHT_JOBS == 2
    assert conc.ACTIVE_JOBS == 2


def test_health_endpoint():
    with patch("worker.config.redis_client.ping", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["redis"] == "connected"
