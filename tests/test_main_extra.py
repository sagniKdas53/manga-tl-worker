from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import worker.concurrency as conc
from worker.main import app

AUTH_HEADERS = {"WORKER_API_SECRET": "test_secret"}


@pytest.fixture(autouse=True)
def configure_api_secret():
    """Authenticated endpoints fail closed now (AUDIT-S3), so the secret has to be configured."""
    previous = conc.WORKER_API_SECRET
    conc.WORKER_API_SECRET = "test_secret"
    yield
    conc.WORKER_API_SECRET = previous


def test_main_health_seeding():
    conc.set_seeding_complete(False)
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "seeding"


def test_main_health_connected():
    conc.set_seeding_complete(True)
    client = TestClient(app)
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    with patch("worker.config.redis_client", mock_redis):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"


def test_main_health_disconnected():
    conc.set_seeding_complete(True)
    client = TestClient(app)
    mock_redis = MagicMock()
    mock_redis.ping.side_effect = Exception("Redis error")
    with patch("worker.config.redis_client", mock_redis):
        res = client.get("/health")
        assert res.status_code == 503
        assert res.json()["status"] == "unhealthy"


def test_main_capabilities():
    client = TestClient(app)
    res = client.get("/capabilities", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert "supported_tasks" in res.json()


def test_submit_job_success():
    client = TestClient(app)
    payload = {"queue_name": "queue:translation", "job_data": {"jobId": "j1", "imageId": "i1"}}
    with patch("threading.Thread"):
        res = client.post("/api/v1/jobs/submit", json=payload, headers=AUTH_HEADERS)
        assert res.status_code == 202
        assert res.json()["status"] == "accepted"
