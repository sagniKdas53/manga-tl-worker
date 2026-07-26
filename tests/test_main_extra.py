from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import worker.concurrency as conc
from worker.main import app


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
    headers = {"WORKER_API_SECRET": conc.WORKER_API_SECRET} if conc.WORKER_API_SECRET else {}
    res = client.get("/capabilities", headers=headers)
    assert res.status_code == 200
    assert "supported_tasks" in res.json()


def test_submit_job_success():
    client = TestClient(app)
    headers = {"WORKER_API_SECRET": conc.WORKER_API_SECRET} if conc.WORKER_API_SECRET else {}
    payload = {"queue_name": "queue:translation", "job_data": {"jobId": "j1", "imageId": "i1"}}
    with patch("threading.Thread"):
        res = client.post("/api/v1/jobs/submit", json=payload, headers=headers)
        assert res.status_code == 202
        assert res.json()["status"] == "accepted"
