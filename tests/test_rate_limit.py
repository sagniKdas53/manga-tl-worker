import os
import json
import pytest
from unittest.mock import patch, MagicMock

from worker.utils.rate_limit import enforce_rate_limit, update_model_costs, reset_job_costs, get_job_costs, estimate_cost

@patch("worker.utils.rate_limit.time")
def test_enforce_rate_limit(mock_time):
    # Test valid limit
    mock_time.time.return_value = 100
    os.environ["RATE_LIMIT"] = "60/min"
    
    import worker.utils.rate_limit as rlimit
    rlimit.LAST_REQUEST_TIME = 99.5
    
    # 60/min = 1/s. Min delay = 1. Elapsed = 0.5. Should sleep for 0.5.
    enforce_rate_limit()
    mock_time.sleep.assert_called_with(0.5)
    
    # Test error fallback
    rlimit.LAST_REQUEST_TIME = "invalid"
    enforce_rate_limit() # should not crash
    
    del os.environ["RATE_LIMIT"]

@patch("worker.utils.rate_limit.requests")
@patch("worker.utils.rate_limit.redis_client")
def test_update_model_costs(mock_redis, mock_req, tmp_path):
    cost_file = tmp_path / "costs.json"
    with patch("worker.utils.rate_limit.COSTS_FILE", str(cost_file)):
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "data": {
                "endpoints": [
                    {"pricing": {"prompt": "0.000001", "completion": "0.000002"}}
                ]
            }
        }
        mock_req.get.return_value = mock_res
        
        update_model_costs(["meta-llama/llama-3-8b-instruct:free"])
        
        assert cost_file.exists()
        costs = json.loads(cost_file.read_text())
        assert "meta-llama/llama-3-8b-instruct:free" in costs
        
        # Test 404
        mock_res.status_code = 404
        with pytest.raises(ValueError):
            update_model_costs(["unknown/model"])

def test_job_costs():
    reset_job_costs()
    estimate_cost("model:free", 100, 50)
    costs = get_job_costs()
    assert len(costs) == 1
    assert costs[0]["model"] == "model:free"
