from unittest.mock import MagicMock
import redis

# Mock redis.Redis globally before any test modules import worker.config
mock_redis = MagicMock()
mock_redis.llen.return_value = 0
mock_redis.ping.return_value = True
redis.Redis = MagicMock(return_value=mock_redis)
