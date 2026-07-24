import copy
from unittest.mock import MagicMock, patch

from worker.services.translation import translate_batch_llm, try_cloud_ai_vision_batch


@patch("worker.services.translation.try_cloud_ai")
def test_translate_batch_llm_disable_fallback(mock_try_cloud_ai):
    # Test that fallback is not used when use_fallback_models=False
    mock_try_cloud_ai.return_value = None  # Simulate primary model failing

    regions = [{"id": "1", "text": "test"}]

    with patch("worker.config.TL_CONFIG") as mock_tl_config:
        mock_tl_config.provider = "openrouter"
        mock_tl_config.resolve_key.return_value = "key"
        mock_tl_config.llm_model = "primary-model"

        # Test with use_fallback_models=False
        res = translate_batch_llm(
            regions,
            provider="openrouter",
            llm_model="user-model",
            use_fallback_models=False,
        )

        # try_cloud_ai should only be called once, for the primary user-model
        assert mock_try_cloud_ai.call_count == 1
        assert mock_try_cloud_ai.call_args[0][2] == "user-model"  # It tried user-model
        assert res is None


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.time.sleep")
def test_try_cloud_ai_vision_batch_degrade_json_schema(mock_sleep, mock_post):
    # Test that 400 error degrades from json_schema to json_object

    # Mock first response as 400 with json_schema
    mock_resp_400 = MagicMock()
    mock_resp_400.status_code = 400
    mock_resp_400.text = "schema not supported"

    # Mock second response as 200 with json_object
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {"choices": [{"message": {"content": "result"}}]}

    call_kwargs_history = []

    def mock_post_side_effect(*args, **kwargs):
        call_kwargs_history.append(copy.deepcopy(kwargs))
        if len(call_kwargs_history) == 1:
            return mock_resp_400
        return mock_resp_200

    mock_post.side_effect = mock_post_side_effect

    with patch("worker.services.translation._get_api_url_and_headers") as mock_get_url:
        mock_get_url.return_value = ("http://test", {}, "test-model")
        with patch("worker.services.translation.estimate_cost") as mock_cost:
            mock_cost.return_value = 0.0

            res = try_cloud_ai_vision_batch(
                provider="openrouter",
                api_key="key",
                model="test-model",
                crops=[{"id": "1", "base64": "test"}],
                response_schema={"type": "object"},
            )

            assert res == "result"
            assert mock_post.call_count == 2

            # Check first request used json_schema
            first_call_kwargs = call_kwargs_history[0]
            assert first_call_kwargs["json"]["response_format"]["type"] == "json_schema"

            # Check second request degraded to json_object
            second_call_kwargs = call_kwargs_history[1]
            assert second_call_kwargs["json"]["response_format"]["type"] == "json_object"
