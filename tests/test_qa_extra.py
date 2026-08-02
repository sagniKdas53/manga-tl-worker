from unittest.mock import MagicMock, patch

import pytest

from worker.handlers import qa
from worker.handlers.qa import (
    process_qa,
)


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_none_mode(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "hello"}]}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "none"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_unknown_mode(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": []}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "unknown"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai")
def test_process_qa_llm_mode(mock_cloud, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "hello", "translatedText": "hi"}]}
    mock_requests.get.return_value = mock_res

    mock_cloud.return_value = (
        '{"results": [{"regionId": "1", "qaStatus": "failed", "qaScore": 0.5, "qaFeedback": "bad translation"}]}'
    )

    with patch("worker.handlers.qa.QA_MODE", "llm"), patch("worker.handlers.qa.QA_CONFIG") as mock_qa:
        mock_qa.provider = "openrouter"
        mock_qa.llm_model = "gpt-4o-mini"
        mock_qa.resolve_key.return_value = "dummy"
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "failed"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_get_error(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 500
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "none"):
        process_qa({"imageId": "img1"})

    assert not mock_requests.post.called


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai")
def test_process_qa_llm_empty_regions(mock_cloud, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": []}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "llm"):
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert not mock_cloud.called


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai_vision")
@patch("worker.handlers.qa.download_image")
def test_process_qa_vlm_mode(mock_dl, mock_cloud_vision, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "ocrRegions": [
            {
                "id": "1",
                "text": "hello",
                "translatedText": "hi",
                "bboxX": 0,
                "bboxY": 0,
                "bboxW": 10,
                "bboxH": 10,
            }
        ]
    }
    mock_requests.get.return_value = mock_res

    import io

    from PIL import Image

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    mock_dl.return_value = valid_bytes

    mock_cloud_vision.return_value = (
        '{"results": [{"regionId": "1", "qaStatus": "passed", "qaScore": 1.0, "qaFeedback": "good"}]}'
    )

    with patch("worker.handlers.qa.QA_MODE", "vlm"), patch("worker.handlers.qa.QA_CONFIG") as mock_qa:
        mock_qa.provider = "openrouter"
        mock_qa.vision_model = "gpt-4o"
        mock_qa.resolve_key.return_value = "dummy"
        with patch("worker.handlers.qa.minio_client.get_object") as mock_minio_get:
            mock_minio_res = MagicMock()
            mock_minio_res.read.return_value = valid_bytes
            mock_minio_get.return_value = mock_minio_res
            process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "passed"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_process_qa_skips_on_attempt_greater_than_zero(mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "hello"}]}
    mock_requests.get.return_value = mock_res

    with patch("worker.handlers.qa.QA_MODE", "llm"):
        process_qa({"imageId": "img1", "qaAttempt": 1})

    # Should have called post with auto-pass fallback (qaStatus == "passed") because qaAttempt > 0 overrides llm mode
    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "passed"
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaFeedback"] == "Auto-passed (QA bypassed)"


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
@patch("worker.handlers.qa.try_cloud_ai")
def test_process_qa_reject_sfx(mock_cloud, mock_redis, mock_requests):
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": [{"id": "1", "text": "boom", "translatedText": "boom"}]}
    mock_requests.get.return_value = mock_res

    mock_cloud.return_value = '{"results": [{"regionId": "1", "qaStatus": "reject_sfx", "qaScore": 1.0, "qaFeedback": "It is a sound effect"}]}'

    with patch("worker.handlers.qa.QA_MODE", "llm"), patch("worker.handlers.qa.QA_CONFIG") as mock_qa:
        mock_qa.provider = "openrouter"
        mock_qa.llm_model = "gpt-4o-mini"
        mock_qa.resolve_key.return_value = "dummy"
        process_qa({"imageId": "img1"})

    mock_requests.post.assert_called()
    assert mock_requests.post.call_args[1]["json"]["qaResults"][0]["qaStatus"] == "reject_sfx"


class TestQaProviderDispatch:
    """AUDIT-W1: QA dispatched on a hardcoded openrouter/gemini/nvidia if/elif in four places.

    cloudflare and neurometric are selectable in the UI and present in config/providers.json, but
    fell off the end of every chain and returned None — which then fell through to the broken local
    fallback and completed QA with zero findings and no error.
    """

    @pytest.mark.parametrize("provider", ["cloudflare", "neurometric", "openrouter", "nvidia"])
    def test_llm_reaches_every_configured_provider(self, provider):
        with patch("worker.handlers.qa.try_cloud_ai", return_value='{"results": []}') as mock_call:
            result = qa._qa_cloud_llm(provider, "key", "some/model", "prompt", "lowest-cost")

        assert result == '{"results": []}'
        assert mock_call.call_args.args[0] == provider
        assert mock_call.call_args.args[2] == "some/model"

    @pytest.mark.parametrize("provider", ["cloudflare", "neurometric", "openrouter", "nvidia"])
    def test_vlm_reaches_every_configured_provider(self, provider):
        with patch("worker.handlers.qa.try_cloud_ai_vision", return_value='{"results": []}') as mock_call:
            result = qa._qa_cloud_vlm(provider, "key", "some/model", "prompt", "b64", "lowest-cost")

        assert result == '{"results": []}'
        assert mock_call.call_args.args[0] == provider

    def test_falls_back_to_the_builtin_default_model(self):
        with patch("worker.handlers.qa.try_cloud_ai", return_value="{}") as mock_call:
            qa._qa_cloud_llm("openrouter", "key", None, "prompt", None)
        assert mock_call.call_args.args[2] == qa.QA_DEFAULT_LLM_MODELS["openrouter"]

    def test_returns_none_without_an_api_key(self):
        with patch("worker.handlers.qa.try_cloud_ai") as mock_call:
            assert qa._qa_cloud_llm("cloudflare", "", "m", "prompt", None) is None
        mock_call.assert_not_called()

    def test_returns_none_when_no_model_can_be_resolved(self):
        # A provider outside the default table must be given an explicit model rather than
        # silently doing nothing.
        with patch("worker.handlers.qa.try_cloud_ai") as mock_call:
            assert qa._qa_cloud_llm("neurometric", "key", None, "prompt", None) is None
        mock_call.assert_not_called()

    def test_provider_exception_is_swallowed_into_none(self):
        with patch("worker.handlers.qa.try_cloud_ai", side_effect=RuntimeError("boom")):
            assert qa._qa_cloud_llm("openrouter", "key", "m", "prompt", None) is None
