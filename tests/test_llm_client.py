from unittest.mock import MagicMock, patch

import pytest

from worker.services.llm_client import LLMClient, LLMResponse, PermanentAPIError, TransientAPIError


@patch("worker.services.llm_client.requests.post")
def test_llm_client_openai_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"translatedText": "Hello"}'}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
        },
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openai", api_key="test_key", model="gpt-4o-mini")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(res, LLMResponse)
    assert res.content == '{"translatedText": "Hello"}'
    assert res.prompt_tokens == 100
    assert res.completion_tokens == 20
    assert res.cached_tokens == 50


@patch("worker.services.llm_client.requests.post")
def test_llm_client_anthropic_prompt_caching(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"text": "Hello Anthropic"}],
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30,
            "cache_read_input_tokens": 150,
        },
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="anthropic", api_key="test_key", model="claude-3-5-sonnet-20241022")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}], system_prompt="System prompt text")

    assert res is not None
    assert res.content == "Hello Anthropic"
    assert res.cached_tokens == 150

    # Verify payload format for Anthropic
    posted_json = mock_post.call_args.kwargs["json"]
    assert "system" in posted_json
    assert posted_json["system"][0]["cache_control"] == {"type": "ephemeral"}


@patch("worker.services.llm_client.requests.post")
def test_llm_client_openrouter_caching_and_session(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter response"}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 25},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(
        provider="openrouter",
        api_key="test_key",
        model="meta-llama/llama-3-8b-instruct:free",
        session_id="chapter-10",
    )
    client.complete(messages=[{"role": "user", "content": "Hi"}], system_prompt="System instructions")

    posted_json = mock_post.call_args.kwargs["json"]
    assert posted_json.get("extra_body", {}).get("session_id") == "chapter-10"
    # System message content should be cache annotated array
    assert posted_json["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
