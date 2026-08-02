from unittest.mock import MagicMock, patch

from worker.services import llm_client
from worker.services.llm_client import LLMClient, LLMResponse


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


@patch("worker.services.llm_client.requests.post")
def test_llm_client_cloudflare_schema_and_session(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"translatedText": "Cloudflare Hi"}'}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 15},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(
        provider="cloudflare",
        api_key="cf_token_test",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        session_id="cf-session-123",
    )
    res = client.complete(
        messages=[{"role": "user", "content": "Hi"}],
        response_schema={"type": "object", "properties": {"translatedText": {"type": "string"}}},
    )

    assert res is not None
    assert res.content == '{"translatedText": "Cloudflare Hi"}'

    posted_json = mock_post.call_args.kwargs["json"]
    # For Cloudflare, json_schema is passed directly, not wrapped under name/schema
    assert posted_json["response_format"] == {
        "type": "json_schema",
        "json_schema": {"type": "object", "properties": {"translatedText": {"type": "string"}}},
    }
    # Check x-session-affinity header
    headers = mock_post.call_args.kwargs["headers"]
    assert headers.get("x-session-affinity") == "cf-session-123"


@patch("worker.services.llm_client.requests.post")
def test_llm_client_null_token_details(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"translatedText": "Success"}'}}],
        "usage": {
            "prompt_tokens": 1408,
            "completion_tokens": 310,
            "total_tokens": 1718,
            "prompt_tokens_details": {"cached_tokens": None, "audio_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": None},
        },
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="neurometric", api_key="test_key", model="clawpack")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(res, LLMResponse)
    assert res.content == '{"translatedText": "Success"}'
    assert res.prompt_tokens == 1408
    assert res.completion_tokens == 310
    assert res.total_tokens == 1718
    assert res.cached_tokens == 0


@patch("worker.services.llm_client.requests.post")
def test_auth_failure_parks_the_provider_instead_of_retrying_it(mock_post):
    """AUDIT triage 2026-08-02: an invalid neurometric key produced 323 identical 401s.

    A bad credential does not heal inside a job, but every layer above the client retries — batch,
    retry pass, per-region fallback, then the RQ job three times. The first 401 must therefore stop
    the provider being asked again rather than being re-attempted at each level.
    """
    llm_client.PROVIDER_AUTH_FAILURES.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error":{"message":"Invalid API key provided."}}'
    mock_post.return_value = mock_resp

    client = LLMClient(provider="neurometric", api_key="bad_key", model="neurometric/clawpack")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert mock_post.call_count == 1
    assert "neurometric" in llm_client.PROVIDER_AUTH_FAILURES

    # A second attempt short-circuits without touching the network — and without sleeping, which
    # is why this uses its own registry rather than the 429 cooldown.
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert mock_post.call_count == 1

    llm_client.PROVIDER_AUTH_FAILURES.clear()


@patch("worker.services.llm_client.requests.post")
def test_auth_failure_is_cleared_by_a_later_success(mock_post):
    llm_client.PROVIDER_AUTH_FAILURES.clear()
    llm_client.PROVIDER_AUTH_FAILURES["openrouter"] = 0.0  # already expired

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="good_key", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is not None
    assert "openrouter" not in llm_client.PROVIDER_AUTH_FAILURES


@patch("worker.services.llm_client.requests.post")
def test_other_4xx_errors_do_not_park_the_provider(mock_post):
    llm_client.PROVIDER_AUTH_FAILURES.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "no such model"
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="missing-model")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert "openrouter" not in llm_client.PROVIDER_AUTH_FAILURES
