import copy
import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from worker import provider_config
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


# ---------------------------------------------------------------------------
# AUDIT-T2 — the error branches
#
# Everything above this line, apart from the auth-failure tests, stubs a 200. The retry ladder,
# the cooldown escalation and the schema degradation had no coverage at all, which is where every
# AUDIT-W8 defect lives. These drive `_execute_with_retry` through each non-200 branch.
# ---------------------------------------------------------------------------


@pytest.fixture
def no_retry_sleep():
    """Neutralise both sleeps so the retry ladder runs at full speed.

    Tenacity naps between attempts via its own `Retrying.sleep`, and `wait_for_cooldown` uses
    `time.sleep` from the module. A 429 test that did not patch both would take ~16s and then block
    for the 10s cooldown it just installed.
    """
    llm_client.PROVIDER_COOLDOWNS.clear()
    llm_client.PROVIDER_CONSECUTIVE_429S.clear()
    llm_client.PROVIDER_AUTH_FAILURES.clear()
    retrying = LLMClient._execute_with_retry.retry
    with (
        patch.object(retrying, "sleep", lambda _: None),
        patch("worker.services.llm_client.time.sleep", lambda _: None),
    ):
        yield
    llm_client.PROVIDER_COOLDOWNS.clear()
    llm_client.PROVIDER_CONSECUTIVE_429S.clear()
    llm_client.PROVIDER_AUTH_FAILURES.clear()


@patch("worker.services.llm_client.requests.post")
def test_429_exhausts_the_retry_ladder_and_installs_a_cooldown(mock_post, no_retry_sleep):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None

    # stop_after_attempt(3) — three requests, then TransientAPIError surfaces and complete() -> None
    assert mock_post.call_count == 3
    assert llm_client.PROVIDER_CONSECUTIVE_429S["openrouter"] == 3
    # Escalation is 2**(n-1) * base, so the third 429 must have installed at least 4x the base.
    remaining = llm_client.PROVIDER_COOLDOWNS["openrouter"] - time.time()
    assert remaining > llm_client.COOLDOWN_BASE_SECONDS * 3


@patch("worker.services.llm_client.requests.post")
def test_429_cooldown_short_circuits_the_next_call(mock_post, no_retry_sleep):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    calls_after_first = mock_post.call_count

    # The cooldown is still in the future, so the second call must not reach the network. Without
    # the patched sleep this is also where a job slot would be blocked for up to 60s (AUDIT-W3).
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert mock_post.call_count == calls_after_first


@patch("worker.services.llm_client.requests.post")
def test_429_honours_a_larger_retry_after_header(mock_post, no_retry_sleep):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "90"}
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    client.complete(messages=[{"role": "user", "content": "Hi"}])

    # 90s beats the escalated default, so the provider's own backpressure wins.
    remaining = llm_client.PROVIDER_COOLDOWNS["openrouter"] - time.time()
    assert 85 < remaining <= 90


@patch("worker.services.llm_client.requests.post")
def test_400_degrades_json_schema_to_json_object_and_succeeds(mock_post, no_retry_sleep):
    """The degradation mutates `payload` in place, so recorded call args all alias the final dict.

    Deep-copying inside the side effect is the only way to prove the *first* attempt really carried
    json_schema — asserting on mock_post.call_args_list here would silently pass either way.
    """
    sent: list[dict] = []

    bad = MagicMock(status_code=400, text="unsupported response_format")
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def record(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return bad if len(sent) == 1 else good

    mock_post.side_effect = record

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    res = client.complete(
        messages=[{"role": "user", "content": "Hi"}],
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )

    assert res is not None and res.content == "ok"
    assert len(sent) == 2
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[1]["response_format"] == {"type": "json_object"}


@patch("worker.services.llm_client.requests.post")
def test_400_without_a_schema_is_permanent_and_not_retried(mock_post, no_retry_sleep):
    mock_resp = MagicMock(status_code=400, text="malformed request")
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    # No json_schema to degrade, so the 400 falls through to PermanentAPIError — one attempt only.
    assert mock_post.call_count == 1


@patch("worker.services.llm_client.requests.post")
def test_5xx_is_transient_and_retried_three_times(mock_post, no_retry_sleep):
    mock_resp = MagicMock(status_code=503, text="upstream unavailable")
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert mock_post.call_count == 3
    # A 5xx must not be confused with rate limiting — no cooldown, no 429 counter.
    assert "openrouter" not in llm_client.PROVIDER_COOLDOWNS
    assert "openrouter" not in llm_client.PROVIDER_CONSECUTIVE_429S


@patch("worker.services.llm_client.requests.post")
def test_5xx_recovers_when_a_retry_succeeds(mock_post, no_retry_sleep):
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "recovered"}}], "usage": {}}
    mock_post.side_effect = [MagicMock(status_code=500, text="boom"), good]

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert res is not None and res.content == "recovered"
    assert mock_post.call_count == 2


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.Timeout("read timed out"),
        requests.exceptions.ConnectionError("connection refused"),
    ],
)
@patch("worker.services.llm_client.requests.post")
def test_network_errors_are_transient_and_retried(mock_post, no_retry_sleep, exc):
    mock_post.side_effect = exc

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    assert client.complete(messages=[{"role": "user", "content": "Hi"}]) is None
    assert mock_post.call_count == 3


# --- AUDIT-W8: the Anthropic branch got no JSON enforcement at all -------------------------------


@patch("worker.services.llm_client.requests.post")
def test_anthropic_gets_structured_output_when_a_schema_is_supplied(mock_post):
    """The whole response_format ladder sat inside the `else`, so Anthropic got nothing.

    Anthropic does not accept `response_format`; the Messages API spells structured output as
    `output_config.format`, and its json_schema variant takes the schema directly rather than the
    OpenAI name/schema/strict wrapper.
    """
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": '{"translatedText": "Hi"}'}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_post.return_value = mock_resp

    schema = {"type": "object", "properties": {"translatedText": {"type": "string"}}}
    client = LLMClient(provider="anthropic", api_key="k", model="claude-sonnet-5")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}], response_schema=schema)

    assert res is not None
    posted = mock_post.call_args.kwargs["json"]
    assert posted["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    # The OpenAI-shaped key must not leak onto an Anthropic payload.
    assert "response_format" not in posted


@patch("worker.services.llm_client.requests.post")
def test_anthropic_400_on_output_config_degrades_by_dropping_it(mock_post, no_retry_sleep):
    """Structured outputs are not on every Anthropic model, so the 400 ladder has to cover them.

    There is no json_object tier to fall back to on Anthropic, so the degrade is a removal: the
    caller's JSON system prompt still stands, which is exactly the pre-fix behaviour.
    """
    sent: list[dict] = []

    bad = MagicMock(status_code=400, text="output_config is not supported for this model")
    good = MagicMock(status_code=200)
    good.json.return_value = {"content": [{"type": "text", "text": "ok"}], "usage": {}}

    def record(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return bad if len(sent) == 1 else good

    mock_post.side_effect = record

    client = LLMClient(provider="anthropic", api_key="k", model="claude-3-opus-20240229")
    res = client.complete(
        messages=[{"role": "user", "content": "Hi"}],
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )

    assert res is not None and res.content == "ok"
    assert len(sent) == 2
    assert sent[0]["output_config"]["format"]["type"] == "json_schema"
    assert "output_config" not in sent[1]


@patch("worker.services.llm_client.requests.post")
def test_null_content_alongside_a_refusal_becomes_empty_string(mock_post):
    """`.get("content", "")` returns None when the key is present and null.

    Providers send exactly that alongside a `refusal`, and the default only applies to a *missing*
    key. Downstream json.loads(None) raises TypeError rather than reporting a parse failure.
    """
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": None, "refusal": "I can't help with that"}}],
        "usage": {},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert res is not None
    assert res.content == ""


@patch("worker.services.llm_client.requests.post")
def test_anthropic_content_skips_non_text_blocks_and_never_returns_none(mock_post):
    """content[0] is not guaranteed to be the text block once thinking is in play."""
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": '{"translatedText": "Hi"}'},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="anthropic", api_key="k", model="claude-sonnet-5")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert res is not None
    assert res.content == '{"translatedText": "Hi"}'


@patch("worker.services.llm_client.requests.post")
def test_anthropic_content_with_no_text_block_is_empty_not_none(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [{"type": "thinking", "thinking": ""}],
        "usage": {},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="anthropic", api_key="k", model="claude-sonnet-5")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert res is not None
    assert res.content == ""


def test_concurrent_429s_do_not_lose_consecutive_increments():
    """PROVIDER_CONSECUTIVE_429S[p] = PROVIDER_CONSECUTIVE_429S.get(p) + 1 is a read-modify-write.

    Job threads share the module-level dicts, so interleaved 429s from the same provider silently
    collapse into one increment and the backoff never escalates. The slow-read dict below forces the
    interleaving that makes the race deterministic rather than luck-dependent.

    Deliberately does *not* take `no_retry_sleep`: that fixture patches `time.sleep` on the shared
    `time` module, which would neutralise the delay below and leave this test passing whether or not
    the lock is there. It makes no HTTP call, so it has no retry ladder to speed up anyway.
    """

    class SlowReadDict(dict):
        """Widens the window between the read and the write.

        The sleep has to come *after* super().get, not before: sleeping first just lets every other
        thread finish its whole read-modify-write before this one reads, which serialises them and
        hides the race. Holding the stale value across the window is what reproduces it.
        """

        def get(self, key, default=None):
            value = super().get(key, default)
            time.sleep(0.002)
            return value

    llm_client.PROVIDER_COOLDOWNS.clear()
    original = llm_client.PROVIDER_CONSECUTIVE_429S
    llm_client.PROVIDER_CONSECUTIVE_429S = SlowReadDict()
    try:
        client = LLMClient(provider="openrouter", api_key="k", model="m")
        threads = [threading.Thread(target=client._register_rate_limit, args=(None,)) for _ in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert llm_client.PROVIDER_CONSECUTIVE_429S["openrouter"] == 24
    finally:
        llm_client.PROVIDER_CONSECUTIVE_429S = original
        llm_client.PROVIDER_COOLDOWNS.clear()


def test_editing_providers_json_is_picked_up_without_a_worker_restart(tmp_path, monkeypatch):
    """PROVIDER_REGISTRY = get_provider_registry() ran at import, so the file was read exactly once.

    The backend has ProviderConfigCache.reload(); the worker had nothing, so a providers.json edit
    needed a container restart to take effect. Asymmetric and surprising.
    """
    cfg = tmp_path / "providers.json"
    cfg.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {},
                "providers": {
                    "acme": {
                        "displayName": "Acme",
                        "type": "openai-compatible",
                        "baseUrl": "https://acme.example/v1/chat/completions",
                        "models": {"tl": [{"id": "acme-1", "name": "Acme 1"}]},
                        "defaults": {"tl": "acme-1"},
                    }
                },
            }
        )
    )
    monkeypatch.setenv("PROVIDERS_CONFIG", str(cfg))
    provider_config.reset_config_loader()
    llm_client.reload_provider_registry()

    assert LLMClient(provider="acme", api_key="k").url == "https://acme.example/v1/chat/completions"

    cfg.write_text(
        json.dumps(
            {
                "version": 2,
                "defaults": {},
                "providers": {
                    "acme": {
                        "displayName": "Acme",
                        "type": "openai-compatible",
                        "baseUrl": "https://acme.example/v2/chat/completions",
                        "models": {"tl": [{"id": "acme-2", "name": "Acme 2"}]},
                        "defaults": {"tl": "acme-2"},
                    }
                },
            }
        )
    )
    os.utime(cfg, (time.time() + 5, time.time() + 5))

    # No restart, no explicit reload call: constructing a client re-reads the changed file.
    assert LLMClient(provider="acme", api_key="k").url == "https://acme.example/v2/chat/completions"

    # Restore PROVIDERS_CONFIG *before* rebuilding. undo() rather than delenv(): conftest.py points
    # that variable at tests/test_providers.json, so deleting it outright would leave every later
    # test in the session with a registry built from the real config/providers.json, which has no
    # anthropic or openai entry.
    monkeypatch.undo()
    provider_config.reset_config_loader()
    llm_client.reload_provider_registry()
