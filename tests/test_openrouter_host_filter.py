"""OpenRouter calls skip the hosts that ignore our request.

Measured 2026-10-06 with a 512-token reasoning budget, one host pinned per call:
- deepseek-v4-flash: Relace reasoned 1,561 tokens and OpenInference 784 (both fp4), at 10-50x the
  cost of the fp8 hosts that kept to the budget (StreamLake 377, Baidu 509, Alibaba 294). In the
  pipeline, OpenInference spent all 8,192 output tokens on reasoning (finish=length, no
  translation) for 216 s a call before the fallback ran.
- deepseek-v4.1-flash: Relace 1,160 and OpenInference 1,098 against 428-608 elsewhere.
- glm-5.3-flash and deepseek-v4.1-flash with the fp4 hosts filtered: Wafer reasoned 6,600-8,192
  tokens past a 4,096 budget, 153-277 s a call, two truncated with no answer.
- mimo-v2.6-flash: cheapest-first routing sent every pipeline call to Darkbloom (fp4) at 15-28
  tokens/s, 13-87 s a call.
Cheapest-first ranks hosts by prompt price, and these are the cheapest on prompt while costing the
most per output token.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from tests.test_llm_client import no_retry_sleep  # noqa: F401  (the retry-ladder fixture)
from worker.services import llm_client
from worker.services.llm_client import LLMClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def _ok():
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    return good


def _sent(mock_post, client, **kwargs):
    sent = []

    def record(*_args, **kw):
        sent.append(copy.deepcopy(kw["json"]))
        return _ok()

    mock_post.side_effect = record
    client.complete(messages=[{"role": "user", "content": "Hi"}], **kwargs)
    return sent


@patch("worker.services.llm_client.requests.post")
def test_openrouter_skips_low_precision_and_listed_hosts(mock_post):
    client = LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")
    provider = _sent(mock_post, client)[0]["provider"]
    assert provider["sort"] == "price"
    assert "fp4" not in provider["quantizations"]
    assert "fp8" in provider["quantizations"] and "unknown" in provider["quantizations"]
    assert provider["ignore"] == ["relace", "open-inference", "wafer"]


@patch("worker.services.llm_client.requests.post")
def test_the_filters_apply_without_a_routing_strategy(mock_post):
    client = LLMClient(provider="openrouter", api_key="k", model="m")
    provider = _sent(mock_post, client)[0]["provider"]
    assert provider["ignore"] == ["relace", "open-inference", "wafer"]
    assert "sort" not in provider


@patch("worker.services.llm_client.requests.post")
def test_a_schema_call_requires_hosts_that_support_every_parameter(mock_post):
    client = LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")
    assert _sent(mock_post, client, response_schema=SCHEMA)[0]["provider"]["require_parameters"] is True
    assert "require_parameters" not in _sent(mock_post, client)[0]["provider"]


@patch("worker.services.llm_client.requests.post")
def test_other_providers_get_no_openrouter_routing(mock_post):
    client = LLMClient(provider="nvidia", api_key="k", model="m")
    assert "provider" not in _sent(mock_post, client)[0]


# OpenRouter's 404 bodies, as returned 2026-10-06 (Mistral Small 3.2, messages trimmed).
BY_PARAMETERS = (
    '{"error":{"message":"No endpoints found that can handle the requested parameters.","code":404,'
    '"metadata":{"failed_routing_step":"Filter by Parameters"}}}'
)
BY_QUANTIZATION = (
    '{"error":{"message":"No endpoints found for the request with quantization: fp32.","code":404,'
    '"metadata":{"failed_routing_step":"Filter by Quantization"}}}'
)
BY_IGNORE = (
    '{"error":{"message":"All providers have been ignored.","code":404,'
    '"metadata":{"failed_routing_step":"Filter by Ignored Providers"}}}'
)
NO_STEP = '{"error":{"message":"No endpoints found that can handle the requested parameters.","code":404}}'
FILTERS = {"quantizations", "ignore"}


def _answers(mock_post, *not_found):
    """Answer with each 404 body in turn, then succeed; record every payload sent."""
    sent = []

    def record(*_args, **kw):
        sent.append(copy.deepcopy(kw["json"]))
        if len(sent) <= len(not_found):
            return MagicMock(status_code=404, text=not_found[len(sent) - 1])
        return _ok()

    mock_post.side_effect = record
    return sent


def _client():
    return LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")


def _complete(client, **kwargs):
    res = client.complete(messages=[{"role": "user", "content": "Hi"}], **kwargs)
    assert res is not None and res.content == "ok"


@pytest.mark.usefixtures("no_retry_sleep")
@patch("worker.services.llm_client.requests.post")
def test_a_model_that_cannot_reason_drops_reasoning_but_keeps_the_schema_guard(mock_post):
    """CodeRabbit on #57: the retry used to drop require_parameters, so a host could ignore the
    schema. Mistral Small 3.2 has hosts for the schema but none for ``reasoning`` (404 with it,
    200 without, measured 2026-10-06), so only reasoning goes."""
    sent = _answers(mock_post, BY_PARAMETERS)
    client = _client()
    _complete(client, response_schema=SCHEMA)

    assert len(sent) == 2
    assert "reasoning" in sent[0] and "reasoning" not in sent[1]
    assert sent[1]["provider"]["require_parameters"] is True
    assert sent[1]["response_format"]["type"] == "json_schema"
    assert set(sent[1]["provider"]) >= FILTERS and sent[1]["provider"]["sort"] == "price"

    # The client's later calls start where the retry left off.
    later = _sent(mock_post, client, response_schema=SCHEMA)[0]
    assert "reasoning" not in later and later["provider"]["require_parameters"] is True


@pytest.mark.usefixtures("no_retry_sleep")
@pytest.mark.parametrize("body", [BY_QUANTIZATION, BY_IGNORE])
@patch("worker.services.llm_client.requests.post")
def test_no_host_past_the_host_filters_drops_only_those(mock_post, body):
    sent = _answers(mock_post, body)
    _complete(_client(), response_schema=SCHEMA)

    assert len(sent) == 2
    assert not FILTERS & set(sent[1]["provider"])
    assert sent[1]["provider"]["require_parameters"] is True
    assert sent[1]["reasoning"] == sent[0]["reasoning"]


@pytest.mark.usefixtures("no_retry_sleep")
@patch("worker.services.llm_client.requests.post")
def test_require_parameters_goes_only_when_the_model_still_has_no_host(mock_post):
    sent = _answers(mock_post, BY_PARAMETERS, BY_PARAMETERS)
    _complete(_client(), response_schema=SCHEMA)

    assert len(sent) == 3
    assert sent[1]["provider"]["require_parameters"] is True and "reasoning" not in sent[1]
    assert "require_parameters" not in sent[2]["provider"]
    # The reasoning cap is back once nothing forces a host to support it.
    assert sent[2]["reasoning"] == sent[0]["reasoning"]


@pytest.mark.usefixtures("no_retry_sleep")
@patch("worker.services.llm_client.requests.post")
def test_a_404_naming_no_step_relaxes_the_host_filters_first(mock_post):
    sent = _answers(mock_post, NO_STEP)
    _complete(_client(), response_schema=SCHEMA)

    assert len(sent) == 2
    assert not FILTERS & set(sent[1]["provider"])
    assert sent[1]["provider"]["require_parameters"] is True and "reasoning" in sent[1]


@pytest.mark.usefixtures("no_retry_sleep")
@patch("worker.services.llm_client.requests.post")
def test_a_call_without_a_schema_keeps_its_reasoning_budget(mock_post):
    sent = _answers(mock_post, BY_QUANTIZATION)
    _complete(_client())

    assert len(sent) == 2
    assert not FILTERS & set(sent[1]["provider"])
    assert sent[1]["reasoning"] == sent[0]["reasoning"]


@patch("worker.services.llm_client.requests.post")
def test_empty_settings_turn_the_filters_off(mock_post, monkeypatch):
    monkeypatch.setattr(llm_client, "OPENROUTER_QUANTIZATIONS", [])
    monkeypatch.setattr(llm_client, "OPENROUTER_IGNORE_PROVIDERS", [])
    monkeypatch.setattr(llm_client, "OPENROUTER_REQUIRE_PARAMETERS", False)
    client = LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")
    assert _sent(mock_post, client, response_schema=SCHEMA)[0]["provider"] == {"allow_fallbacks": True, "sort": "price"}
