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


@pytest.mark.usefixtures("no_retry_sleep")
@patch("worker.services.llm_client.requests.post")
def test_no_matching_host_retries_once_without_the_filters(mock_post):
    sent = []
    no_endpoints = MagicMock(
        status_code=404,
        text='{"error":{"message":"No endpoints found that can handle the requested parameters.","code":404}}',
    )

    def record(*_args, **kw):
        sent.append(copy.deepcopy(kw["json"]))
        return no_endpoints if len(sent) == 1 else _ok()

    mock_post.side_effect = record
    client = LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")
    res = client.complete(messages=[{"role": "user", "content": "Hi"}], response_schema=SCHEMA)

    assert res is not None and res.content == "ok"
    assert len(sent) == 2
    assert {"quantizations", "ignore", "require_parameters"} <= set(sent[0]["provider"])
    assert not {"quantizations", "ignore", "require_parameters"} & set(sent[1]["provider"])
    assert sent[1]["provider"]["sort"] == "price"


@patch("worker.services.llm_client.requests.post")
def test_empty_settings_turn_the_filters_off(mock_post, monkeypatch):
    monkeypatch.setattr(llm_client, "OPENROUTER_QUANTIZATIONS", [])
    monkeypatch.setattr(llm_client, "OPENROUTER_IGNORE_PROVIDERS", [])
    monkeypatch.setattr(llm_client, "OPENROUTER_REQUIRE_PARAMETERS", False)
    client = LLMClient(provider="openrouter", api_key="k", model="m", routing_strategy="lowest-cost")
    assert _sent(mock_post, client, response_schema=SCHEMA)[0]["provider"] == {"allow_fallbacks": True, "sort": "price"}
