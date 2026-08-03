"""AUDIT-W11 — a chapter pinned to a dead provider must have an escape hatch.

Fallback deliberately does not cross provider boundaries: a chapter pinned to a provider
presumably wants that provider. The exception is a pinned provider whose API key has been
rejected. It is parked in PROVIDER_AUTH_FAILURES and short-circuits every later call, so
refusing to cross means failing 100% of the chapter's translations instead of degrading to a
provider that works. On the 2026-08-02 run one invalid `neurometric` key produced 401 x 323 and
failed 11 of 50 translation jobs, every traceback carrying "No fallback applied".
"""

import time

import pytest

import worker.services.llm_client as llm_client
from worker.services.llm_client import AUTH_FAILURE_COOLDOWN_SECONDS, is_provider_auth_parked
from worker.services.translation import resolve_fallback_target


@pytest.fixture(autouse=True)
def clear_parked():
    llm_client.PROVIDER_AUTH_FAILURES.clear()
    yield
    llm_client.PROVIDER_AUTH_FAILURES.clear()


@pytest.fixture
def global_openrouter(monkeypatch):
    from worker.config import TL_CONFIG

    monkeypatch.setattr(TL_CONFIG, "provider", "openrouter")
    monkeypatch.setattr(TL_CONFIG, "llm_model", "openai/gpt-5.6-luna")
    monkeypatch.setattr(TL_CONFIG, "resolve_key", lambda provider=None: "global-key")
    return TL_CONFIG


def park(provider):
    llm_client.PROVIDER_AUTH_FAILURES[provider] = time.time() + AUTH_FAILURE_COOLDOWN_SECONDS


def test_is_parked_tracks_the_cooldown_window():
    assert not is_provider_auth_parked("neurometric")
    park("neurometric")
    assert is_provider_auth_parked("neurometric")
    assert is_provider_auth_parked("NeuroMetric"), "provider names are compared case-insensitively"
    assert not is_provider_auth_parked(None)

    llm_client.PROVIDER_AUTH_FAILURES["neurometric"] = time.time() - 1
    assert not is_provider_auth_parked("neurometric"), "an elapsed cooldown must release"


def test_does_not_cross_providers_while_the_pinned_one_still_answers(global_openrouter):
    # The pin is honoured: nothing is wrong with 'neurometric' beyond this one model failing.
    assert resolve_fallback_target("neurometric", "some-model", "pinned-key", True) is None


def test_crosses_to_the_global_provider_once_the_pinned_one_is_parked(global_openrouter):
    park("neurometric")

    target = resolve_fallback_target("neurometric", "some-model", "pinned-key", True)

    assert target == ("openrouter", "global-key", "openai/gpt-5.6-luna")


def test_swaps_model_within_the_pinned_provider_when_it_is_the_global_one(global_openrouter):
    target = resolve_fallback_target("openrouter", "some-other-model", "pinned-key", True)

    assert target == ("openrouter", "pinned-key", "openai/gpt-5.6-luna")


def test_no_fallback_when_the_pinned_model_is_already_the_global_default(global_openrouter):
    assert resolve_fallback_target("openrouter", "openai/gpt-5.6-luna", "k", True) is None


def test_respects_use_fallback_models_even_when_parked(global_openrouter):
    park("neurometric")

    assert resolve_fallback_target("neurometric", "some-model", "pinned-key", False) is None


def test_no_crossing_when_the_global_provider_has_no_usable_key(monkeypatch):
    from worker.config import TL_CONFIG

    monkeypatch.setattr(TL_CONFIG, "provider", "openrouter")
    monkeypatch.setattr(TL_CONFIG, "llm_model", "openai/gpt-5.6-luna")
    monkeypatch.setattr(TL_CONFIG, "resolve_key", lambda provider=None: "")
    park("neurometric")

    assert resolve_fallback_target("neurometric", "some-model", "pinned-key", True) is None


def test_no_fallback_without_a_configured_global_default(monkeypatch):
    from worker.config import TL_CONFIG

    monkeypatch.setattr(TL_CONFIG, "provider", "openrouter")
    monkeypatch.setattr(TL_CONFIG, "llm_model", "")
    park("neurometric")

    assert resolve_fallback_target("neurometric", "some-model", "pinned-key", True) is None
