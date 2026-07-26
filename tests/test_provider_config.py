from unittest.mock import MagicMock

from worker.provider_config import ProviderConfigLoader, normalize_model_name


def test_normalize_model_name():
    assert normalize_model_name("gemini", "google/gemini-2.5-flash:free") == "gemini-2.5-flash"
    assert (
        normalize_model_name("nvidia", "nvidia/riva-translate-4b-instruct-v1.1:free")
        == "nvidia/riva-translate-4b-instruct-v1.1"
    )
    assert normalize_model_name("neurometric", "neurometric/clawpack") == "clawpack"
    assert normalize_model_name("openai", "gpt-4o-mini:free") == "gpt-4o-mini"
    assert normalize_model_name("anthropic", "claude-3-5-sonnet-20241022:free") == "claude-3-5-sonnet-20241022"


def test_provider_config_loader():
    loader = ProviderConfigLoader()
    assert loader.providers is not None
    assert "openrouter" in loader.providers
    assert "gemini" in loader.providers
    assert "nvidia" in loader.providers

    registry = loader.get_provider_registry()
    assert "openrouter" in registry
    assert "openai" in registry
    assert "anthropic" in registry
    assert "url" in registry["openrouter"]


def test_publish_config_to_redis():
    loader = ProviderConfigLoader()
    mock_redis = MagicMock()
    loader.publish_config_to_redis(mock_redis)

    assert mock_redis.set.called
    assert mock_redis.publish.called
    args = mock_redis.set.call_args[0]
    assert args[0] == "system:providers:config"
