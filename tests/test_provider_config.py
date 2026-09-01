import ast
import json
import pathlib
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


def test_model_pricing_is_preserved_in_published_catalog(tmp_path):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "test": {
                        "models": {
                            "tl": [
                                {
                                    "id": "priced/model",
                                    "name": "Priced model",
                                    "pricing": {
                                        "currency": "USD",
                                        "promptPerMillion": 0.25,
                                        "completionPerMillion": 1.5,
                                        "source": "test",
                                    },
                                }
                            ]
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loader = ProviderConfigLoader(str(config_path))
    mock_redis = MagicMock()

    loader.publish_config_to_redis(mock_redis)

    published = json.loads(mock_redis.set.call_args.args[1])
    pricing = published["providers"]["test"]["models"]["tl"][0]["pricing"]
    assert pricing["promptPerMillion"] == 0.25
    assert pricing["completionPerMillion"] == 1.5


def test_importing_worker_config_does_not_publish():
    """Importing worker.config must not touch the provider config key.

    system:providers:config is the only thing the backend builds the settings UI's provider
    dropdowns from, and it is filtered to the providers the *publishing* process holds keys for.
    Every worker module imports worker.config, so publishing at import time let any host-run script
    (a probe, a benchmark) overwrite the deployment's providers with its own key-less environment —
    which emptied the Translation and QA provider dropdowns and left OCR with nothing but `local`.

    Asserted over the AST rather than by reloading the module: re-executing worker.config would
    re-run every other import side effect it has, one of which can call sys.exit.
    """
    import worker.config as worker_config

    assert callable(getattr(worker_config, "publish_provider_config", None)), (
        "worker.config must expose publish_provider_config() for the service to call explicitly"
    )

    config_path = worker_config.__file__
    assert config_path, "worker.config has no __file__ to inspect"
    source = pathlib.Path(config_path).read_text(encoding="utf-8")
    # Statements that run on import: everything at module level except the bodies of defs, which
    # only run when something calls them.
    executed_on_import = [
        stmt
        for stmt in ast.parse(source).body
        if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    for stmt in executed_on_import:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "publish_config_to_redis", (
                    "worker/config.py publishes provider config at import time; move the call into "
                    "the worker service's startup (see main.py's lifespan)."
                )
