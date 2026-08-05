"""Provider configuration loader, validator, and Redis sync publisher."""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from worker.config import logger


@dataclass
class ModelEntry:
    id: str
    name: str
    free: bool = False


@dataclass
class ProviderConfig:
    name: str
    display_name: str
    type: str
    base_url: str
    auth_header: str | None
    auth_prefix: str | None
    extra_headers: dict[str, str]
    key_env_var: str | None
    api_key: str
    free_tier: bool
    rate_limits: int | None
    priority: int
    models: dict[str, list[ModelEntry] | None]
    defaults: dict[str, str | None]
    active: bool = False


def interpolate_env_vars(raw_str: str) -> str:
    """Replace ${VAR:-default} or ${VAR} in strings."""
    if not raw_str:
        return ""

    def replace_match(match):
        var_name = match.group(1)
        default_val = match.group(3) if match.group(3) is not None else ""
        return os.environ.get(var_name, default_val)

    pattern = r"\$\{([A-Za-z0-9_]+)(:-([^}]+))?\}"
    return re.sub(pattern, replace_match, raw_str)


def normalize_model_name(provider: str, model_name: str) -> str:
    """Normalize model identifier for provider payload sending."""
    if not model_name:
        return ""
    name = model_name.removesuffix(":free")
    p = (provider or "").lower()
    if p == "gemini" and name.startswith("google/"):
        name = name[len("google/") :]
    elif p == "neurometric" and name.startswith("neurometric/"):
        name = name[len("neurometric/") :]
    return name


def find_providers_json_path() -> str:
    """Find the path to providers.json from environment or standard locations."""
    env_path = os.environ.get("PROVIDERS_CONFIG", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        "/app/config/providers.json",
        os.path.join(os.path.dirname(__file__), "../../../config/providers.json"),
        os.path.join(os.path.dirname(__file__), "../../config/providers.json"),
        os.path.join(os.getcwd(), "config/providers.json"),
        os.path.join(os.getcwd(), "../config/providers.json"),
    ]
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.exists(abs_path):
            return abs_path

    return ""


class ProviderConfigLoader:
    """Loads and manages provider configurations."""

    def __init__(self, config_path: str = ""):
        self.config_path = config_path or find_providers_json_path()
        self.providers: dict[str, ProviderConfig] = {}
        self.defaults: dict[str, str | None] = {}
        self.version: int = 1
        self.raw_data: dict[str, Any] = {}
        self._loaded_mtime: float | None = None
        self.load_and_validate()

    def _config_mtime(self) -> float | None:
        try:
            return os.stat(self.config_path).st_mtime
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        """Re-read providers.json when it has been edited since the last load.

        Cheap enough to call per LLMClient construction — one stat against a call that then spends
        seconds in an HTTP request. Returns True when a reload actually happened.
        """
        if not self.config_path:
            return False
        mtime = self._config_mtime()
        if mtime is None or mtime == self._loaded_mtime:
            return False
        logger.info(f"providers.json changed on disk (mtime {mtime}); reloading provider configuration.")
        self.load_and_validate()
        return True

    def load_and_validate(self):
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"providers.json not found at '{self.config_path}'. Provider config loader idle.")
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                self.raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse providers.json at '{self.config_path}': {e}")
            return

        # Reload has to drop providers that were deleted from the file, not just overwrite the ones
        # that remain — the loop below only ever adds.
        self.providers = {}
        self._loaded_mtime = self._config_mtime()
        self.version = self.raw_data.get("version", 1)
        self.defaults = self.raw_data.get("defaults", {})

        providers_raw = self.raw_data.get("providers", {})
        for name, pdata in providers_raw.items():
            key_env_var = pdata.get("keyEnvVar")
            api_key = ""
            active = False

            if key_env_var:
                api_key = os.environ.get(key_env_var, "").strip()
                if not api_key:
                    api_key = os.environ.get("API_KEY", "").strip()
                active = bool(api_key)
                if name == "cloudflare":
                    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
                    active = bool(api_key and account_id)
            else:
                # Local or keyless provider
                active = True

            base_url = interpolate_env_vars(pdata.get("baseUrl", ""))

            models_dict: dict[str, list[ModelEntry] | None] = {}
            for task, task_models in pdata.get("models", {}).items():
                if task_models is None:
                    models_dict[task] = None
                else:
                    models_dict[task] = [
                        ModelEntry(
                            id=m["id"],
                            name=m.get("name", m["id"]),
                            free=m.get("free", False),
                        )
                        for m in task_models
                    ]

            defaults_dict = {
                "tl": pdata.get("defaultTLModel"),
                "qaLLM": pdata.get("defaultQALLMModel"),
                "qaVLM": pdata.get("defaultQAVLMModel"),
                "ocr": pdata.get("defaultOCRModel"),
            }

            # Validate defaults exist in model list if specified
            for task, default_model in defaults_dict.items():
                task_models = models_dict.get(task)
                if default_model and task_models is not None:
                    model_ids = [m.id for m in task_models]
                    if default_model not in model_ids:
                        logger.warning(
                            f"Provider '{name}' default model '{default_model}' for task '{task}' not in models list"
                        )

            prov_config = ProviderConfig(
                name=name,
                display_name=pdata.get("displayName", name),
                type=pdata.get("type", "openai-compatible"),
                base_url=base_url,
                auth_header=pdata.get("authHeader"),
                auth_prefix=pdata.get("authPrefix"),
                extra_headers=pdata.get("extraHeaders", {}),
                key_env_var=key_env_var,
                api_key=api_key,
                free_tier=pdata.get("freeTier", False),
                rate_limits=pdata.get("rateLimits"),
                priority=pdata.get("priority", 99),
                models=models_dict,
                defaults=defaults_dict,
                active=active,
            )
            self.providers[name] = prov_config

        logger.info(
            f"Successfully loaded providers.json v{self.version} ({len(self.providers)} providers, "
            f"{sum(1 for p in self.providers.values() if p.active)} active)"
        )

    def get_active_providers(self) -> dict[str, ProviderConfig]:
        return {k: v for k, v in self.providers.items() if v.active}

    def get_provider_registry(self) -> dict[str, dict]:
        """Convert loaded ProviderConfigs into format expected by LLMClient / PROVIDER_REGISTRY."""
        registry = {}
        for name, pconfig in self.providers.items():
            registry[name] = {
                "url": pconfig.base_url,
                "auth_header": pconfig.auth_header or "Authorization",
                "auth_prefix": pconfig.auth_prefix if pconfig.auth_prefix is not None else "Bearer ",
                "extra_headers": pconfig.extra_headers,
                "rate_limits": pconfig.rate_limits,
                "default_model": pconfig.defaults.get("tl") or "",
                "default_vision_model": pconfig.defaults.get("ocr") or pconfig.defaults.get("qaVLM") or "",
                "is_anthropic": pconfig.type == "anthropic",
            }
        return registry

    def publish_config_to_redis(self, redis_client):
        """Publish resolved provider/model map to Redis for backend consumption."""
        if not redis_client:
            return

        try:
            active_provs = self.get_active_providers()
            config_map = {
                "version": self.version,
                "defaults": self.defaults,
                "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "providers": {},
            }

            for name, prov in active_provs.items():
                models_export = {}
                capabilities = []
                for task, model_list in prov.models.items():
                    if model_list is not None:
                        capabilities.append(task)
                        models_export[task] = [{"id": m.id, "name": m.name, "free": m.free} for m in model_list]

                config_map["providers"][name] = {
                    "displayName": prov.display_name,
                    "type": prov.type,
                    "freeTier": prov.free_tier,
                    "priority": prov.priority,
                    "rateLimits": prov.rate_limits,
                    "models": models_export,
                    "defaults": prov.defaults,
                    "capabilities": capabilities,
                }

            json_blob = json.dumps(config_map)
            redis_client.set("system:providers:config", json_blob)
            redis_client.publish(
                "provider:config:updated",
                json.dumps({"event": "config_reload", "timestamp": config_map["timestamp"]}),
            )
            logger.info("Published resolved provider configuration to Redis key system:providers:config")
        except Exception as e:
            logger.error(f"Failed to publish provider config to Redis: {e}")


# Singleton loader instance
_global_loader: ProviderConfigLoader | None = None


def get_config_loader() -> ProviderConfigLoader:
    global _global_loader
    if _global_loader is None:
        _global_loader = ProviderConfigLoader()
    return _global_loader


def reset_config_loader() -> None:
    """Drop the singleton so the next get_config_loader() re-resolves the path and reloads."""
    global _global_loader
    _global_loader = None


def get_provider_registry() -> dict[str, dict]:
    loader = get_config_loader()
    registry = loader.get_provider_registry()
    if not registry:
        # Fallback to hardcoded registry if loading failed or returned empty
        logger.warning("Dynamic provider registry empty, falling back to built-in default registry.")
        return {
            "openrouter": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "extra_headers": {"HTTP-Referer": "https://manga-library"},
                "default_model": "deepseek/deepseek-v4-pro",
            },
            "gemini": {
                "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "default_model": "gemini-3.5-flash",
            },
            "cloudflare": {
                "url": "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "default_vision_model": "@cf/meta/llama-3.2-11b-vision-instruct",
            },
            "nvidia": {
                "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "default_model": "nvidia/riva-translate-4b-instruct-v1.1",
                "default_vision_model": "nvidia/nemotron-nano-12b-v2-vl",
            },
            "neurometric": {
                "url": "https://api.neurometric.ai/v1/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "default_model": "clawpack",
            },
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer ",
                "default_model": "gpt-4o-mini",
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1/messages",
                "auth_header": "x-api-key",
                "auth_prefix": "",
                "extra_headers": {"anthropic-version": "2023-06-01"},
                # claude-3-5-sonnet-20241022 was retired on 2025-10-28 and now 404s, so this
                # fallback branch could only ever fail. Sonnet is the right tier here: this is
                # high-volume per-page translation/OCR, not deep reasoning.
                "default_model": "claude-sonnet-5",
                "is_anthropic": True,
            },
        }
    return registry
