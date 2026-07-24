"""Thin LLM HTTP client. Replaces duplicated request/retry/parse logic in

try_cloud_ai, try_cloud_ai_vision, and try_cloud_ai_vision_batch.
Includes native prompt caching support for OpenRouter and Anthropic.
"""

import time
from dataclasses import dataclass

import requests
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

from worker.config import logger
from worker.utils.rate_limit import enforce_rate_limit, estimate_cost


class TransientAPIError(Exception):
    """Raised on retryable HTTP errors (429, 5xx, timeouts)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class PermanentAPIError(Exception):
    """Raised on non-retryable HTTP errors (400, 401, 403, etc.)."""

    pass


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    provider: str = ""
    cost: float | None = None


# Provider cooldown registry
PROVIDER_COOLDOWNS: dict[str, float] = {}


def wait_for_cooldown(provider: str, max_wait: float = 60.0):
    """Block if provider is in cooldown."""
    cooldown_until = PROVIDER_COOLDOWNS.get(provider, 0.0)
    remaining = cooldown_until - time.time()
    if remaining > 0:
        sleep_time = min(remaining, max_wait)
        logger.info(f"Provider '{provider}' is on cooldown. Sleeping for {sleep_time:.1f}s...")
        time.sleep(sleep_time)


# Provider endpoint registry
PROVIDER_REGISTRY: dict[str, dict] = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {"HTTP-Referer": "https://manga-library"},
        "default_model": "meta-llama/llama-3-8b-instruct:free",
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
        "default_model": "claude-3-5-sonnet-20241022",
        "is_anthropic": True,
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "default_model": "gemini-1.5-flash",
    },
    "nvidia": {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "default_model": "nvidia/riva-translate-4b-instruct-v1.1",
        "default_vision_model": "nvidia/nemotron-nano-12b-v2-vl",
    },
}


class LLMClient:
    """Thin HTTP client for LLM providers with automatic retry and prompt caching."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str = "",
        request_id: str = "",
        routing_strategy: str | None = None,
        session_id: str | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.request_id = request_id
        self.routing_strategy = routing_strategy
        self.session_id = session_id
        self.req_prefix = f"[{request_id}] " if request_id else ""

        provider_info = PROVIDER_REGISTRY.get(provider, {})
        self.url = provider_info.get("url", "")
        self.is_anthropic = provider_info.get("is_anthropic", False)
        self.model = model or provider_info.get("default_model", "")

        self.headers = {"Content-Type": "application/json"}
        auth_header = provider_info.get("auth_header", "Authorization")
        auth_prefix = provider_info.get("auth_prefix", "Bearer ")
        if api_key:
            self.headers[auth_header] = f"{auth_prefix}{api_key}"
        if extra := provider_info.get("extra_headers"):
            self.headers.update(extra)

        self._degraded_format = False

    def complete(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse | None:
        """Send a completion request with retries."""
        if not self.url or not self.api_key:
            logger.warning(f"{self.req_prefix}Missing URL or API key for provider '{self.provider}'")
            return None

        wait_for_cooldown(self.provider)
        cooldown_until = PROVIDER_COOLDOWNS.get(self.provider, 0.0)
        if time.time() < cooldown_until:
            logger.warning(f"{self.req_prefix}Skipping provider '{self.provider}' — still in cooldown")
            return None

        enforce_rate_limit()

        payload = self._build_payload(messages, system_prompt, response_schema)
        self._inject_routing_and_caching(payload)

        try:
            return self._execute_with_retry(payload)
        except (TransientAPIError, PermanentAPIError) as e:
            logger.error(f"{self.req_prefix}LLM call failed for provider '{self.provider}': {e}")
            return None

    def _build_payload(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        response_schema: dict | None = None,
    ) -> dict:
        """Build provider-specific request payload."""
        payload = {}
        if self.is_anthropic:
            payload = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if system_prompt:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
        else:
            payload = {
                "model": self.model,
                "messages": list(messages),
            }
            if system_prompt:
                payload["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

            if response_schema and not self._degraded_format:
                if self.provider == "nvidia":
                    payload["response_format"] = {"type": "json_object"}
                else:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": response_schema,
                            "strict": True,
                        },
                    }
                    if self.provider == "openrouter":
                        payload["plugins"] = [{"id": "response-healing"}]

        return payload

    def _inject_routing_and_caching(self, payload: dict):
        """Inject routing parameters, OpenRouter prompt caching, and session tracking."""
        if self.provider != "openrouter":
            return

        if self.routing_strategy == "lowest-cost":
            payload["provider"] = {
                "allow_fallbacks": False,
                "sort": "price",
                "order": ["StreamLake", "NovitaAI", "Baidu Qianfan", "Decart"],
            }
        elif self.routing_strategy == "highest-throughput":
            payload["provider"] = {"allow_fallbacks": True, "sort": "throughput"}

        # Inject OpenRouter cache_control on system prompt if present
        if "messages" in payload:
            for msg in payload["messages"]:
                if msg.get("role") == "system":
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    break

        if self.session_id:
            payload.setdefault("extra_body", {})["session_id"] = self.session_id

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(TransientAPIError),
        reraise=True,
    )
    def _execute_with_retry(self, payload: dict) -> LLMResponse:
        """Execute HTTP request with Tenacity backoff."""
        start = time.perf_counter()
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=(10, 45))
        except requests.exceptions.Timeout as e:
            raise TransientAPIError(f"Timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise TransientAPIError(f"Connection error: {e}") from e

        if response.status_code == 429:
            PROVIDER_COOLDOWNS[self.provider] = time.time() + 5.0
            raise TransientAPIError("Rate limited (429)", status_code=429)

        if response.status_code == 400 and not self._degraded_format:
            current_rf = payload.get("response_format", {})
            if current_rf.get("type") == "json_schema":
                logger.warning(f"{self.req_prefix}400 with json_schema — degrading to json_object")
                payload["response_format"] = {"type": "json_object"}
                self._degraded_format = True
                raise TransientAPIError("Degrading json_schema to json_object", status_code=400)

        if response.status_code >= 500:
            raise TransientAPIError(f"Server error: {response.status_code}", status_code=response.status_code)
        if response.status_code >= 400:
            raise PermanentAPIError(f"Client error: {response.status_code} — {response.text}")

        return self._parse_response(response.json(), time.perf_counter() - start)

    def _parse_response(self, data: dict, elapsed: float) -> LLMResponse:
        """Normalize response JSON from Anthropic or OpenAI format."""
        if self.is_anthropic:
            content_list = data.get("content", [])
            content = content_list[0].get("text", "") if content_list else ""
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            cached_tokens = usage.get("cache_read_input_tokens", 0)
        else:
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens", 0)

        logger.info(f"{self.req_prefix}Provider={self.provider} Model={self.model} Time={elapsed:.2f}s")
        logger.info(f"{self.req_prefix}Tokens in={prompt_tokens} out={completion_tokens} total={total_tokens}")

        if cached_tokens > 0:
            cache_ratio = cached_tokens / max(prompt_tokens, 1)
            logger.info(
                f"{self.req_prefix}Cache hit: {cached_tokens}/{prompt_tokens} tokens ({cache_ratio:.0%} cached)"
            )

        cost = estimate_cost(self.model, prompt_tokens, completion_tokens, self.provider)
        if cost is not None:
            logger.info(f"{self.req_prefix}Estimated cost: ${cost:.5f}")

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            model=self.model,
            provider=self.provider,
            cost=cost,
        )
