"""Thin LLM HTTP client. Replaces duplicated request/retry/parse logic in

try_cloud_ai, try_cloud_ai_vision, and try_cloud_ai_vision_batch.
Includes native prompt caching support for OpenRouter and Anthropic.
"""

import threading
import time
from dataclasses import dataclass

import requests
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

from worker.config import logger
from worker.provider_config import get_config_loader, get_provider_registry
from worker.utils.rate_limit import enforce_rate_limit, record_llm_call

# Anthropic already sent max_tokens; everyone else was sending none at all, so the ceiling was
# whatever the routed model happened to default to. A QA pass over a dense page blew through it and
# came back truncated (out=3408) with no way for the caller to tell. Explicit and generous: a full
# QA verdict for ~16 regions runs well under this.
DEFAULT_MAX_OUTPUT_TOKENS = 8192


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
    # Anthropic bills cache *writes* at a premium and reports them separately from input_tokens.
    # Nothing read them before, so the premium on the first call of every session was invisible.
    cache_write_tokens: int = 0
    model: str = ""
    provider: str = ""
    cost: float | None = None
    # "authoritative" when the provider reported the cost itself, "estimated" from the local rate
    # table, "free"/"unknown" otherwise. A total built from mixed sources is only as good as its
    # worst entry, which is why the source travels with the number.
    cost_source: str = ""
    # OpenRouter's generation id (gen-...). Without it a call cannot be looked up after the fact,
    # which is what made spend impossible to reconcile against an invoice.
    generation_id: str = ""
    # Which upstream endpoint OpenRouter actually routed to. Same model id, different silicon —
    # the first thing worth knowing when a page is unexpectedly slow, costly, or garbled.
    upstream_provider: str = ""
    # OpenRouter rewrites :floor/:nitro suffixes, so what ran is not always what was asked for.
    model_resolved: str = ""
    duration_ms: int | None = None
    # "stop" on a complete answer, "length" when the model ran out of output budget. Callers that
    # parse structured JSON must check this: OpenRouter's response-healing plugin closes a truncated
    # object so json.loads() still succeeds, and the result looks valid while having silently lost
    # its trailing fields.
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# Provider cooldown registry.
#
# These are read and written from every job thread. Individual dict operations are atomic under the
# GIL, but the 429 bookkeeping is a read-modify-write (get, +1, set) and interleaved 429s from the
# same provider collapsed into a single increment — so the exponential backoff never escalated
# under exactly the load that produces concurrent 429s. _STATE_LOCK guards the compound updates;
# plain single-key reads elsewhere are left unlocked deliberately, since a stale cooldown read only
# costs one extra attempt.
_STATE_LOCK = threading.Lock()
PROVIDER_COOLDOWNS: dict[str, float] = {}
PROVIDER_CONSECUTIVE_429S: dict[str, int] = {}

# Providers that answered 401/403, and until when to stop asking. Deliberately separate from
# PROVIDER_COOLDOWNS: a rate-limit cooldown is worth waiting out, a bad credential is not, so this
# one short-circuits without sleeping.
PROVIDER_AUTH_FAILURES: dict[str, float] = {}

COOLDOWN_BASE_SECONDS = 10.0
COOLDOWN_MAX_SECONDS = 120.0
# Long enough to cover the whole retry ladder above a single job, short enough that fixing the key
# recovers without a worker restart.
AUTH_FAILURE_COOLDOWN_SECONDS = 300.0


def is_provider_auth_parked(provider: str | None) -> bool:
    """True while a provider's key is known-rejected and calls to it short-circuit.

    Callers use this to decide whether falling back across a provider boundary is warranted.
    Normally it is not — a chapter pinned to a provider presumably wants that provider — but a
    pinned provider parked here will refuse every request until the cooldown lapses, so staying
    on it means failing the whole chapter rather than degrading to a working one.
    """
    if not provider:
        return False
    return time.time() < PROVIDER_AUTH_FAILURES.get(provider.lower().strip(), 0.0)


def wait_for_cooldown(provider: str, max_wait: float = 60.0):
    """Block if provider is in cooldown."""
    cooldown_until = PROVIDER_COOLDOWNS.get(provider, 0.0)
    remaining = cooldown_until - time.time()
    if remaining > 0:
        sleep_time = min(remaining, max_wait)
        logger.info(f"Provider '{provider}' is on cooldown. Sleeping for {sleep_time:.1f}s...")
        time.sleep(sleep_time)


# Provider endpoint registry.
#
# Mutated in place rather than rebound, because translation.py and the test suite hold references
# to this exact dict.
PROVIDER_REGISTRY: dict[str, dict] = get_provider_registry()


def reload_provider_registry() -> None:
    """Rebuild PROVIDER_REGISTRY from the current provider configuration."""
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(get_provider_registry())


def normalize_model_name(provider: str, model: str) -> str:
    """Normalize model name for specific direct provider endpoints."""
    prov = provider.lower().strip()
    norm = model.strip()
    if prov == "gemini":
        norm = norm.removeprefix("google/").removeprefix("models/").removesuffix(":free")
    elif prov == "nvidia":
        norm = norm.removesuffix(":free")
    elif prov == "neurometric":
        norm = norm.removeprefix("neurometric/")
    elif prov in ("openai", "anthropic"):
        norm = norm.removesuffix(":free")
    return norm


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

        # PROVIDER_REGISTRY was built once at import, so editing providers.json needed a worker
        # restart to take effect — while the backend has had ProviderConfigCache.reload() all
        # along. Pick up an edited file here instead: one stat on a call that is about to spend
        # seconds in HTTP.
        if get_config_loader().reload_if_changed():
            reload_provider_registry()

        provider_info = PROVIDER_REGISTRY.get(provider, {})
        self.url = provider_info.get("url", "")
        self.is_anthropic = provider_info.get("is_anthropic", False)
        self.rate_limits = provider_info.get("rate_limits")
        raw_model = model or provider_info.get("default_model", "")
        self.model = normalize_model_name(provider, raw_model)

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

        auth_failed_until = PROVIDER_AUTH_FAILURES.get(self.provider, 0.0)
        if time.time() < auth_failed_until:
            logger.warning(
                f"{self.req_prefix}Skipping provider '{self.provider}' — its API key was rejected; "
                f"not retrying for another {auth_failed_until - time.time():.0f}s."
            )
            return None

        wait_for_cooldown(self.provider)
        cooldown_until = PROVIDER_COOLDOWNS.get(self.provider, 0.0)
        if time.time() < cooldown_until:
            logger.warning(f"{self.req_prefix}Skipping provider '{self.provider}' — still in cooldown")
            return None

        enforce_rate_limit(self.provider, self.rate_limits)

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
                "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
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
            if response_schema and not self._degraded_format:
                # Anthropic has no `response_format`; the Messages API spells structured output as
                # output_config.format, and its json_schema variant takes the schema directly
                # rather than the OpenAI name/schema/strict wrapper used below.
                payload["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": response_schema,
                    }
                }
        else:
            payload = {
                "model": self.model,
                "messages": list(messages),
                "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
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

        # Ask OpenRouter to report what the call actually cost. It knows which endpoint served the
        # request and what the cache discount came to; a local rate table can only guess at both.
        payload["usage"] = {"include": True}

        if self.routing_strategy == "lowest-cost":
            payload["provider"] = {
                "allow_fallbacks": True,
                "sort": "price",
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
        stop=stop_after_attempt(3),
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
            cooldown_time = self._register_rate_limit(response.headers.get("Retry-After"))
            raise TransientAPIError(f"Rate limited (429), cooldown: {cooldown_time}s", status_code=429)

        if response.status_code == 400 and not self._degraded_format:
            if payload.get("response_format", {}).get("type") == "json_schema":
                logger.warning(f"{self.req_prefix}400 with json_schema — degrading to json_object")
                payload["response_format"] = {"type": "json_object"}
                self._degraded_format = True
                raise TransientAPIError("Degrading json_schema to json_object", status_code=400)
            if "output_config" in payload:
                # Structured outputs are not available on every Anthropic model, and there is no
                # json_object tier to step down to. Drop the constraint and let the caller's JSON
                # system prompt carry it — which is exactly the behaviour before it was added.
                logger.warning(f"{self.req_prefix}400 with output_config — dropping structured output")
                payload.pop("output_config")
                self._degraded_format = True
                raise TransientAPIError("Dropping unsupported output_config", status_code=400)

        if response.status_code >= 500:
            raise TransientAPIError(f"Server error: {response.status_code}", status_code=response.status_code)

        if response.status_code in (401, 403):
            # A bad credential does not heal inside a job, but every layer above this one retries:
            # batch, then a retry pass, then per-region individual fallback, then the RQ job itself
            # up to 3 times. On the 2026-08-02 drained run one invalid neurometric key produced 323
            # identical 401s and failed 11 of 50 translation jobs. Parking the provider on cooldown
            # makes call() short-circuit the rest of those attempts and says why, once, loudly.
            with _STATE_LOCK:
                PROVIDER_AUTH_FAILURES[self.provider] = time.time() + AUTH_FAILURE_COOLDOWN_SECONDS
            raise PermanentAPIError(
                f"Authentication failed for provider '{self.provider}' ({response.status_code}) — check its API "
                f"key. Skipping this provider for {AUTH_FAILURE_COOLDOWN_SECONDS:.0f}s. {response.text}"
            )

        if response.status_code >= 400:
            raise PermanentAPIError(f"Client error: {response.status_code} — {response.text}")

        return self._parse_response(response.json(), time.perf_counter() - start)

    def _register_rate_limit(self, retry_after: str | None) -> float:
        """Record a 429 and return the cooldown it installs.

        The consecutive-429 count is a read-modify-write over a dict shared by every job thread, so
        it has to be done under the lock or concurrent 429s lose increments and the backoff stays
        flat at exactly the moment it should be escalating.
        """
        cooldown_time = 10.0
        if retry_after and retry_after.isdigit():
            cooldown_time = float(retry_after)
        with _STATE_LOCK:
            consecutive = PROVIDER_CONSECUTIVE_429S.get(self.provider, 0) + 1
            PROVIDER_CONSECUTIVE_429S[self.provider] = consecutive
            multiplier = min(2 ** (consecutive - 1), COOLDOWN_MAX_SECONDS / COOLDOWN_BASE_SECONDS)
            cooldown_time = max(cooldown_time, COOLDOWN_BASE_SECONDS * multiplier)
            PROVIDER_COOLDOWNS[self.provider] = time.time() + cooldown_time
        return cooldown_time

    def _parse_response(self, data: dict, elapsed: float) -> LLMResponse:
        """Normalize response JSON from Anthropic or OpenAI format."""
        with _STATE_LOCK:
            PROVIDER_CONSECUTIVE_429S.pop(self.provider, None)
            PROVIDER_AUTH_FAILURES.pop(self.provider, None)

        # Only the OpenAI-shaped response carries these; the Anthropic branch leaves them at their
        # defaults and falls back to the local estimator.
        authoritative_cost = None
        generation_id = ""
        upstream_provider = ""
        model_resolved = ""

        if self.is_anthropic:
            # content[0] is not reliably the text block — a thinking block precedes it whenever
            # thinking is on — and .get("text") yields None on any block that is not text.
            content = next(
                (b.get("text") or "" for b in data.get("content", []) if b.get("type", "text") == "text"),
                "",
            )
            usage = data.get("usage", {})
            cached_tokens = usage.get("cache_read_input_tokens", 0) or 0
            cache_write_tokens = usage.get("cache_creation_input_tokens", 0) or 0
            # Anthropic reports input_tokens EXCLUDING cache reads and cache writes; the OpenAI
            # shape reports prompt_tokens INCLUDING them. Normalise onto the OpenAI convention so
            # cached_tokens is always a subset of prompt_tokens — the cache-hit ratio logged below
            # and the pricing downstream both depend on that being true.
            prompt_tokens = (usage.get("input_tokens", 0) or 0) + cached_tokens + cache_write_tokens
            completion_tokens = usage.get("output_tokens", 0) or 0
            total_tokens = prompt_tokens + completion_tokens
            # Anthropic spells it "max_tokens"; normalize onto the OpenAI vocabulary.
            finish_reason = "length" if data.get("stop_reason") == "max_tokens" else "stop"
        else:
            choices = data.get("choices", [])
            # `or ""` rather than a .get default: providers send an explicit null content alongside
            # a refusal, and a default only applies to a *missing* key. json.loads(None) downstream
            # raises TypeError instead of reporting a parse failure.
            content = (choices[0].get("message", {}).get("content") or "") if choices else ""
            finish_reason = choices[0].get("finish_reason") or "" if choices else ""
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens") or 0
            cache_write_tokens = details.get("cache_write_tokens") or 0
            # Present only because _inject_routing_and_caching asks for it; absent on providers
            # that do not report a cost, which is what the estimator fallback is for.
            authoritative_cost = usage.get("cost")
            generation_id = data.get("id") or ""
            upstream_provider = data.get("provider") or ""
            model_resolved = data.get("model") or ""

        logger.info(f"{self.req_prefix}Provider={self.provider} Model={self.model} Time={elapsed:.2f}s")
        logger.info(f"{self.req_prefix}Tokens in={prompt_tokens} out={completion_tokens} total={total_tokens}")

        if finish_reason == "length":
            logger.warning(
                f"{self.req_prefix}Response truncated at the output token limit "
                f"(out={completion_tokens}). Structured output past this point is lost."
            )

        if cached_tokens > 0:
            cache_ratio = cached_tokens / max(prompt_tokens, 1)
            logger.info(
                f"{self.req_prefix}Cache hit: {cached_tokens}/{prompt_tokens} tokens ({cache_ratio:.0%} cached)"
            )

        duration_ms = int(elapsed * 1000)
        cost, cost_source = record_llm_call(
            self.model,
            prompt_tokens,
            completion_tokens,
            provider=self.provider,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
            generation_id=generation_id,
            upstream_provider=upstream_provider,
            model_resolved=model_resolved,
            duration_ms=duration_ms,
            authoritative_cost=authoritative_cost,
        )
        if cost is not None:
            logger.info(f"{self.req_prefix}Cost ({cost_source}): ${cost:.5f}")
        else:
            logger.warning(f"{self.req_prefix}Cost unknown for {self.model} — no price available")
        if generation_id:
            logger.info(f"{self.req_prefix}Generation {generation_id} via {upstream_provider or 'unknown'}")

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            model=self.model,
            provider=self.provider,
            cost=cost,
            cost_source=cost_source,
            generation_id=generation_id,
            upstream_provider=upstream_provider,
            model_resolved=model_resolved or self.model,
            duration_ms=duration_ms,
            finish_reason=finish_reason,
        )
