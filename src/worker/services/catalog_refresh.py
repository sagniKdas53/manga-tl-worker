"""Keep the provider catalog current from inside the worker.

This replaces the ``refresh-provider-models`` GitHub workflow, which fetched the OpenRouter and
NVIDIA catalogs twice a week and committed ``config/providers.json`` to ``main``. The same merge
runs here on a timer instead, and the result goes to Redis — the place the backend and the
settings UI already read the catalog from — rather than into git. A catalog commit, when one is
wanted, is a deliberate act: ``scripts/dump_provider_catalog.py`` in the parent repository writes
the refreshed document back to ``config/providers.json``.

What a refresh does (unchanged from the workflow):

* paid OpenRouter entries are curated production choices; they keep their place, get current
  prices, and are dropped only when OpenRouter no longer lists them;
* free OpenRouter entries are discovery, replaced on every run with the strongest current free
  models for each task (``FREE_LIMIT`` per task);
* NVIDIA publishes no price or capability data, so its shortlist is only verified against the
  live catalog and labelled as free credits;
* a default model missing from a live catalog aborts the whole refresh — a partial catalog must
  never reach the UI.

What is new here: a best-value report per task (paid models ranked by intelligence per dollar),
stored alongside the refresh report so cheap, strong models can be found without reading the
OpenRouter site. It is a report, not a change to the model lists.
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from worker.config import logger

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
NVIDIA_MODELS = "https://integrate.api.nvidia.com/v1/models"
MANAGED_BY = "provider-catalog-refresh"

TEXT_TASKS = ("tl", "qaLLM")
VISION_TASKS = ("qaVLM", "ocr")
FREE_LIMIT = {"tl": 8, "qaLLM": 8, "qaVLM": 6, "ocr": 6}
VALUE_LIMIT = 5

#: Redis keys. ``DOCUMENT_KEY`` holds the last refreshed providers.json object (with the mtime of
#: the file it was derived from); ``REPORT_KEY`` holds what the last run did or why it failed.
DOCUMENT_KEY = "system:providers:document"
REPORT_KEY = "system:providers:catalog-refresh"

# Free catalogs contain classifiers, role-play models, and narrow domain models. They may be
# useful elsewhere, but putting them in a production translation fallback list is misleading.
FREE_EXCLUSIONS = (
    "content-safety",
    "guard",
    "moderation",
    "roleplay",
    "role-play",
    "finance",
    "-fin:",
    "-fin-",
    "medical",
    "-med-",
    "music",
    "audio",
    "code",
)


class RefreshError(RuntimeError):
    """Raised when a refresh would leave the production config invalid."""


# --- catalog fetch -----------------------------------------------------------------------------


def fetch(url: str, *, authorization: str | None = None, timeout: int = 45) -> dict[str, Any]:
    headers = {"User-Agent": "manga-library/provider-catalog-refresh"}
    if authorization:
        headers["Authorization"] = authorization
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_catalogs(nvidia_api_key: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return ``(openrouter_models, nvidia_models)``; NVIDIA is ``None`` when there is no key.

    An empty OpenRouter catalog is an error rather than "nothing available": refreshing against
    it would drop every model.
    """
    openrouter = fetch(OPENROUTER_MODELS).get("data") or []
    if not openrouter:
        raise RefreshError("OpenRouter returned an empty catalog; refusing a partial refresh")
    if not nvidia_api_key:
        return openrouter, None
    nvidia = fetch(NVIDIA_MODELS, authorization=f"Bearer {nvidia_api_key}").get("data") or []
    if not nvidia:
        raise RefreshError("NVIDIA returned an empty catalog; refusing a partial refresh")
    return openrouter, nvidia


# --- pure merge --------------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _per_million(value: Any) -> float | None:
    number = _number(value)
    return round(number * 1_000_000, 9) if number is not None else None


def openrouter_pricing(model: dict[str, Any]) -> dict[str, Any]:
    raw = model.get("pricing") or {}
    pricing: dict[str, Any] = {"currency": "USD", "source": "openrouter"}
    for source_key, output_key in (
        ("prompt", "promptPerMillion"),
        ("completion", "completionPerMillion"),
        ("input_cache_read", "cacheReadPerMillion"),
    ):
        value = _per_million(raw.get(source_key))
        if value is not None:
            pricing[output_key] = value
    for source_key, output_key in (("request", "request"), ("image", "image")):
        value = _number(raw.get(source_key))
        if value is not None:
            pricing[output_key] = value
    return pricing


def is_openrouter_free(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    prompt = _number(pricing.get("prompt"))
    completion = _number(pricing.get("completion"))
    if prompt != 0 or completion != 0:
        return False
    optional_charges = (
        "request",
        "image",
        "web_search",
        "internal_reasoning",
        "input_cache_read",
        "input_cache_write",
    )
    return all((_number(pricing.get(key)) or 0) == 0 for key in optional_charges)


def supports_task(model: dict[str, Any], task: str) -> bool:
    architecture = model.get("architecture") or {}
    inputs = architecture.get("input_modalities") or []
    outputs = architecture.get("output_modalities") or []
    if "text" not in inputs or sorted(outputs) != ["text"]:
        return False
    if architecture.get("tokenizer") == "Router":
        return False
    return task in TEXT_TASKS or "image" in inputs


def free_candidate(model: dict[str, Any], task: str) -> bool:
    model_id = str(model.get("id") or "").lower()
    return (
        bool(model_id)
        and is_openrouter_free(model)
        and supports_task(model, task)
        and not any(fragment in model_id for fragment in FREE_EXCLUSIONS)
    )


def intelligence_index(model: dict[str, Any]) -> float | None:
    benchmarks = model.get("benchmarks") or {}
    return _number((benchmarks.get("artificial_analysis") or {}).get("intelligence_index"))


def candidate_rank(model: dict[str, Any]) -> tuple[float, int, str]:
    score = intelligence_index(model)
    return (score if score is not None else -1.0, int(model.get("created") or 0), str(model.get("id") or ""))


def openrouter_entry(model: dict[str, Any], *, managed: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model["id"],
        "name": model.get("name") or model["id"],
        "free": is_openrouter_free(model),
        "pricing": openrouter_pricing(model),
    }
    if managed:
        entry["managedBy"] = MANAGED_BY
    return entry


def _defaults(provider: dict[str, Any]) -> set[str]:
    return {
        value
        for key, value in provider.items()
        if key.startswith("default") and key.endswith("Model") and isinstance(value, str) and value
    }


def refresh_openrouter(provider: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    live = {model.get("id"): model for model in catalog if model.get("id")}
    defaults = _defaults(provider)
    changes: list[str] = []

    missing_defaults = sorted(defaults - live.keys())
    if missing_defaults:
        raise RefreshError(f"OpenRouter defaults missing from the live catalog: {', '.join(missing_defaults)}")

    for task, entries in (provider.get("models") or {}).items():
        if task not in FREE_LIMIT or not isinstance(entries, list):
            continue

        # Paid entries are deliberate production choices. Free entries belong to discovery and
        # are replaced on every run, including legacy entries from before managedBy was added.
        curated: list[dict[str, Any]] = []
        previous_free = {
            entry.get("id")
            for entry in entries
            if entry.get("free") and not entry.get("pinned") and entry.get("id") not in defaults
        }
        for entry in entries:
            model_id = entry.get("id")
            model = live.get(model_id)
            if entry.get("pinned") or model_id in defaults:
                if model is None:
                    changes.append(f"openrouter/{task}: removed unavailable pinned model {model_id}")
                    continue
                refreshed = openrouter_entry(model, managed=False)
                refreshed["pinned"] = True
                if "tier" in entry:
                    refreshed["tier"] = entry["tier"]
                curated.append(refreshed)
                continue
            if entry.get("free") or entry.get("managedBy") == MANAGED_BY:
                continue
            if model is None:
                changes.append(f"openrouter/{task}: removed unavailable curated model {model_id}")
                continue
            refreshed = openrouter_entry(model, managed=False)
            for key in ("tier", "pinned"):
                if key in entry:
                    refreshed[key] = entry[key]
            curated.append(refreshed)

        candidates = [model for model in catalog if free_candidate(model, task)]
        candidates.sort(key=candidate_rank, reverse=True)
        selected = candidates[: FREE_LIMIT[task]]
        managed = [openrouter_entry(model, managed=True) for model in selected]

        # A manually curated entry wins if the same id appears in the free selection.
        curated_ids = {entry["id"] for entry in curated}
        managed = [entry for entry in managed if entry["id"] not in curated_ids]
        provider["models"][task] = curated + managed

        current_free = {entry["id"] for entry in managed}
        for model_id in sorted(previous_free - current_free):
            changes.append(f"openrouter/{task}: removed free model {model_id}")
        for model_id in sorted(current_free - previous_free):
            changes.append(f"openrouter/{task}: added free model {model_id}")

    return changes


def refresh_nvidia(provider: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    live = {model.get("id"): model for model in catalog if model.get("id")}
    defaults = _defaults(provider)
    changes: list[str] = []

    missing_defaults = sorted(defaults - live.keys())
    if missing_defaults:
        raise RefreshError(f"NVIDIA defaults missing from the live catalog: {', '.join(missing_defaults)}")

    for task, entries in (provider.get("models") or {}).items():
        if not isinstance(entries, list):
            continue
        refreshed: list[dict[str, Any]] = []
        for entry in entries:
            model_id = entry.get("id")
            if model_id not in live:
                changes.append(f"nvidia/{task}: removed unavailable model {model_id}")
                continue
            updated = copy.deepcopy(entry)
            updated["free"] = True
            updated["pricing"] = {"currency": "USD", "source": "nvidia-catalog", "note": "Free credits"}
            refreshed.append(updated)
        provider["models"][task] = refreshed
    return changes


def refresh_document(
    document: dict[str, Any],
    openrouter_catalog: list[dict[str, Any]],
    nvidia_catalog: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(refreshed copy, change lines)``.

    ``nvidia_catalog=None`` leaves the NVIDIA block untouched — the worker has no NVIDIA key, so
    it could not verify the shortlist and must not guess.
    """
    updated = copy.deepcopy(document)
    providers = updated.get("providers") or {}
    if "openrouter" not in providers:
        raise RefreshError("providers.json is missing provider openrouter")

    changes = refresh_openrouter(providers["openrouter"], openrouter_catalog)
    if nvidia_catalog is not None:
        if "nvidia" not in providers:
            raise RefreshError("providers.json is missing provider nvidia")
        changes.extend(refresh_nvidia(providers["nvidia"], nvidia_catalog))
    return updated, changes


def blended_price_per_million(model: dict[str, Any]) -> float | None:
    """Mean of prompt and completion price per million tokens, or None when either is unknown."""
    pricing = model.get("pricing") or {}
    prompt = _per_million(pricing.get("prompt"))
    completion = _per_million(pricing.get("completion"))
    if prompt is None or completion is None:
        return None
    return round((prompt + completion) / 2, 6)


def value_candidates(catalog: list[dict[str, Any]], task: str, limit: int = VALUE_LIMIT) -> list[dict[str, Any]]:
    """Paid models for ``task`` ranked by intelligence index per dollar of blended price.

    Free models are excluded (they are already selected by :func:`refresh_openrouter`) and so are
    models with no published intelligence index, since a price alone says nothing about worth.
    """
    ranked: list[tuple[float, dict[str, Any]]] = []
    for model in catalog:
        if not model.get("id") or is_openrouter_free(model) or not supports_task(model, task):
            continue
        score = intelligence_index(model)
        price = blended_price_per_million(model)
        if score is None or price is None or price <= 0:
            continue
        ranked.append(
            (
                score / price,
                {
                    "id": model["id"],
                    "name": model.get("name") or model["id"],
                    "intelligenceIndex": score,
                    "blendedPerMillion": price,
                },
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]["intelligenceIndex"], item[1]["id"]), reverse=True)
    return [entry for _, entry in ranked[:limit]]


# --- runner ------------------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


class CatalogRefresher:
    """Runs the refresh against a live ``ProviderConfigLoader`` and records the result in Redis."""

    def __init__(self, loader, redis_client):
        self.loader = loader
        self.redis = redis_client

    def _nvidia_key(self) -> str:
        provider = self.loader.providers.get("nvidia")
        if provider is not None and provider.api_key:
            return provider.api_key
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    def restore(self) -> bool:
        """Re-apply the last refreshed document after a restart.

        Only when it was derived from the providers.json now on disk: an edited file is newer
        information than any refresh, so the file wins and the next run rebases onto it.
        """
        if not self.redis:
            return False
        try:
            raw = self.redis.get(DOCUMENT_KEY)
        except Exception as e:
            logger.warning(f"Catalog refresh: could not read {DOCUMENT_KEY} from Redis: {e}")
            return False
        if not raw:
            return False
        try:
            stored = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Catalog refresh: {DOCUMENT_KEY} in Redis is not JSON; ignoring it")
            return False
        if stored.get("sourceMtime") != self.loader._loaded_mtime:
            logger.info(
                "Catalog refresh: providers.json changed since the last refresh; the file wins until the next run"
            )
            return False
        document = stored.get("document")
        if not isinstance(document, dict):
            return False
        self.loader.apply_document(document)
        logger.info(f"Catalog refresh: restored the document refreshed at {stored.get('refreshedAt')}")
        return True

    def run_once(self) -> list[str]:
        """Fetch, merge, apply, publish. Never raises; the outcome is logged and stored under REPORT_KEY."""
        started = time.monotonic()
        report: dict[str, Any] = {"refreshedAt": _utcnow(), "ok": False, "changes": [], "valueCandidates": {}}
        try:
            # A hand edit to providers.json since the last load is picked up first, so the refresh
            # always rebases onto what the operator last wrote.
            self.loader.reload_if_changed()
            base = self.loader.raw_data
            if not base:
                raise RefreshError("no providers.json is loaded")

            openrouter, nvidia = fetch_catalogs(self._nvidia_key())
            updated, changes = refresh_document(base, openrouter, nvidia)
            report["valueCandidates"] = {task: value_candidates(openrouter, task) for task in FREE_LIMIT}
            if nvidia is None:
                report["nvidia"] = "skipped: no NVIDIA_API_KEY, shortlist left as configured"

            changed = json.dumps(updated, sort_keys=True) != json.dumps(base, sort_keys=True)
            if changed:
                self.loader.apply_document(updated)
                self.loader.publish_config_to_redis(self.redis)
            self._store(
                DOCUMENT_KEY,
                {"sourceMtime": self.loader._loaded_mtime, "refreshedAt": report["refreshedAt"], "document": updated},
            )

            report["ok"] = True
            report["changes"] = changes
            report["changed"] = changed
            for line in changes:
                logger.info(f"Catalog refresh: {line}")
            logger.info(
                f"Catalog refresh: {'updated' if changed else 'already current'} "
                f"({len(changes)} change(s), {time.monotonic() - started:.1f}s)"
            )
            return changes
        except (RefreshError, requests.RequestException, ValueError) as e:
            report["error"] = str(e)
            logger.warning(f"Catalog refresh failed; keeping the current catalog: {e}")
            return []
        except Exception as e:  # never take the worker down over a catalog
            report["error"] = f"{type(e).__name__}: {e}"
            logger.error(f"Catalog refresh failed unexpectedly; keeping the current catalog: {e}")
            return []
        finally:
            self._store(REPORT_KEY, report)

    def _store(self, key: str, value: dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            self.redis.set(key, json.dumps(value))
        except Exception as e:
            logger.warning(f"Catalog refresh: could not write {key} to Redis: {e}")
