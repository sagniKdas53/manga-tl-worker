# Worker Quality Gate & Test Suite Alignment Plan

> **Status: executed — kept for the root-cause write-up below, not as a worklist.** The gate is
> green as of 2026-08-17 (`cd worker && ../.venv/bin/python -m pytest -q` -> **415 passed** in
> ~7s), and the quality-gate doc this plan asks to fix already says `cd worker`. The remaining
> `unified-workers` mentions below are quotations of the old state, not outstanding work.

This plan addresses all quality gate check failures for the Python worker (`worker/`) according to [quality_gate.md](../docs/guides/quality_gate.md).

---

## Detailed Explanation: Why Provider Registry Broke Recently

### How it worked previously:
In `src/worker/provider_config.py`, `get_provider_registry()` calls `loader.get_provider_registry()`. `loader.get_provider_registry()` iterates through loaded providers and skips any where `pconfig.active` is `False`.
Previously, when unit tests or local dev environments ran without cloud API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.), `loader.get_provider_registry()` returned an **empty dictionary `{}`**.
`get_provider_registry()` had a fallback check:
```python
if not registry:
    # Fallback to hardcoded registry
    return {"openrouter": {...}, "gemini": {...}, "openai": {...}, "anthropic": {...}}
```
Because `{}` is falsy, `if not registry:` was `True`, and the hardcoded fallback provided the endpoint URLs and headers to `LLMClient`.

### Why it broke recently:
`ollama` and `lmstudio` were added to `config/providers.json` with `"keyEnvVar": null` (keyless local providers).
Because they require no key, `ProviderConfigLoader` marks them as `active = True` by default.
Consequently:
1. `loader.get_provider_registry()` returns `{'ollama': {...}, 'lmstudio': {...}}` even when no API keys exist in the environment.
2. `if not registry:` evaluates to `False` (since `{'ollama': ...}` is non-empty).
3. The hardcoded fallback for cloud providers (`openrouter`, `gemini`, `nvidia`, `openai`, `anthropic`) is **bypassed**.
4. `PROVIDER_REGISTRY` lacks entries for `openai`, `anthropic`, `openrouter`, `gemini`, `nvidia`, `neurometric`.
5. When `LLMClient(provider="openai", api_key="...")` initializes, `self.url` is `""`, leading to `LLMClient.complete()` returning `None` and cascading failures across 9 unit tests.

### Fix Strategy:
`loader.get_provider_registry()` will provide endpoint definitions for **all** configured providers in `config/providers.json` (not just currently active ones), so `LLMClient` always knows the target base URL and header requirements when an API key is supplied at runtime or during testing.

---

## Comprehensive List of Quality Gate Failures

Below is the complete inventory of all failures observed during the quality gate check, categorized by tool:

### 1. Ruff (Linting & Formatting — 6 errors)
- `src/worker/provider_config.py:6:36` — `F401`: `dataclasses.field` imported but unused
- `src/worker/provider_config.py:103:41` — `UP015`: Unnecessary mode argument `"r"` in `open()`
- `src/worker/provider_config.py:213:43` — `UP017`: Use `datetime.UTC` alias instead of `timezone.utc`
- `src/worker/services/llm_client.py:62:1` — `E402`: Module level import not at top of file
- `tests/test_provider_config.py:1:1` — `I001`: Import block is un-sorted / un-formatted
- `tests/test_provider_config.py:3:40` — `F401`: `worker.services.llm_client.LLMClient` imported but unused

### 2. Pyright (Static Type Checker — 1 error + 1 config warning)
- `src/worker/provider_config.py:153:48` — `reportOptionalIterable`: `models_dict[task]` can be `None` (for providers like `neurometric` where `qaVLM: null`). Pyright flags indexing `None` as non-iterable.
- `pyrightconfig.json` — Warning: `venv .venv subdirectory not found in venv path /home/sagnik/Projects/docker-composes/manga-library/worker`. (`venvPath` should point to `".."`).

### 3. Pytest (Unit Test Failures — 9 failed out of 156 executed)
- `tests/test_llm_client.py::test_llm_client_openai_success` — `AssertionError`: `LLMClient.complete()` returned `None` due to missing `openai` URL in `PROVIDER_REGISTRY`.
- `tests/test_llm_client.py::test_llm_client_anthropic_prompt_caching` — `AssertionError`: `LLMClient.complete()` returned `None` due to missing `anthropic` URL.
- `tests/test_llm_client.py::test_llm_client_openrouter_caching_and_session` — `AssertionError`: `LLMClient.complete()` returned `None` due to missing `openrouter` URL.
- `tests/test_provider_config.py::test_provider_config_loader` — `AssertionError: assert 'openrouter' in registry`.
- `tests/test_redo_pipeline.py::test_process_region_redo_translation` — `AssertionError`: Redo translation step returned `None` due to missing provider registry URL.
- `tests/test_translation_extra.py::test_get_api_url_and_headers` — `AssertionError`: API URL resolution returned empty string.
- `tests/test_translation_extra.py::test_try_cloud_ai` — `AssertionError`: `try_cloud_ai` returned `None`.
- `tests/test_translation_extra.py::test_try_cloud_ai_vision` — `AssertionError`: `try_cloud_ai_vision` returned `None`.
- `tests/test_translation_fallback.py::test_try_cloud_ai_vision_batch_degrade_json_schema` — `AssertionError: assert None == 'result'`.

### 4. Code Coverage (72% vs 80% Target)
- Overall coverage is at 72%. Areas with low coverage:
  - `src/worker/rq_tasks.py` (46%)
  - `src/worker/services/llm_client.py` (43%)
  - `src/worker/handlers/stub.py` (31%)
  - `src/worker/utils/image.py` (27%)
  - `src/worker/handlers/qa.py` (58%)

### 5. Documentation Parity
- `docs/quality_gate.md:51` — References `cd unified-workers`, needs update to `cd worker`.

---

## Proposed Changes

### Worker Configuration & Quality Gate Fixes

#### [MODIFY] [provider_config.py](src/worker/provider_config.py)
- Update `get_provider_registry()` to include all defined providers from `providers.json` so `LLMClient` can route requests and format payloads correctly for all providers regardless of startup environment variables.
- Fix pyright `reportOptionalIterable` error at line 153 by safely checking `task_models` before iterating.
- Fix ruff lint issues (remove unused `field` import, remove redundant `"r"` mode in `open()`, use `datetime.UTC`).

#### [MODIFY] [pyrightconfig.json](pyrightconfig.json)
- Update `venvPath` to `".."` so Pyright accurately recognizes `.venv` located at the root directory.

#### [MODIFY] [llm_client.py](src/worker/services/llm_client.py)
- Move import to top of file to resolve ruff `E402` lint warning.

---

### Worker Tests & Coverage

#### [MODIFY] [test_provider_config.py](tests/test_provider_config.py)
- Clean up unused import `LLMClient` and sort import block according to ruff formatting standards.
- Add test assertions verifying that all configured providers are present in the provider registry.

#### [NEW] Additional Unit Tests for Coverage
- Add unit tests for low-coverage modules (`services/llm_client.py`, `utils/image.py`, `rq_tasks.py`) to raise total test coverage to 80%+.

---

### Documentation

#### [MODIFY] [quality_gate.md](../docs/guides/quality_gate.md)
- Update directory path in worker quality gate section from `cd unified-workers` to `cd worker`.

---

## Verification Plan

### Automated Verification Commands
- `cd worker && ../.venv/bin/ruff check .` (Must output 0 errors)
- `cd worker && ../.venv/bin/ruff check . --fix && ../.venv/bin/ruff format .`
- `cd worker && ../.venv/bin/pyright .` (Must output 0 errors)
- `cd worker && PYTHONPATH=src ../.venv/bin/python -m pytest tests/ --cov=src/worker --cov-report=term-missing` (100% tests passing, coverage ≥80%)
