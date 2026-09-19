import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from worker.provider_config import ProviderConfigLoader
from worker.services import catalog_refresh
from worker.services.catalog_refresh import (
    DOCUMENT_KEY,
    MANAGED_BY,
    REPORT_KEY,
    CatalogRefresher,
    RefreshError,
    fetch,
    is_openrouter_free,
    refresh_document,
    value_candidates,
)


def openrouter_model(model_id, *, free=False, vision=False, score=10, price="0.000001"):
    price = "0" if free else price
    return {
        "id": model_id,
        "name": model_id.replace("/", ": "),
        "created": score,
        "architecture": {
            "input_modalities": ["text", "image"] if vision else ["text"],
            "output_modalities": ["text"],
            "tokenizer": "Other",
        },
        "pricing": {"prompt": price, "completion": price},
        "benchmarks": {"artificial_analysis": {"intelligence_index": score}},
    }


@pytest.fixture
def document():
    return {
        "version": 1,
        "providers": {
            "openrouter": {
                "models": {
                    "tl": [
                        {"id": "paid/model", "name": "Old name", "free": False},
                        {"id": "free/pinned:free", "name": "Pinned winner", "free": True, "pinned": True},
                        {"id": "gone/free:free", "name": "Gone", "free": True},
                    ],
                    "qaLLM": [],
                    "qaVLM": [],
                    "ocr": [],
                },
                "defaultTLModel": "paid/model",
            },
            "nvidia": {
                "models": {
                    "tl": [
                        {"id": "nvidia/live", "name": "Live", "free": True},
                        {"id": "nvidia/gone", "name": "Gone", "free": True},
                    ]
                },
                "defaultTLModel": "nvidia/live",
            },
        },
    }


@pytest.fixture
def openrouter():
    return [
        openrouter_model("paid/model"),
        openrouter_model("free/pinned:free", free=True, score=1),
        openrouter_model("free/text:free", free=True, score=50),
        openrouter_model("free/vision:free", free=True, vision=True, score=60),
        openrouter_model("free/code-model:free", free=True, score=100),
    ]


@pytest.fixture
def nvidia():
    return [{"id": "nvidia/live", "owned_by": "nvidia"}]


def test_fetch_sends_authorization_header():
    with patch.object(catalog_refresh.requests, "get") as get:
        get.return_value.json.return_value = {"data": []}
        fetch("https://example.test/models", authorization="Bearer nvidia-key")
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer nvidia-key"


def test_cache_charges_prevent_a_model_from_being_marked_free():
    model = openrouter_model("cached/model:free", free=True)
    model["pricing"]["input_cache_read"] = "0.0000001"
    assert not is_openrouter_free(model)


def test_refreshes_prices_and_replaces_free_entries(document, openrouter, nvidia):
    updated, changes = refresh_document(document, openrouter, nvidia)

    tl = updated["providers"]["openrouter"]["models"]["tl"]
    assert [entry["id"] for entry in tl] == ["paid/model", "free/pinned:free", "free/vision:free", "free/text:free"]
    assert tl[0]["pricing"]["promptPerMillion"] == 1.0
    assert tl[1]["pinned"]
    assert tl[2]["managedBy"] == MANAGED_BY
    assert "free/code-model:free" not in [entry["id"] for entry in tl]
    assert "openrouter/tl: removed free model gone/free:free" in changes

    vision = updated["providers"]["openrouter"]["models"]["ocr"]
    assert [entry["id"] for entry in vision] == ["free/vision:free"]


def test_refresh_is_idempotent(document, openrouter, nvidia):
    once, _ = refresh_document(document, openrouter, nvidia)
    twice, changes = refresh_document(once, openrouter, nvidia)
    assert twice == once
    assert changes == []


def test_nvidia_shortlist_is_verified_not_expanded(document, openrouter, nvidia):
    updated, changes = refresh_document(document, openrouter, nvidia)
    models = updated["providers"]["nvidia"]["models"]["tl"]
    assert [entry["id"] for entry in models] == ["nvidia/live"]
    assert models[0]["pricing"]["note"] == "Free credits"
    assert "nvidia/tl: removed unavailable model nvidia/gone" in changes


def test_without_an_nvidia_catalog_the_nvidia_block_is_left_alone(document, openrouter):
    updated, changes = refresh_document(document, openrouter, None)
    assert updated["providers"]["nvidia"] == document["providers"]["nvidia"]
    assert not any(line.startswith("nvidia/") for line in changes)


def test_missing_default_aborts_the_whole_refresh(document, openrouter, nvidia):
    document = copy.deepcopy(document)
    document["providers"]["openrouter"]["defaultTLModel"] = "missing/default"
    with pytest.raises(RefreshError, match="defaults missing"):
        refresh_document(document, openrouter, nvidia)


def test_value_candidates_rank_paid_models_by_index_per_dollar():
    catalog = [
        openrouter_model("cheap/strong", score=60, price="0.000001"),
        openrouter_model("dear/strong", score=80, price="0.00001"),
        openrouter_model("free/strong:free", free=True, score=90),
        openrouter_model("cheap/unscored", price="0.000001") | {"benchmarks": {}},
        openrouter_model("vision/only", vision=True, score=70, price="0.000002"),
    ]
    ranked = value_candidates(catalog, "tl", limit=10)
    assert [entry["id"] for entry in ranked] == ["cheap/strong", "vision/only", "dear/strong"]
    assert ranked[0]["blendedPerMillion"] == 1.0
    assert value_candidates(catalog, "ocr", limit=10)[0]["id"] == "vision/only"


# --- runner --------------------------------------------------------------------------------------


def make_loader(tmp_path, document):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(document))
    return ProviderConfigLoader(str(path))


def model_ids(loader: ProviderConfigLoader, provider: str, purpose: str) -> list[str]:
    models = loader.providers[provider].models[purpose]
    assert models is not None, f"{provider}/{purpose} has no model list"
    return [m.id for m in models]


def test_run_once_applies_publishes_and_stores(tmp_path, document, openrouter, nvidia, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    loader = make_loader(tmp_path, document)
    redis = MagicMock()
    with patch.object(catalog_refresh, "fetch_catalogs", return_value=(openrouter, nvidia)):
        changes = CatalogRefresher(loader, redis).run_once()

    assert "openrouter/tl: added free model free/text:free" in changes
    tl_ids = model_ids(loader, "openrouter", "tl")
    assert "free/text:free" in tl_ids and "gone/free:free" not in tl_ids

    stored = {call.args[0]: json.loads(call.args[1]) for call in redis.set.call_args_list}
    assert "system:providers:config" in stored  # the published catalog the UI reads
    assert stored[DOCUMENT_KEY]["sourceMtime"] == loader._loaded_mtime
    assert stored[DOCUMENT_KEY]["document"] == loader.raw_data
    report = stored[REPORT_KEY]
    assert report["ok"] and report["changed"]
    assert set(report["valueCandidates"]) == {"tl", "qaLLM", "qaVLM", "ocr"}


def test_run_once_keeps_the_catalog_when_a_fetch_fails(tmp_path, document):
    loader = make_loader(tmp_path, document)
    before = copy.deepcopy(loader.raw_data)
    redis = MagicMock()
    with patch.object(
        catalog_refresh, "fetch_catalogs", side_effect=RefreshError("OpenRouter returned an empty catalog")
    ):
        assert CatalogRefresher(loader, redis).run_once() == []

    assert loader.raw_data == before
    stored = {call.args[0]: json.loads(call.args[1]) for call in redis.set.call_args_list}
    assert list(stored) == [REPORT_KEY]
    assert not stored[REPORT_KEY]["ok"]
    assert "empty catalog" in stored[REPORT_KEY]["error"]


def test_restore_reapplies_the_last_refresh_only_for_the_same_file(tmp_path, document, openrouter, nvidia):
    loader = make_loader(tmp_path, document)
    refreshed, _ = refresh_document(document, openrouter, nvidia)
    redis = MagicMock()
    redis.get.return_value = json.dumps(
        {"sourceMtime": loader._loaded_mtime, "refreshedAt": "2026-09-18T00:00:00+00:00", "document": refreshed}
    )
    assert CatalogRefresher(loader, redis).restore()
    assert "free/text:free" in model_ids(loader, "openrouter", "tl")

    (tmp_path / "edited").mkdir()
    edited = make_loader(tmp_path / "edited", document)
    redis.get.return_value = json.dumps({"sourceMtime": -1.0, "document": refreshed})
    assert not CatalogRefresher(edited, redis).restore()
    assert "free/text:free" not in model_ids(edited, "openrouter", "tl")


def test_a_hand_edit_wins_over_the_restored_document(tmp_path, document):
    loader = make_loader(tmp_path, document)
    loader.apply_document(copy.deepcopy(document) | {"version": 99})
    assert loader.version == 99

    edited = copy.deepcopy(document) | {"version": 2}
    (tmp_path / "providers.json").write_text(json.dumps(edited))
    loader._loaded_mtime = None  # force the mtime check to see a change
    assert loader.reload_if_changed()
    assert loader.version == 2
