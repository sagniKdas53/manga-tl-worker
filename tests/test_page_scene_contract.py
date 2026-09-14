import hashlib
import json
import os
from pathlib import Path

import pytest

from worker.page_scene import CONTRACT_SCHEMA_SHA256, PageSceneValidationError, validate_page_scene
from worker.schemas import PageSceneRenderRequest

FIXTURES = Path(os.environ.get("PAGE_SCENE_FIXTURES", Path(__file__).parents[2] / "contracts" / "fixtures" / "page-scene-v1"))


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_worker_contract_model_is_pinned_to_authoritative_schema():
    schema = Path(__file__).parents[2] / "contracts" / "page-scene-v1.schema.json"
    assert hashlib.sha256(schema.read_bytes()).hexdigest() == CONTRACT_SCHEMA_SHA256


def patch(document, operation):
    parts = operation["path"].lstrip("/").split("/")
    parent = document
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = parts[-1]
    if operation["op"] in {"replace", "add"}:
        parent[int(key) if isinstance(parent, list) else key] = operation["value"]
    elif isinstance(parent, list):
        parent.pop(int(key))
    else:
        parent.pop(key)


def test_shared_valid_scenes_validate():
    for name in ("logical-valid.json", "resolved-valid.json", "overlap-preserve-valid.json"):
        assert PageSceneRenderRequest(contract_version="page-scene/v1", page_scene=load(name)).page_scene["contract_version"] == "page-scene/v1"
        assert validate_page_scene(load(name)).document["contract_version"] == "page-scene/v1"


def test_shared_invalid_scenes_reject():
    with pytest.raises(PageSceneValidationError):
        validate_page_scene(load("non-finite-geometry.json"))
    for case_file in ("invalid-cases.json", "resolved-invalid-cases.json"):
        cases = load(case_file)
        for case in cases["cases"]:
            scene = load(cases["base_fixture"])
            for operation in case["operations"]:
                patch(scene, operation)
            with pytest.raises(PageSceneValidationError):
                validate_page_scene(scene)
