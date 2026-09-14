"""Standalone `page-scene/v1` artifact validation; no parent-checkout imports."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

CONTRACT_VERSION = "page-scene/v1"


class PageSceneValidationError(ValueError):
    """A contract violation that must fail the worker job before rendering."""


@dataclass(frozen=True)
class PageSceneArtifact:
    document: dict[str, Any]
    logical_scene_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def logical_scene_digest(document: dict[str, Any]) -> str:
    logical = {key: value for key, value in document.items() if key != "resolved_layout"}
    logical["scene_kind"] = "logical"
    return hashlib.sha256(_canonical_json(logical).encode()).hexdigest()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PageSceneValidationError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PageSceneValidationError(f"{label} must be lowercase SHA-256")
    return value


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PageSceneValidationError("non-finite geometry")
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)


def validate_page_scene(document: Any) -> PageSceneArtifact:
    if not isinstance(document, dict):
        raise PageSceneValidationError("scene must be an object")
    _reject_nonfinite(document)
    required = {"contract_version", "scene_kind", "page", "provenance", "fragments", "owners", "policies", "assets", "cleanup_artifacts", "objects"}
    if required - document.keys() or set(document) - (required | {"resolved_layout"}):
        raise PageSceneValidationError("scene has missing or unknown fields")
    if document["contract_version"] != CONTRACT_VERSION:
        raise PageSceneValidationError("unsupported contract_version")
    kind = document["scene_kind"]
    if kind not in {"logical", "resolved"} or (kind == "logical" and "resolved_layout" in document) or (kind == "resolved" and "resolved_layout" not in document):
        raise PageSceneValidationError("invalid logical/resolved scene boundary")
    page = document["page"]
    if not isinstance(page, dict) or not isinstance(page.get("revision"), int) or page["revision"] < 0:
        raise PageSceneValidationError("invalid page revision")
    source = page.get("source")
    if not isinstance(source, dict):
        raise PageSceneValidationError("missing page source")
    source_sha256 = _require_sha256(source.get("sha256"), "page.source.sha256")
    fragments = document["fragments"]
    owners = document["owners"]
    policies = document["policies"]
    assets = document["assets"]
    cleanups = document["cleanup_artifacts"]
    objects = document["objects"]
    if not all(isinstance(value, list) for value in (fragments, owners, policies, assets, cleanups, objects)):
        raise PageSceneValidationError("scene collections must be arrays")
    fragment_ids = {_require_string(item.get("fragment_id"), "fragment_id") for item in fragments if isinstance(item, dict)}
    if len(fragment_ids) != len(fragments):
        raise PageSceneValidationError("duplicate or invalid fragment")
    owner_ids: set[str] = set()
    owned_fragments: set[str] = set()
    for owner in owners:
        if not isinstance(owner, dict):
            raise PageSceneValidationError("invalid owner")
        owner_id = _require_string(owner.get("owner_id"), "owner_id")
        if owner_id in owner_ids:
            raise PageSceneValidationError("duplicate owner")
        owner_ids.add(owner_id)
        for fragment_id in owner.get("fragment_ids", []):
            if fragment_id not in fragment_ids or fragment_id in owned_fragments:
                raise PageSceneValidationError("fragments need one owner")
            owned_fragments.add(fragment_id)
    if owned_fragments != fragment_ids:
        raise PageSceneValidationError("fragments need one owner")
    actions: dict[str, str] = {}
    for policy in policies:
        if not isinstance(policy, dict):
            raise PageSceneValidationError("invalid policy")
        owner_id = _require_string(policy.get("owner_id"), "policy.owner_id")
        action = policy.get("user_override") or policy.get("action")
        if owner_id not in owner_ids or owner_id in actions or action not in {"preserve", "explain", "replace", "review"}:
            raise PageSceneValidationError("owners need one valid policy")
        actions[owner_id] = action
    if set(actions) != owner_ids:
        raise PageSceneValidationError("owners need one valid policy")
    asset_kinds = {_require_string(asset.get("asset_id"), "asset_id"): asset.get("kind") for asset in assets if isinstance(asset, dict)}
    if len(asset_kinds) != len(assets):
        raise PageSceneValidationError("duplicate or invalid asset")
    cleanup_owners: dict[str, set[str]] = {}
    for cleanup in cleanups:
        if not isinstance(cleanup, dict):
            raise PageSceneValidationError("invalid cleanup")
        cleanup_id = _require_string(cleanup.get("cleanup_id"), "cleanup_id")
        cleanup_owner_ids = set(cleanup.get("owner_ids", []))
        if (cleanup_id in cleanup_owners or cleanup.get("source_sha256") != source_sha256 or asset_kinds.get(cleanup.get("mask_asset_id")) != "glyph_mask" or asset_kinds.get(cleanup.get("patch_asset_id")) != "cleanup_patch" or not cleanup_owner_ids or any(actions.get(owner_id) != "replace" for owner_id in cleanup_owner_ids)):
            raise PageSceneValidationError("cleanup is not source/asset/policy authorized")
        cleanup_owners[cleanup_id] = cleanup_owner_ids
    for item in objects:
        if not isinstance(item, dict):
            raise PageSceneValidationError("invalid object")
        if item.get("kind") != "manual_cleanup":
            transform = item.get("transform")
            if not isinstance(transform, dict) or any(
                not isinstance(transform.get(field), (int, float))
                or isinstance(transform.get(field), bool)
                or not math.isfinite(transform[field])
                for field in ("x", "y", "width", "height", "rotation_degrees")
            ) or transform["width"] <= 0 or transform["height"] <= 0:
                raise PageSceneValidationError("invalid object transform")
        if item.get("kind") != "automatic_text":
            continue
        owner_id = item.get("owner_id")
        cleanup_ids = item.get("cleanup_ids")
        if actions.get(owner_id) != "replace" or not isinstance(item.get("text"), str) or not item["text"].strip() or not isinstance(cleanup_ids, list) or not cleanup_ids or any(owner_id not in cleanup_owners.get(cleanup_id, set()) for cleanup_id in cleanup_ids):
            raise PageSceneValidationError("automatic text is not policy/cleanup authorized")
    digest = logical_scene_digest(document)
    if kind == "resolved" and document["resolved_layout"].get("logical_scene_sha256") != digest:
        raise PageSceneValidationError("resolved layout digest mismatch")
    return PageSceneArtifact(document=document, logical_scene_sha256=digest)
