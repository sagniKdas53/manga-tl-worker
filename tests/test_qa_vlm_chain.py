"""Vision QA: number labels, key repair, the fallback model chain and uncertain-region checks."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tests.qa_binding import bound_qa_job
from worker.handlers.qa import _normalize_qa_items, _region_labels, process_qa


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _region(uuid, order, *, status=None, translated="Hello"):
    return {
        "id": uuid,
        "text": "テキスト",
        "bboxX": 10 * order,
        "bboxY": 20,
        "bboxW": 30,
        "bboxH": 40,
        "translatedText": translated,
        "bubbleReadingOrder": order,
        "qaStatus": status,
    }


def _verdict(label, status="passed", **extra):
    return {
        "regionId": label,
        "qaStatus": status,
        "qaScore": 0.9,
        "qaFeedback": "ok",
        "directFix": {"correctedText": "", "suggestedFontSize": 0},
        "escalation": {
            "ocrBad": False,
            "correctedSourceText": "",
            "needsReOcr": False,
            "needsManualIntervention": False,
            "orderBad": False,
            "suggestedReadingOrderIndex": 0,
        },
        **extra,
    }


@pytest.fixture
def qa_env():
    """Run process_qa in VLM mode with the backend, storage and models mocked."""
    with (
        patch("worker.handlers.qa.try_cloud_ai_vision") as vision,
        patch("worker.handlers.qa.download_image", return_value=_png()),
        patch("worker.handlers.qa.minio_client") as minio,
        patch("worker.handlers.qa.requests.get") as get,
        patch("worker.handlers.qa.requests.post") as post,
        patch("worker.handlers.qa.QA_MODE", "vlm"),
        patch("worker.handlers.qa.QA_CONFIG") as config,
        patch("worker.handlers.qa.QA_VLM_FALLBACK_MODELS", ["fallback/b", "fallback/c"]),
    ):
        config.provider = "openrouter"
        config.resolve_key.return_value = "key"
        config.vlm_model = "global/a"
        minio.get_object.return_value.read.return_value = _png()
        post.return_value = MagicMock(status_code=200)

        def run(regions, replies, job_extra=None):
            get.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"ocrRegions": regions}))
            vision.side_effect = [r if r is None else json.dumps(r) for r in replies]
            job = bound_qa_job({"imageId": "img", "qaVlmModel": "series/q", **(job_extra or {})}, _png())
            process_qa(job)
            payload = post.call_args.kwargs["json"]
            prompts = [call.args[3] for call in vision.call_args_list]
            models = [call.args[2] for call in vision.call_args_list]
            return payload, prompts, models

        yield run


def test_labels_follow_reader_numbers():
    regions = [_region("u-b", 2), _region("u-a", 1)]
    assert _region_labels(regions) == {"2": "u-b", "1": "u-a"}


def test_labels_fall_back_to_position_when_orders_collide():
    regions = [_region("u-b", 1), _region("u-a", 1)]
    regions[1]["bboxY"] = 5
    assert _region_labels(regions) == {"1": "u-a", "2": "u-b"}


def test_aliased_keys_and_numeric_ids_are_repaired():
    items = _normalize_qa_items([{"regionId": 3, "status": "Passed", "score": 1}], {"3": "u-3"})
    assert items == [{"regionId": "u-3", "qaStatus": "passed", "qaScore": 1}]


def test_status_key_drift_no_longer_discards_verdicts(qa_env):
    regions = [_region("u-1", 1), _region("u-2", 2)]
    drifted = [{**_verdict(label), "status": "passed"} for label in ("1", "2")]
    for item in drifted:
        del item["qaStatus"]
    payload, prompts, models = qa_env(regions, [{"results": drifted}])

    assert models == ["series/q"]
    assert {r["regionId"] for r in payload["qaResults"]} == {"u-1", "u-2"}
    assert payload["qaResponseIntegrity"]["complete"] is True
    assert '"regionId": "1"' in prompts[0] and "u-1" not in prompts[0]


def test_refusal_falls_through_the_chain(qa_env):
    regions = [_region("u-1", 1)]
    payload, _, models = qa_env(regions, [None, {"results": [_verdict("1")], "uncertainChecks": []}])

    assert models == ["series/q", "global/a"]
    assert [r["regionId"] for r in payload["qaResults"]] == ["u-1"]
    assert payload["qaResponseIntegrity"]["complete"] is True


def test_fallback_is_asked_only_about_missing_regions(qa_env):
    regions = [_region("u-1", 1), _region("u-2", 2)]
    payload, prompts, models = qa_env(
        regions,
        [{"results": [_verdict("1")]}, {"results": [_verdict("2", "direct_fix")]}],
    )

    assert models == ["series/q", "global/a"]
    assert '"regionId": "2"' in prompts[1] and '"regionId": "1"' not in prompts[1]
    assert {r["regionId"]: r["qaStatus"] for r in payload["qaResults"]} == {"u-1": "passed", "u-2": "direct_fix"}
    assert payload["qaResponseIntegrity"]["complete"] is True


def test_duplicate_verdict_is_re_asked_not_trusted(qa_env):
    regions = [_region("u-1", 1)]
    payload, _, models = qa_env(
        regions,
        [{"results": [_verdict("1"), _verdict("1", "failed")]}, {"results": [_verdict("1", "direct_fix")]}],
    )

    assert models == ["series/q", "global/a"]
    assert [r["qaStatus"] for r in payload["qaResults"]] == ["direct_fix"]


def test_chain_is_bounded_and_reports_incomplete(qa_env):
    regions = [_region("u-1", 1)]
    payload, _, models = qa_env(regions, [None, None, None, None])

    assert models == ["series/q", "global/a", "fallback/b", "fallback/c"]
    assert payload["qaResults"] == []
    assert payload["qaResponseIntegrity"]["complete"] is False


def test_no_fallback_when_disabled(qa_env):
    payload, _, models = qa_env([_region("u-1", 1)], [None], {"useFallbackModels": False})

    assert models == ["series/q"]
    assert payload["qaResponseIntegrity"]["complete"] is False


def test_uncertain_regions_are_checked_not_judged(qa_env):
    regions = [_region("u-1", 1, status="cleanup_review"), _region("u-2", 2)]
    payload, prompts, _ = qa_env(
        regions,
        [
            {
                "results": [_verdict("2")],
                "uncertainChecks": [{"regionId": "1", "kind": "background_text", "reason": "a shop sign"}],
            }
        ],
    )

    assert payload["qaTargetIds"] == ["u-2"]
    assert payload["uncertainChecks"] == [
        {"regionId": "u-1", "kind": "background_text", "reason": "a shop sign", "model": "series/q"}
    ]
    assert "UNCERTAIN REGIONS" in prompts[0]


def test_unanswered_uncertain_region_is_asked_again(qa_env):
    regions = [_region("u-1", 1, status="cleanup_review"), _region("u-2", 2)]
    payload, prompts, models = qa_env(
        regions,
        [
            {"results": [_verdict("2")], "uncertainChecks": [{"regionId": "1", "kind": "maybe", "reason": "?"}]},
            {"results": [], "uncertainChecks": [{"regionId": "1", "kind": "dialogue", "reason": "speech"}]},
        ],
    )

    assert models == ["series/q", "global/a"]
    assert [c["kind"] for c in payload["uncertainChecks"]] == ["dialogue"]
    assert "Region Metadata:\n[]" in prompts[1]


def test_qa_bypass_leaves_rejected_and_uncertain_regions_alone():
    regions = [_region("u-1", 1, status="rejected"), _region("u-2", 2, status="cleanup_review"), _region("u-3", 3)]
    with (
        patch("worker.handlers.qa.requests.get") as get,
        patch("worker.handlers.qa.requests.post") as post,
        patch("worker.handlers.qa.QA_MODE", "none"),
    ):
        get.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"ocrRegions": regions}))
        post.return_value = MagicMock(status_code=200)
        process_qa(bound_qa_job({"imageId": "img", "qaMode": "none"}, _png()))
    payload = post.call_args.kwargs["json"]
    assert [r["regionId"] for r in payload["qaResults"]] == ["u-3"]
    assert payload["qaTargetIds"] == ["u-3"]
