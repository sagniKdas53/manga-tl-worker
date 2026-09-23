"""
Guards for the silent-QA-pass failure mode found in run 20260803-084755.

A QA call that hit the output token limit came back truncated. OpenRouter's response-healing plugin
closed the JSON, so it parsed cleanly into a single `{"qaFeedback": "..."}` entry with no regionId.
The worker forwarded it, the backend could not apply it, and — because nothing was scored — recorded
the page as a clean QA pass and completed the pipeline.
"""

import json
from unittest.mock import MagicMock, patch

from tests.qa_binding import bound_qa_job
from worker.handlers.qa import QA_JSON_SCHEMA, _sanitize_qa_results
from worker.services.llm_client import LLMClient

REGIONS = [{"id": "r1", "text": "こんにちは"}, {"id": "r2", "text": "さようなら"}]


def _result(region_id, status="passed", **extra):
    base = {"regionId": region_id, "qaStatus": status, "qaScore": 1.0, "qaFeedback": "ok"}
    base.update(extra)
    return base


def test_truncated_result_without_region_id_is_dropped():
    healed = [{"qaFeedback": "The translation accurately conveys the meaning of the original"}]
    assert _sanitize_qa_results(healed, REGIONS) == []


def test_valid_results_survive():
    results = [_result("r1"), _result("r2", "failed")]
    assert _sanitize_qa_results(results, REGIONS) == results


def test_partial_truncation_keeps_the_regions_that_did_arrive():
    results = [_result("r1"), {"qaFeedback": "cut off here"}]
    kept = _sanitize_qa_results(results, REGIONS)
    assert [r["regionId"] for r in kept] == ["r1"]


def test_unknown_region_id_is_dropped():
    assert _sanitize_qa_results([_result("not-a-region")], REGIONS) == []


def test_invalid_status_is_dropped():
    assert _sanitize_qa_results([_result("r1", "looks-fine")], REGIONS) == []


def test_non_dict_entries_are_dropped():
    assert _sanitize_qa_results(["garbage", None, _result("r1")], REGIONS) == [_result("r1")]


def test_schema_requires_the_objects_the_backend_routes_on():
    """
    Both were optional, and the model emitted neither — 10 "direct_fix" verdicts with no directFix
    payload and 10 "failed" verdicts with no escalation block. The backend keys its direct-fix and
    re-OCR branches on the object being present, so both were dead paths.
    """
    item = QA_JSON_SCHEMA["properties"]["results"]["items"]
    assert "directFix" in item["required"]
    assert "escalation" in item["required"]
    assert "needsReOcr" in item["properties"]["escalation"]["required"]


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_unusable_qa_response_is_not_reported_as_a_pass(mock_redis, mock_requests):
    """The old behaviour fabricated a "passed" verdict for every region here."""
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"ocrRegions": REGIONS}
    mock_requests.get.return_value = mock_res

    healed = '{"results":[{"qaFeedback":"truncated mid-sentence and healed shut"}]}'

    with (
        patch("worker.handlers.qa.QA_MODE", "llm"),
        patch("worker.handlers.qa.try_cloud_ai", return_value=healed),
        patch("worker.handlers.qa.try_local_ai", return_value=healed),
    ):
        from worker.handlers.qa import process_qa

        process_qa(bound_qa_job({"imageId": "img1"}))

    posted = mock_requests.post.call_args[1]["json"]["qaResults"]
    assert posted == [], "an unusable QA response must report no verdict, not a pass"


@patch("worker.services.llm_client.requests.post")
def test_finish_reason_length_is_surfaced(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"results":[]}'}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2027, "completion_tokens": 3408, "total_tokens": 5435},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    result = client.complete([{"role": "user", "content": "hi"}])

    assert result is not None
    assert result.finish_reason == "length"
    assert result.truncated is True


@patch("worker.services.llm_client.requests.post")
def test_complete_response_is_not_flagged_truncated(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result is not None
    assert result.truncated is False


@patch("worker.services.llm_client.requests.post")
def test_every_provider_gets_an_explicit_output_budget(mock_post):
    """Only Anthropic used to send max_tokens; everyone else inherited the model's default."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {},
    }
    mock_post.return_value = mock_resp

    client = LLMClient(provider="openrouter", api_key="k", model="m")
    client.complete([{"role": "user", "content": "hi"}])

    assert mock_post.call_args[1]["json"]["max_tokens"] > 4096


@patch("worker.handlers.qa.requests")
@patch("worker.handlers.qa.redis_client")
def test_truncated_response_is_reported_incomplete_with_its_targets(mock_redis, mock_requests):
    """OQ-02: the 38-of-63 case. The backend, not the worker, decides the pass, so the callback
    states what was asked and what came back wrong, and names the scene it judged."""
    mock_redis.llen.return_value = 0
    mock_res = MagicMock()
    mock_res.status_code = 200
    translated = [{**r, "translatedText": f"english {r['id']}"} for r in REGIONS]
    mock_res.json.return_value = {"ocrRegions": translated}
    mock_requests.get.return_value = mock_res
    judged = [r["id"] for r in translated]
    partial = json.dumps({"results": [{"regionId": judged[0], "qaStatus": "passed", "qaScore": 1}]})

    with (
        patch("worker.handlers.qa.QA_MODE", "llm"),
        patch("worker.handlers.qa.try_cloud_ai", return_value=partial),
        patch("worker.handlers.qa.try_local_ai", return_value=partial),
    ):
        from worker.handlers.qa import process_qa

        job = bound_qa_job({"imageId": "img1"})
        process_qa(job)

    posted = mock_requests.post.call_args[1]["json"]
    assert sorted(posted["qaTargetIds"]) == sorted(judged)
    assert posted["qaResponseIntegrity"]["complete"] is False
    assert any(judged[1] in e for e in posted["qaResponseIntegrity"]["errors"])
    assert posted["judgedArtifact"] == {
        "artifact": job["renderArtifact"],
        "pageRevision": job["pageRevision"],
        "logicalSceneSha256": job["logicalSceneSha256"],
    }


def test_duplicate_and_foreign_verdicts_are_integrity_errors():
    from worker.handlers.qa import _qa_response_integrity

    regions = [{"id": "a"}, {"id": "b"}]
    report = _qa_response_integrity(
        [
            {"regionId": "a", "qaStatus": "passed"},
            {"regionId": "a", "qaStatus": "failed"},
            {"regionId": "b", "qaStatus": "passed"},
            {"regionId": "zzz", "qaStatus": "passed"},
        ],
        regions,
    )
    assert report["complete"] is False
    assert "duplicate verdict for a" in report["errors"]
    assert any("foreign regionId zzz" in e for e in report["errors"])
    complete = _qa_response_integrity(
        [{"regionId": "a", "qaStatus": "passed"}, {"regionId": "b", "qaStatus": "reject_sfx"}], regions
    )
    assert complete == {"complete": True, "errors": []}


@patch("worker.handlers.qa.minio_client")
def test_vlm_refuses_to_judge_bytes_that_are_not_the_bound_artifact(mock_minio):
    """OQ-01: the VLM judges exactly the artifact the render bound to the job, or nothing."""
    import pytest

    from worker.handlers.qa import _read_render_artifact

    job = bound_qa_job({"imageId": "img1"}, rendered=b"final")
    mock_minio.get_object.return_value.read.return_value = b"initial"
    with pytest.raises(ValueError):
        _read_render_artifact(job["renderArtifact"])
    mock_minio.get_object.return_value.read.return_value = b"final"
    assert _read_render_artifact(job["renderArtifact"]) == b"final"
    with pytest.raises(ValueError):
        _read_render_artifact(None)
