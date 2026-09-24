from unittest.mock import MagicMock, patch

import pytest

from worker.handlers.translation import process_translation


def _run(regions):
    response = MagicMock(status_code=200)
    response.json.return_value = {"ocrRegions": regions, "conversations": []}
    with (
        patch("worker.handlers.translation.requests.get", return_value=response),
        patch("worker.handlers.translation.requests.post") as post,
        patch("worker.handlers.translation.should_translate_region", return_value=True),
        patch("worker.handlers.translation.chunk_regions_by_conversation", side_effect=lambda rs, *_: [rs]),
        patch("worker.handlers.translation.translate_batch_llm", return_value={}) as provider,
        patch(
            "worker.handlers.translation.parse_and_validate_batch",
            side_effect=lambda _raw, batch, *_a, **_k: {r["id"]: {"translatedText": "Text"} for r in batch},
        ),
        patch("worker.handlers.translation.is_valid_translation", return_value=True),
    ):
        process_translation({"imageId": "image", "pageId": "page"})
    return post.call_args.kwargs["json"], provider


@pytest.mark.parametrize("include_valid", [False, True])
def test_cleanup_review_region_is_translated_for_qa_to_judge(include_valid):
    """Uncertain regions are translated; the backend keeps them hidden until vision QA decides."""
    regions: list[dict] = [
        {"id": "review", "text": "看板", "detectedLanguage": "ja", "confidence": 0.9, "qaStatus": "cleanup_review"}
    ]
    if include_valid:
        regions.append({"id": "valid", "text": "文字", "detectedLanguage": "ja", "confidence": 0.99})
    payload, provider = _run(regions)

    expected = ["review", "valid"] if include_valid else ["review"]
    assert sorted(r["regionId"] for r in payload["translations"]) == expected
    assert sorted(r["id"] for r in provider.call_args.args[0]) == expected
    assert payload["policy"]["skipped"] == []


def test_rejected_region_never_reaches_provider():
    regions = [
        {"id": "gone", "text": "R", "qaStatus": "rejected"},
        {"id": "valid", "text": "文字", "detectedLanguage": "ja", "confidence": 0.99},
    ]
    payload, provider = _run(regions)

    assert [r["regionId"] for r in payload["translations"]] == ["valid"]
    assert [r["id"] for r in provider.call_args.args[0]] == ["valid"]
    assert payload["policy"]["skipped"] == [{"regionId": "gone", "policyAction": "review", "policyReason": "rejected"}]
