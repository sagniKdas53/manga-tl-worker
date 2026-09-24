from unittest.mock import MagicMock, patch

import pytest

from worker.handlers.translation import process_translation


@pytest.mark.parametrize("include_valid", [False, True])
def test_cleanup_review_never_reaches_provider_or_translation_callback(include_valid):
    regions: list[dict] = [{"id": "review", "text": "uncertain", "qaStatus": "cleanup_review"}]
    if include_valid:
        regions.append({"id": "valid", "text": "文字", "detectedLanguage": "ja", "confidence": 0.99})
    response = MagicMock(status_code=200)
    response.json.return_value = {"ocrRegions": regions, "conversations": []}
    with (
        patch("worker.handlers.translation.requests.get", return_value=response),
        patch("worker.handlers.translation.requests.post") as post,
        patch("worker.handlers.translation.should_translate_region", return_value=True),
        patch("worker.handlers.translation.chunk_regions_by_conversation", side_effect=lambda rs, *_: [rs]),
        patch("worker.handlers.translation.translate_batch_llm", return_value={}) as provider,
        patch(
            "worker.handlers.translation.parse_and_validate_batch", return_value={"valid": {"translatedText": "Text"}}
        ),
        patch("worker.handlers.translation.is_valid_translation", return_value=True),
    ):
        process_translation({"imageId": "image", "pageId": "page"})
    payload = post.call_args.kwargs["json"]
    assert payload["policy"]["skipped"] == [
        {"regionId": "review", "policyAction": "review", "policyReason": "cleanup-review-required"}
    ]
    assert [r["regionId"] for r in payload["translations"]] == (["valid"] if include_valid else [])
    if include_valid:
        assert [r["id"] for r in provider.call_args.args[0]] == ["valid"]
    else:
        provider.assert_not_called()
