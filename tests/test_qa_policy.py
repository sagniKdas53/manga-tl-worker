import json
from unittest.mock import MagicMock, patch

from worker.handlers.qa import _process_qa_llm, _translation_qa_regions


def test_translation_qa_regions_selects_only_effective_replace():
    regions = [
        {"id": "review-sfx", "regionType": "sfx"},
        {"id": "preserved", "regionType": "speech", "user_override": "preserve"},
        {"id": "replace-dialogue", "regionType": "speech", "user_override": "replace"},
    ]

    assert [region["id"] for region in _translation_qa_regions(regions)] == ["replace-dialogue"]


@patch("worker.handlers.qa.QA_CONFIG")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
def test_policy_skipped_regions_never_enter_per_region_translation_qa(
    mock_post,
    mock_get,
    mock_try_cloud_ai,
    mock_qa_config,
):
    image_info = {
        "ocrRegions": [
            {
                "id": "review-sfx",
                "text": "ドキ",
                "regionType": "sfx",
                "translatedText": "",
                "confidence": 0.9,
            },
            {
                "id": "replace-dialogue",
                "text": "こんにちは",
                "regionType": "speech",
                "user_override": "replace",
                "translatedText": "Hello",
                "confidence": 0.9,
            },
        ]
    }
    mock_get.return_value = MagicMock(status_code=200, json=lambda: image_info)
    mock_post.return_value = MagicMock(status_code=200)
    mock_qa_config.provider = "openrouter"
    mock_qa_config.llm_model = "qa-model"
    mock_qa_config.resolve_key.return_value = "qa-key"
    mock_try_cloud_ai.return_value = json.dumps(
        {
            "results": [
                {"regionId": "review-sfx", "qaStatus": "reject_sfx", "qaScore": 1, "qaFeedback": "wrong target"},
                {"regionId": "replace-dialogue", "qaStatus": "passed", "qaScore": 1, "qaFeedback": "good"},
            ]
        }
    )

    _process_qa_llm({"imageId": "image-1"})

    prompt = mock_try_cloud_ai.call_args.args[3]
    assert "review-sfx" not in prompt
    assert "ドキ" not in prompt
    assert "replace-dialogue" in prompt
    callback = mock_post.call_args.kwargs["json"]
    assert [result["regionId"] for result in callback["qaResults"]] == ["replace-dialogue"]
    assert image_info["ocrRegions"][0]["text"] == "ドキ", "page-wide visual QA retains source pixels"
