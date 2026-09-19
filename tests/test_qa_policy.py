import json
from unittest.mock import MagicMock, patch

from worker.handlers.qa import _process_qa_llm, _translation_qa_regions


def test_translation_qa_regions_selects_successful_unreviewed_elements():
    regions = [
        {"id": "review-sfx", "regionType": "sfx", "translatedText": "Heartbeat"},
        {
            "id": "preserved",
            "regionType": "speech",
            "user_override": "preserve",
            "translatedText": "Do not show",
        },
        {"id": "review-dialogue", "regionType": "speech", "translatedText": "Hello"},
        {
            "id": "failed-dialogue",
            "regionType": "speech",
            "translatedText": "Missing",
            "translationFailed": True,
        },
    ]

    assert [region["id"] for region in _translation_qa_regions(regions)] == [
        "review-sfx",
        "review-dialogue",
    ]


@patch("worker.handlers.qa.QA_CONFIG")
@patch("worker.handlers.qa.try_cloud_ai")
@patch("worker.handlers.qa.requests.get")
@patch("worker.handlers.qa.requests.post")
def test_unreviewed_sfx_enters_translation_qa_for_rejection(
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
                "translatedText": "Heartbeat",
                "confidence": 0.9,
            },
            {
                "id": "review-dialogue",
                "text": "こんにちは",
                "regionType": "speech",
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
                {"regionId": "review-sfx", "qaStatus": "reject_sfx", "qaScore": 1, "qaFeedback": "SFX"},
                {"regionId": "review-dialogue", "qaStatus": "passed", "qaScore": 1, "qaFeedback": "Dialogue"},
            ]
        }
    )

    _process_qa_llm({"imageId": "image-1"})

    prompt = mock_try_cloud_ai.call_args.args[3]
    assert "review-sfx" in prompt
    assert "ドキ" in prompt
    assert "review-dialogue" in prompt
    callback = mock_post.call_args.kwargs["json"]
    assert [result["regionId"] for result in callback["qaResults"]] == ["review-sfx", "review-dialogue"]
