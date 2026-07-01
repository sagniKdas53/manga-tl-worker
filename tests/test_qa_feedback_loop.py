from unittest.mock import patch
import json
import os
from worker.services.translation import translate_batch_llm


@patch("worker.services.translation.try_cloud_ai")
def test_translate_batch_llm_handles_qa_feedback(mock_try_cloud_ai):
    # Set up mock response
    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-1",
                    "translation": "Corrected translation",
                    "translationNotes": "notes",
                    "emotion": "neutral",
                    "tone": "neutral",
                    "translationScore": 0.9,
                }
            ]
        }
    )

    unmatched_regions = [
        {
            "id": "region-1",
            "text": "原テキスト",
            "translatedText": "Failed previous translation",
            "qaStatus": "failed",
            "qaFeedback": "It should be more polite.",
        }
    ]

    # Set up environment variables to trigger try_cloud_ai
    os.environ["MODEL_PROVIDER"] = "openrouter"
    os.environ["API_KEY"] = "fake-key"

    res = translate_batch_llm(unmatched_regions, context_str="", response_schema=None)
    assert res is not None

    # Check that mock_try_cloud_ai was called with the prompt containing qaFeedback
    args, kwargs = mock_try_cloud_ai.call_args
    prompt = args[3]  # The prompt parameter
    assert "previousTranslation" in prompt or "qaFeedback" in prompt
    assert "It should be more polite." in prompt
    assert "Failed previous translation" in prompt
