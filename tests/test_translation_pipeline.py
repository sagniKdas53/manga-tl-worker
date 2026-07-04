import json
import os
from unittest.mock import patch, MagicMock

from worker.handlers.translation import process_translation


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "gemini",
        "GEMINI_API_KEY": "fake-gemini-key",
        "PREFERRED_LLM_MODEL": "gemini-1.5-pro",
    },
)
def test_process_translation_gemini(mock_post, mock_get, mock_try_cloud_ai):
    # Setup mock backend image details response
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    # Mock batch LLM translation response
    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    # Invoke process_translation
    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    # Assertions
    mock_get.assert_called_once()
    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "gemini"
    assert args[1] == "fake-gemini-key"
    assert args[2] == "gemini-1.5-pro"

    mock_post.assert_called_once()
    post_args, post_kwargs = mock_post.call_args
    assert "translation" in post_args[0]
    payload = post_kwargs["json"]
    assert payload["imageId"] == "image-uuid-1"
    assert len(payload["translations"]) == 1
    assert payload["translations"][0]["regionId"] == "region-uuid-1"
    assert payload["translations"][0]["translatedText"] == "Hello"
    assert payload["translations"][0]["emotion"] == "neutral"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "fake-openrouter-key",
        "PREFERRED_LLM_MODEL": "meta-llama/llama-3-8b-instruct:free",
    },
)
def test_process_translation_openrouter(mock_post, mock_get, mock_try_cloud_ai):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "openrouter"
    assert args[1] == "fake-openrouter-key"
    assert args[2] == "meta-llama/llama-3-8b-instruct:free"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "openai",
        "API_KEY": "fake-openai-key",
        "PREFERRED_LLM_MODEL": "gpt-4o-mini",
    },
)
def test_process_translation_openai(mock_post, mock_get, mock_try_cloud_ai):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "openai"
    assert args[1] == "fake-openai-key"
    assert args[2] == "gpt-4o-mini"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "anthropic",
        "API_KEY": "fake-anthropic-key",
        "PREFERRED_LLM_MODEL": "claude-3-5-sonnet-20241022",
    },
)
def test_process_translation_anthropic(mock_post, mock_get, mock_try_cloud_ai):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "anthropic"
    assert args[1] == "fake-anthropic-key"
    assert args[2] == "claude-3-5-sonnet-20241022"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "nvidia",
        "NVIDIA_API_KEY": "fake-nvidia-key",
        "PREFERRED_LLM_MODEL": "google/gemma-3n-e4b-it",
    },
)
def test_process_translation_nvidia(mock_post, mock_get, mock_try_cloud_ai):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_cloud_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    mock_try_cloud_ai.assert_called_once()
    args, kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "google/gemma-3n-e4b-it"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_local_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "ollama",
        "API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "GEMINI_API_KEY": "",
        "NVIDIA_API_KEY": "",
        "LOCAL_LLM_PROVIDER": "ollama",
        "LOCAL_LLM_MODEL": "gemma4:e4b",
    },
)
def test_process_translation_local_fallback(mock_post, mock_get, mock_try_local_ai):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    mock_try_local_ai.return_value = json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "Greeting",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    mock_try_local_ai.assert_called_once()
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_local_ai")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.translate_text")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch.dict(
    os.environ,
    {
        "MODEL_PROVIDER": "gemini",
        "GEMINI_API_KEY": "fake-gemini-key",
        "PREFERRED_LLM_MODEL": "gemini-1.5-pro",
    },
)
def test_process_translation_retry_individual_fallback(
    mock_post, mock_get, mock_translate_text, mock_try_cloud_ai, mock_try_local_ai
):
    mock_image_info = {
        "id": "image-uuid-1",
        "ocrRegions": [
            {
                "id": "region-uuid-1",
                "text": "こんにちは",
                "detectedLanguage": "ja",
                "confidence": 0.9,
                "width": 100,
                "height": 100,
                "bubbleReadingOrder": 1,
            }
        ],
        "conversations": [],
    }
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = mock_image_info
    mock_get.return_value = mock_get_res

    # Force batch translation to fail (return None or empty)
    mock_try_cloud_ai.return_value = None
    mock_try_local_ai.return_value = None

    # Individual retry fallback returns translation
    mock_translate_text.return_value = "Hello"

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    job_data = {
        "imageId": "image-uuid-1",
        "sourceLanguage": "ja",
        "targetLanguage": "en",
    }
    process_translation(job_data)

    # Verifies both retry and individual fallback were triggered
    assert mock_try_cloud_ai.call_count > 0
    mock_translate_text.assert_called_once()
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"
    assert (
        payload["translations"][0]["translationNotes"]
        == "Individual translation fallback"
    )
