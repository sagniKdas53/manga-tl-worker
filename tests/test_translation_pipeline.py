import json
from unittest.mock import MagicMock, patch

import pytest

from worker.handlers.translation import process_translation


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_process_translation_gemini(mock_tl_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_tl_config.provider = "gemini"
    mock_tl_config.resolve_key.return_value = "fake-gemini-key"
    mock_tl_config.llm_model = "gemini-1.5-pro"

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
    args, _kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "gemini"
    assert args[1] == "fake-gemini-key"
    assert args[2] == "gemini-1.5-pro"
    assert _kwargs["routing_strategy"] == "lowest-cost"

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
@patch("worker.config.TL_CONFIG")
def test_process_translation_openrouter(mock_tl_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_tl_config.provider = "openrouter"
    mock_tl_config.resolve_key.return_value = "fake-openrouter-key"
    mock_tl_config.llm_model = "meta-llama/llama-3-8b-instruct:free"

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
    args, _kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "openrouter"
    assert args[1] == "fake-openrouter-key"
    assert args[2] == "meta-llama/llama-3-8b-instruct:free"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_process_translation_openai(mock_tl_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_tl_config.provider = "openai"
    mock_tl_config.resolve_key.return_value = "fake-openai-key"
    mock_tl_config.llm_model = "gpt-4o-mini"

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
    args, _kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "openai"
    assert args[1] == "fake-openai-key"
    assert args[2] == "gpt-4o-mini"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_process_translation_anthropic(mock_tl_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_tl_config.provider = "anthropic"
    mock_tl_config.resolve_key.return_value = "fake-anthropic-key"
    mock_tl_config.llm_model = "claude-3-5-sonnet-20241022"

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
    args, _kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "anthropic"
    assert args[1] == "fake-anthropic-key"
    assert args[2] == "claude-3-5-sonnet-20241022"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_process_translation_nvidia(mock_tl_config, mock_post, mock_get, mock_try_cloud_ai):
    mock_tl_config.provider = "nvidia"
    mock_tl_config.resolve_key.return_value = "fake-nvidia-key"
    mock_tl_config.llm_model = "google/gemma-3n-e4b-it"

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
    args, _kwargs = mock_try_cloud_ai.call_args
    assert args[0] == "nvidia"
    assert args[1] == "fake-nvidia-key"
    assert args[2] == "google/gemma-3n-e4b-it"

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["translations"][0]["translatedText"] == "Hello"


@patch("worker.services.translation.try_local_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_process_translation_local_fallback(mock_tl_config, mock_post, mock_get, mock_try_local_ai):
    mock_tl_config.provider = "ollama"
    mock_tl_config.resolve_key.return_value = ""
    mock_tl_config.llm_model = "gemma4:e4b"

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
@patch("worker.config.TL_CONFIG")
def test_process_translation_retry_individual_fallback(
    mock_tl_config,
    mock_post,
    mock_get,
    mock_translate_text,
    mock_try_cloud_ai,
    mock_try_local_ai,
):
    mock_tl_config.provider = "gemini"
    mock_tl_config.resolve_key.return_value = "fake-gemini-key"
    mock_tl_config.llm_model = "gemini-1.5-pro"

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
    assert payload["translations"][0]["translationNotes"] == "Individual translation fallback"


def _image_info_one_region():
    return {
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


def _batch_reply():
    return json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "Hello",
                    "translationNotes": "",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.98,
                }
            ]
        }
    )


@patch("worker.utils.rate_limit.get_job_costs")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.handlers.translation.TL_CONFIG")
@patch("worker.config.TL_CONFIG")
def test_model_identifier_reports_the_model_that_actually_ran(
    mock_service_config,
    mock_handler_config,
    mock_post,
    mock_get,
    mock_try_cloud_ai,
    mock_get_job_costs,
):
    """The worker's static default must not masquerade as the model that served the job.

    modelIdentifier used to be built from the handler's TL_CONFIG, which is frozen at import,
    so a page translated by gpt-5.6-luna was recorded as whatever the worker was configured
    with. The corpus benchmark reads this field to attribute quality to a model, so a wrong
    value there silently confounds the comparison. The two configs are patched to *different*
    models on purpose: the handler's is the one the bug used.
    """
    for cfg, model in (
        (mock_service_config, "openai/gpt-5.6-luna"),
        (mock_handler_config, "deepseek/deepseek-v4-pro"),
    ):
        cfg.provider = "openrouter"
        cfg.llm_model = model
        cfg.resolve_key.return_value = "fake-key"

    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = _image_info_one_region()
    mock_get.return_value = mock_get_res
    mock_try_cloud_ai.return_value = _batch_reply()
    mock_post.return_value = MagicMock(status_code=200)

    # What the provider actually billed for this job.
    mock_get_job_costs.return_value = [
        {
            "model": "openai/gpt-5.6-luna",
            "provider": "openrouter",
            "prompt_tokens": 1747,
            "completion_tokens": 648,
            "estimated_cost": None,
        }
    ]

    process_translation(
        {
            "imageId": "image-uuid-1",
            "sourceLanguage": "ja",
            "targetLanguage": "en",
            "tlProvider": "openrouter",
            "tlModel": "openai/gpt-5.6-luna",
        }
    )

    payload = mock_post.call_args.kwargs["json"]
    identifiers = {t["modelIdentifier"] for t in payload["translations"]}
    assert identifiers == {"openrouter/openai/gpt-5.6-luna"}
    assert not any("deepseek" in i for i in identifiers), (
        "modelIdentifier fell back to the worker's static TL_CONFIG default"
    )


@patch("worker.utils.rate_limit.get_job_costs")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.handlers.translation.TL_CONFIG")
@patch("worker.config.TL_CONFIG")
def test_model_identifier_falls_back_to_the_requested_model_when_no_cost_was_recorded(
    mock_service_config,
    mock_handler_config,
    mock_post,
    mock_get,
    mock_try_cloud_ai,
    mock_get_job_costs,
):
    """No cost records (cost calculation disabled) must still beat the static default."""
    for cfg, model in (
        (mock_service_config, "openai/gpt-5.6-luna"),
        (mock_handler_config, "deepseek/deepseek-v4-pro"),
    ):
        cfg.provider = "openrouter"
        cfg.llm_model = model
        cfg.resolve_key.return_value = "fake-key"

    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = _image_info_one_region()
    mock_get.return_value = mock_get_res
    mock_try_cloud_ai.return_value = _batch_reply()
    mock_post.return_value = MagicMock(status_code=200)
    mock_get_job_costs.return_value = []

    process_translation(
        {
            "imageId": "image-uuid-1",
            "sourceLanguage": "ja",
            "targetLanguage": "en",
            "tlProvider": "openrouter",
            "tlModel": "openai/gpt-5.6-luna",
        }
    )

    payload = mock_post.call_args.kwargs["json"]
    assert {t["modelIdentifier"] for t in payload["translations"]} == {"openrouter/openai/gpt-5.6-luna"}


def _echoed_batch_reply():
    """A batch answer that is not a translation: the model handed the Japanese straight back.

    `is_valid_translation` rejects this as identical_to_source. It is what an OCR misfire on a
    texture actually produces -- the model has nothing to translate and echoes the input.
    """
    return json.dumps(
        {
            "translations": [
                {
                    "id": "region-uuid-1",
                    "translation": "こんにちは",
                    "translationNotes": "",
                    "emotion": "neutral",
                    "tone": "polite",
                    "translationScore": 0.5,
                }
            ]
        }
    )


@patch("worker.services.translation.try_local_ai")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.translate_text")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_a_page_whose_text_is_rejected_finishes_instead_of_raising(
    mock_tl_config,
    mock_post,
    mock_get,
    mock_translate_text,
    mock_try_cloud_ai,
    mock_try_local_ai,
):
    """AUDIT-B13, the case it is actually about.

    Every tier answers, and every answer is rejected as not-a-translation. Re-running the job
    asks the same question of the same models, so it must finish and report rather than raise:
    raising made process_job_rq retry to maxAttempts and leave a red FAILED row to dismiss by
    hand. The callback must still go out, and must say allFailed.
    """
    mock_tl_config.provider = "gemini"
    mock_tl_config.resolve_key.return_value = "fake-gemini-key"
    mock_tl_config.llm_model = "gemini-1.5-pro"

    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = _image_info_one_region()
    mock_get.return_value = mock_get_res

    # Answers arrive at every tier; none of them is a translation.
    mock_try_cloud_ai.return_value = _echoed_batch_reply()
    mock_try_local_ai.return_value = _echoed_batch_reply()
    mock_translate_text.return_value = "こんにちは"

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    process_translation(
        {
            "imageId": "image-uuid-1",
            "sourceLanguage": "ja",
            "targetLanguage": "en",
        }
    )

    payload = mock_post.call_args[1]["json"]
    assert payload["allFailed"] is True
    assert payload["failedCount"] == 1
    assert payload["totalCount"] == 1
    assert payload["translations"][0]["translationFailed"] is True


@patch("worker.services.translation.try_local_ai")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.handlers.translation.translate_text")
@patch("worker.handlers.translation.requests.get")
@patch("worker.handlers.translation.requests.post")
@patch("worker.config.TL_CONFIG")
def test_a_page_that_got_no_answer_at_all_still_raises_for_retry(
    mock_tl_config,
    mock_post,
    mock_get,
    mock_translate_text,
    mock_try_cloud_ai,
    mock_try_local_ai,
):
    """The other half of AUDIT-B13, and the reason the first fix was wrong.

    Nothing answers -- which is what a provider timeout, a 429, a 5xx or a malformed response
    looks like from here, because `process_chunk` swallows the exception and returns None. That
    is not an untranslatable page; it is a page nobody was asked about successfully, and a later
    attempt can genuinely succeed. Suppressing this marked the job COMPLETED and left the page
    permanently untranslated with no retry and no failed row.

    The callback still goes out first, so the backend sees the attempt either way.
    """
    mock_tl_config.provider = "gemini"
    mock_tl_config.resolve_key.return_value = "fake-gemini-key"
    mock_tl_config.llm_model = "gemini-1.5-pro"

    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.json.return_value = _image_info_one_region()
    mock_get.return_value = mock_get_res

    # No answer at any tier.
    mock_try_cloud_ai.return_value = None
    mock_try_local_ai.return_value = None
    mock_translate_text.return_value = None

    mock_post_res = MagicMock()
    mock_post_res.status_code = 200
    mock_post.return_value = mock_post_res

    with pytest.raises(RuntimeError, match="no answer at all"):
        process_translation(
            {
                "imageId": "image-uuid-1",
                "sourceLanguage": "ja",
                "targetLanguage": "en",
            }
        )

    # The backend was still told, before the raise.
    payload = mock_post.call_args[1]["json"]
    assert payload["allFailed"] is True
    assert payload["failedCount"] == 1
