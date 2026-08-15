import logging
from unittest.mock import patch

import pytest

from worker.config import is_usable_model
from worker.handlers.qa import process_qa


@pytest.mark.parametrize(
    "model",
    ["N/A", "n/a", "", "  ", None, "default", "inherit", "[ORPHANED] foo", 123],
)
def test_is_usable_model_rejects_sentinels(model):
    assert is_usable_model(model) is False


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-4o", "gpt-4o-mini", " Default-Settings "],
)
def test_is_usable_model_accepts_real_models(model):
    assert is_usable_model(model) is True


def _run_process_qa(job_data, provider=None, vlm_model=None, llm_model=None):
    with (
        patch("worker.handlers.qa.redis_client") as mock_redis,
        patch("worker.handlers.qa.QA_CONFIG") as mock_qa,
        patch("worker.handlers.qa._auto_pass_all") as mock_pass,
        patch("worker.handlers.qa._process_qa_llm") as mock_llm,
        patch("worker.handlers.qa._process_qa_vlm") as mock_vlm,
        patch("worker.handlers.qa._process_qa_hybrid") as mock_hybrid,
    ):
        mock_redis.llen.return_value = 0
        mock_qa.provider = provider
        mock_qa.vlm_model = vlm_model
        mock_qa.llm_model = llm_model
        process_qa(job_data)
    return mock_pass, mock_llm, mock_vlm, mock_hybrid


def test_auto_routes_to_llm_when_vlm_sentinel():
    mock_pass, mock_llm, mock_vlm, mock_hybrid = _run_process_qa(
        {
            "imageId": "img1",
            "qaMode": "auto",
            "qaProvider": "openrouter",
            "qaVlmModel": "N/A",
            "qaLlmModel": "openai/gpt-4o-mini",
        }
    )
    mock_llm.assert_called_once()
    assert not mock_vlm.called
    assert not mock_pass.called
    assert not mock_hybrid.called


def test_auto_prefers_vlm_when_both_models_usable():
    mock_pass, mock_llm, mock_vlm, mock_hybrid = _run_process_qa(
        {
            "imageId": "img1",
            "qaMode": "auto",
            "qaProvider": "openrouter",
            "qaVlmModel": "openai/gpt-4o",
            "qaLlmModel": "openai/gpt-4o-mini",
        }
    )
    mock_vlm.assert_called_once()
    assert not mock_llm.called
    assert not mock_pass.called
    assert not mock_hybrid.called


def test_auto_routes_to_none_when_no_usable_models():
    mock_pass, mock_llm, mock_vlm, mock_hybrid = _run_process_qa(
        {
            "imageId": "img1",
            "qaMode": "auto",
            "qaProvider": "openrouter",
            "qaVlmModel": "N/A",
            "qaLlmModel": "N/A",
        }
    )
    mock_pass.assert_called_once()
    assert not mock_llm.called
    assert not mock_vlm.called
    assert not mock_hybrid.called


def test_auto_routes_to_none_without_provider():
    mock_pass, mock_llm, mock_vlm, mock_hybrid = _run_process_qa(
        {
            "imageId": "img1",
            "qaMode": "auto",
            "qaProvider": None,
            "qaVlmModel": "openai/gpt-4o",
            "qaLlmModel": "openai/gpt-4o-mini",
        },
        provider=None,
    )
    mock_pass.assert_called_once()
    assert not mock_llm.called
    assert not mock_vlm.called
    assert not mock_hybrid.called


def test_auto_logs_resolution(caplog):
    with caplog.at_level(logging.INFO):
        _run_process_qa(
            {
                "imageId": "img1",
                "qaMode": "auto",
                "qaProvider": "openrouter",
                "qaVlmModel": "N/A",
                "qaLlmModel": "openai/gpt-4o-mini",
            }
        )
    assert "[QA] AUTO mode resolved to 'llm'" in caplog.text


def test_explicit_mode_untouched_by_auto_branch(capsys):
    mock_pass, mock_llm, mock_vlm, mock_hybrid = _run_process_qa(
        {
            "imageId": "img1",
            "qaMode": "llm",
            "qaProvider": "openrouter",
            "qaVlmModel": "N/A",
            "qaLlmModel": "N/A",
        }
    )
    mock_llm.assert_called_once()
    assert not mock_vlm.called
    assert not mock_pass.called
    assert not mock_hybrid.called
    out = capsys.readouterr().out
    assert "AUTO mode resolved" not in out
