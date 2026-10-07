"""OpenRouter session_id: one per chapter and stage, sent where OpenRouter reads it.

Price routing sends each call to whichever host is cheapest, and every host keeps its own prompt
cache. Measured 2026-10-06 with GLM 5.3 Flash and a 2.8k-token shared prefix: without a session the
calls alternated Novita / StreamLake and missed the cache; with one session_id they stayed on
StreamLake and cached 2,752 tokens each, 78% cheaper per call. The client used to put session_id
under "extra_body" (an OpenAI-SDK idiom) and nothing set one, so OpenRouter never saw it.
"""

import copy
from unittest.mock import MagicMock, patch

from worker.config import get_llm_session, reset_llm_session, set_llm_session
from worker.services.llm_client import LLMClient


def _sent(client):
    sent = []

    def record(*_args, **kw):
        sent.append(copy.deepcopy(kw["json"]))
        good = MagicMock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        return good

    with patch("worker.services.llm_client.requests.post", side_effect=record):
        client.complete(messages=[{"role": "user", "content": "Hi"}])
    return sent[0]


def test_the_bound_session_reaches_openrouter_at_the_top_level():
    token = set_llm_session("tlhub:chapter-1:translation")
    try:
        payload = _sent(LLMClient(provider="openrouter", api_key="k", model="m"))
    finally:
        reset_llm_session(token)
    assert payload["session_id"] == "tlhub:chapter-1:translation"
    assert "extra_body" not in payload


def test_an_explicit_session_wins_and_no_session_sends_none():
    token = set_llm_session("tlhub:chapter-1:qa")
    try:
        payload = _sent(LLMClient(provider="openrouter", api_key="k", model="m", session_id="mine"))
    finally:
        reset_llm_session(token)
    assert payload["session_id"] == "mine"
    assert "session_id" not in _sent(LLMClient(provider="openrouter", api_key="k", model="m"))


def test_other_providers_get_no_session_field():
    token = set_llm_session("tlhub:chapter-1:translation")
    try:
        payload = _sent(LLMClient(provider="nvidia", api_key="k", model="m"))
    finally:
        reset_llm_session(token)
    assert "session_id" not in payload


def test_process_job_rq_binds_one_session_per_chapter_and_stage():
    from worker.rq_tasks import process_job_rq

    mock_res = MagicMock(status_code=200)
    mock_res.json.return_value = {"status": "PENDING"}
    seen = {}

    for queue, target, job_data in (
        (
            "queue:translation",
            "worker.rq_tasks.process_translation",
            {"jobId": "j", "chapterId": "c1", "imageId": "i1"},
        ),
        ("queue:qa", "worker.rq_tasks.process_qa", {"jobId": "j", "chapterId": "c1", "imageId": "i1"}),
        ("queue:ocr", "worker.rq_tasks.process_ocr", {"jobId": "j", "imageId": "i2"}),
    ):
        with (
            patch("worker.rq_tasks.check_stale_job", return_value=False),
            patch("requests.get", return_value=mock_res),
            patch("worker.rq_tasks.update_job_status"),
            patch(target, side_effect=lambda _d, q=queue: seen.__setitem__(q, get_llm_session())),
        ):
            process_job_rq(queue, job_data)
        assert get_llm_session() == ""

    assert seen == {
        "queue:translation": "tlhub:c1:translation",
        "queue:qa": "tlhub:c1:qa",
        "queue:ocr": "tlhub:i2:ocr",
    }


def test_qa_asks_for_no_reasoning_on_passed_regions():
    """Passed regions' qaFeedback is never read; the "detailed explanation" every prompt asked for was
    most of QA's output, and output is 76% of a QA call's price (Gemini 3.5 Flash Lite, 2026-10-06)."""
    import inspect

    import worker.handlers.qa as qa

    source = inspect.getsource(qa)
    assert "You MUST still provide a detailed explanation" not in source
    assert source.count('Leave "qaFeedback" empty, or a few words at most.') == 3
