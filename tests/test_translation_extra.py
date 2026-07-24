from unittest.mock import MagicMock, patch

from worker.services.translation import (
    _get_api_url_and_headers,
    build_context_string,
    parse_and_validate_batch,
    translate_batch_deepl,
    translate_batch_llm,
    translate_text,
    try_cloud_ai,
    try_cloud_ai_vision,
    try_google_translate,
    try_local_ai,
    try_local_vlm_vision,
    validate_translation_response,
    wait_for_cooldown,
)


def test_validate_translation_response():
    # test dict format
    data = {
        "translations": [{"id": "1", "translation": "hello", "translationScore": 0.9}]
    }
    res = validate_translation_response(data)
    assert "1" in res  # type: ignore
    assert res["1"]["translatedText"] == "hello"  # type: ignore

    # test items format
    data = {"items": [{"id": "1", "translation": "hello"}]}
    res = validate_translation_response(data)
    assert "1" in res  # type: ignore

    # test simple dict
    data = {"1": "hello"}
    res = validate_translation_response(data)
    assert "1" in res  # type: ignore

    # test invalid
    assert validate_translation_response("not_a_list_or_dict") is None


def test_parse_and_validate_batch():
    text = '```json\n{"translations": [{"id": "1", "translation": "hello"}]}\n```'
    res = parse_and_validate_batch(text, [])
    assert "1" in res
    assert parse_and_validate_batch("", []) is None
    assert parse_and_validate_batch("invalid json", []) is None


@patch("worker.services.translation.time.sleep")
@patch("worker.services.translation.time.time")
def test_wait_for_cooldown(mock_time, mock_sleep):
    from worker.services.translation import PROVIDER_COOLDOWNS

    mock_time.return_value = 100.0
    PROVIDER_COOLDOWNS["test_prov"] = 105.0
    wait_for_cooldown("test_prov")
    mock_sleep.assert_called_with(5.0)


def test_get_api_url_and_headers():
    url, hdrs, _model = _get_api_url_and_headers("openrouter", "key", "")
    assert "openrouter" in url
    assert "Authorization" in hdrs

    url, hdrs, _model = _get_api_url_and_headers("anthropic", "key", "claude")
    assert "anthropic" in url
    assert "x-api-key" in hdrs


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.time.time")
def test_try_cloud_ai(mock_time, mock_post):
    from worker.services.translation import PROVIDER_COOLDOWNS

    mock_time.return_value = 100.0
    PROVIDER_COOLDOWNS["openai"] = 0.0  # reset cooldown

    # success
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    mock_post.return_value = mock_resp

    res = try_cloud_ai("openai", "key", "gpt-4", "prompt", {"type": "object"})
    assert res == "hello"

    # 429
    mock_resp.status_code = 429
    mock_post.return_value = mock_resp
    with patch("worker.services.translation.time.sleep"):
        res = try_cloud_ai("openai", "key", "gpt-4", "prompt")
        assert res is None


@patch("worker.services.translation.requests.post")
def test_try_cloud_ai_vision(mock_post):
    from worker.services.translation import PROVIDER_COOLDOWNS

    PROVIDER_COOLDOWNS["anthropic"] = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"content": [{"text": "hello vision"}]}
    mock_post.return_value = mock_resp

    res = try_cloud_ai_vision(
        "anthropic", "key", "claude", "prompt", "base64", system_prompt="sys"
    )
    assert res == "hello vision"


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.os.environ.get")
@patch("worker.utils.lock.acquire_lock")
def test_try_local_ai(mock_lock, mock_env, mock_post):
    # Setup mock config
    def env_get(k, default=""):
        if k == "LOCAL_LLM_PROVIDER":
            return "ollama"
        if k == "LOCAL_LLM_ENDPOINT":
            return "http://test:11434"
        return default

    mock_env.side_effect = env_get

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hello local"}}]}
    mock_post.return_value = mock_resp

    res = try_local_ai("prompt", "text", response_schema={})
    assert res == "hello local"


@patch("worker.services.translation.requests.get")
def test_try_google_translate(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [[["hello ", None, None], ["world", None, None]]]
    mock_get.return_value = mock_resp

    res = try_google_translate("こんにちは世界")
    assert res == "hello world"


@patch("worker.services.translation.try_deepl")
@patch("worker.services.translation.try_cloud_ai")
@patch("worker.services.translation.try_google_translate")
@patch("worker.services.translation.os.environ.get")
def test_translate_text(mock_env, mock_google, mock_cloud, mock_deepl):
    # Cross-provider translation fallbacks are intentionally disabled.
    mock_deepl.return_value = None
    mock_cloud.return_value = None
    mock_google.return_value = "hello from google"

    res = translate_text("test")
    assert res is None

    # DeepL is no longer part of the strict fallback chain.
    mock_deepl.return_value = "hello deepl"

    def env_get(k, default=""):
        if k == "TRANSLATION_PROVIDER":
            return "deepl"
        return default

    mock_env.side_effect = env_get

    res = translate_text("test")
    assert res is None


@patch("worker.services.translation.try_cloud_ai")
@patch("worker.services.translation.os.environ.get")
@patch("worker.services.translation.time.time")
def test_translate_batch_llm(mock_time, mock_env, mock_cloud):
    regions = [{"id": "1", "text": "hello"}]
    mock_time.return_value = 100.0

    with patch("worker.config.TL_CONFIG") as mock_tl:
        mock_tl.provider = "openai"
        mock_tl.resolve_key.return_value = "key"

        mock_cloud.return_value = '{"translations": [{"id": "1", "translation": "world", "translationScore": 0.9}]}'

        res = translate_batch_llm(regions)
        assert res is not None
        assert "1" in res


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.os.environ.get")
def test_translate_batch_deepl(mock_env, mock_post):
    def env_get(k, default=""):
        if k in ("DEEPL_API_KEY", "DEEPL_KEY"):
            return "dummy_key"
        return default

    mock_env.side_effect = env_get

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "translations": [{"text": "hello deepl 1"}, {"text": "hello deepl 2"}]
    }
    mock_post.return_value = mock_resp

    regions = [{"id": "1", "text": "j1"}, {"id": "2", "text": "j2"}]

    res = translate_batch_deepl(regions)
    assert res is not None
    assert res["1"] == "hello deepl 1"
    assert res["2"] == "hello deepl 2"


def test_build_context_string():
    info = {
        "seriesMetadata": {
            "title": "My Manga",
            "originalLanguage": "ja",
            "metadataJson": {"characters": ["Alice"]},
        },
        "chapterSummary": "Some summary",
        "previousPageText": "hello | world",
    }
    s = build_context_string(info)
    assert "Series Title: My Manga" in s
    assert "Some summary" in s
    assert "- hello" in s


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.os.environ.get")
def test_try_local_vlm_vision(mock_env, mock_post):
    def env_get(k, default=""):
        if k == "LOCAL_LLM_PROVIDER":
            return "ollama"
        if k == "LOCAL_LLM_ENDPOINT":
            return "http://test:11434"
        return default

    mock_env.side_effect = env_get

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hello vlm"}}]}
    mock_post.return_value = mock_resp

    res = try_local_vlm_vision("model", "prompt", "base64")
    assert res == "hello vlm"


@patch("worker.services.translation.try_cloud_ai")
def test_translate_text_providers(mock_cloud):
    from worker.services.translation import translate_text

    mock_cloud.return_value = "hello translation"

    with patch("worker.config.TL_CONFIG") as mock_tl:
        mock_tl.resolve_key.return_value = "dummy"

        for prov in ["openrouter", "gemini", "openai", "nvidia", "anthropic"]:
            mock_tl.provider = prov
            res = translate_text("ja text", source_lang="ja", target_lang="en")
            assert "hello" in res
