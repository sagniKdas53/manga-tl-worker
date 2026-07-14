from unittest.mock import MagicMock, patch

from worker.services.translation import (
    is_valid_translation,
    should_translate_region,
    try_deepl,
)


def test_is_valid_translation():
    # Valid translation
    assert is_valid_translation("こんにちは", "Hello")

    # Boilerplate check
    assert is_valid_translation("こんにちは", "Here is the translation: Hello")
    assert not is_valid_translation("こんにちは", "translate the following text: Hello")

    # Identical to Japanese source check
    assert not is_valid_translation("こんにちは", "こんにちは")

    # Pathologically long
    assert not is_valid_translation(
        "hi",
        "This is an extremely long translation for a very short text which should definitely fail the validation check because it exceeds the length ratio by a huge margin.",
    )


def test_should_translate_region():
    # Reject too small
    region_small = {"width": 5, "height": 5, "text": "a"}
    assert not should_translate_region(region_small)

    # Reject low confidence
    region_low_conf = {"width": 20, "height": 20, "text": "hello", "confidence": 0.2}
    assert not should_translate_region(region_low_conf)

    # SFX whitelist
    region_sfx = {
        "width": 20,
        "height": 20,
        "text": "ドン",
        "confidence": 0.2,
    }  # Should pass despite low conf due to whitelist
    assert should_translate_region(region_sfx)

    # Alphanumeric with low confidence
    region_alpha = {"width": 20, "height": 20, "text": "AB12", "confidence": 0.4}
    assert not should_translate_region(region_alpha)


@patch("worker.services.translation.requests.post")
@patch("worker.services.translation.os.environ")
def test_try_deepl(mock_env, mock_post):
    mock_env.get.return_value = "dummy_key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"translations": [{"text": "Hello"}]}
    mock_post.return_value = mock_resp

    res = try_deepl("こんにちは", "en")
    assert res == "Hello"
    mock_post.assert_called_once()
