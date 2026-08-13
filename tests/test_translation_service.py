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


# --- R3 (docs/issues.md) ----------------------------------------------------------------------


def test_sound_effects_are_left_as_the_artist_drew_them():
    """Every reference output leaves sfx in the artwork untouched. We set English over them, which
    means painting a slab on the artwork to do it."""
    from worker.services.translation import should_typeset_region

    assert not should_typeset_region({"text": "ギチィ", "regionType": "sfx", "confidence": 0.9})
    assert should_typeset_region({"text": "もう家に帰して！", "regionType": "speech", "confidence": 0.9})


def test_unenclosed_low_confidence_text_is_not_typeset():
    """sample10's `cu3gichi` -- a misread of the artwork's ギチィ -- became the sentence "Deadline
    countdown activated!", which appears nowhere in the manga, and we painted a lavender box on a
    desk to hold it. No balloon plus a recogniser that could not read it is the signature."""
    from worker.services.translation import should_typeset_region

    assert not should_typeset_region(
        {"text": "cu3gichi", "bubbleId": "direct_text_3", "confidence": 0.31},
    )


def test_both_halves_are_required_before_dropping_a_region():
    """The half that protects real dialogue.

    sample10's yellow region is lettered straight onto a character's blanket -- no balloon at all --
    and it is a whole line of dialogue. A low score inside a balloon is still a line somebody said.
    Neither signal alone may drop a region.
    """
    from worker.services.translation import should_typeset_region

    assert should_typeset_region(
        {"text": "もう家に帰して！", "bubbleId": "direct_text_1", "confidence": 0.95},
    ), "unenclosed but read cleanly: this is dialogue"
    assert should_typeset_region(
        {"text": "ええ。ですから", "bubbleId": "bubble_2", "confidence": 0.20},
    ), "poorly read but inside a balloon: still a line somebody said"


def test_the_sfx_prompt_no_longer_contradicts_the_romaji_rule():
    """28 of the corpus's 29 parenthetical glosses were the exact shape the prompt asked for: it
    required "DOKAA (WHAM)" eleven lines above forbidding "ERUFU (ELF!)", which is the same string
    in the same shape. The model was obeying, not drifting."""
    from worker.services.translation import (
        MANGA_TRANSLATION_JSON_SYSTEM_PROMPT,
        MANGA_TRANSLATION_SYSTEM_PROMPT,
    )

    for prompt in (MANGA_TRANSLATION_JSON_SYSTEM_PROMPT,):
        assert "Transliterate the sound effect" not in prompt
        assert "NEVER include romanized text" in prompt

    assert MANGA_TRANSLATION_SYSTEM_PROMPT.count("- Do not explain.") == 1
