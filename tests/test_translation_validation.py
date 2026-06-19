from worker.services.translation import is_valid_translation


def test_valid_translation():
    # Standard valid translation
    assert is_valid_translation("こんにちは", "Hello") is True


def test_cjk_leak_translation():
    # Has a lot of raw Japanese mixed in
    assert (
        is_valid_translation("こんにちは、お元気ですか", "Hello, お元気ですか") is False
    )
    # Only small ratio of CJK characters is fine (e.g. quote, symbol, or a single char)
    assert is_valid_translation("こんにちは", "Hello (Japanese)") is True


def test_length_ratio_translation():
    # Normal length is fine
    assert is_valid_translation("これはペンです", "This is a pen.") is True
    # Suspiciously long translation
    assert (
        is_valid_translation(
            "こんにちは",
            "This is a super long translation text that goes on and on and on and on and on and on and on and on and on and on and on and on and on.",
        )
        is False
    )


def test_excessive_repetition_translation():
    # Repeating words
    assert (
        is_valid_translation("あそこ", "Over there Over there Over there Over there")
        is False
    )
    # Normal repeated words are fine if context allows
    assert is_valid_translation("バイバイ", "Bye bye") is True
