from worker.utils.text import (
    clean_translated_text,
    contains_japanese,
    detect_language,
)


def test_detect_language():
    assert detect_language("こんにちは") == "ja"
    assert detect_language("日本語です") == "ja"
    assert detect_language("中文") == "zh-TW"
    assert detect_language("Hello world") == "en"


def test_contains_japanese():
    assert contains_japanese("こんにちは") is True
    assert contains_japanese("Hello") is False


def test_clean_translated_text():
    assert clean_translated_text(None) is None
    assert clean_translated_text("") == ""
    assert clean_translated_text([{"content": ' "Hello" '}]) == "Hello"
    assert clean_translated_text([" 'Hello' "]) == "Hello"
    assert clean_translated_text('"Quoted Text"') == "Quoted Text"
    assert clean_translated_text("'Single Quoted'") == "Single Quoted"
    assert clean_translated_text(123) == 123
