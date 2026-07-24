import re


def detect_language(text):
    # Regex ranges for CJK
    # Japanese Hiragana/Katakana
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
        return "ja"
    # Chinese Hanzi (CJK Unified Ideographs)
    elif re.search(r"[\u4E00-\u9FFF]", text):
        return "zh-TW"
    # Otherwise fallback to English
    return "en"


def contains_japanese(text):
    return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))


def clean_translated_text(translated):
    if not translated:
        return translated
    if isinstance(translated, list) and len(translated) > 0:
        if isinstance(translated[0], dict) and "content" in translated[0]:
            translated = translated[0]["content"]
        elif isinstance(translated[0], str):
            translated = translated[0]
    if isinstance(translated, str):
        translated = translated.strip()
        if (translated.startswith('"') and translated.endswith('"')) or (
            translated.startswith("'") and translated.endswith("'")
        ):
            translated = translated[1:-1].strip()
        return translated
    return translated
