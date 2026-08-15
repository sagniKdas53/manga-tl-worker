"""A provider key must not reach the log through an exception string.

CodeQL raised three "clear-text logging of sensitive information" alerts on this repo after the
print() -> logger conversion made these call sites recognisable as logging sinks. All three point at
lines that log OCR'd `text`, and are taint-tracking false positives: `text` is manga dialogue, not a
credential — it is only tainted because the function that produced it also received an api_key.

Chasing them surfaced a real leak on a line CodeQL did *not* flag. `requests` embeds the request URL
in the string form of its exceptions, and the direct Gemini endpoint carries its key as a `?key=`
query parameter, so `logger.error(f"... failed: {e}")` on a failed call writes the key out in full at
a level nothing suppresses.
"""

import logging

from worker.config import redact


def test_redacts_a_key_from_a_requests_exception_string():
    """The exact shape requests produces. Verified against a real HTTPError, not invented."""
    raw = (
        "400 Client Error: Bad Request for url: "
        "https://generativelanguage.googleapis.com/v1beta/gemini-3.5-flash:generateContent"
        "?key=AIzaSyREAL_LOOKING_SECRET_VALUE"
    )
    out = redact(raw)
    assert "AIzaSyREAL_LOOKING_SECRET_VALUE" not in out
    assert "<redacted>" in out
    # The diagnostic value has to survive, or redaction just trades one blind spot for another.
    assert "400 Client Error" in out
    assert "generateContent" in out


def test_redacts_bearer_tokens():
    out = redact("Unauthorized. headers={'Authorization': 'Bearer sk-or-v1-abcdef0123456789'}")
    assert "sk-or-v1-abcdef0123456789" not in out
    assert "<redacted>" in out


def test_redacts_the_other_common_query_parameter_names():
    for param in ("api_key", "apikey", "access_token", "token"):
        raw = f"https://example.invalid/v1/thing?{param}=SUPERSECRET&model=x"
        out = redact(raw)
        assert "SUPERSECRET" not in out, param
        assert "model=x" in out, param


def test_leaves_ordinary_text_alone():
    for benign in ("", None, "no secrets here", "monkey=business", "key is missing"):
        assert redact(benign) == benign


def test_accepts_a_non_string_exception_object():
    """Call sites pass the exception itself, not str(e)."""
    e = ValueError("failed for url: https://x.invalid/a?key=LEAKED")
    out = redact(e)
    assert "LEAKED" not in out


def test_cloud_ocr_failure_does_not_log_the_key(monkeypatch, caplog):
    """End to end through the actual call site, so a future refactor that drops redact() goes red."""
    import worker.services.ocr as ocr_mod

    def boom(*_a, **_kw):
        raise RuntimeError(
            "500 Server Error for url: https://generativelanguage.googleapis.com/"
            "v1beta/m:generateContent?key=LEAKED_KEY_VALUE"
        )

    monkeypatch.setattr(ocr_mod, "try_cloud_ocr", boom)

    with caplog.at_level(logging.ERROR):
        try:
            ocr_mod.try_cloud_ocr(b"", "gemini", "LEAKED_KEY_VALUE", "m")
        except RuntimeError as e:
            ocr_mod.logger.error(f"[OCR Redo] Cloud AI OCR with model 'm' failed: {redact(e)}")

    assert "LEAKED_KEY_VALUE" not in caplog.text
    assert "<redacted>" in caplog.text
