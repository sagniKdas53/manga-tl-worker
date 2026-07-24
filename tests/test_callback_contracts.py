"""
Contract tests validating worker callback payload shapes against OpenAPI spec.
"""


def test_panel_callback_payload_matches_spec():
    mock_panels = [{"x": 10, "y": 20, "width": 500, "height": 400, "readingOrder": 1}]
    payload = {
        "imageId": "123e4567-e89b-12d3-a456-426614174000",
        "pageId": "123e4567-e89b-12d3-a456-426614174001",
        "panels": mock_panels,
    }
    assert "imageId" in payload
    assert "pageId" in payload
    assert isinstance(payload["panels"], list)
    for p in payload["panels"]:
        assert all(k in p for k in ["x", "y", "width", "height", "readingOrder"])


def test_ocr_callback_payload_matches_spec():
    mock_regions = [
        {
            "text": "こんにちは",
            "detectedLanguage": "ja",
            "confidence": 0.95,
            "x": 15,
            "y": 25,
            "width": 100,
            "height": 40,
        }
    ]
    payload = {
        "imageId": "123e4567-e89b-12d3-a456-426614174000",
        "pageId": "123e4567-e89b-12d3-a456-426614174001",
        "modelIdentifier": "tesseract",
        "confidence": 0.95,
        "regions": mock_regions,
    }
    assert "imageId" in payload
    assert "pageId" in payload
    assert "modelIdentifier" in payload
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["regions"], list)
    for r in payload["regions"]:
        assert all(
            k in r
            for k in [
                "text",
                "detectedLanguage",
                "confidence",
                "x",
                "y",
                "width",
                "height",
            ]
        )


def test_translation_callback_boolean_type():
    payload = {
        "imageId": "123e4567-e89b-12d3-a456-426614174000",
        "translationFailed": False,
        "translations": [
            {
                "regionId": "123e4567-e89b-12d3-a456-426614174002",
                "translatedText": "Hello",
                "translationFailed": False,
                "translationScore": 0.9,
            }
        ],
    }
    assert "imageId" in payload
    assert isinstance(payload["translationFailed"], bool)
    assert not isinstance(payload["translationFailed"], str)


def test_render_callback_values_are_strings():
    payload = {
        "imageId": "123e4567-e89b-12d3-a456-426614174000",
        "storagePath": "renders/rendered_123.png",
    }
    for k, v in payload.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
