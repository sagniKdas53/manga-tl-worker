"""Local OCR catalog: language routing, det+rec validation, and the environment escape hatch."""

import pytest

from worker.ocr_models import (
    BUILTIN_LOCAL_OCR_MODELS,
    LocalOcrCatalog,
    LocalOcrModel,
    parse_local_ocr_models,
)

V6 = "PP-OCRv6"
V5 = "PP-OCRv5"


@pytest.fixture
def catalog():
    return LocalOcrCatalog.from_entries(BUILTIN_LOCAL_OCR_MODELS)


@pytest.fixture(autouse=True)
def _clear_env_pin(monkeypatch):
    """The env pin short-circuits resolution, so it must not leak in from the developer's shell."""
    monkeypatch.delenv("PADDLEOCR_DET_MODEL", raising=False)
    monkeypatch.delenv("PADDLEOCR_REC_MODEL", raising=False)
    monkeypatch.delenv("PADDLEOCR_LOCAL_MODEL", raising=False)


def test_v6_does_not_claim_korean():
    """The premise of the whole module: PP-OCRv6 has no Korean recognition model."""
    v6 = next(m for m in parse_local_ocr_models(BUILTIN_LOCAL_OCR_MODELS) if m.id == V6)
    assert not v6.supports("ko")
    v5 = next(m for m in parse_local_ocr_models(BUILTIN_LOCAL_OCR_MODELS) if m.id == V5)
    assert v5.supports("ko")


def test_korean_routes_to_v5(catalog):
    resolved = catalog.resolve(None, "ko")
    assert resolved is not None
    assert resolved.model_id == V5
    assert resolved.rec == "korean_PP-OCRv5_mobile_rec"


def test_japanese_stays_on_v6(catalog):
    """v6 remains the default everywhere it is actually capable — this is not a blanket downgrade."""
    resolved = catalog.resolve(None, "ja")
    assert resolved is not None
    assert resolved.model_id == V6
    assert resolved.rec == "PP-OCRv6_medium_rec"


def test_explicit_choice_is_honoured_when_capable(catalog):
    resolved = catalog.resolve(V5, "ja")
    assert resolved is not None
    assert resolved.model_id == V5
    assert resolved.rec == "PP-OCRv5_server_rec"
    assert not resolved.auto_routed


def test_explicit_choice_is_overridden_when_incapable(catalog):
    resolved = catalog.resolve(V6, "ko")
    assert resolved is not None
    assert resolved.model_id == V5
    assert resolved.auto_routed
    assert resolved.requested_model_id == V6


def test_unknown_language_falls_back_rather_than_failing(catalog):
    """PP-OCRv6 reads Latin scripts the catalog does not enumerate, so an unlisted language must
    degrade to the default pair instead of failing the job."""
    resolved = catalog.resolve(None, "fr")
    assert resolved is not None
    assert resolved.model_id == V6
    assert resolved.rec == "PP-OCRv6_medium_rec"


def test_language_casing_and_blank_are_normalised(catalog):
    assert catalog.resolve(None, "KO") is not None
    assert catalog.resolve(None, "KO").rec == "korean_PP-OCRv5_mobile_rec"  # type: ignore[union-attr]
    # A missing language defaults to Japanese, the pipeline's historical default.
    assert catalog.resolve(None, None).model_id == V6  # type: ignore[union-attr]


def test_readers_are_cached_per_pair_not_per_language(catalog):
    ja = catalog.resolve(None, "ja")
    ko = catalog.resolve(None, "ko")
    assert ja is not None and ko is not None
    assert ja.cache_key != ko.cache_key


class TestValidation:
    """Both halves of a pair are checked, because a missing rec half fails silently at runtime."""

    def test_entry_with_unknown_rec_is_not_offered(self):
        catalog = LocalOcrCatalog.from_entries(
            [
                {
                    "id": "bogus",
                    "name": "Bogus",
                    "det": "PP-OCRv6_medium_det",
                    "rec": {"ja": "definitely_not_a_real_rec_model"},
                }
            ]
        )
        assert [m.id for m in catalog.available()] == []

    def test_entry_with_unknown_det_is_not_offered(self):
        catalog = LocalOcrCatalog.from_entries(
            [
                {
                    "id": "bogus",
                    "name": "Bogus",
                    "det": "definitely_not_a_real_det_model",
                    "rec": {"ja": "PP-OCRv6_medium_rec"},
                }
            ]
        )
        assert [m.id for m in catalog.available()] == []

    def test_builtin_pairs_are_all_available(self, catalog):
        """Guards against a PaddleOCR upgrade quietly dropping a model the catalog still names."""
        assert {m.id for m in catalog.available()} == {V6, V5}


class TestParsing:
    def test_string_rec_requires_langs(self):
        assert parse_local_ocr_models([{"id": "x", "det": "d", "rec": "r"}]) == []

    def test_string_rec_with_langs_expands(self):
        models = parse_local_ocr_models([{"id": "x", "det": "d", "rec": "r", "langs": ["ja", "KO"]}])
        assert models[0].rec == {"ja": "r", "ko": "r"}

    def test_incomplete_entries_are_skipped(self):
        assert parse_local_ocr_models([{"id": "x"}, {"det": "d", "rec": "r"}, {}]) == []

    def test_empty_config_falls_back_to_builtins(self):
        """A deployment with no `local` block in providers.json still gets working choices."""
        assert {m.id for m in LocalOcrCatalog.from_entries(None).models} == {V6, V5}


class TestDisplayName:
    """The UI showed only the rec model, which says nothing about detection. Show both."""

    def test_uniform_pair_names_both_halves(self):
        model = LocalOcrModel(id="x", name="X", det="D", rec={"ja": "R"})
        assert model.display_name == "X (D + R)"

    def test_multi_rec_pair_is_summarised(self):
        model = LocalOcrModel(id="x", name="X", det="D", rec={"ja": "R1", "ko": "R2"})
        assert model.display_name == "X (D + per-language rec)"


class TestEnvironmentPin:
    def test_pin_is_honoured_when_it_can_read_the_language(self, catalog, monkeypatch):
        monkeypatch.setenv("PADDLEOCR_DET_MODEL", "PP-OCRv5_server_det")
        monkeypatch.setenv("PADDLEOCR_REC_MODEL", "PP-OCRv5_server_rec")
        resolved = catalog.resolve(None, "ja")
        assert resolved is not None
        assert resolved.rec == "PP-OCRv5_server_rec"

    def test_pin_is_ignored_when_it_cannot_read_the_language(self, catalog, monkeypatch):
        """An existing .env pinning the v6 pair must not keep breaking Korean after this fix."""
        monkeypatch.setenv("PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det")
        monkeypatch.setenv("PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec")
        resolved = catalog.resolve(None, "ko")
        assert resolved is not None
        assert resolved.rec == "korean_PP-OCRv5_mobile_rec"

    def test_custom_unknown_model_is_left_alone(self, catalog, monkeypatch):
        """An operator's own model has unknowable coverage, so it is never second-guessed."""
        monkeypatch.setenv("PADDLEOCR_DET_MODEL", "my_det")
        monkeypatch.setenv("PADDLEOCR_REC_MODEL", "my_rec")
        resolved = catalog.resolve(None, "ko")
        assert resolved is not None
        assert resolved.rec == "my_rec"

    def test_half_a_pin_is_ignored(self, catalog, monkeypatch):
        monkeypatch.setenv("PADDLEOCR_DET_MODEL", "my_det")
        resolved = catalog.resolve(None, "ja")
        assert resolved is not None
        assert resolved.model_id == V6

    def test_local_model_env_selects_default(self, catalog, monkeypatch):
        monkeypatch.setenv("PADDLEOCR_LOCAL_MODEL", V5)
        resolved = catalog.resolve(None, "ja")
        assert resolved is not None
        assert resolved.model_id == V5
