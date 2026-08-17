"""Local (PaddleOCR) model catalog: det+rec pairs resolved per source language.

PaddleOCR ships text *detection* and text *recognition* as two separate models, and only the
recognition model carries a character set. PP-OCRv6 has no Korean recognition model at all — its
supported set is ch / chinese_cht / en / japan plus Latin (see paddleocr's own
``_pipelines/ocr.py::_get_ocr_model_names``). Running Korean through PP-OCRv6_medium_rec therefore
does not fail, it *hallucinates*: Hangul comes back as CJK and Latin noise. PP-OCRv5 does have
``korean_PP-OCRv5_mobile_rec``.

The worker used to read one global ``PADDLEOCR_DET_MODEL``/``PADDLEOCR_REC_MODEL`` pair from the
environment and apply it to every language, which is what made Korean pages unreadable. This module
replaces that with a catalog, declared in ``providers.json`` under the ``local`` provider exactly
like every cloud provider's model list, where each entry names a detection model and a *per-language*
recognition model. Resolution then picks a pair that can actually read the page's script, falling
back to another catalog entry when the selected one cannot (Korean on PP-OCRv6 -> PP-OCRv5).

Both halves of a pair are validated against PaddleX's on-disk model registry before an entry is
offered as a choice, so the UI never lists a model the host cannot load.
"""

import importlib.util
import os
from dataclasses import dataclass, field
from functools import lru_cache

from worker.config import logger

# Recognition models are language-specific, so a catalog entry maps ISO 639-1 source language ->
# recognition model name. The keys of that map *are* the entry's supported languages: an entry
# without a "ko" key cannot read Korean, which is the whole point of this module.
#
# These built-ins mirror upstream PaddleOCR's own language->model routing and are used only when
# providers.json declares no `local` provider, so an existing deployment keeps working unchanged.
BUILTIN_LOCAL_OCR_MODELS: list[dict] = [
    {
        "id": "PP-OCRv6",
        "name": "PP-OCRv6 Medium",
        "det": "PP-OCRv6_medium_det",
        "rec": {
            "ja": "PP-OCRv6_medium_rec",
            "zh": "PP-OCRv6_medium_rec",
            "zh-cn": "PP-OCRv6_medium_rec",
            "zh-tw": "PP-OCRv6_medium_rec",
            "en": "PP-OCRv6_medium_rec",
        },
    },
    # The detector is deliberately the *mobile* one even though ja/zh recognition is server-tier:
    # PP-OCRv5_server_det peaks at ~3.9 GB RSS on a 1024px page, which exceeds the worker's 4 GiB
    # cap once the process baseline is counted and OOM-kills the whole container mid-page. The
    # mobile detector peaks at ~0.5 GB for effectively the same boxes.
    {
        "id": "PP-OCRv5",
        "name": "PP-OCRv5",
        "det": "PP-OCRv5_mobile_det",
        "rec": {
            "ja": "PP-OCRv5_server_rec",
            "zh": "PP-OCRv5_server_rec",
            "zh-cn": "PP-OCRv5_server_rec",
            "zh-tw": "PP-OCRv5_server_rec",
            "ko": "korean_PP-OCRv5_mobile_rec",
            "en": "en_PP-OCRv5_mobile_rec",
        },
    },
]

# Order in which to look for a replacement when the requested entry cannot read the language. The
# catalog's own declaration order wins; this is only a tiebreak so behaviour is deterministic.
_DEFAULT_MODEL_ID = "PP-OCRv6"


def normalize_lang(source_language: str | None) -> str:
    """Normalise a source language to the lowercase ISO key used by catalog rec maps."""
    return (source_language or "ja").strip().lower()


def env_override() -> tuple[str, str] | None:
    """The explicit det/rec pair pinned by the environment, if any.

    Both halves must be set: pinning only one would silently pair a custom model with a catalog
    default. This is the manual escape hatch — it is deliberately *not* set by docker-compose any
    more, because a value that is always present cannot express "let the worker choose".
    """
    det = os.environ.get("PADDLEOCR_DET_MODEL", "").strip()
    rec = os.environ.get("PADDLEOCR_REC_MODEL", "").strip()
    return (det, rec) if det and rec else None


@dataclass(frozen=True)
class LocalOcrModel:
    """One selectable local OCR choice: a detection model plus per-language recognition models."""

    id: str
    name: str
    det: str
    rec: dict[str, str]
    free: bool = True

    @property
    def languages(self) -> list[str]:
        """ISO codes this entry can actually recognise, in declaration order."""
        return list(self.rec.keys())

    def supports(self, source_language: str | None) -> bool:
        return normalize_lang(source_language) in self.rec

    def rec_for(self, source_language: str | None) -> str | None:
        return self.rec.get(normalize_lang(source_language))

    @property
    def rec_models(self) -> list[str]:
        """Every distinct recognition model this entry can load, in declaration order."""
        seen: dict[str, None] = {}
        for model in self.rec.values():
            seen.setdefault(model, None)
        return list(seen)

    @property
    def default_rec(self) -> str | None:
        """The recognition model to reach for when the language is not one this entry names."""
        recs = self.rec_models
        return recs[0] if recs else None

    @property
    def display_name(self) -> str:
        """Label shown in the UI. Names *both* halves, because a det model alone says nothing
        about which scripts the choice can read — the recognition model is what has the charset."""
        recs = self.rec_models
        if len(recs) == 1:
            return f"{self.name} ({self.det} + {recs[0]})"
        return f"{self.name} ({self.det} + per-language rec)"


@dataclass(frozen=True)
class ResolvedOcrModel:
    """The concrete det/rec pair to hand PaddleOCR for one job."""

    model_id: str
    det: str
    rec: str
    language: str
    #: True when the requested entry could not read this language and another was substituted.
    auto_routed: bool = False
    requested_model_id: str | None = None

    @property
    def cache_key(self) -> str:
        return f"{self.model_id}:{self.det}:{self.rec}"


@lru_cache(maxsize=1)
def _paddlex_config_root() -> str | None:
    """Locate PaddleX's bundled module configs without importing PaddleX (which is expensive)."""
    try:
        spec = importlib.util.find_spec("paddlex")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = os.path.join(spec.submodule_search_locations[0], "configs", "modules")
    return root if os.path.isdir(root) else None


@lru_cache(maxsize=2)
def _known_model_names(kind: str) -> frozenset[str]:
    """Model names PaddleX knows about for ``kind`` ("text_detection" / "text_recognition").

    Empty when the registry cannot be read, which callers treat as "cannot validate" rather than
    "nothing is valid" — a worker without PaddleX installed (unit tests) must not publish an empty
    catalog.
    """
    root = _paddlex_config_root()
    if not root:
        return frozenset()
    directory = os.path.join(root, kind)
    try:
        return frozenset(f.removesuffix(".yaml") for f in os.listdir(directory) if f.endswith(".yaml"))
    except OSError:
        return frozenset()


def validation_available() -> bool:
    """True when both halves of a pair can be checked against the PaddleX registry."""
    return bool(_known_model_names("text_detection")) and bool(_known_model_names("text_recognition"))


def missing_model_files(model: LocalOcrModel) -> list[str]:
    """Names from *model* that PaddleX does not recognise. Empty when it is fully loadable.

    Checks the detection model *and* every recognition model, because a pair whose rec half is
    missing loads fine and then silently transcribes nothing useful.
    """
    if not validation_available():
        return []

    dets = _known_model_names("text_detection")
    recs = _known_model_names("text_recognition")

    missing = []
    if model.det not in dets:
        missing.append(model.det)
    missing.extend(name for name in model.rec_models if name not in recs)
    return missing


def parse_local_ocr_models(entries: list[dict] | None) -> list[LocalOcrModel]:
    """Build catalog entries from providers.json ``local.models.ocr`` rows.

    ``rec`` accepts either a language map or a bare string; a bare string needs a ``langs`` list to
    say which languages it covers, since a recognition model's charset is not discoverable from its
    name.
    """
    models: list[LocalOcrModel] = []
    for entry in entries or []:
        model_id = str(entry.get("id") or "").strip()
        det = str(entry.get("det") or "").strip()
        raw_rec = entry.get("rec")

        if not model_id or not det or not raw_rec:
            logger.warning(f"[OCR Models] Skipping local OCR entry {entry!r}: needs 'id', 'det' and 'rec'.")
            continue

        if isinstance(raw_rec, str):
            langs = entry.get("langs") or []
            if not langs:
                logger.warning(
                    f"[OCR Models] Skipping local OCR entry '{model_id}': a string 'rec' also needs "
                    "'langs' listing the languages it can read."
                )
                continue
            rec = {normalize_lang(lang): raw_rec.strip() for lang in langs}
        elif isinstance(raw_rec, dict):
            rec = {normalize_lang(lang): str(name).strip() for lang, name in raw_rec.items() if name}
        else:
            logger.warning(f"[OCR Models] Skipping local OCR entry '{model_id}': 'rec' must be a string or object.")
            continue

        if not rec:
            logger.warning(f"[OCR Models] Skipping local OCR entry '{model_id}': no recognition models resolved.")
            continue

        models.append(
            LocalOcrModel(
                id=model_id,
                name=str(entry.get("name") or model_id),
                det=det,
                rec=rec,
                free=bool(entry.get("free", True)),
            )
        )
    return models


@dataclass
class LocalOcrCatalog:
    """The set of local OCR choices this host can offer, and the language routing over them."""

    models: list[LocalOcrModel] = field(default_factory=list)

    @classmethod
    def from_entries(cls, entries: list[dict] | None) -> "LocalOcrCatalog":
        parsed = parse_local_ocr_models(entries)
        if not parsed:
            parsed = parse_local_ocr_models(BUILTIN_LOCAL_OCR_MODELS)
        return cls(models=parsed)

    def available(self) -> list[LocalOcrModel]:
        """Entries whose det *and* rec models are all present in the PaddleX registry."""
        usable = []
        for model in self.models:
            missing = missing_model_files(model)
            if missing:
                logger.warning(
                    f"[OCR Models] Local OCR choice '{model.id}' is unavailable — PaddleX does not "
                    f"provide: {', '.join(missing)}. It will not be offered as a choice."
                )
                continue
            usable.append(model)
        return usable

    def get(self, model_id: str | None) -> LocalOcrModel | None:
        if not model_id:
            return None
        wanted = model_id.strip().lower()
        for model in self.available():
            if model.id.lower() == wanted:
                return model
        return None

    def default_for(self, source_language: str | None) -> LocalOcrModel | None:
        """The entry to use when nothing was explicitly selected.

        Prefers the configured default when it can read the language, then falls back to declaration
        order — which is what routes Korean to PP-OCRv5 without anyone having to ask for it.
        """
        usable = self.available()
        if not usable:
            return None

        preferred_id = os.environ.get("PADDLEOCR_LOCAL_MODEL", "").strip() or _DEFAULT_MODEL_ID
        preferred = next((m for m in usable if m.id.lower() == preferred_id.lower()), None)
        if preferred is not None and preferred.supports(source_language):
            return preferred

        for model in usable:
            if model.supports(source_language):
                return model
        return preferred or usable[0]

    def rec_covers(self, rec_name: str, lang: str) -> bool | None:
        """Whether *rec_name* is declared for *lang*.

        Returns None when no catalog entry mentions the model at all — an operator's own custom
        recognition model, whose language coverage this code has no way to know and must not
        second-guess.
        """
        known = False
        for model in self.models:
            if rec_name in model.rec_models:
                known = True
                if model.rec.get(lang) == rec_name:
                    return True
        return False if known else None

    def resolve(self, model_id: str | None, source_language: str | None) -> ResolvedOcrModel | None:
        """Pick the det/rec pair to run for *source_language*.

        An explicit *model_id* is honoured when that entry can read the language. When it cannot —
        the PP-OCRv6/Korean case — routing falls through to an entry that can, loudly, rather than
        returning a pair that would transcribe Hangul as noise.

        Precedence is: explicit per-job choice, then the environment pin, then the catalog default.
        """
        lang = normalize_lang(source_language)

        requested = self.get(model_id)

        # An explicit per-job choice outranks the environment pin. The pin is a deployment-wide
        # *default* for jobs that express no preference, not a veto over one that does: a caller
        # that asks for PP-OCRv5 on a page has to get PP-OCRv5, or the choice is decorative.
        if requested is not None and requested.supports(lang):
            rec = requested.rec_for(lang)
            assert rec is not None  # supports() just checked the key exists
            return ResolvedOcrModel(
                model_id=requested.id,
                det=requested.det,
                rec=rec,
                language=lang,
                requested_model_id=requested.id,
            )

        # An environment pin beats the catalog default, except when the catalog positively knows
        # the pinned recognition model cannot read this script. Honouring it there would reinstate
        # the exact bug this module exists to fix for every deployment that still has the old vars
        # in its .env.
        override = env_override()
        if override is not None:
            det, rec = override
            if self.rec_covers(rec, lang) is not False:
                return ResolvedOcrModel(model_id="env-override", det=det, rec=rec, language=lang)
            logger.warning(
                f"[OCR Models] PADDLEOCR_DET_MODEL/PADDLEOCR_REC_MODEL pin '{det}' + '{rec}', but "
                f"'{rec}' has no character set for '{lang}'. Ignoring the pin for this job and "
                "choosing a model that can read it. Unset those vars to silence this."
            )

        chosen = self.default_for(lang)
        if chosen is None:
            return None

        rec = chosen.rec_for(lang)
        if rec is None:
            # No entry claims this language. That is not necessarily fatal — PP-OCRv6 reads the whole
            # Latin set without the catalog enumerating every one of them — so fall back to the
            # entry's own default pair rather than failing the job outright. Languages that *are*
            # claimed by another entry (Korean, by PP-OCRv5) never reach here: default_for picked
            # that entry above.
            rec = chosen.default_rec
            if rec is None:
                logger.error(
                    f"[OCR Models] No local OCR model can recognise language '{lang}'. "
                    f"Available: {', '.join(m.id for m in self.available())}."
                )
                return None
            logger.warning(
                f"[OCR Models] No local OCR entry declares language '{lang}'; falling back to "
                f"'{chosen.id}' ({chosen.det} + {rec}). Add '{lang}' to that entry's 'rec' map in "
                "providers.json if the transcription comes back wrong."
            )

        auto_routed = False
        requested_id = None
        if requested is not None and requested.id != chosen.id:
            auto_routed = True
            requested_id = requested.id
            logger.warning(
                f"[OCR Models] Local OCR model '{requested.id}' has no recognition model for "
                f"'{lang}' (it covers {', '.join(requested.languages)}); routing to '{chosen.id}' "
                f"({chosen.det} + {rec}) instead so the text is actually readable."
            )
        elif model_id and requested is None:
            logger.warning(
                f"[OCR Models] Local OCR model '{model_id}' is not a known local choice; falling back to '{chosen.id}'."
            )

        return ResolvedOcrModel(
            model_id=chosen.id,
            det=chosen.det,
            rec=rec,
            language=lang,
            auto_routed=auto_routed,
            requested_model_id=requested_id,
        )
