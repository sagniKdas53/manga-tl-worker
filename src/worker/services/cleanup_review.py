"""Bounded local evidence check for a CTD-empty region; never authorizes erasure."""

import numpy as np

from worker.model_manager import get_local_ocr_backend, model_manager
from worker.services.ocr import parse_paddle_ocr_results, parse_rapid_ocr_results
from worker.utils.image import downscale_for_ocr


def assess_empty_mask(image, geometry, source_language="ja", ocr_model="", expected_text="") -> list[str]:
    """Recheck one native-scale crop using the existing local OCR model.

    Glyph size is an estimate from the recognized line geometry, not a detector verdict.
    A mismatch is evidence of unstable OCR, not proof that an image contains no text.
    The caller already holds the node OCR lock. No cloud calls or repeated CTD passes.
    """
    x, y, width, height = geometry
    x0, y0 = max(0, int(x) - 32), max(0, int(y) - 32)
    crop = image[y0 : min(image.shape[0], int(y + height) + 32), x0 : min(image.shape[1], int(x + width) + 32)]
    if not crop.size:
        return ["Local OCR check: invalid crop; geometry needs review"]
    crop, source_scale = downscale_for_ocr(crop, max_dim=1024)
    try:
        backend = get_local_ocr_backend()
        if backend == "rapidocr":
            reader = model_manager.get_rapid_ocr_reader(source_language)
            if reader is None:
                raise RuntimeError("Local OCR reader unavailable")
            results = parse_rapid_ocr_results(reader(crop))
        else:
            reader = model_manager.get_paddle_ocr_reader(source_language, ocr_model)
            if reader is None:
                raise RuntimeError("Local OCR reader unavailable")
            results = parse_paddle_ocr_results(reader.predict(crop))
        matched = []
        for quad, text, score in results:
            points = np.asarray(quad, dtype=float) * source_scale
            if points.shape != (4, 2) or not np.isfinite(points).all() or not text.strip():
                continue
            cx, cy = np.add(points.mean(axis=0), (x0, y0))
            if not (x <= cx <= x + width and y <= cy <= y + height):
                continue
            edges = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
            glyph_px = min(float(edges.min()), float(edges.max()) / max(1, len(text.strip())))
            matched.append((text, float(score), glyph_px))
        if not matched:
            return [
                "Local OCR check: no text confirmed inside the region; possible false-positive OCR or detector miss"
            ]
        glyph_px = min(item[2] for item in matched)
        score = max(item[1] for item in matched)
        scale = "small-text candidate" if glyph_px < 16 else "not a small-text candidate"
        diagnostics = [f"Local OCR check: estimated glyph size {glyph_px:.1f}px ({scale}); confidence {score:.2f}"]
        expected = "".join(c for c in expected_text if c.isalnum()).casefold()
        recognized = "".join(c for text, _, _ in matched for c in text if c.isalnum()).casefold()
        if expected and expected == recognized:
            diagnostics.append(
                "Local OCR agrees with the original text; CTD missed lettering, source retained for review"
            )
        elif expected:
            diagnostics.append(
                "Local OCR disagrees with the original text; possible false-positive region or recognition error"
            )
        else:
            diagnostics.append(
                "Original OCR text unavailable for comparison; recognition alone cannot authorize cleanup"
            )
        return diagnostics
    except Exception as exc:
        return [f"Local OCR check unavailable ({type(exc).__name__}); source retained for review"]
