import io
import logging
import math
import os

import requests
from PIL import Image, ImageDraw, ImageFont

from worker.config import (
    CALLBACK_URL,
    CONTRAST_FLOOR,
    backend_headers,
    minio_client,
    redis_client,
)
from worker.utils.image import download_image

logger = logging.getLogger(__name__)

# Font registry: map display names to filesystem paths
FONT_REGISTRY = {
    "Comic Neue": {
        "normal": "/usr/share/fonts/opentype/comic-neue/ComicNeue-Regular.otf",
        "bold": "/usr/share/fonts/opentype/comic-neue/ComicNeue-Bold.otf",
        "italic": "/usr/share/fonts/opentype/comic-neue/ComicNeue-Italic.otf",
        "bolditalic": "/usr/share/fonts/opentype/comic-neue/ComicNeue-BoldItalic.otf",
    },
    "Bangers": {
        "normal": "/usr/share/fonts/truetype/google/Bangers-Regular.ttf",
        "bold": "/usr/share/fonts/truetype/google/Bangers-Regular.ttf",  # Bangers has one weight
        "italic": "/usr/share/fonts/truetype/google/Bangers-Regular.ttf",
        "bolditalic": "/usr/share/fonts/truetype/google/Bangers-Regular.ttf",
    },
    "Luckiest Guy": {
        "normal": "/usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf",
        "bold": "/usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf",
        "italic": "/usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf",
        "bolditalic": "/usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf",
    },
    # AUDIT-D2: the display names stay "Arial" and "Courier New" because they are persisted
    # on existing layers — renaming the keys would strand every chapter already using them.
    # The files behind them are now Liberation Sans and Liberation Mono, metric-compatible
    # substitutes with the same advance widths, so fitting and line breaking are unchanged.
    # The originals were Monotype fonts scraped from third-party repos that had no right to
    # redistribute them. Unlike those single files, Liberation has genuine bold and italic
    # faces, so the four style keys below now resolve to four different fonts.
    "Arial": {
        "normal": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "italic": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "bolditalic": "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    },
    "Courier New": {
        "normal": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "bold": "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "italic": "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
        "bolditalic": "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
    },
    "WenQuanYi Micro Hei": {
        "normal": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "bold": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "italic": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "bolditalic": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    },
    "NanumGothic": {
        "normal": "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "bold": "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "italic": "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "bolditalic": "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    },
    "IPAGothic": {
        "normal": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "bold": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "italic": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "bolditalic": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    },
}
DEFAULT_FONT_FALLBACK_ORDER = [
    "Comic Neue",
    "Luckiest Guy",
    "Bangers",
    "IPAGothic",
    "WenQuanYi Micro Hei",
    "NanumGothic",
]


def load_font(size, font_name="Comic Neue", bold=False, italic=False):
    # Determine the style key
    if bold and italic:
        style_key = "bolditalic"
    elif bold:
        style_key = "bold"
    elif italic:
        style_key = "italic"
    else:
        style_key = "normal"

    # Helper function to load a font from a path
    def try_load(path):
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, int(size))
            except Exception:
                pass
        return None

    # 1. Try requested font from registry
    if font_name in FONT_REGISTRY:
        path = FONT_REGISTRY[font_name].get(style_key) or FONT_REGISTRY[font_name].get("normal")
        font = try_load(path)
        if font:
            return font

    # 2. Try fallbacks from registry in order
    for fallback in DEFAULT_FONT_FALLBACK_ORDER:
        if fallback in FONT_REGISTRY:
            path = FONT_REGISTRY[fallback].get(style_key) or FONT_REGISTRY[fallback].get("normal")
            font = try_load(path)
            if font:
                return font

    # 3. Fallback to the original system fonts
    font_paths = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ),
    ]
    for path in font_paths:
        font = try_load(path)
        if font:
            return font

    # Try general system names via suffixes
    if bold and italic:
        suffixes = ["BoldItalic.ttf", "-BoldItalic.ttf", "BI.ttf"]
    elif bold:
        suffixes = ["Bold.ttf", "-Bold.ttf", "B.ttf"]
    elif italic:
        suffixes = ["Italic.ttf", "-Italic.ttf", "I.ttf"]
    else:
        suffixes = ["Regular.ttf", ".ttf", "R.ttf"]
    font_names = ["DejaVuSans", "LiberationSans", "FreeSans", "Arial"]
    for name in font_names:
        for suffix in suffixes:
            try:
                return ImageFont.truetype(f"{name}{suffix}", int(size))
            except Exception:
                pass

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def wrap_text(text, font, max_width):
    if not text:
        return []
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join([*current_line, word])
        try:
            w = font.getbbox(test_line)[2]
        except Exception:
            try:
                w = font.getsize(test_line)[0]
            except Exception:
                w = len(test_line) * 6

        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def draw_wrapped_text(draw, text, font, text_color, x, y, max_width, max_height, alignment="center"):
    lines = wrap_text(text, font, max_width)
    if not lines:
        return

    line_heights = []
    for line in lines:
        try:
            bbox = font.getbbox(line)
            line_heights.append(bbox[3] - bbox[1] + 2)
        except Exception:
            try:
                line_heights.append(font.getsize(line)[1] + 2)
            except Exception:
                line_heights.append(14)

    total_height = sum(line_heights)
    start_y = y + (max_height - total_height) / 2

    current_y = start_y
    for i, line in enumerate(lines):
        try:
            line_width = font.getbbox(line)[2]
        except Exception:
            try:
                line_width = font.getsize(line)[0]
            except Exception:
                line_width = len(line) * 6

        if alignment == "center":
            line_x = x + (max_width - line_width) / 2
        elif alignment == "right":
            line_x = x + max_width - line_width
        else:
            line_x = x

        draw.text((line_x, current_y), line, fill=text_color, font=font)
        current_y += line_heights[i]


# Where across a line's box the mask is measured, as fractions of the box.
#
# The centre alone is what D6 was: it misses that a curved balloon closes in above and below the
# midline, so a line sized to its own midline poked out through the outline. Measuring the whole
# 1.2em line box is the opposite error -- the top and bottom of that box are leading, no glyph
# reaches them, and a balloon that pinches at a tail row there would veto a line that in fact has
# nothing to draw at that height. These fractions bound the band the ink actually occupies,
# ascender to descender.
BAND_SAMPLE_FRACTIONS = (0.15, 0.35, 0.5, 0.65, 0.85)


# Hyphenation dictionaries are a few hundred KB each and stateless once built.
_HYPHENATORS: dict[str, object] = {}
# Typographic minimums: never leave fewer than 2 characters before the break or 3 after it.
HYPHEN_MIN_LEFT = 2
HYPHEN_MIN_RIGHT = 3


def hyphen_positions(word, lang="en_US"):
    """Indices in `word` where a hyphen may legally be inserted, ascending.

    Positions are computed on the word's alphabetic core, so surrounding punctuation cannot buy a
    break that the letters do not justify -- `"Otaku-kun` would otherwise be offered a break after
    `"O`, the quote counting toward the two-character minimum.
    """
    if not word:
        return []
    start = 0
    end = len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    core = word[start:end]
    if len(core) < HYPHEN_MIN_LEFT + HYPHEN_MIN_RIGHT:
        return []

    if lang not in _HYPHENATORS:
        try:
            import pyphen

            _HYPHENATORS[lang] = pyphen.Pyphen(lang=lang, left=HYPHEN_MIN_LEFT, right=HYPHEN_MIN_RIGHT)
        except Exception as e:
            # No dictionary for this language, or pyphen missing. Fall back to not hyphenating,
            # which is what the renderer did before this existed.
            logger.warning("Hyphenation unavailable for %s (%s); setting without it", lang, e)
            _HYPHENATORS[lang] = None
    hyphenator = _HYPHENATORS[lang]
    if hyphenator is None:
        return []

    try:
        return [start + p for p in hyphenator.positions(core)]  # type: ignore[attr-defined]
    except Exception:
        return []


def break_word_to_width(word, measure, width, lang="en_US"):
    """Split `word` into a head that fits `width` and the rest.

    Prefers the largest legal hyphenation point, so the head comes back with its hyphen attached.
    Falls back to breaking between characters -- what the renderer did for every over-long word
    before -- only when no legal point fits, which is the narrow-box case where the alternative is
    text outside the region.
    """
    for position in sorted(hyphen_positions(word, lang), reverse=True):
        head = word[:position] + "-"
        if measure(head) <= width:
            return head, word[position:]

    # No legal break fits. Take as many characters as will.
    cut = 1
    while cut < len(word) and measure(word[: cut + 1]) <= width:
        cut += 1
    return word[:cut], word[cut:]


def reassemble(lines):
    """The words a set of wrapped lines carries, with hyphenated breaks put back together.

    A line ending in a hyphen continues into the next without a space. Hyphens are then dropped
    from the comparison entirely, because an inserted hyphen and a word's own hyphen are
    indistinguishable once drawn -- and breaking at an existing hyphen is legal anyway. A break
    that is *not* at a hyphen still shows up as two words and is still caught.
    """
    joined = ""
    for i, line in enumerate(lines):
        joined += line
        if not line.endswith("-") and i < len(lines) - 1:
            joined += " "
    return [w.replace("-", "") for w in joined.split()]


def mask_span_for_band(polygon, box_x, box_width, y_top, y_bottom):
    """The horizontal span a line occupying `y_top..y_bottom` can be set in.

    The mask's widest run at each sampled row, clipped to the box, intersected across the band --
    so the answer is the width available over the line's whole height, not just at its midline.
    Returns None when no sampled row reaches the mask, which is the caller's cue to fall back to
    the box.
    """
    left = None
    right = None
    n_pts = len(polygon)
    for t in BAND_SAMPLE_FRACTIONS:
        y = y_top + t * (y_bottom - y_top)
        crossings = []
        for i in range(n_pts):
            x1, y1 = polygon[i][0], polygon[i][1]
            x2, y2 = polygon[(i + 1) % n_pts][0], polygon[(i + 1) % n_pts][1]
            if (y1 <= y < y2) or (y2 <= y < y1):
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if len(crossings) < 2:
            continue
        crossings.sort()
        row = None
        widest = 0
        for i in range(0, len(crossings) - 1, 2):
            run_left = max(crossings[i], box_x)
            run_right = min(crossings[i + 1], box_x + box_width)
            if run_right - run_left > widest:
                widest = run_right - run_left
                row = (run_left, run_right)
        if row is None:
            continue
        left = row[0] if left is None else max(left, row[0])
        right = row[1] if right is None else min(right, row[1])

    if left is None or right is None or right - left <= 0:
        return None
    return (left, right)


# Glyphs drawn past the erased plate need their own backdrop.
#
# `free_text_box` (backend `coordinator.rs`) squares a tall vertical Japanese column into a box
# English can be set across -- equal area, up to 2.5x the column's width. The erase plate does not
# grow with it; it stays the tight shape that covered the source text. For a balloon the two agree,
# because the box is an inset of the bubble the plate fills. For free-floating text they do not, and
# the difference is drawn straight onto artwork -- 329 of the 552 free-floating elements in the
# 400-page corpus escape their plate, by a median 42% of the box width. HKXfexLbAAAN7IE p4 is the
# case: a 91x293 column becomes a 186x187 box, and half of "Talk about peak laziness!" lands on the
# character's bow and hand.
#
# Widening the plate to the box instead would cover her chin and shoulder, and clamping the box back
# to the plate costs five font sizes and a hyphen break. Outlining the glyphs is what a human
# typesetter does with a caption over art: the text reads anywhere and the art survives.
#
# The stroke is the plate's own colour, so it can only be seen where the pixels differ from the
# plate -- which is exactly where the plate is not.
#
# Gated on the region having no detected bubble, and not on the geometry alone. A box inside a
# balloon also escapes its mask, because the box is the bubble inset while the mask hugs the
# glyphs; on the corpus that is 1122 of the 1504 elements a geometry-only gate would outline. There
# the overhang lands on the balloon's own blank interior and needs no backdrop, and stroking it
# would bet 4 px around every letter on `backgroundColor` having sampled the interior exactly.
HALO_STROKE_RATIO = 0.10
HALO_MIN_STROKE = 2
HALO_TOLERANCE = 2.0


def has_detected_bubble(region):
    """The backend's `has_detected_bubble`, in Python.

    bubble* describes a real container only when it is larger than the bbox; when the two match it
    is the echo the worker writes for text YOLO matched to no balloon.
    """
    bubble_w = region.get("bubbleW")
    bubble_h = region.get("bubbleH")
    if bubble_w is None or bubble_h is None:
        return False
    return not (bubble_w == region.get("bboxW") and bubble_h == region.get("bboxH"))


def halo_stroke_for(mask_polygon, box, font_size, in_bubble=False):
    """Stroke width for text at `box` given the erased `mask_polygon`, or 0 when none is needed.

    `box` is `(x, y, w, h)` in page pixels. Returns 0 when the box sits within the mask's extent --
    the free-text case where the column was never widened -- and 0 for anything inside a detected
    balloon, where whatever the box overhangs is the balloon's own blank interior.
    """
    if not font_size or font_size <= 0:
        return 0
    if in_bubble:
        return 0

    stroke = max(HALO_MIN_STROKE, round(font_size * HALO_STROKE_RATIO))

    points = mask_polygon
    if isinstance(points, str):
        import json

        try:
            points = json.loads(points)
        except Exception:
            return 0
    if not isinstance(points, list) or not points:
        # Nothing was erased, so every glyph is sitting on artwork.
        return stroke

    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return 0

    x, y, w, h = (float(v) for v in box)
    escapes = (
        x < min(xs) - HALO_TOLERANCE
        or y < min(ys) - HALO_TOLERANCE
        or x + w > max(xs) + HALO_TOLERANCE
        or y + h > max(ys) + HALO_TOLERANCE
    )
    return stroke if escapes else 0


def fit_text_in_box_py(
    text,
    max_width,
    max_height,
    font_name,
    default_font_size=16,
    shape="rectangular",
    box_x=0,
    box_y=0,
    mask_polygon=None,
    bold=False,
    italic=False,
    lang="en_US",
):
    clean_text = (text or "").replace("\r\n", "\n")
    paragraphs = clean_text.split("\n")

    polygon_points = None
    if mask_polygon:
        try:
            import json

            parsed = json.loads(mask_polygon) if isinstance(mask_polygon, str) else mask_polygon
            if isinstance(parsed, list) and all(isinstance(p, list) and len(p) == 2 for p in parsed):
                polygon_points = parsed
        except Exception:
            pass

    # The mask is a flow constraint only when it is wider than the box it is being flowed into.
    #
    # It serves two purposes at once: it is the shape painted over the source text, and -- below --
    # the shape the lines are wrapped to. Those agree for a balloon, where the mask is the outline
    # and the box is an inset of it. They disagree for free-floating text, where the mask is the
    # tight rectangle round the vertical Japanese column: wrapping to it would set English back down
    # that column and undo the box the caller asked for. When the mask does not span the box, the box
    # is the deliberate one of the two, so the mask keeps its erasing job and loses its typesetting
    # one.
    if polygon_points:
        xs = [p[0] for p in polygon_points]
        if min(xs) > box_x + 2 or max(xs) < box_x + max_width - 2:
            polygon_points = None

    # One font instance per size for this box. The size search wraps and then measures at each
    # candidate size, and every load_font miss is a font file read; there are a dozen or so
    # candidates per box. Scoped to this call so it stays a pure function of its arguments.
    fonts = {}

    def font_at(f_size):
        if f_size not in fonts:
            fonts[f_size] = load_font(f_size, font_name=font_name, bold=bold, italic=italic)
        return fonts[f_size]

    def wrap_text_py(txt, f_size):
        font = font_at(f_size)
        if not font:
            return {"lines": [txt], "line_centers": [box_x + max_width / 2]}

        def get_text_width(t):
            try:
                return font.getlength(t)
            except Exception:
                try:
                    bbox = font.getbbox(t)
                    return bbox[2] - bbox[0]
                except Exception:
                    try:
                        return font.getsize(t)[0]  # type: ignore
                    except Exception:
                        return len(t) * (f_size * 0.5)

        # 1. Polygon-aware wrapping
        if polygon_points and len(polygon_points) > 0:
            line_height = f_size * 1.2

            def try_wrap_for_n_lines(N):
                tentative_lines = []
                tentative_centers = []
                line_index = 0
                current_line = ""

                def get_line_span(idx):
                    """The span a line at `idx` can actually occupy.

                    D6 (docs/render_quality_gap_2026-08-05.md): this used to measure the mask at
                    the line's *centre* y, but glyphs occupy the whole line band, and a curved
                    balloon is narrower at the band's edges than at its middle. A line sized to
                    the centre row therefore pokes out through the outline above and below its
                    own midline -- 286 lines across the 40-page corpus, in 137 of 351 elements.
                    Sampling the band and keeping the narrowest run makes the span the text can
                    actually be set in.
                    """
                    total_text_height = N * line_height
                    y_start = box_y + (max_height - total_text_height) / 2
                    line_top = y_start + idx * line_height
                    span = mask_span_for_band(polygon_points, box_x, max_width, line_top, line_top + line_height)

                    # No sampled row reaches the mask (the line sits off the end of a balloon
                    # shorter than its text), or the band pinches to nothing at a spike or tail.
                    # The box is the honest answer in both cases -- it is what the caller asked
                    # for and what the pre-band code returned.
                    if span is None:
                        return {"left": box_x, "right": box_x + max_width}
                    return {"left": span[0], "right": span[1]}

                for para in paragraphs:
                    if not para:
                        tentative_lines.append("")
                        span = get_line_span(line_index)
                        tentative_centers.append((span["left"] + span["right"]) / 2)
                        line_index += 1
                        if line_index >= N:
                            return None
                        continue

                    words = para.split(" ")
                    for word in words:
                        span = get_line_span(line_index)
                        allowed_w = (span["right"] - span["left"]) * 0.95
                        word_width = get_text_width(word)

                        if word_width > allowed_w:
                            if current_line:
                                tentative_lines.append(current_line)
                                tentative_centers.append((span["left"] + span["right"]) / 2)
                                line_index += 1
                                if line_index >= N:
                                    return None

                            # The width available is re-read for every piece: consecutive lines of a
                            # curved balloon can differ by a lot, and a piece measured against the
                            # line above it is how text ends up outside the outline.
                            remaining = word
                            while True:
                                piece_span = get_line_span(line_index)
                                piece_allowed = (piece_span["right"] - piece_span["left"]) * 0.95
                                if get_text_width(remaining) <= piece_allowed or len(remaining) <= 1:
                                    break
                                head, remaining = break_word_to_width(remaining, get_text_width, piece_allowed, lang)
                                tentative_lines.append(head)
                                tentative_centers.append((piece_span["left"] + piece_span["right"]) / 2)
                                line_index += 1
                                if line_index >= N:
                                    return None
                            current_line = remaining
                        else:
                            test_line = (current_line + " " + word) if current_line else word
                            if get_text_width(test_line) > allowed_w and current_line:
                                tentative_lines.append(current_line)
                                tentative_centers.append((span["left"] + span["right"]) / 2)
                                current_line = word
                                line_index += 1
                                if line_index >= N:
                                    return None
                            else:
                                current_line = test_line

                    if current_line:
                        span = get_line_span(line_index)
                        tentative_lines.append(current_line)
                        tentative_centers.append((span["left"] + span["right"]) / 2)
                        current_line = ""
                        line_index += 1
                        if line_index >= N and paragraphs.index(para) < len(paragraphs) - 1:
                            return None

                return (
                    {"lines": tentative_lines, "line_centers": tentative_centers} if len(tentative_lines) <= N else None
                )

            max_possible_lines = int(max_height // line_height)
            if max_possible_lines > 0:
                for N in range(1, max_possible_lines + 1):
                    wrapped = try_wrap_for_n_lines(N)
                    if wrapped is not None:
                        return wrapped

            # Fallback if fits failed
            fallback_lines = []
            fallback_centers = []
            for para in paragraphs:
                if not para:
                    fallback_lines.append("")
                    fallback_centers.append(box_x + max_width / 2)
                    continue
                words = para.split(" ")
                current_line = ""
                for word in words:
                    test_line = (current_line + " " + word) if current_line else word
                    if get_text_width(test_line) > max_width and current_line:
                        fallback_lines.append(current_line)
                        fallback_centers.append(box_x + max_width / 2)
                        current_line = word
                    else:
                        current_line = test_line
                if current_line:
                    fallback_lines.append(current_line)
                    fallback_centers.append(box_x + max_width / 2)
            return {"lines": fallback_lines, "line_centers": fallback_centers}

        # 2. Rectangular wrapping
        if shape != "elliptical":
            result_lines = []
            for para in paragraphs:
                if not para:
                    result_lines.append("")
                    continue
                words = para.split(" ")
                current_line = ""
                for word in words:
                    word_width = get_text_width(word)
                    if word_width > max_width:
                        if current_line:
                            result_lines.append(current_line)
                        remaining = word
                        while get_text_width(remaining) > max_width and len(remaining) > 1:
                            head, remaining = break_word_to_width(remaining, get_text_width, max_width, lang)
                            result_lines.append(head)
                        current_line = remaining
                    else:
                        test_line = (current_line + " " + word) if current_line else word
                        if get_text_width(test_line) > max_width and current_line:
                            result_lines.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                if current_line:
                    result_lines.append(current_line)
            line_centers = [box_x + max_width / 2] * len(result_lines)
            return {"lines": result_lines, "line_centers": line_centers}

        # 3. Elliptical wrapping
        line_height = f_size * 1.2
        half_h = max_height / 2
        half_w = max_width / 2

        def try_wrap_for_n_lines_ellipse(N):
            tentative_lines = []
            current_line = ""
            line_index = 0

            def get_line_allowed_width(idx):
                dy = (idx + 0.5 - N / 2) * line_height
                ratio = dy / half_h
                if abs(ratio) >= 1.0:
                    return 0
                import math

                return 2.0 * half_w * math.sqrt(1.0 - ratio * ratio) * 0.95

            for para in paragraphs:
                if not para:
                    tentative_lines.append("")
                    line_index += 1
                    if line_index >= N:
                        return None
                    continue

                words = para.split(" ")
                for word in words:
                    allowed_w = get_line_allowed_width(line_index)
                    if allowed_w <= 0:
                        return None
                    word_width = get_text_width(word)
                    if word_width > allowed_w:
                        if current_line:
                            tentative_lines.append(current_line)
                            line_index += 1
                            if line_index >= N:
                                return None
                        remaining = word
                        while True:
                            piece_allowed = get_line_allowed_width(line_index)
                            if piece_allowed <= 0:
                                return None
                            if get_text_width(remaining) <= piece_allowed or len(remaining) <= 1:
                                break
                            head, remaining = break_word_to_width(remaining, get_text_width, piece_allowed, lang)
                            tentative_lines.append(head)
                            line_index += 1
                            if line_index >= N:
                                return None
                        current_line = remaining
                    else:
                        test_line = (current_line + " " + word) if current_line else word
                        if get_text_width(test_line) > allowed_w and current_line:
                            tentative_lines.append(current_line)
                            current_line = word
                            line_index += 1
                            if line_index >= N:
                                return None
                        else:
                            current_line = test_line

                if current_line:
                    tentative_lines.append(current_line)
                    current_line = ""
                    line_index += 1
                    if line_index >= N and paragraphs.index(para) < len(paragraphs) - 1:
                        return None
            return tentative_lines if len(tentative_lines) <= N else None

        max_possible_lines = int(max_height // line_height)
        if max_possible_lines > 0:
            for N in range(1, max_possible_lines + 1):
                wrapped = try_wrap_for_n_lines_ellipse(N)
                if wrapped is not None:
                    return {
                        "lines": wrapped,
                        "line_centers": [box_x + max_width / 2] * len(wrapped),
                    }

        fallback_lines = []
        for para in paragraphs:
            if not para:
                fallback_lines.append("")
                continue
            words = para.split(" ")
            current_line = ""
            for word in words:
                test_line = (current_line + " " + word) if current_line else word
                if get_text_width(test_line) > max_width and current_line:
                    fallback_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                fallback_lines.append(current_line)
        return {
            "lines": fallback_lines,
            "line_centers": [box_x + max_width / 2] * len(fallback_lines),
        }

    # D7 (docs/render_quality_gap_2026-08-05.md): `max_width // 3` assumed roughly 3 characters
    # per line and used that to cap the search before it ever ran. `fits_clean` below already
    # rejects any size that overflows the box or breaks a word, so this pre-cap adds no safety --
    # it only ever forecloses sizes the search would otherwise have accepted. It hit hardest on
    # tall narrow boxes (vertical Japanese speech bubbles): sample1's `safe_text_w=145` capped the
    # search at 48px before it could even try the ~60-70px mangatranslator.ai used for the same
    # balloon.
    # The search only needs an upper *bound*; every candidate is still checked against the height,
    # the width, the word rule and the mask below, so a generous bound cannot produce a bad layout,
    # only a slower search (a few more bisection steps).
    #
    # Both of the old terms were doing damage. `72` is an absolute pixel count applied to pages
    # from 832px to 6905px wide, so on the big ones it capped the type at a size the page's own
    # lettering dwarfs. `max_height // 2` looks conservative and is worse: one line at h/2 with a
    # 1.2 line-height fills exactly 60% of the box and no more, which is why the elements this cap
    # bound sat at a median fill of exactly 0.60. A single line is bounded by the height check at
    # h/1.2 anyway -- the halving was pure loss.
    max_start_size = int(max_height)
    start_size = max(max_start_size, default_font_size)

    min_font_size = 6
    line_height_multiplier = 1.2

    def broke_a_word(res):
        """True when the wrap cut a word apart in a way a reader cannot put back together.

        Every wrapping path above splits an over-wide word, and since hyphenation that split is
        usually at a legal point and carries a hyphen -- "Op-" / "portunities" is a word set across
        two lines, not a broken one. What is still broken is a split with no hyphen at all, which
        is what happens when no legal point fits: "collect" / "ion". `reassemble` puts the
        hyphenated case back and leaves the other one showing.
        """
        return reassemble(res["lines"]) != [w.replace("-", "") for w in clean_text.split()]

    def widest_line(res, f_size):
        font = font_at(f_size)
        if not font:
            return 0.0
        widest = 0.0
        for line in res["lines"]:
            try:
                w = font.getlength(line)
            except Exception:
                try:
                    bbox = font.getbbox(line)
                    w = bbox[2] - bbox[0]
                except Exception:
                    w = len(line) * (f_size * 0.5)
            widest = max(widest, w)
        return widest

    def stays_inside_mask(res, f_size):
        """True when every line, as drawn, stays within the mask across its own band.

        The box-width test above cannot see this: a line can be well inside a box and still cross
        the balloon outline, because the box is the mask's bounding rectangle and the mask curves
        in. Measured here against the same geometry the draw uses -- each line's centre, its width
        at this size, and the span available over the band it occupies.
        """
        if not polygon_points:
            return True
        font = font_at(f_size)
        if not font:
            return True
        lines = res["lines"]
        centers = res.get("line_centers") or []
        line_height = f_size * line_height_multiplier
        y_start = box_y + (max_height - len(lines) * line_height) / 2
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                w = font.getlength(line)
            except Exception:
                w = len(line) * (f_size * 0.5)
            center = centers[idx] if idx < len(centers) else box_x + max_width / 2
            top = y_start + idx * line_height
            span = mask_span_for_band(polygon_points, box_x, max_width, top, top + line_height)
            if span is None:
                continue
            if center - w / 2 < span[0] - 0.5 or center + w / 2 > span[1] + 0.5:
                return False
        return True

    evaluated = {}

    def evaluate(f_size):
        """(wrap, fits height, fits cleanly) at this size."""
        if f_size in evaluated:
            return evaluated[f_size]
        res = wrap_text_py(clean_text, f_size)
        total = len(res["lines"]) * f_size * line_height_multiplier
        fits_height = total <= max_height
        fits_clean = (
            fits_height
            and not broke_a_word(res)
            and widest_line(res, f_size) <= max_width
            and stays_inside_mask(res, f_size)
        )
        # Contained is the weaker promise: the text is inside its box, but a word may have been
        # broken to get it there. It is what the last-resort tier below searches on.
        fits_contained = fits_height and widest_line(res, f_size) <= max_width
        evaluated[f_size] = (res, fits_height, fits_clean, fits_contained)
        return evaluated[f_size]

    def largest_size_where(criterion):
        low = min_font_size
        high = start_size
        best = None
        while low <= high:
            mid = (low + high) // 2
            res, fits_height, fits_clean, fits_contained = evaluate(mid)
            ok = {"clean": fits_clean, "contained": fits_contained, "height": fits_height}[criterion]
            if ok:
                best = (mid, res)
                low = mid + 1
            else:
                high = mid - 1
        return best

    # Largest size that lays the text out whole: every word intact and every line inside the box.
    #
    # Height used to be the only test, which made both of the ways text can fail horizontally
    # invisible to the search. A word wider than the line gets split per character, and the
    # polygon/ellipse fallback wraps to the box width without splitting anything, so a size that
    # spills sideways out of the bubble still reported the same "fits" as one that does not. The
    # search then kept growing the font until the *height* ran out and returned the largest size
    # that mangles the text rather than the largest size that sets it.
    best = largest_size_where("clean")

    # Nothing sets cleanly -- a single word longer than the box is wide even at 6px, say. Keep the
    # text inside its box and break the word, rather than the other way round.
    #
    # Height alone used to be the fallback, and it is a bad trade: it grows the type until the
    # *height* runs out while the lines get arbitrarily wider than the box. sample27's
    # "...Just kidding! (PLAYFUL RETRACTION)" is a 44px-wide box that came out at 43px type --
    # lines twice the width of the region, drawn across the neighbouring balloon and the art. A
    # broken word is ugly and local; text outside its box lands on someone else's panel.
    if best is None:
        best = largest_size_where("contained")

    # And if even that is impossible -- a box narrower than a single glyph -- the height rule is
    # all that is left, at which point the region is too small to set anything in legibly and the
    # real answer is upstream (D10: it should not have been typeset at all).
    if best is None:
        best = largest_size_where("height")

    if best is None:
        best_fs = min_font_size
        best_res = wrap_text_py(clean_text, min_font_size)
    else:
        best_fs, best_res = best

    def limiting_constraint(f_size):
        """Which rule stopped the search one step above the size it chose.

        Underfill (D7) is not one problem: a balloon can be two-thirds empty because a single long
        word will not fit its width, because the absolute size cap bit, or because the text really
        does fill it. Those want different fixes, and from outside the function they are
        indistinguishable -- hence reporting it rather than guessing. Cheap: the size above the
        chosen one has usually been evaluated already by the search.
        """
        if f_size >= start_size:
            return "size_cap"
        res, fits_height, _, _ = evaluate(f_size + 1)
        if not fits_height:
            return "height"
        if widest_line(res, f_size + 1) > max_width:
            return "width"
        if broke_a_word(res):
            return "unbreakable_word"
        if not stays_inside_mask(res, f_size + 1):
            return "mask"
        return "none"

    total_height = len(best_res["lines"]) * best_fs * line_height_multiplier
    return {
        "fontSize": best_fs,
        "lines": best_res["lines"],
        "overflow": total_height > max_height,
        "lineCenters": best_res["line_centers"],
        "limitedBy": limiting_constraint(best_fs),
    }


def _relative_luminance(hex_color):
    """WCAG relative luminance of an `#rrggbb` string, or None if it is not one."""
    if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return None
    try:
        channels = [int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def readable_text_color(bg_hex, fg_hex, floor=CONTRAST_FLOOR):
    """`fg_hex`, unless it is illegible on `bg_hex`, in which case black or white.

    R2 gave the renderer a backdrop colour it never used to see. A region's text colour is decided
    without reference to it -- the default is black -- so a covering balloon sampled from a dark
    panel arrived as black text on a near-black fill. sample10's bottom-left corner is the case:
    white-stroked lettering on a dark panel, no balloon, dominant colour #33272d.

    Only fires below the floor, so a deliberate colour pairing that is merely low-contrast is left
    alone; black on black is never deliberate. 3.0 is WCAG's large-text threshold, and lettering
    set to fill a balloon is large text.
    """
    bg = _relative_luminance(bg_hex)
    fg = _relative_luminance(fg_hex)
    if bg is None or fg is None:
        return fg_hex
    lighter, darker = max(bg, fg), min(bg, fg)
    if (lighter + 0.05) / (darker + 0.05) >= floor:
        return fg_hex
    return "#000000" if bg > 0.4 else "#ffffff"


# ---------------------------------------------------------------------------
# AUDIT-R5 — an element's angle
#
# `rotation` is the angle of the element's *box*, in degrees, about the box centre. The box itself
# (x/y/maxWidth/maxHeight) is always stored unrotated; a `maskPolygon` is the opposite, already in
# absolute page coordinates with the angle baked in. So the plate drawn from a polygon must not be
# turned again, and everything derived from the box must.
#
# Sign convention follows the editor, which is SVG/canvas: y grows downward and a positive angle
# turns clockwise on screen. PIL's `Image.rotate` turns counter-clockwise, hence the negation where
# it is used.
# ---------------------------------------------------------------------------


def rotate_point_deg(px, py, cx, cy, degrees):
    """Rotate (px, py) about (cx, cy). Matches `rotatePoint` in the frontend's polygonUtils.ts."""
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def box_outline_points(ex, ey, ew, eh, degrees, elliptical=False, segments=32):
    """The element box as a point list, turned to `degrees`.

    A rotated rectangle is no longer expressible as a PIL rectangle and a rotated ellipse is not
    expressible as a PIL ellipse, so both become polygons. The ellipse is sampled rather than
    approximated by its bounding box, because the plate has to follow the balloon.
    """
    cx, cy = ex + ew / 2.0, ey + eh / 2.0
    if elliptical:
        rx, ry = ew / 2.0, eh / 2.0
        raw = [
            (
                cx + rx * math.cos(2.0 * math.pi * i / segments),
                cy + ry * math.sin(2.0 * math.pi * i / segments),
            )
            for i in range(segments)
        ]
    else:
        raw = [(ex, ey), (ex + ew, ey), (ex + ew, ey + eh), (ex, ey + eh)]
    if not degrees:
        return raw
    return [rotate_point_deg(px, py, cx, cy, degrees) for px, py in raw]


DEFAULT_TEXT_BOX_PADDING_PX = 4
DEFAULT_TEXT_BOX_SAFETY_PERCENT = 95


def resolve_text_box_inset(job_data):
    """`(padding_px, safety_percent)` for this job, defaulted and clamped.

    AUDIT-R1/F16. The clamps are not decoration: a safety percent of 0, or a padding wider than
    half the box, fits every element into a zero-width rectangle, and a job payload is not a
    trusted source of numbers.
    """
    job_data = job_data or {}

    def read(key, default, low, high):
        try:
            value = round(float(job_data.get(key, default)))
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    return (
        read("textBoxPaddingPx", DEFAULT_TEXT_BOX_PADDING_PX, 0, 64),
        read("textBoxSafetyPercent", DEFAULT_TEXT_BOX_SAFETY_PERCENT, 1, 100),
    )


def text_fit_box(ex, ey, ew, eh, padding_px, safety_percent):
    """The rectangle text is fitted into, given an element's box.

    Mirrors `textFitBox` in `frontend/src/utils/textFitBox.ts`; the two must agree, which is the
    entire point of both existing. Truncating (not rounding) the scaled extents is what keeps the
    integer results identical to the frontend's `Math.floor`.
    """
    # QA re-renders a page without a job payload (qa.py), so an unset inset means "whatever the
    # pipeline has always used" rather than an error.
    usable_w = max(1, ew - padding_px * 2)
    usable_h = max(1, eh - padding_px * 2)
    return (
        ex + padding_px,
        ey + padding_px,
        max(1, int(usable_w * safety_percent / 100)),
        max(1, int(usable_h * safety_percent / 100)),
    )


def element_rotation(el):
    """An element's angle in degrees, or 0.0. Treated as unrotated below a tenth of a degree."""
    try:
        degrees = float(el.get("rotation") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if abs(degrees) < 0.1 or not math.isfinite(degrees) else degrees


def render_image_core(image_id, page_id=None, chapter_id=None, text_box_inset=None):
    try:
        render_target_id = image_id
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        if page_id:
            backend_url += f"?pageId={page_id}"
            if chapter_id:
                backend_url += f"&chapterId={chapter_id}"
        elif chapter_id:
            backend_url += f"?chapterId={chapter_id}"
        res = requests.get(backend_url, headers=backend_headers())
        if res.status_code != 200:
            logger.error(f"[Render] Failed to get image info: {res.status_code}")
            return False
        image_info = res.json()
        layer_elements = image_info.get("layerElements", [])
        regions_by_id = {r["id"]: r for r in image_info.get("ocrRegions", []) if r.get("id")}
    except Exception as e:
        logger.error(f"[Render] Error fetching image details: {e}")
        raise e

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        logger.error(f"[Render] Error downloading image: {e}")
        raise e

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # AUDIT-R1/F16: the inset applied before fitting. It used to be the literals 4 and 0.95
        # here and three *different* answers in the frontend, one of them the live reader -- the
        # screen the typesetting was being judged on. One value now, from settings, sent on the
        # job.
        box_padding_px, box_safety_percent = text_box_inset or (
            DEFAULT_TEXT_BOX_PADDING_PX,
            DEFAULT_TEXT_BOX_SAFETY_PERCENT,
        )

        # Render only visible elements from visible translation/sfx layers.
        #
        # This used to also accept `layerType is None`, which was meant as a backwards-compatible
        # fallback but behaved as "render anything unlabelled". The backend never sent layerType at
        # all -- `Layer` is @JsonIgnore'd on LayerElement -- so *every* element matched, and each
        # rendered page had its Japanese OCR text drawn underneath the translated bubbles, where it
        # overflowed. Layer visibility set in the reader was ignored for the same reason.
        #
        # Now fails closed: an element whose layer type is unknown is not drawn. Rendering nothing
        # extra is always recoverable; baking source text into an export is not.
        translation_elements = [
            el
            for el in layer_elements
            if el.get("visible", True)
            and el.get("layerVisible", True)
            and el.get("layerType") in ("translation", "sfx")
        ]
        skipped = len(layer_elements) - len(translation_elements)
        if skipped:
            logger.warning(
                f"[Render] {len(translation_elements)} element(s) to draw, {skipped} skipped "
                f"(not a visible translation/sfx layer)"
            )

        for el in translation_elements:
            text = el.get("text", "")
            if not text:
                continue

            box_shape = el.get("boxShape") or "rectangular"
            # Auto-uppercase for speech bubbles
            region_type = el.get("regionType")
            if (region_type == "speech" or (region_type is None and box_shape == "elliptical")) and os.environ.get(
                "USE_UPPERCASE_SPEECH", "true"
            ).lower() in (
                "true",
                "1",
                "t",
            ):
                text = text.upper()

            ex = float(el.get("x") or 0.0)
            ey = float(el.get("y") or 0.0)
            ew = int(el.get("maxWidth") or 100)
            eh = int(el.get("maxHeight") or 50)

            bg_color_hex = el.get("backgroundColor")
            text_color_hex = readable_text_color(el.get("backgroundColor"), el.get("textColor") or "#000000")

            font_size = float(el.get("size") or 12.0)
            font_weight = el.get("fontWeight") or "normal"
            font_style = el.get("fontStyle") or "normal"

            bold = "bold" in font_weight.lower()
            italic = "italic" in font_style.lower()
            mask_polygon = el.get("maskPolygon")
            el_region = regions_by_id.get(el.get("regionId"))
            in_bubble = bool(el_region) and has_detected_bubble(el_region)

            # Erase what we are about to typeset.
            #
            # The mask covers the source text; the box is where the English goes. Inside a balloon
            # those agree closely enough -- the box is an inset of the bubble the mask fills, and
            # what it overhangs is blank interior. For free-floating text they are two different
            # rectangles: `free_text_box` pads the column and, for a column too narrow to set a
            # word in, widens it. Filling only the mask left that difference on bare artwork.
            #
            # Filling the box as well costs nothing where the two overlap (same colour) and is
            # bounded by the widening cap, which is why this became affordable only once
            # `free_text_box` stopped squaring the column away.
            # AUDIT-R5: the box turns with the element, the mask polygon does not. The polygon is
            # page-space and already carries the angle; the box is stored unrotated. Filling the
            # box axis-aligned beside a turned mask is what laid a straight white rectangle across
            # artwork on every rotated caption.
            rotation_deg = element_rotation(el)
            painted_box = False
            if bg_color_hex and bg_color_hex.startswith("#"):
                # Draw mask
                if mask_polygon:
                    try:
                        import json

                        pts = json.loads(mask_polygon) if isinstance(mask_polygon, str) else mask_polygon
                        if isinstance(pts, list) and len(pts) > 0:
                            poly_tuples = [(float(p[0]), float(p[1])) for p in pts]
                            draw.polygon(poly_tuples, fill=bg_color_hex)
                    except Exception as e:
                        logger.error(f"[Render] Failed to draw polygon mask: {e}")
                    if not in_bubble:
                        draw.polygon(
                            box_outline_points(ex, ey, ew, eh, rotation_deg),
                            fill=bg_color_hex,
                        )
                        painted_box = True
                elif box_shape == "elliptical" and not rotation_deg:
                    draw.ellipse([ex, ey, ex + ew, ey + eh], fill=bg_color_hex)
                    painted_box = True
                else:
                    draw.polygon(
                        box_outline_points(ex, ey, ew, eh, rotation_deg, elliptical=(box_shape == "elliptical")),
                        fill=bg_color_hex,
                    )
                    painted_box = True

            # Draw Text
            #
            # This inset box is what fit_text_in_box_py actually wraps and centres lines against
            # (its polygon path computes each line's horizontal span from box_y/max_height, not
            # from ex/ey/ew/eh) -- the draw step below must centre and clamp against the exact
            # same box, or a line the fit computed for the bubble's wide middle can end up drawn a
            # few pixels off, at a height where an oval mask has already narrowed. That was
            # invisible at the small font sizes the old width//3 cap produced; larger text reaches
            # further toward the taper, where the same few-pixel drift becomes visible overflow.
            text_box_x, text_box_y, text_box_w, text_box_h = text_fit_box(
                ex, ey, ew, eh, box_padding_px, box_safety_percent
            )

            font_name = el.get("font") or "Comic Neue"
            fit = fit_text_in_box_py(
                text,
                text_box_w,
                text_box_h,
                font_name=font_name,
                default_font_size=int(font_size),
                shape=("elliptical" if box_shape == "elliptical" else "rectangular"),
                box_x=text_box_x,  # type: ignore
                box_y=text_box_y,  # type: ignore
                mask_polygon=mask_polygon,
                bold=bold,
                italic=italic,
            )

            f_size = fit["fontSize"]
            font = load_font(f_size, font_name=font_name, bold=bold, italic=italic)
            # The plate is only a backdrop where it actually reaches; see `halo_stroke_for`. When
            # the box itself was filled above there is nothing left to escape onto.
            halo_stroke = (
                0 if painted_box else halo_stroke_for(mask_polygon, (ex, ey, ew, eh), f_size, in_bubble=in_bubble)
            )
            halo_fill = bg_color_hex if bg_color_hex and bg_color_hex.startswith("#") else "#ffffff"
            if halo_stroke:
                logger.info(
                    f"[Render] Haloing '{text[:24]}' ({halo_stroke}px, {halo_fill}) -- its "
                    f"{ew}x{eh} box escapes the erased area"
                )
            if font:
                line_height = f_size * 1.2
                total_height = len(fit["lines"]) * line_height
                start_y = text_box_y + (text_box_h - total_height) / 2

                # Where each line goes, in page coordinates, before any rotation. Collected rather
                # than drawn straight away so a rotated element can render the same placements onto
                # its own tile — the fitting and centring logic below must not have to know whether
                # the element is turned.
                placements = []

                for i, line in enumerate(fit["lines"]):
                    line_center_x = (
                        fit["lineCenters"][i]
                        if (fit.get("lineCenters") and i < len(fit["lineCenters"]))
                        else (text_box_x + text_box_w / 2)
                    )

                    try:
                        line_width = font.getlength(line)
                    except Exception:
                        try:
                            bbox = font.getbbox(line)
                            line_width = bbox[2] - bbox[0]
                        except Exception:
                            try:
                                line_width = font.getsize(line)[0]  # type: ignore
                            except Exception:
                                line_width = len(line) * (f_size * 0.5)

                    line_x = line_center_x - line_width / 2
                    # A line centred on the shape it was wrapped to can still start left of the box
                    # or end right of it -- the centre comes from the mask span, the width from the
                    # glyphs. Keep whatever fits inside the box inside it, so an off-centre line
                    # cannot walk into the next panel or off the page.
                    if line_width <= text_box_w:
                        line_x = min(max(line_x, text_box_x), text_box_x + text_box_w - line_width)
                    line_y = start_y + i * line_height
                    placements.append((line_x, line_y, line, line_width))

                if not rotation_deg:
                    for line_x, line_y, line, _ in placements:
                        draw.text(
                            (line_x, line_y),
                            line,
                            fill=text_color_hex,
                            font=font,
                            stroke_width=halo_stroke,
                            stroke_fill=halo_fill if halo_stroke else None,
                        )
                elif placements:
                    # AUDIT-R5: draw the lines level onto a transparent tile, turn the tile, and
                    # composite. Rotating the text rather than the coordinates is what keeps the
                    # glyphs themselves upright-relative-to-the-box instead of merely repositioned.
                    #
                    # The tile is sized from the actual placements unioned with the element box,
                    # not from the box alone: `fit` is allowed to produce a line wider than the box
                    # (the clamp above only fires when the line already fits), and a tile cut to
                    # the box would silently crop exactly those overflowing lines that the halo
                    # exists to make readable.
                    pad = math.ceil(halo_stroke * 2) + 4
                    min_x = min([ex] + [px for px, _, _, _ in placements]) - pad
                    min_y = min([ey] + [py for _, py, _, _ in placements]) - pad
                    max_x = max([ex + ew] + [px + w for px, _, _, w in placements]) + pad
                    max_y = max([ey + eh] + [py + line_height for _, py, _, _ in placements]) + pad
                    tile_w = max(1, math.ceil(max_x - min_x))
                    tile_h = max(1, math.ceil(max_y - min_y))

                    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                    tile_draw = ImageDraw.Draw(tile)
                    for line_x, line_y, line, _ in placements:
                        tile_draw.text(
                            (line_x - min_x, line_y - min_y),
                            line,
                            fill=text_color_hex,
                            font=font,
                            stroke_width=halo_stroke,
                            stroke_fill=halo_fill if halo_stroke else None,
                        )

                    # Negated: PIL turns counter-clockwise, the editor (SVG/canvas, y down) turns
                    # clockwise for a positive angle.
                    rotated = tile.rotate(-rotation_deg, resample=Image.Resampling.BICUBIC, expand=True)
                    # The tile turns about the element box's centre, not the tile's own, so the
                    # text ends up where the rotated plate is.
                    box_cx, box_cy = ex + ew / 2.0, ey + eh / 2.0
                    tile_cx, tile_cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                    spun_cx, spun_cy = rotate_point_deg(tile_cx, tile_cy, box_cx, box_cy, rotation_deg)
                    img.paste(
                        rotated,
                        (
                            round(spun_cx - rotated.width / 2.0),
                            round(spun_cy - rotated.height / 2.0),
                        ),
                        rotated,
                    )

        # Save flattened image
        out_buf = io.BytesIO()
        img.save(out_buf, format="PNG")
        out_bytes = out_buf.getvalue()

        # Upload to MinIO under rendered/{render_target_id}.png
        storage_path = f"rendered/{render_target_id}.png"
        minio_client.put_object(
            "manga-library",
            storage_path,
            io.BytesIO(out_bytes),
            len(out_bytes),
            content_type="image/png",
        )
        logger.info(f"[Render] Flattened image uploaded to MinIO: {storage_path}")

        # Save local copy in render cache
        from worker.config import RENDER_CACHE_DIR

        if os.environ.get("ENABLE_QA_AUDIT_CACHE", "false").lower() in (
            "true",
            "1",
            "yes",
        ):
            os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(RENDER_CACHE_DIR, f"{render_target_id}.png")
            with open(cache_path, "wb") as f:
                f.write(out_bytes)
            logger.info(f"[Render] Cached rendered image to {cache_path}")
        return True

    except Exception as e:
        logger.error(f"[Render] Error rendering typeset: {e}")
        import traceback

        traceback.print_exc()
        raise e


def process_render(job_data):
    image_id = job_data.get("imageId")
    page_id = job_data.get("pageId")

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:render")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    logger.info(f"[Render] Processing page: {page_id or image_id}{progress_str}")

    from worker.config import QA_MODE

    qa_mode_resolved = job_data.get("qaMode") or QA_MODE

    if qa_mode_resolved == "auto":
        from worker.config import QA_CONFIG, is_usable_model

        provider = job_data.get("qaProvider") or getattr(QA_CONFIG, "provider", None)
        has_vlm = is_usable_model(job_data.get("qaVlmModel")) or is_usable_model(getattr(QA_CONFIG, "vlm_model", None))
        has_llm = is_usable_model(job_data.get("qaLlmModel")) or is_usable_model(getattr(QA_CONFIG, "llm_model", None))
        if has_vlm and provider:
            qa_mode_resolved = "vlm"
        elif has_llm and provider:
            qa_mode_resolved = "llm"
        else:
            qa_mode_resolved = "none"

    if not render_image_core(
        image_id,
        page_id=page_id,
        chapter_id=job_data.get("chapterId"),
        text_box_inset=resolve_text_box_inset(job_data),
    ):
        raise Exception("Render failed")

    # Trigger callback
    callback_payload = {
        "jobId": job_data.get("jobId"),
        "imageId": image_id,
        "pageId": page_id,
    }
    try:
        res = requests.post(f"{CALLBACK_URL}/render", json=callback_payload, headers=backend_headers())
        logger.debug(f"[Render] Callback status code: {res.status_code}")
    except Exception as e:
        logger.error(f"[Render] Failed to post callback: {e}")
