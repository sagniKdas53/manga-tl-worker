import io
import requests
from PIL import Image, ImageDraw, ImageFont
from worker.config import (
    CALLBACK_URL,
    BACKEND_HEADERS,
    minio_client,
    logger,
    redis_client,
)
from worker.utils.image import download_image
import os

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
    "Arial": {
        "normal": "/usr/share/fonts/truetype/google/Arial.ttf",
        "bold": "/usr/share/fonts/truetype/google/Arial.ttf",
        "italic": "/usr/share/fonts/truetype/google/Arial.ttf",
        "bolditalic": "/usr/share/fonts/truetype/google/Arial.ttf",
    },
    "Courier New": {
        "normal": "/usr/share/fonts/truetype/google/CourierNew.ttf",
        "bold": "/usr/share/fonts/truetype/google/CourierNew.ttf",
        "italic": "/usr/share/fonts/truetype/google/CourierNew.ttf",
        "bolditalic": "/usr/share/fonts/truetype/google/CourierNew.ttf",
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
        path = FONT_REGISTRY[font_name].get(style_key) or FONT_REGISTRY[font_name].get(
            "normal"
        )
        font = try_load(path)
        if font:
            return font

    # 2. Try fallbacks from registry in order
    for fallback in DEFAULT_FONT_FALLBACK_ORDER:
        if fallback in FONT_REGISTRY:
            path = FONT_REGISTRY[fallback].get(style_key) or FONT_REGISTRY[
                fallback
            ].get("normal")
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
        test_line = " ".join(current_line + [word])
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


def draw_wrapped_text(
    draw, text, font, text_color, x, y, max_width, max_height, alignment="center"
):
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
):
    clean_text = (text or "").replace("\r\n", "\n")
    paragraphs = clean_text.split("\n")

    polygon_points = None
    if mask_polygon:
        try:
            import json

            parsed = (
                json.loads(mask_polygon)
                if isinstance(mask_polygon, str)
                else mask_polygon
            )
            if isinstance(parsed, list) and all(
                isinstance(p, list) and len(p) == 2 for p in parsed
            ):
                polygon_points = parsed
        except Exception:
            pass

    def wrap_text_py(txt, f_size):
        font = load_font(f_size, font_name=font_name, bold=bold, italic=italic)
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
                        return font.getsize(t)[0]
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
                    total_text_height = N * line_height
                    y_start = box_y + (max_height - total_text_height) / 2
                    line_center_y = y_start + (idx + 0.5) * line_height

                    intersects = []
                    n_pts = len(polygon_points)
                    for i in range(n_pts):
                        p1 = polygon_points[i]
                        p2 = polygon_points[(i + 1) % n_pts]
                        x1, y1 = p1[0], p1[1]
                        x2, y2 = p2[0], p2[1]
                        if (y1 <= line_center_y < y2) or (y2 <= line_center_y < y1):
                            ix = x1 + (line_center_y - y1) * (x2 - x1) / (y2 - y1)
                            intersects.append(ix)

                    if len(intersects) >= 2:
                        intersects.sort()
                        best_span = {"left": box_x, "right": box_x + max_width}
                        max_overlap_len = 0
                        for i in range(0, len(intersects) - 1, 2):
                            segment_left = intersects[i]
                            segment_right = intersects[i + 1]
                            overlap_left = max(segment_left, box_x)
                            overlap_right = min(segment_right, box_x + max_width)
                            overlap_len = overlap_right - overlap_left
                            if overlap_len > max_overlap_len:
                                max_overlap_len = overlap_len
                                best_span = {
                                    "left": overlap_left,
                                    "right": overlap_right,
                                }
                        if max_overlap_len > 0:
                            return best_span
                    return {"left": box_x, "right": box_x + max_width}

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
                                tentative_centers.append(
                                    (span["left"] + span["right"]) / 2
                                )
                                line_index += 1
                                if line_index >= N:
                                    return None

                            current_word_part = ""
                            for char in word:
                                test_part = current_word_part + char
                                next_span = get_line_span(line_index)
                                next_allowed_w = (
                                    next_span["right"] - next_span["left"]
                                ) * 0.95
                                if (
                                    get_text_width(test_part) > next_allowed_w
                                    and current_word_part
                                ):
                                    tentative_lines.append(current_word_part)
                                    tentative_centers.append(
                                        (next_span["left"] + next_span["right"]) / 2
                                    )
                                    current_word_part = char
                                    line_index += 1
                                    if line_index >= N:
                                        return None
                                else:
                                    current_word_part = test_part
                            current_line = current_word_part
                        else:
                            test_line = (
                                (current_line + " " + word) if current_line else word
                            )
                            if get_text_width(test_line) > allowed_w and current_line:
                                tentative_lines.append(current_line)
                                tentative_centers.append(
                                    (span["left"] + span["right"]) / 2
                                )
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
                        if (
                            line_index >= N
                            and paragraphs.index(para) < len(paragraphs) - 1
                        ):
                            return None

                return (
                    {"lines": tentative_lines, "line_centers": tentative_centers}
                    if len(tentative_lines) <= N
                    else None
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
                        current_word_part = ""
                        for char in word:
                            test_part = current_word_part + char
                            if (
                                get_text_width(test_part) > max_width
                                and current_word_part
                            ):
                                result_lines.append(current_word_part)
                                current_word_part = char
                            else:
                                current_word_part = test_part
                        current_line = current_word_part
                    else:
                        test_line = (
                            (current_line + " " + word) if current_line else word
                        )
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
                        current_word_part = ""
                        for char in word:
                            test_part = current_word_part + char
                            current_allowed_w = get_line_allowed_width(line_index)
                            if (
                                get_text_width(test_part) > current_allowed_w
                                and current_word_part
                            ):
                                tentative_lines.append(current_word_part)
                                current_word_part = char
                                line_index += 1
                                if line_index >= N:
                                    return None
                            else:
                                current_word_part = test_part
                        current_line = current_word_part
                    else:
                        test_line = (
                            (current_line + " " + word) if current_line else word
                        )
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

    max_start_size = min(max_height // 2, max_width // 3, 72)
    start_size = max(max_start_size, default_font_size)

    low = 6
    high = start_size
    best_fs = 6
    best_res = None
    line_height_multiplier = 1.2

    while low <= high:
        mid = (low + high) // 2
        res = wrap_text_py(clean_text, mid)
        total_height = len(res["lines"]) * mid * line_height_multiplier
        if total_height <= max_height:
            best_fs = mid
            best_res = res
            low = mid + 1
        else:
            high = mid - 1

    if best_res is None:
        best_res = wrap_text_py(clean_text, 6)
        best_fs = 6

    total_height = len(best_res["lines"]) * best_fs * line_height_multiplier
    return {
        "fontSize": best_fs,
        "lines": best_res["lines"],
        "overflow": total_height > max_height,
        "lineCenters": best_res["line_centers"],
    }


def render_image_core(image_id):
    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[Render] Failed to get image info: {res.status_code}", flush=True)
            return False
        image_info = res.json()
        layer_elements = image_info.get("layerElements", [])
    except Exception as e:
        print(f"[Render] Error fetching image details: {e}", flush=True)
        raise e

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[Render] Error downloading image: {e}", flush=True)
        raise e

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Render only visible elements from visible translation/sfx layers
        translation_elements = [
            el
            for el in layer_elements
            if el.get("visible", True)
            and el.get("layerVisible", True)
            and (
                el.get("layerType") in ("translation", "sfx")
                or el.get("layerType") is None
            )
        ]

        for el in translation_elements:
            text = el.get("text", "")
            if not text:
                continue

            box_shape = el.get("boxShape") or "rectangular"
            # Auto-uppercase for speech bubbles
            region_type = el.get("regionType")
            if region_type == "speech" or (
                region_type is None and box_shape == "elliptical"
            ):
                if os.environ.get("USE_UPPERCASE_SPEECH", "true").lower() in (
                    "true",
                    "1",
                    "t",
                ):
                    text = text.upper()

            ex = float(el.get("x", 0.0))
            ey = float(el.get("y", 0.0))
            ew = int(el.get("maxWidth") or 100)
            eh = int(el.get("maxHeight") or 50)

            bg_color_hex = el.get("backgroundColor")
            text_color_hex = el.get("textColor") or "#000000"

            font_size = float(el.get("size") or 12.0)
            font_weight = el.get("fontWeight") or "normal"
            font_style = el.get("fontStyle") or "normal"

            bold = "bold" in font_weight.lower()
            italic = "italic" in font_style.lower()
            mask_polygon = el.get("maskPolygon")

            # Masking
            if bg_color_hex and bg_color_hex.startswith("#"):
                # Draw mask
                if mask_polygon:
                    try:
                        import json

                        pts = (
                            json.loads(mask_polygon)
                            if isinstance(mask_polygon, str)
                            else mask_polygon
                        )
                        if isinstance(pts, list) and len(pts) > 0:
                            poly_tuples = [(float(p[0]), float(p[1])) for p in pts]
                            draw.polygon(poly_tuples, fill=bg_color_hex)
                    except Exception as e:
                        print(f"[Render] Failed to draw polygon mask: {e}", flush=True)
                elif box_shape == "elliptical":
                    draw.ellipse([ex, ey, ex + ew, ey + eh], fill=bg_color_hex)
                else:
                    draw.rectangle([ex, ey, ex + ew, ey + eh], fill=bg_color_hex)

            # Draw Text
            font_name = el.get("font") or "Comic Neue"
            fit = fit_text_in_box_py(
                text,
                int((ew - 8) * 0.95),  # 5% safety margin
                int((eh - 8) * 0.95),  # 5% safety margin
                font_name=font_name,
                default_font_size=int(font_size),
                shape=("elliptical" if box_shape == "elliptical" else "rectangular"),
                box_x=ex + 4,
                box_y=ey + 4,
                mask_polygon=mask_polygon,
                bold=bold,
                italic=italic,
            )

            f_size = fit["fontSize"]
            font = load_font(f_size, font_name=font_name, bold=bold, italic=italic)
            if font:
                line_height = f_size * 1.2
                total_height = len(fit["lines"]) * line_height
                start_y = ey + (eh - total_height) / 2

                for i, line in enumerate(fit["lines"]):
                    line_center_x = (
                        fit["lineCenters"][i]
                        if (fit.get("lineCenters") and i < len(fit["lineCenters"]))
                        else (ex + ew / 2)
                    )

                    try:
                        line_width = font.getlength(line)
                    except Exception:
                        try:
                            bbox = font.getbbox(line)
                            line_width = bbox[2] - bbox[0]
                        except Exception:
                            try:
                                line_width = font.getsize(line)[0]
                            except Exception:
                                line_width = len(line) * (f_size * 0.5)

                    line_x = line_center_x - line_width / 2
                    line_y = start_y + i * line_height
                    draw.text((line_x, line_y), line, fill=text_color_hex, font=font)

        # Save flattened image
        out_buf = io.BytesIO()
        img.save(out_buf, format="PNG")
        out_bytes = out_buf.getvalue()

        # Upload to MinIO under rendered/{imageId}.png
        storage_path = f"rendered/{image_id}.png"
        minio_client.put_object(
            "manga-library",
            storage_path,
            io.BytesIO(out_bytes),
            len(out_bytes),
            content_type="image/png",
        )
        print(f"[Render] Flattened image uploaded to MinIO: {storage_path}", flush=True)

        # Save local copy in render cache
        from worker.config import RENDER_CACHE_DIR

        os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(RENDER_CACHE_DIR, f"{image_id}.png")
        with open(cache_path, "wb") as f:
            f.write(out_bytes)
        logger.info(f"[Render] Cached rendered image to {cache_path}")
        return True

    except Exception as e:
        print(f"[Render] Error rendering typeset: {e}", flush=True)
        import traceback

        traceback.print_exc()
        raise e


def process_render(job_data):
    image_id = job_data["imageId"]

    page_num = job_data.get("pageNumber")
    chapter_num = job_data.get("chapterNumber")
    queue_len = redis_client.llen("queue:render")

    progress_str = ""
    if page_num is not None:
        progress_str = f" | Page {page_num}"
        if chapter_num is not None:
            progress_str += f" of Chapter {chapter_num}"
        progress_str += f" (Queue: {queue_len} remaining)"

    print(f"[Render] Processing image: {image_id}{progress_str}", flush=True)

    from worker.config import QA_MODE

    qa_mode_resolved = job_data.get("qaMode") or QA_MODE

    if qa_mode_resolved == "auto":
        from worker.config import QA_CONFIG

        provider = job_data.get("qaProvider") or getattr(QA_CONFIG, "provider", None)
        has_vlm = job_data.get("qaVlmModel") or getattr(QA_CONFIG, "vlm_model", None)
        has_llm = job_data.get("qaLlmModel") or getattr(QA_CONFIG, "llm_model", None)
        if has_vlm and provider:
            qa_mode_resolved = "vlm"
        elif has_llm and provider:
            qa_mode_resolved = "llm"
        else:
            qa_mode_resolved = "none"

    if not render_image_core(image_id):
        raise Exception("Render failed")

    # Trigger callback
    callback_payload = {"imageId": image_id}
    try:
        res = requests.post(
            f"{CALLBACK_URL}/render", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[Render] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[Render] Failed to post callback: {e}", flush=True)
