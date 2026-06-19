import io
import requests
from PIL import Image, ImageDraw, ImageFont
from worker.config import CALLBACK_URL, BACKEND_HEADERS, minio_client, logger
from worker.utils.image import download_image
import os


def load_font(size, bold=False, italic=False):
    # Try common font paths on Debian/Ubuntu/Alpine systems
    # Bold and Italic fonts
    if bold and italic:
        suffixes = ["BoldItalic.ttf", "-BoldItalic.ttf", "BI.ttf"]
    elif bold:
        suffixes = ["Bold.ttf", "-Bold.ttf", "B.ttf"]
    elif italic:
        suffixes = ["Italic.ttf", "-Italic.ttf", "I.ttf"]
    else:
        suffixes = ["Regular.ttf", ".ttf", "R.ttf"]

    font_names = ["DejaVuSans", "LiberationSans", "FreeSans", "Arial"]

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
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, int(size))
            except Exception:
                pass

    # Fallback system names
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


def process_render(job_data):
    image_id = job_data["imageId"]
    print(f"[Render] Processing image: {image_id}", flush=True)

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[Render] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        layer_elements = image_info.get("layerElements", [])
    except Exception as e:
        print(f"[Render] Error fetching image details: {e}", flush=True)
        return

    try:
        img_bytes = download_image(image_info)
    except Exception as e:
        print(f"[Render] Error downloading image: {e}", flush=True)
        return

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # We render elements belonging to 'translation' layer
        translation_elements = [el for el in layer_elements if el.get("visible", True)]

        for el in translation_elements:
            text = el.get("text", "")
            if not text:
                continue

            ex = float(el.get("x", 0.0))
            ey = float(el.get("y", 0.0))
            ew = int(el.get("maxWidth") or 100)
            eh = int(el.get("maxHeight") or 50)

            bg_color_hex = el.get("backgroundColor")
            text_color_hex = el.get("textColor") or "#000000"

            box_shape = el.get("boxShape") or "rectangular"
            font_size = float(el.get("size") or 12.0)
            font_weight = el.get("fontWeight") or "normal"
            font_style = el.get("fontStyle") or "normal"

            bold = "bold" in font_weight.lower()
            italic = "italic" in font_style.lower()

            # Masking
            if bg_color_hex and bg_color_hex.startswith("#"):
                # Draw mask
                if box_shape == "elliptical":
                    draw.ellipse([ex, ey, ex + ew, ey + eh], fill=bg_color_hex)
                else:
                    draw.rectangle([ex, ey, ex + ew, ey + eh], fill=bg_color_hex)

            # Draw Text
            font = load_font(font_size, bold=bold, italic=italic)
            if font:
                draw_wrapped_text(
                    draw, text, font, text_color_hex, ex, ey, ew, eh, alignment="center"
                )

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

    except Exception as e:
        print(f"[Render] Error rendering typeset: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return

    # Trigger callback
    callback_payload = {"imageId": image_id}
    try:
        res = requests.post(
            f"{CALLBACK_URL}/render", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[Render] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[Render] Failed to post callback: {e}", flush=True)
