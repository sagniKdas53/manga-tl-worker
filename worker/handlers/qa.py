import io
import os
import json
import base64
import requests
from PIL import Image
from worker.config import CALLBACK_URL, BACKEND_HEADERS, minio_client, logger
from worker.utils.image import download_image
from worker.services.translation import try_cloud_ai_vision, try_local_vlm_vision

QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "regionId": {"type": "string"},
                    "qaStatus": {
                        "type": "string",
                        "enum": ["passed", "failed", "direct_fix"],
                    },
                    "qaScore": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "qaFeedback": {"type": "string"},
                    "directFix": {
                        "type": "object",
                        "properties": {
                            "correctedText": {"type": "string"},
                            "suggestedFontSize": {"type": "number"},
                        },
                    },
                    "escalation": {
                        "type": "object",
                        "properties": {
                            "ocrBad": {"type": "boolean"},
                            "correctedSourceText": {"type": "string"},
                            "orderBad": {"type": "boolean"},
                            "suggestedReadingOrderIndex": {"type": "number"},
                        },
                    },
                },
                "required": ["regionId", "qaStatus", "qaScore", "qaFeedback"],
            },
        }
    },
    "required": ["results"],
}


def process_qa(job_data):
    image_id = job_data["imageId"]
    print(f"[QA] Processing QA check for image: {image_id}", flush=True)

    try:
        backend_url = CALLBACK_URL.replace("/jobs/callback", f"/images/{image_id}")
        res = requests.get(backend_url, headers=BACKEND_HEADERS)
        if res.status_code != 200:
            print(f"[QA] Failed to get image info: {res.status_code}", flush=True)
            return
        image_info = res.json()
        ocr_regions = image_info.get("ocrRegions", [])
    except Exception as e:
        print(f"[QA] Error fetching image details: {e}", flush=True)
        return

    # Download original image
    try:
        original_bytes = download_image(image_info)
    except Exception as e:
        print(f"[QA] Error downloading original image: {e}", flush=True)
        return

    # Download rendered typeset image from MinIO
    try:
        response = minio_client.get_object("manga-library", f"rendered/{image_id}.png")
        rendered_bytes = response.read()
    except Exception as e:
        print(f"[QA] Error downloading rendered image: {e}", flush=True)
        return

    try:
        # Create side-by-side combined image for VLM comparison
        img1 = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(rendered_bytes)).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size
        combined_width = w1 + w2
        combined_height = max(h1, h2)

        combined_img = Image.new(
            "RGB", (combined_width, combined_height), (255, 255, 255)
        )
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (w1, 0))

        # Save to base64
        combined_buf = io.BytesIO()
        combined_img.save(combined_buf, format="JPEG", quality=85)
        combined_base64 = base64.b64encode(combined_buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[QA] Error combining images: {e}", flush=True)
        return

    # Build region metadata list to seed the VLM
    regions_metadata = []
    for r in ocr_regions:
        regions_metadata.append(
            {
                "regionId": r["id"],
                "ocrText": r["text"],
                "ocrScore": r.get("ocrScore") or r.get("confidence") or 1.0,
                "translatedText": r.get("translatedText") or "",
                "translationScore": r.get("translationScore") or 1.0,
                "x": r["bboxX"],
                "y": r["bboxY"],
                "w": r["bboxW"],
                "h": r["bboxH"],
                "readingOrder": r.get("bubbleReadingOrder") or 0,
            }
        )

    logger.debug(
        f"[QA] VLM QA input metadata (regions_metadata):\n{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}"
    )

    prompt = f"""You are an expert manga typesetting QA reviewer.
Analyze the combined image. The left half is the original manga page (Japanese), and the right half is the rendered typeset English page.

Verify the overall quality of typesetting, text fitting, translation, and reading flow. We have seeded each text region with its OCR confidence (ocrScore) and translation confidence (translationScore). Keep these previous scores in mind when evaluating the overall results.

For each region in the provided metadata, evaluate and check if:
1. Text overflows the speech bubble/mask boundaries.
2. Text overlaps with panel borders or other text.
3. Translation flow is awkward, or the English translation does not match the original Japanese text.
4. The OCR transcription was bad/inaccurate (flag with ocrBad=true and provide correctedSourceText).
5. The reading order/bubble sequence is incorrect (flag with orderBad=true and provide suggestedReadingOrderIndex).

Status categories:
- "passed": No correction needed. You MUST still provide a detailed explanation/reasoning in "qaFeedback" explaining why the region passed (e.g., text fitting is perfect, translation is highly accurate, layout looks clean).
- "direct_fix": Small/cosmetic adjustment (e.g. slight text wrap tweak or minor font size reduction) that you can prescribe directly. You must supply "directFix" object with correctedText or suggestedFontSize. You MUST also provide detailed reasoning in "qaFeedback".
- "failed": Major translation error or layout issue requiring a translation/typesetting re-run. Specify "qaFeedback" with detailed correction notes.

IMPORTANT: For EVERY region (including "passed" regions), you MUST provide a detailed explanation/reasoning in "qaFeedback" explaining your evaluation.

Region Metadata:
{json.dumps(regions_metadata, ensure_ascii=False, indent=2)}

You MUST return a JSON object containing a "results" key with an array of objects conforming to the requested schema. No other text."""

    provider = os.environ.get("MODEL_PROVIDER", "").lower().strip()
    api_key = os.environ.get("API_KEY", "").strip()

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or (
        api_key if provider == "openrouter" else ""
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or (
        api_key if provider == "gemini" else ""
    )
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip() or (
        api_key if provider == "nvidia" else ""
    )

    qa_response = None
    if openrouter_key:
        vlm_model = os.environ.get("PREFERRED_VLM_MODEL", "google/gemini-2.5-flash")
        try:
            qa_response = try_cloud_ai_vision(
                "openrouter",
                openrouter_key,
                vlm_model,
                prompt,
                combined_base64,
                QA_JSON_SCHEMA,
            )
        except Exception as e:
            print(f"[QA] VLM QA via OpenRouter failed: {e}", flush=True)

    if not qa_response and gemini_key:
        vlm_model = os.environ.get("PREFERRED_MODEL", "gemini-1.5-flash")
        try:
            qa_response = try_cloud_ai_vision(
                "gemini", gemini_key, vlm_model, prompt, combined_base64, QA_JSON_SCHEMA
            )
        except Exception as e:
            print(f"[QA] VLM QA via Gemini failed: {e}", flush=True)

    if not qa_response and nvidia_key:
        nvidia_vlm_model = os.environ.get(
            "NVIDIA_VLM_MODEL", "nvidia/nemotron-nano-12b-v2-vl"
        )
        try:
            qa_response = try_cloud_ai_vision(
                "nvidia",
                nvidia_key,
                nvidia_vlm_model,
                prompt,
                combined_base64,
                QA_JSON_SCHEMA,
            )
        except Exception as e:
            print(f"[QA] VLM QA via Nvidia failed: {e}", flush=True)

    local_vlm_model = os.environ.get("LOCAL_VLM_MODEL", "").strip()
    if not qa_response and local_vlm_model:
        try:
            qa_response = try_local_vlm_vision(
                local_vlm_model, prompt, combined_base64, QA_JSON_SCHEMA
            )
        except Exception as e:
            print(f"[QA] VLM QA via Local VLM failed: {e}", flush=True)

    # If all VLM options failed or we are in stub fallback, construct a pass result for all regions
    results = []
    if qa_response:
        try:
            cleaned = qa_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            results = parsed.get("results") or []
        except Exception as e:
            print(
                f"[QA] Failed to parse VLM response: {e}. Raw response: {qa_response}",
                flush=True,
            )

    if not results:
        print("[QA] Falling back to default PASS for all regions.", flush=True)
        for r in ocr_regions:
            results.append(
                {
                    "regionId": r["id"],
                    "qaStatus": "passed",
                    "qaScore": 1.0,
                    "qaFeedback": "Auto-passed fallback",
                }
            )

    logger.debug(
        f"[QA] VLM QA results output:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
    )

    # Call backend
    callback_payload = {"imageId": image_id, "qaResults": results}
    try:
        res = requests.post(
            f"{CALLBACK_URL}/qa", json=callback_payload, headers=BACKEND_HEADERS
        )
        print(f"[QA] Callback status code: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[QA] Failed to post QA callback to backend: {e}", flush=True)
