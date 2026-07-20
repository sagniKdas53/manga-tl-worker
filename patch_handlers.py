import re
import glob

for handler_file in glob.glob("worker/handlers/*.py"):
    with open(handler_file, "r") as f:
        content = f.read()

    # Extract routing strategy from job_data if it exists
    # If the file handles job processing, job_data is usually available.
    
    # In translation.py:
    if "translation.py" in handler_file:
        content = content.replace(
            "tl_model = job_data.get(\"tlModel\")",
            "tl_model = job_data.get(\"tlModel\")\n    routing_strategy = job_data.get(\"routingStrategy\")"
        )
        content = content.replace(
            "try_cloud_ai(prov, api_key, tl_model, final_prompt, TRANSLATION_JSON_SCHEMA, request_id)",
            "try_cloud_ai(prov, api_key, tl_model, final_prompt, TRANSLATION_JSON_SCHEMA, request_id, routing_strategy=routing_strategy)"
        )
        content = content.replace(
            "try_cloud_ai_vision_batch(prov, api_key, tl_model, final_prompt, batch_data, TRANSLATION_JSON_SCHEMA, request_id)",
            "try_cloud_ai_vision_batch(prov, api_key, tl_model, final_prompt, batch_data, TRANSLATION_JSON_SCHEMA, request_id, routing_strategy=routing_strategy)"
        )

    # In ocr.py:
    if "ocr.py" in handler_file:
        content = content.replace(
            "ocr_model = job_data.get(\"ocrModel\")",
            "ocr_model = job_data.get(\"ocrModel\")\n    routing_strategy = job_data.get(\"routingStrategy\")"
        )
        # Note: ocr.py uses try_cloud_ocr which is inside worker/services/ocr.py!
        content = content.replace(
            "try_cloud_ocr(crop_bytes, provider, api_key, ocr_model, qa_feedback=qa_feedback)",
            "try_cloud_ocr(crop_bytes, provider, api_key, ocr_model, qa_feedback=qa_feedback, routing_strategy=routing_strategy)"
        )

    # In qa.py:
    if "qa.py" in handler_file:
        content = content.replace(
            "llm_model = job_data.get(\"qaLlmModel\")",
            "llm_model = job_data.get(\"qaLlmModel\")\n    routing_strategy = job_data.get(\"routingStrategy\")"
        )
        content = content.replace(
            "try_cloud_ai(\"openrouter\", api_key, llm_model, prompt, QA_JSON_SCHEMA)",
            "try_cloud_ai(\"openrouter\", api_key, llm_model, prompt, QA_JSON_SCHEMA, routing_strategy=routing_strategy)"
        )
        content = content.replace(
            "try_cloud_ai(prov, api_key, llm_model, prompt, QA_JSON_SCHEMA)",
            "try_cloud_ai(prov, api_key, llm_model, prompt, QA_JSON_SCHEMA, routing_strategy=routing_strategy)"
        )
        content = content.replace(
            "try_cloud_ai_vision(\n                        \"openrouter\",",
            "try_cloud_ai_vision(\n                        \"openrouter\","
        )
        # We can just replace all try_cloud_ai_vision with routing_strategy
        content = content.replace(
            "mime_type=\"image/jpeg\",\n                        response_schema=QA_JSON_SCHEMA,\n                    )",
            "mime_type=\"image/jpeg\",\n                        response_schema=QA_JSON_SCHEMA,\n                        routing_strategy=routing_strategy,\n                    )"
        )

    with open(handler_file, "w") as f:
        f.write(content)

