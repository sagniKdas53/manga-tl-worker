import requests, base64

invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"

with open("paddleocr1.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

assert (
    len(image_b64) < 180_000
), "To upload larger images, use the assets API (see docs)"

headers = {"Authorization": "Bearer $NVIDIA_API_KEY", "Accept": "application/json"}

payload = {
    "input": [{"type": "image_url", "url": f"data:image/png;base64,{image_b64}"}]
}

response = requests.post(invoke_url, headers=headers, json=payload)

print(response.json())
