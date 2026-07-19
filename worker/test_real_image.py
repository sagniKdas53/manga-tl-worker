import onnxruntime as ort
import numpy as np
import cv2
import urllib.request

session = ort.InferenceSession("/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface/models/yolo26s_manga109.onnx")

req = urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Manga_style_drawing.jpg/800px-Manga_style_drawing.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

h, w = img.shape[:2]
scale = 1280 / max(h, w)
nh, nw = int(h * scale), int(w * scale)
img_resized = cv2.resize(img, (nw, nh))
padded_img = np.zeros((1280, 1280, 3), dtype=np.uint8)
padded_img[:nh, :nw] = img_resized

input_tensor = padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

out = session.run(None, {session.get_inputs()[0].name: input_tensor})
preds = out[0][0]
print("Shape:", preds.shape)
for i in range(preds.shape[0]):
    score = float(preds[i, 4])
    if score > 0.2:
        print(f"Det {i}: score={score:.3f}, class={preds[i, 5]}, box={preds[i, 0:4]}")
