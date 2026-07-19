import onnxruntime as ort
import numpy as np
session = ort.InferenceSession("/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface/models/yolo26s_manga109.onnx")
x = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
out = session.run(None, {session.get_inputs()[0].name: x})
print("preds shape:", out[0].shape)
print("first det:", out[0][0, 0, :])
