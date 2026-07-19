from ultralytics import YOLO
import sys

model_path = "/home/sagnik/Projects/docker-composes/manga-library/unified-workers/models_backup/models--ShadowB--Manga109-panel-balloon-text-yolov26-segmentation/snapshots/3a860269ee0beb43ce9f31d82c7851441eb178ae/best.pt"
try:
    model = YOLO(model_path)
    print("Class names:", model.names)
except Exception as e:
    print("Error:", e)
